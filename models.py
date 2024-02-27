import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers.models.switch_transformers.modeling_switch_transformers import SwitchTransformersLayerFF, SwitchTransformersLayerSelfAttention, SwitchTransformersLayerCrossAttention, SwitchTransformersAttention

from qualcomm_functions import clipped_softmax


def get_text_model(dense=False, num_experts=8):
    # Pretrained Switch Transformers Model from HuggingFace
    # https://huggingface.co/collections/google/switch-transformers-release-6548c35c6507968374b56d1f

    print("Loading model into memory...")

    if dense:
        checkpoint = "google-t5/t5-small"
    else:
        checkpoint = f"google/switch-base-{num_experts}"
    
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=True)
    # For finetuning non NLP tasks, can change below to use plain SwitchTransformersModel
    # which is the bare transformer outputting raw hidden-states without any specific head on top
    # SwitchTransformersForConditionalGeneration is specific to text tasks
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
    
    
    return model, tokenizer

def freeze_mixture_of_experts(model):
    # Freeze the Mixture of Experts SwitchTransformer Forward layer in the decoder
    # found to be best for finetuning: https://huggingface.co/blog/moe#fine-tuning-moes
    print("Freezing MOE Layers...")
    for switch_block in model.decoder.block:
        # Each decoder block has a Self Attention layer, a Cross Attention layer, and a Fully Connected layer
        for switch_layer in switch_block.layer: 
            if isinstance(switch_layer, SwitchTransformersLayerFF):
                if switch_layer.is_sparse:
                    # layer contains a Mixture of Experts
                    for param in switch_layer.parameters():
                        # Set the pytorch no gradient flag
                        param.requires_grad = False
    return model

def prepare_for_quantization(model):
    # replace the forward function in the attention layers with the forward_with_clipped_softmax function
    for switch_block in model.encoder.block:
        for switch_layer in switch_block.layer:
            if isinstance(switch_layer, SwitchTransformersLayerSelfAttention):
                funcType = type(switch_layer.SelfAttention.forward) 
                switch_layer.SelfAttention.forward = funcType(forward_with_clipped_softmax, switch_layer.SelfAttention, SwitchTransformersAttention)
            elif isinstance(switch_layer, SwitchTransformersLayerCrossAttention):
                funcType = type(switch_layer.EncDecAttention.forward)
                switch_layer.EncDecAttention.forward = funcType(forward_with_clipped_softmax, switch_layer.EncDecAttention, SwitchTransformersAttention)
    for switch_block in model.decoder.block:
        for switch_layer in switch_block.layer:
            if isinstance(switch_layer, SwitchTransformersLayerSelfAttention):
                funcType = type(switch_layer.SelfAttention.forward) 
                switch_layer.SelfAttention.forward = funcType(forward_with_clipped_softmax, switch_layer.SelfAttention, SwitchTransformersAttention)
            elif isinstance(switch_layer, SwitchTransformersLayerCrossAttention):
                funcType = type(switch_layer.EncDecAttention.forward)
                switch_layer.EncDecAttention.forward = funcType(forward_with_clipped_softmax, switch_layer.EncDecAttention, SwitchTransformersAttention)


def forward_with_clipped_softmax(
    self,
    hidden_states,
    mask=None,
    key_value_states=None,
    position_bias=None,
    past_key_value=None,
    layer_head_mask=None,
    query_length=None,
    use_cache=False,
    output_attentions=False,
):
    """
    Self-attention (if key_value_states is None) or attention over source sentence (provided by key_value_states).
    """
    # Input is (batch_size, seq_length, dim)
    # Mask is (batch_size, key_length) (non-causal) or (batch_size, key_length, key_length)
    # past_key_value[0] is (batch_size, n_heads, q_len - 1, dim_per_head)
    batch_size, seq_length = hidden_states.shape[:2]

    real_seq_length = seq_length

    if past_key_value is not None:
        if len(past_key_value) != 2:
            raise ValueError(
                f"past_key_value should have 2 past states: keys and values. Got { len(past_key_value)} past states"
            )
        real_seq_length += past_key_value[0].shape[2] if query_length is None else query_length

    key_length = real_seq_length if key_value_states is None else key_value_states.shape[1]

    def shape(states):
        """projection"""
        return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)

    def unshape(states):
        """reshape"""
        return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)

    def project(hidden_states, proj_layer, key_value_states, past_key_value):
        """projects hidden states correctly to key/query states"""
        if key_value_states is None:
            # self-attn
            # (batch_size, n_heads, seq_length, dim_per_head)
            hidden_states = shape(proj_layer(hidden_states))
        elif past_key_value is None:
            # cross-attn
            # (batch_size, n_heads, seq_length, dim_per_head)
            hidden_states = shape(proj_layer(key_value_states))

        if past_key_value is not None:
            if key_value_states is None:
                # self-attn
                # (batch_size, n_heads, key_length, dim_per_head)
                hidden_states = torch.cat([past_key_value, hidden_states], dim=2)
            elif past_key_value.shape[2] != key_value_states.shape[1]:
                # checking that the `sequence_length` of the `past_key_value` is the same as
                # the provided `key_value_states` to support prefix tuning
                # cross-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(key_value_states))
            else:
                # cross-attn
                hidden_states = past_key_value
        return hidden_states

    # get query states
    query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)

    # get key/value states
    key_states = project(
        hidden_states, self.k, key_value_states, past_key_value[0] if past_key_value is not None else None
    )
    value_states = project(
        hidden_states, self.v, key_value_states, past_key_value[1] if past_key_value is not None else None
    )

    # compute scores
    scores = torch.matmul(
        query_states, key_states.transpose(3, 2)
    )  # equivalent of torch.einsum("bnqd,bnkd->bnqk", query_states, key_states), compatible with onnx op>9

    if position_bias is None:
        if not self.has_relative_attention_bias:
            position_bias = torch.zeros(
                (1, self.n_heads, real_seq_length, key_length), device=scores.device, dtype=scores.dtype
            )
            if self.gradient_checkpointing and self.training:
                position_bias.requires_grad = True
        else:
            position_bias = self.compute_bias(real_seq_length, key_length, device=scores.device)

        # if key and values are already calculated
        # we want only the last query position bias
        if past_key_value is not None:
            position_bias = position_bias[:, :, -hidden_states.size(1) :, :]

        if mask is not None:
            position_bias = position_bias + mask  # (batch_size, n_heads, seq_length, key_length)

    if self.pruned_heads:
        mask = torch.ones(position_bias.shape[1])
        mask[list(self.pruned_heads)] = 0
        position_bias_masked = position_bias[:, mask.bool()]
    else:
        position_bias_masked = position_bias

    scores += position_bias_masked
    ######################################################################
    #       Replace softmax with clipped_softmax
    ######################################################################
    attn_weights = clipped_softmax(scores.float(), dim=-1).type_as(
        scores
    )  # (batch_size, n_heads, seq_length, key_length)
    attn_weights = nn.functional.dropout(
        attn_weights, p=self.dropout, training=self.training
    )  # (batch_size, n_heads, seq_length, key_length)

    # Mask heads if we want to
    if layer_head_mask is not None:
        attn_weights = attn_weights * layer_head_mask
    
    attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    attn_output = self.o(attn_output)

    present_key_value_state = (key_states, value_states) if (self.is_decoder and use_cache) else None
    outputs = (attn_output,) + (present_key_value_state,) + (position_bias,)

    if output_attentions:
        outputs = outputs + (attn_weights,)
    return outputs

def test_text_generation(model, tokenizer):
    # Test Text Generation capabilities of a pretrained model
    print("Testing text generation...")
    input_text = "A <extra_id_0> walks into a bar and orders a <extra_id_1> with <extra_id_2> pinch of <extra_id_3>."
    print("\nInput text:", input_text)
    print("...Generating output...")
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(0)
    outputs = model.generate(input_ids, max_length=20)
    token_output = tokenizer.decode(outputs[0])
    output = input_text
    word_index_start = 0
    word_index_end = 0
    for i in range(4):
        token_marker = f"<extra_id_{i}>"
        next_token_marker = f"<extra_id_{i+1}>"
        word_index_start = token_output.index(token_marker) + len(token_marker) + 1
        word_index_end = token_output.index(next_token_marker)
        word = token_output[word_index_start:word_index_end]
        output = output.replace(token_marker, word)

    print("Output text:", output)
    