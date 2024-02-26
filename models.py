from transformers import AutoTokenizer, Adafactor, AutoModelForSeq2SeqLM
from transformers.models.switch_transformers.modeling_switch_transformers import SwitchTransformersLayerFF

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
    #optimizer = Adafactor(model.parameters())
    
    
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
    