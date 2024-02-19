from transformers import AutoTokenizer, SwitchTransformersForConditionalGeneration
from transformers.models.switch_transformers.modeling_switch_transformers import SwitchTransformersLayerFF


# Download pretrained Switch Transformers Model from HuggingFace
print("Loading model into memory...")
tokenizer = AutoTokenizer.from_pretrained("google/switch-base-32")
# For finetuning, can change below to use plain SwitchTransformersModel
# which is the bare transformer outputting raw hidden-states without any specific head on top
# SwitchTransformersForConditionalGeneration is specific to text generation tasks
model = SwitchTransformersForConditionalGeneration.from_pretrained("google/switch-base-32", device_map="auto")


# Test Text Generation capabilities of the pretrained model
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



# Freeze the Mixture of Experts SwitchTransformer Forward layer in the decoder
# found to be best for finetuning: https://huggingface.co/blog/moe#fine-tuning-moes
print("\nFreezing MOE Layers...")
for switch_block in model.decoder.block:
    # Each decoder block has a Self Attention layer, a Cross Attention layer, and a Fully Connected layer
    for switch_layer in switch_block.layer: 
        if isinstance(switch_layer, SwitchTransformersLayerFF):
            if switch_layer.is_sparse:
                # layer contains a Mixture of Experts
                for param in switch_layer.parameters():
                    # Set the pytorch no gradient flag
                    param.requires_grad = False
                    



# TODO finetune on datasets
# https://huggingface.co/datasets
# Use device-distributed (Multi-GPU) training api for fine-tuning
# https://huggingface.co/docs/transformers/accelerate
