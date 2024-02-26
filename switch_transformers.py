import argparse
from accelerate import Accelerator
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer

# Local modules
import models
import data



# Get command line arguments
parser = argparse.ArgumentParser(
    prog="SwitchTransformersFineTuner", 
    description="Runs the hugging face transformers Trainer to fine tune the Switch Transformer networks")
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--experts', type=int, default=32)
parser.add_argument('--task', choices=['translation', 'summarization'], default='translation', help='Choice of dataset type to fine tune with')
parser.add_argument('--dense', action='store_true', help='Run the T5 dense, non Switch Transformer network for comparison')
args = parser.parse_args()



# Setup model
model, tokenizer = models.get_text_model(num_experts=args.experts, dense=args.dense)
if not args.dense:
    model = models.freeze_mixture_of_experts(model)


# Finetune on datasets
epochs = args.epochs
run_name = f"switch_base_{args.experts}_{args.task}" if not args.dense else f"t5-small_{args.task}"

if args.task == 'summarization':
    processed_data = data.get_summarization_dataset(model, tokenizer)  
else: 
    processed_data = data.get_translation_dataset(model, tokenizer) 

tokenized_data, data_collator, compute_metrics, batch_size = processed_data


# Use device-distributed (Multi-GPU) training api for fine-tuning
# https://huggingface.co/docs/transformers/accelerate
accelerator = Accelerator(log_with="wandb")
accelerator.init_trackers(
    project_name="thesis2", 
    config={ 
        "model":"switch_transformers_8", 
        "dataset": "summarization_billsum", 
        "epochs": epochs,
        },
    init_kwargs={"wandb": {"entity": "awesomepossum", "name": run_name}}
)


training_args = Seq2SeqTrainingArguments(
    output_dir= f"runs/{run_name}",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=epochs,
    predict_with_generate=True,
    fp16=True,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data["train"],
    eval_dataset=tokenized_data["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

"""
Links for reference

https://huggingface.co/docs/transformers/main/en/quantization#4-bit
https://huggingface.co/docs/transformers/accelerate
https://huggingface.co/docs/transformers/main/en/perf_train_gpu_many
https://github.com/huggingface/notebooks/blob/main/examples/text_classification.ipynb
https://huggingface.co/docs/transformers/main_classes/optimizer_schedules
https://github.com/google-research/t5x/blob/main/t5x/notebooks/training.ipynb
https://docs.wandb.ai/guides/integrations/accelerate
https://github.com/Qualcomm-AI-research/outlier-free-transformers/blob/main/transformers_language/models/softmax.py
https://huggingface.co/datasets/super_glue
"""