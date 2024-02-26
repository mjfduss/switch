import evaluate
import numpy as np
from transformers import DataCollatorForSeq2Seq
from datasets import load_dataset

def get_translation_dataset(model, tokenizer):
    # https://huggingface.co/docs/transformers/tasks/translation
    print("Loading the Opus Books dataset...")
    books = load_dataset("opus_books", "en-fr")
    books = books["train"].train_test_split(test_size=0.2)
    
    print("Preprocessing dataset...")
    preprocess_translation_dataset = _load_tokenizer(tokenizer)
    tokenized_books = books.map(preprocess_translation_dataset, batched=True)

    print("Setting data collator...")
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    compute_metrics_function = _load_translation_metric()
    
    batch_size = 16

    return tokenized_books, data_collator, compute_metrics_function, batch_size

def _load_tokenizer(tokenizer):
    def _preprocess_translation_dataset(examples):
        source_lang = "en"
        target_lang = "fr"
        prefix = "translate English to French: "

        inputs = [prefix + example[source_lang] for example in examples["translation"]]
        targets = [example[target_lang] for example in examples["translation"]]
        model_inputs = tokenizer(inputs, text_target=targets, max_length=128, truncation=True)
        return model_inputs
    return _preprocess_translation_dataset


def _postprocess_translation_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels

def _load_translation_metric():
    print("Loading evaluation metric...")
    metric = evaluate.load("sacrebleu")

    def _compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds, decoded_labels = _postprocess_translation_text(decoded_preds, decoded_labels)

        result = metric.compute(predictions=decoded_preds, references=decoded_labels)
        result = {"bleu": result["score"]}

        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result

    return _compute_metrics


def get_summarization_dataset(model, tokenizer):
    print("Loading the Bill Sum dataset...")
    billsum = load_dataset("billsum")
    
    print("Preprocessing dataset...")
    preprocess_function = _load_tokenizer_summarization(tokenizer)
    tokenized_billsum = billsum.map(preprocess_function, batched=True)
    
    print("Setting data collator...")
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    compute_metrics_function = _load_summarization_metric(tokenizer)

    # Will run out of memory on 24GB Nvidia cards if batch size is 8 or higher
    batch_size = 4

    return tokenized_billsum, data_collator, compute_metrics_function, batch_size



def _load_tokenizer_summarization(tokenizer):
    prefix = "summarize: "
    def _preprocess_summarization_dataset(examples):
        inputs = [prefix + doc for doc in examples["text"]]
        model_inputs = tokenizer(inputs, max_length=1024, truncation=True)

        labels = tokenizer(text_target=examples["summary"], max_length=128, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    return _preprocess_summarization_dataset


def _load_summarization_metric(tokenizer):
    print("Loading evaluation metric...")
    rouge = evaluate.load("rouge")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
        result["gen_len"] = np.mean(prediction_lens)

        return {k: round(v, 4) for k, v in result.items()}

    return compute_metrics