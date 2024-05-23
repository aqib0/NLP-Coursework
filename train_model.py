import joblib
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, EarlyStoppingCallback
from transformers import DataCollatorForTokenClassification
import numpy as np
from seqeval.metrics import classification_report, f1_score, accuracy_score
import torch

# Function to load the dataset with a fallback option
def load_dataset_with_fallback(dataset_name, fallback_name):
    try:
        dataset = load_dataset(dataset_name, download_mode="force_redownload")
    except Exception as e:
        print(f"Error loading dataset '{dataset_name}': {e}. Using '{fallback_name}' as a fallback.")
        dataset = load_dataset(fallback_name, download_mode="force_redownload")
    return dataset

def tokenize_and_align_labels(short_dataset, list_name, tokenizer, label_encoding):
    tokenized_inputs = tokenizer(short_dataset["tokens"], truncation=True, is_split_into_words=True)

    labels = []
    for i, label in enumerate(list_name):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(label[word_idx])
            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

def turn_dict_to_list_of_dict(d):
    new_list = []
    for labels, inputs in zip(d["labels"], d["input_ids"]):
        entry = {"input_ids": inputs, "labels": labels}
        new_list.append(entry)
    return new_list

def compute_metrics(p, label_list):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

def main():
    # Load the dataset with a fallback
    dataset = load_dataset_with_fallback("surrey-nlp/PLOD-CW", "conll2003")

    # Load the DistilBERT tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModelForTokenClassification.from_pretrained("distilbert-base-uncased", num_labels=4)

    # Extract the train, validation, and test datasets
    short_dataset = dataset["train"]
    val_dataset = dataset["validation"]
    test_dataset = dataset["test"]

    # Tokenize the input data
    tokenized_input = tokenizer(short_dataset["tokens"], is_split_into_words=True)

    # Example single sentence example
    for token in tokenized_input["input_ids"]:
        print(tokenizer.convert_ids_to_tokens(token))
        break

    # Define the label encoding
    label_encoding = {"B-O": 0, "B-AC": 1, "B-LF": 2, "I-LF": 3}

    # Create the label lists for train, validation, and test datasets
    label_list = [[label_encoding.get(tag, 0) for tag in sample] for sample in short_dataset["ner_tags"]]
    val_label_list = [[label_encoding.get(tag, 0) for tag in sample] for sample in val_dataset["ner_tags"]]
    test_label_list = [[label_encoding.get(tag, 0) for tag in sample] for sample in test_dataset["ner_tags"]]

    # Tokenize and align labels for train, validation, and test datasets
    tokenized_datasets = tokenize_and_align_labels(short_dataset, label_list, tokenizer, label_encoding)
    tokenized_val_datasets = tokenize_and_align_labels(val_dataset, val_label_list, tokenizer, label_encoding)
    tokenized_test_datasets = tokenize_and_align_labels(test_dataset, test_label_list, tokenizer, label_encoding)

    # Convert tokenized datasets
    tokenised_train = turn_dict_to_list_of_dict(tokenized_datasets)
    tokenised_val = turn_dict_to_list_of_dict(tokenized_val_datasets)
    tokenised_test = turn_dict_to_list_of_dict(tokenized_test_datasets)

    # Load the DataCollator
    data_collator = DataCollatorForTokenClassification(tokenizer)

    # Load the metric
    global metric
    metric = load_metric("seqeval")

    # Define training arguments
    args = TrainingArguments(
        f"DistilALBERT-finetuned-NER",
        evaluation_strategy="steps",
        eval_steps=7000,
        save_total_limit=3,
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=6,
        weight_decay=0.001,
        save_steps=35000,
        metric_for_best_model='f1',
        load_best_model_at_end=True
    )

    # Create the Trainer
    trainer = Trainer(
        model,
        args,
        train_dataset=tokenised_train,
        eval_dataset=tokenised_val,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=lambda p: compute_metrics(p, label_list),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    # Train the model
    trainer.train()

    # Save the trained model and tokenizer using joblib
    joblib.dump(model, "trained_model.joblib")
    joblib.dump(tokenizer, "tokenizer.joblib")

    # Prepare the test data for evaluation
    predictions, labels, _ = trainer.predict(tokenised_test)
    predictions = np.argmax(predictions, axis=2)

    # Remove the predictions for the special tokens
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    # Compute the metrics on the test results
    results = metric.compute(predictions=true_predictions, references=true_labels)
    print(results)

if __name__ == "__main__":
    main()

