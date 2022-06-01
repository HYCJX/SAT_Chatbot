import json
import logging
import numpy as np
import os

from datasets import Dataset, load_metric
from transformers import AutoModelForTokenClassification, AutoTokenizer, DataCollatorForTokenClassification, Trainer, TrainingArguments

from emotion_cause_datasets.emocause import EmoCauseDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

label_names = ["O", "I"]
id2label = {str(i): label for i, label in enumerate(label_names)}
label2id = {v: k for k, v in id2label.items()}


def compute_metrics(eval_predictions) -> dict:
    logits, labels = eval_predictions
    predictions = np.argmax(logits, axis=-1)
    io_predictions = []
    io_labels = []
    for prediction, label in zip(predictions, labels):
        for predicted_idx, label_idx in zip(prediction, label):
            if label_idx == -100:
                continue
            io_predictions.append(label_names[predicted_idx])
            io_labels.append(label_names[label_idx])
    metric = load_metric("seqeval")
    return metric.compute(predictions=[io_predictions], references=[io_labels])


def finetune_roberta(
    dataset_train: Dataset,
    dataset_valid: Dataset,
    dataset_test: Dataset,
    batch_size: int,
    learning_rate: float,
    num_train_epoch: int,
    output_dir: str,
    warmup_ratio: float,
    weight_decay: float,
):

    args = TrainingArguments(
        evaluation_strategy="epoch",
        fp16=True,
        greater_is_better=True,
        learning_rate=learning_rate,
        load_best_model_at_end=True,
        logging_strategy="epoch",
        num_train_epochs=num_train_epoch,
        output_dir=output_dir,
        per_device_eval_batch_size=(batch_size * 2),
        per_device_train_batch_size=batch_size,
        save_strategy="epoch",
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
    )

    model = AutoModelForTokenClassification.from_pretrained("roberta-base",
                                                            id2label=id2label,
                                                            label2id=label2id)
    tokenizer = AutoTokenizer.from_pretrained("roberta-base", use_fast=True)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset_train,
        eval_dataset=dataset_valid,
        tokenizer=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=compute_metrics,
    )
    logging.info(f"Training RoBERTa model ...")
    trainer.train()

    logging.info(f"Evaluating ...")
    val_results = trainer.evaluate()
    with open(os.path.join(output_dir, "val-results.json"), "w") as stream:
        json.dump(val_results, stream, indent=4)
    logging.info(f"Evaluation results: {val_results}")

    logging.info(f"Testing ...")
    test_results = trainer.predict(dataset_test)
    with open(os.path.join(output_dir, "test-results.json"), "w") as stream:
        json.dump(test_results.metrics, stream, indent=4)
    logging.info(f"Test results: {test_results.metrics}")


finetune_roberta(
    EmoCauseDataset("train"),
    EmoCauseDataset("test"),
    EmoCauseDataset("test"),
    16,
    1e-5,
    5,
    "EmoCause_outputs",
    0.2,
    0.01,
)
