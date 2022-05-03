import json
import logging
import numpy as np
import os

from sklearn.metrics import f1_score
from torch.utils.data import Dataset
from transformers import AutoModelForSequenceClassification, DataCollator, Trainer, TrainingArguments

from emotion_datasets.meld_dataset import Meld_Dataset, NUM_CLASSES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

def compute_metrics(eval_predictions) -> dict:
    """
    Return f1_weighted, f1_micro, and f1_macro scores.
    """
    predictions, label_ids = eval_predictions
    preds = np.argmax(predictions, axis=1)
    f1_weighted = f1_score(label_ids, preds, average="weighted")
    f1_micro = f1_score(label_ids, preds, average="micro")
    f1_macro = f1_score(label_ids, preds, average="macro")

    return {"f1_weighted": f1_weighted, "f1_micro": f1_micro, "f1_macro": f1_macro}

def finetune_roberta(
    num_classes: int,
    data_collator: DataCollator,
    dataset_train: Dataset,
    dataset_valid: Dataset,
    dataset_test: Dataset,
    batch_size: int,
    learning_rate: float,
    model_checkpoint: str,
    num_train_epoch: int,
    output_dir: str,
    warmup_ratio: float,
    weight_decay: float,
    # speaker_mode: str,
    # num_past_utterances: int,
    # num_future_utterances: int,
):

    args = TrainingArguments(
        evaluation_strategy="epoch",
        fp16=True,
        greater_is_better=True,
        learning_rate=learning_rate,
        load_best_model_at_end=True,
        logging_strategy="epoch",
        metric_for_best_model="eval_f1_weighted",
        num_train_epochs=num_train_epoch,
        output_dir=output_dir,
        per_device_eval_batch_size=(batch_size * 2),
        per_device_train_batch_size=batch_size,
        save_strategy="epoch",
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
    )

    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_classes)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset_train,
        eval_dataset=dataset_valid,
        data_collator=data_collator,
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
    NUM_CLASSES,
    None,
    Meld_Dataset("train"),
    Meld_Dataset("val"),
    Meld_Dataset("test"),
    16,
    1e-6,
    "roberta-base",
    15,
    "outputs",
    0.2,
    0.01,
)