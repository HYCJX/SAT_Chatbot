import json
import logging
import numpy as np
import optuna
import os

from torch.utils.data import Dataset
from sklearn.metrics import f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollator, Trainer, TrainingArguments
from typing import Optional

from emotion_cause_datasets.emocause_cl import NUM_LABELS
from emotion_cause_datasets.emocause_cl import EmocauseClDataset, EmocauseDataCollator

logging.basicConfig(
    filename="emotion_classification.log",
    filemode="w",
    level=logging.INFO,
    format="%(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
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


class EmocauseRecognitionTrainer():
    def __init__(self,
                 model_type: str,
                 num_labels: int,
                 dataset_train: Dataset,
                 dataset_valid: Dataset,
                 dataset_test: Dataset,
                 batch_size: Optional[int] = 64,
                 output_dir: Optional[str] = "emocause_classifier_outputs") -> None:
        self.model_type = model_type
        self.tokenizer = AutoTokenizer.from_pretrained(model_type,
                                                       use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_type,
                                                                        num_labels=num_labels)
        self.best_params = {
            "learning_rate": 1e-5,
            "warmup_ratio": 0.2,
            "weight_decay": 0.01
        }
        self.dataset_train = dataset_train
        self.dataset_valid = dataset_valid
        self.dataset_test = dataset_test
        self.data_collator = EmocauseDataCollator(self.tokenizer)
        self.batch_size = batch_size
        self.output_dir = output_dir

    def tune_hyperparameters(self):
        def objective(trial: optuna.Trial):
            training_args = TrainingArguments(num_train_epochs=10,
                                              learning_rate=trial.suggest_loguniform("learning_rate",
                                                                                     low=5e-7,
                                                                                     high=5e-4),
                                              warmup_ratio=trial.suggest_uniform("warmup_ratio",
                                                                                 low=0.1,
                                                                                 high=0.3),
                                              weight_decay=trial.suggest_loguniform("weight_decay",
                                                                                    0.001,
                                                                                    0.1),
                                              max_grad_norm=1.0,
                                              output_dir=self.output_dir,
                                              per_device_eval_batch_size=self.batch_size,
                                              per_device_train_batch_size=self.batch_size)
            trainer = Trainer(model=self.model,
                              args=training_args,
                              train_dataset=self.dataset_train,
                              eval_dataset=self.dataset_valid,
                              tokenizer=self.tokenizer,
                              data_collator=self.data_collator,
                              compute_metrics=compute_metrics)
            trainer.train()
            evaluation_metrics = trainer.evaluate()
            return evaluation_metrics["eval_f1_weighted"]

        logging.info(f"Fintuning {self.model_type} model ...")
        study = optuna.create_study(study_name="hyper-parameter-search",
                                    direction="maximize")
        study.optimize(func=objective, n_trials=25)
        logging.info(study.best_value)
        logging.info(study.best_params)
        logging.info(study.best_trial)
        self.best_params = study.best_params

    def train(self):
        training_args = TrainingArguments(num_train_epochs=10,
                                          learning_rate=self.best_params["learning_rate"],
                                          warmup_ratio=self.best_params["warmup_ratio"],
                                          weight_decay=self.best_params["weight_decay"],
                                          max_grad_norm=1.0,
                                          output_dir=self.output_dir,
                                          per_device_train_batch_size=self.batch_size,
                                          per_device_eval_batch_size=self.batch_size,
                                          evaluation_strategy="epoch",
                                          logging_strategy="epoch",
                                          save_strategy="epoch",
                                          metric_for_best_model="eval_f1_weighted",
                                          greater_is_better=True,
                                          load_best_model_at_end=True)
        trainer = Trainer(model=self.model,
                          args=training_args,
                          train_dataset=self.dataset_train,
                          eval_dataset=self.dataset_valid,
                          tokenizer=self.tokenizer,
                          data_collator=self.data_collator,
                          compute_metrics=compute_metrics)
        logging.info(f"Training {self.model_type} model ...")
        trainer.train()
        logging.info(f"Evaluating ...")
        val_results = trainer.evaluate()
        checkpoint_name = self.model_type.replace("/", "")
        with open(os.path.join(self.output_dir, f"{checkpoint_name}__val-results.json"), "w") as stream:
            json.dump(val_results, stream, indent=4)
        logging.info(f"Evaluation results: {val_results}")
        logging.info(f"Testing ...")
        test_results = trainer.predict(self.dataset_test)
        with open(os.path.join(self.output_dir, f"{checkpoint_name}_test-results.json"), "w") as stream:
            json.dump(test_results.metrics, stream, indent=4)
        logging.info(f"Test results: {test_results.metrics}")
        trainer.save_model(f"{self.output_dir}/{checkpoint_name}")
        logging.info(
            "*"*10 + "Current best checkpoint is saved." + "*"*10
        )


if __name__ == "__main__":
    trainer = EmocauseRecognitionTrainer(model_type="roberta-base",
                                         num_labels=NUM_LABELS,
                                         dataset_train=EmocauseClDataset(
                                             "train"),
                                         dataset_valid=EmocauseClDataset(
                                             "valid"),
                                         dataset_test=EmocauseClDataset(
                                             "test"))
    trainer.train()
