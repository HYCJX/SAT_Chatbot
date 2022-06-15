import json
import logging
import numpy as np
import optuna
import os

from sklearn.metrics import mean_squared_error, f1_score
from torch.utils.data import Dataset, Subset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollator, Trainer, TrainingArguments
from typing import Optional

from emotion_datasets.sst import SSTDataCollator, SST2DataCollator, StanfordSentimentTreebank, StanfordSentimentTreebankV2

logging.basicConfig(
    filename="sentiment_analysis.log",
    filemode="w",
    level=logging.INFO,
    format="%(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)


def compute_metrics(evaluation_predictions) -> dict:
    predictions, labels = evaluation_predictions
    mse = mean_squared_error(labels, predictions)
    return {"MSE": mse}


def compute_classification_metrics(evaluation_predictions) -> dict:
    predictions, labels = evaluation_predictions
    preds = np.argmax(predictions, axis=1)
    f1_weighted = f1_score(labels, preds, average="weighted")
    f1_micro = f1_score(labels, preds, average="micro")
    f1_macro = f1_score(labels, preds, average="macro")
    return {"f1_weighted": f1_weighted, "f1_micro": f1_micro, "f1_macro": f1_macro}


class SentimentModelTrainer():
    def __init__(self,
                 model_checkpoint: str,
                 num_labels: int,
                 dataset_train: Dataset,
                 dataset_valid: Dataset,
                 dataset_test: Dataset,
                 is_classification: Optional[bool] = False,
                 output_dir: Optional[str] = "sentiment_analysis_outputs") -> None:
        self.model_checkpoint = model_checkpoint
        self.is_classification = is_classification
        self.model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint,
                                                                        num_labels=num_labels)
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint,
                                                       use_fast=True)
        self.data_collator = SST2DataCollator(self.tokenizer)
        self.best_params = {
            "learning_rate": 1e-5,
            "warmup_ratio": 0.2,
            "weight_decay": 0.01
        }
        self.dataset_train = dataset_train
        self.dataset_valid = dataset_valid
        self.dataset_test = dataset_test
        self.output_dir = output_dir

    def tune_hyperparameters(self):
        def objective(trial: optuna.Trial):
            training_args = TrainingArguments(num_train_epochs=10,
                                              learning_rate=trial.suggest_uniform("learning_rate",
                                                                                  low=1e-5,
                                                                                  high=5e-5),
                                              warmup_ratio=trial.suggest_uniform("warmup_ratio",
                                                                                 low=0.1,
                                                                                 high=0.3),
                                              weight_decay=trial.suggest_loguniform("weight_decay",
                                                                                    0.001,
                                                                                    0.1),
                                              max_grad_norm=1.0,
                                              output_dir=self.output_dir,
                                              per_device_train_batch_size=16,
                                              per_device_eval_batch_size=32,
                                              )
            dataset_finetune = Subset(self.dataset_train,
                                      [*range(0, 1000, 1)])
            if not self.is_classification:
                trainer = Trainer(model=self.model,
                                  args=training_args,
                                  train_dataset=dataset_finetune,
                                  eval_dataset=self.dataset_valid,
                                  data_collator=self.data_collator,
                                  compute_metrics=compute_metrics)
            else:
                trainer = Trainer(model=self.model,
                                  args=training_args,
                                  train_dataset=dataset_finetune,
                                  eval_dataset=self.dataset_valid,
                                  data_collator=self.data_collator,
                                  compute_metrics=compute_classification_metrics)
            trainer.train()
            evaluation_metrics = trainer.evaluate()
            if not self.is_classification:
                return evaluation_metrics["eval_MSE"]
            else:
                return evaluation_metrics["eval_f1_weighted"]

        logging.info(f"Fintuning {self.model_checkpoint} model ...")
        direction = "maximize" if self.is_classification else "minimize"
        study = optuna.create_study(study_name="hyper-parameter-search",
                                    direction=direction)
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
                                          per_device_train_batch_size=16,
                                          per_device_eval_batch_size=32,
                                          evaluation_strategy="epoch",
                                          logging_strategy="epoch",
                                          save_strategy="epoch",
                                          metric_for_best_model="eval_f1_weighted",
                                          greater_is_better=True,
                                          load_best_model_at_end=True)
        if not self.is_classification:
            trainer = Trainer(model=self.model,
                              args=training_args,
                              train_dataset=self.dataset_train,
                              eval_dataset=self.dataset_valid,
                              data_collator=self.data_collator,
                              compute_metrics=compute_metrics)
        else:
            trainer = Trainer(model=self.model,
                              args=training_args,
                              train_dataset=self.dataset_train,
                              eval_dataset=self.dataset_valid,
                              data_collator=self.data_collator,
                              compute_metrics=compute_classification_metrics)
        logging.info(f"Training {self.model_checkpoint} model ...")
        trainer.train()
        logging.info(f"Evaluating ...")
        val_results = trainer.evaluate()
        checkpoint_name = self.model_checkpoint.replace("/", "")
        with open(os.path.join(self.output_dir, f"{checkpoint_name}_val-results.json"), "w") as stream:
            json.dump(val_results, stream, indent=4)
        logging.info(f"Evaluation results: {val_results}")
        logging.info(f"Testing ...")
        test_results = trainer.predict(self.dataset_test)
        with open(os.path.join(self.output_dir, f"{checkpoint_name}_test-results.json"), "w") as stream:
            json.dump(test_results.metrics, stream, indent=4)
        logging.info(f"Test results: {test_results.metrics}")
        trainer.save_model(
            f"{self.output_dir}/{checkpoint_name}"
        )
        logging.info(
            "*"*10 + "Current best checkpoint is saved." + "*"*10
        )


if __name__ == "__main__":
    model_list = ["albert-base-v2",
                  "roberta-base",
                  "facebook/muppet-roberta-base",
                  "xlnet-base-cased",
                  "roberta-large"]
    for model in model_list:
        sentiment_model_trainer = SentimentModelTrainer(model_checkpoint=model,
                                                        num_labels=2,
                                                        dataset_train=StanfordSentimentTreebankV2(
                                                            "train"
                                                        ),
                                                        dataset_valid=StanfordSentimentTreebankV2(
                                                            "validation"
                                                        ),
                                                        dataset_test=StanfordSentimentTreebankV2(
                                                            "test"
                                                        ),
                                                        is_classification=True)
        sentiment_model_trainer.train()
