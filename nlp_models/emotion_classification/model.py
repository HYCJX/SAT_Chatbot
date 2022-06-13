import json
import logging
import optuna
import os
import torch

from sklearn.metrics import mean_squared_error
from torch.utils.data import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from typing import Optional

from emotion_datasets.sst import DataCollator, StanfordSentimentTreebank

logging.basicConfig(
    filename='sentiment_analysis.log',
    filemode='w',
    level=logging.DEBUG,
    format='%(name)s - %(levelname)s - %(message)s',
    datefmt="%Y-%m-%d %H:%M:%S",
)


def compute_metrics(evaluation_predictions) -> dict:
    predictions, labels = evaluation_predictions
    mse = mean_squared_error(labels, predictions)
    return {"MSE": mse}


class SentimentModelTrainer():
    def __init__(self,
                 model_checkpoint: str,
                 dataset_train: Dataset,
                 dataset_valid: Dataset,
                 dataset_test: Dataset,
                 output_dir: Optional[str] = "sentiment_analysis_outputs") -> None:
        self.model_checkpoint = model_checkpoint
        self.model = None
        self.model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint,
                                                                        num_labels=1)
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint,
                                                       use_fast=True)
        self.data_collator = DataCollator(self.tokenizer)
        self.best_params = {}
        self.dataset_train = dataset_train
        self.dataset_valid = dataset_valid
        self.dataset_test = dataset_test
        self.output_dir = output_dir

    def tune_hyperparameters(self):
        def objective(trial: optuna.Trial):
            training_args = TrainingArguments(
                num_train_epochs=trial.suggest_int('num_train_epochs',
                                                   low=2,
                                                   high=10),
                learning_rate=trial.suggest_loguniform("learning_rate",
                                                       low=1e-5,
                                                       high=1e-3),
                warmup_ratio=trial.suggest_loguniform("warmup_ratio",
                                                      low=0.01,
                                                      high=0.1),
                weight_decay=trial.suggest_loguniform('weight_decay',
                                                      0.001,
                                                      0.1),
                max_grad_norm=1.0,
                output_dir=self.output_dir,
                per_device_train_batch_size=16,
                per_device_eval_batch_size=32,
            )
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=self.dataset_train,
                eval_dataset=self.dataset_valid,
                data_collator=self.data_collator,
                compute_metrics=compute_metrics,
            )
            trainer.train()
            evaluation_metrics = trainer.evaluate()
            return evaluation_metrics["eval_MSE"]

        logging.info(f"Fintuning RoBERTa model ...")
        study = optuna.create_study(study_name='hyper-parameter-search',
                                    direction='minimize')
        study.optimize(func=objective, n_trials=25)
        logging.info(study.best_value)
        logging.info(study.best_params)
        logging.info(study.best_trial)
        self.best_params = study.best_params

    def train(self):
        training_args = TrainingArguments(
            num_train_epochs=self.best_params["num_train_epochs"],
            learning_rate=self.best_params["learning_rate"],
            warmup_ratio=self.best_params["warmup_ratio"],
            weight_decay=self.best_params["weight_decay"],
            max_grad_norm=1.0,
            output_dir=self.output_dir,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=32,
        )
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.dataset_train,
            eval_dataset=self.dataset_valid,
            data_collator=self.data_collator,
            compute_metrics=compute_metrics,
        )
        logging.info(f"Training RoBERTa model ...")
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
        self.model.save_pretrained(
            f"{self.output_dir}/{checkpoint_name}.ckpt"
        )
        logging.info(
            "*"*10 + "Current best checkpoint is saved." + "*"*10
        )


if __name__ == "__main__":
    model_list = ["albert-base-v2",
                  "roberta-base",
                  "facebook/muppet-roberta-base",
                  "xlnet-base-cased",
                  "roberta-large",
                  "facebook/muppet-roberta-large"]
    for model in model_list:
        sentiment_model_trainer = SentimentModelTrainer(model,
                                                        StanfordSentimentTreebank(
                                                            "train"
                                                        ),
                                                        StanfordSentimentTreebank(
                                                            "validation"
                                                        ),
                                                        StanfordSentimentTreebank(
                                                            "test"
                                                        ))
        sentiment_model_trainer.tune_hyperparameters()
        sentiment_model_trainer.train()
