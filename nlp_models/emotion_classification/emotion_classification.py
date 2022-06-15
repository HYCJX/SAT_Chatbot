import json
import logging
import numpy as np
import optuna
import os

from sklearn.metrics import f1_score
from torch.utils.data import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollator, Trainer, TrainingArguments
from typing import Optional

from emotion_datasets.meld_dataset import Meld_Dataset, NUM_CLASSES

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


class EmotionClassifierTrainer():
    def __init__(self,
                 model_checkpoint: str,
                 num_labels: int,
                 is_using_augmentation: bool,
                 dataset_hp: Dataset,
                 dataset_train: Dataset,
                 dataset_valid: Dataset,
                 dataset_test: Dataset,
                 data_collator: Optional[DataCollator] = None,
                 batch_size: Optional[int] = 16,
                 output_dir: Optional[str] = "emotion_classifier_outputs") -> None:
        self.model_checkpoint = model_checkpoint
        self.model = None
        self.model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint,
                                                                        num_labels=num_labels)
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint,
                                                       use_fast=True)
        self.best_params = {}
        self.is_using_augmentation = is_using_augmentation
        self.dataset_hp = dataset_hp
        self.dataset_train = dataset_train
        self.dataset_valid = dataset_valid
        self.dataset_test = dataset_test
        self.data_collator = data_collator
        self.batch_size = batch_size
        self.output_dir = output_dir

    def tune_hyperparameters(self):
        def objective(trial: optuna.Trial):
            training_args = TrainingArguments(
                num_train_epochs=trial.suggest_int("num_train_epochs",
                                                   low=10,
                                                   high=15),
                learning_rate=trial.suggest_loguniform("learning_rate",
                                                       low=5e-7,
                                                       high=5e-4),
                warmup_ratio=trial.suggest_uniform("warmup_ratio",
                                                   low=0.1,
                                                   high=0.3),
                weight_decay=0.01,
                max_grad_norm=1.0,
                output_dir=self.output_dir,
                per_device_eval_batch_size=(self.batch_size * 2),
                per_device_train_batch_size=self.batch_size,
            )
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=self.dataset_hp,
                eval_dataset=self.dataset_valid,
                tokenizer=self.tokenizer,
                data_collator=self.data_collator,
                compute_metrics=compute_metrics,
            )
            trainer.train()
            evaluation_metrics = trainer.evaluate()
            return evaluation_metrics["eval_loss"]

        logging.info(f"Fintuning {self.model_checkpoint} model ...")
        study = optuna.create_study(study_name="hyper-parameter-search",
                                    direction="minimize")
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
            weight_decay=0.01,
            max_grad_norm=1.0,
            output_dir=self.output_dir,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=32
        )
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.dataset_train,
            eval_dataset=self.dataset_valid,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            compute_metrics=compute_metrics,
        )
        logging.info(f"Training {self.model_checkpoint} model ...")
        trainer.train()
        logging.info(f"Evaluating ...")
        val_results = trainer.evaluate()
        checkpoint_name = self.model_checkpoint.replace("/", "")
        with open(os.path.join(self.output_dir, f"{checkpoint_name}_{self.is_using_augmentation}_val-results.json"), "w") as stream:
            json.dump(val_results, stream, indent=4)
        logging.info(f"Evaluation results: {val_results}")
        logging.info(f"Testing ...")
        test_results = trainer.predict(self.dataset_test)
        with open(os.path.join(self.output_dir, f"{checkpoint_name}_{self.is_using_augmentation}_test-results.json"), "w") as stream:
            json.dump(test_results.metrics, stream, indent=4)
        logging.info(f"Test results: {test_results.metrics}")
        self.model.save_pretrained(
            f"{self.output_dir}/{checkpoint_name}_{self.is_using_augmentation}"
        )
        logging.info(
            "*"*10 + "Current best checkpoint is saved." + "*"*10
        )


if __name__ == "__main__":
    model_list = ["roberta-base",
                  "facebook/muppet-roberta-base",
                  "xlnet-base-cased"]
    for model in model_list:
        for use_data_augmentation in [False, True]:
            if use_data_augmentation:
                dataset_hp = Meld_Dataset("train",
                                          model_checkpoint=model,
                                          num_future_utterances=1000,
                                          num_past_utterances=1000,
                                          speaker_mode="upper",
                                          hp_up_to=150)
                dataset_train = Meld_Dataset("train",
                                             model_checkpoint=model,
                                             num_future_utterances=1000,
                                             num_past_utterances=1000,
                                             speaker_mode="upper")
            else:
                dataset_hp = Meld_Dataset("train",
                                          model_checkpoint=model,
                                          num_future_utterances=0,
                                          num_past_utterances=0,
                                          speaker_mode=None,
                                          hp_up_to=150)
                dataset_train = Meld_Dataset("train",
                                             model_checkpoint=model,
                                             num_future_utterances=0,
                                             num_past_utterances=0,
                                             speaker_mode=None)
            dataset_valid = Meld_Dataset("val",
                                         model_checkpoint=model,
                                         num_future_utterances=0,
                                         num_past_utterances=0,
                                         speaker_mode=None)
            dataset_test = Meld_Dataset("test",
                                        model_checkpoint=model,
                                        num_future_utterances=0,
                                        num_past_utterances=0,
                                        speaker_mode=None)
            emotion_classifier_trainer = EmotionClassifierTrainer(model_checkpoint=model,
                                                                  num_labels=NUM_CLASSES,
                                                                  is_using_augmentation=use_data_augmentation,
                                                                  dataset_hp=dataset_hp,
                                                                  dataset_train=dataset_train,
                                                                  dataset_valid=dataset_valid,
                                                                  dataset_test=dataset_test,
                                                                  output_dir="emotion_classification_outputs")
            emotion_classifier_trainer.tune_hyperparameters()
            emotion_classifier_trainer.train()
