import argparse
import logging
import math
import torch.nn.functional as F
import numpy as np
import os
import random
import sys
import torch

from itertools import chain
from pathlib import Path
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2Tokenizer, get_polynomial_decay_schedule_with_warmup
from typing import Optional, Tuple

from dataset import ResponseGenerationDataset
from dataset import PadCollate


logging.basicConfig(filename="response_generation.log",
                    filemode="w",
                    level=logging.INFO,
                    format="%(name)s - %(levelname)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
logging.info("This will get logged to a file")

SPECIAL_TOKENS = {
    "bos_token": "<bos>",
    "eos_token": "<eos>",
    "additional_special_tokens": ["<speaker1>", "<speaker2>"]
}

SPECIAL_TOKEN_NAMES = ["<bos>", "<eos>", "<speaker1>", "<speaker2>"]


class Trainer():
    def __init__(self,
                 output_dir: str,
                 mode: Optional[str] = "train",
                 dataset_name="empathetic_dialogues",
                 model_type: Optional[str] = "gpt2",
                 checkpoint_path: Optional[str] = None,
                 seed: Optional[float] = 0,
                 max_length: Optional[int] = 1024,
                 max_history: Optional[int] = 5,
                 learning_rate: Optional[float] = 1e-5,
                 warmup_ratio: Optional[float] = 0.1,
                 batch_size: Optional[int] = 8,
                 num_epochs: Optional[int] = 10
                 ):
        """
        Arguments:
            output_dir: Output directory.
            mode: train or infer.
            model_type: Transformer model.
            checkpoint_path: Model checkpoint to start with.
            seed: Seed for random generations in the trainer.
            max_length: Maximum length of transformer inputs.
            max_history: Maximum number of turns of conversations used as history.

            Hyper-parameters:
                learning_rate: Learning rate of the optimizer.
                warmup_ratio: Warm-up hyperparameter of the scheduler.
                batch_size: Batch size of the trainer.
                num_epochs: Number of epochs to train the transformer.

        Class fields:
            self.output_dir: Output directory.
            self.mode: Mode.
            self.device: Device for training (cpu or cuda).
            self.seed: Seed.
            self.tokenizer: Transformer's tokenizer.
            self.model: Transformer Model.
            self.max_length: Maximum length of transformer inputs.
            self.max_history: Maximum number of turns of conversations used as history.
            self.best_loss: Current best loss to start with.
            self.last_epoch: Current epoch to start with.

            Training components:
                self.optimizer: Optimizer.
                self.scheduler: Scheduler.
                self.data_collator: Data collator.
                self.train_loader: Data loader of training data.
                self.valid_loader: Data loader of validation data.

            Hyper-parameters:
                self.learning_rate: Learning rate of the optimizer.
                self.warmup_ratio: Warm-up hyperparameter of the scheduler.
                self.batch_size: Batch size.
                self.num_epochs: Number of epochs to train the transformer.
        """
        # Set Output Directory:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.output_dir = output_dir

        # Set Mode:
        self.mode = mode

        # Set Up Device:
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        logging.info(f"Using {self.device}")

        # Fix Seed for Random Generations:
        if use_cuda:
            torch.backends.cudnn.deterministic = True
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        self.seed = seed

        # Tokenizer & Vocabulary:
        logging.info("Loading the tokenizer...")
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_type)
        orig_num_tokens = len(self.tokenizer.encoder)
        num_added_tokens = self.tokenizer.add_special_tokens(SPECIAL_TOKENS)

        # Load Model:
        logging.info("Loading the model...")
        self.model = GPT2LMHeadModel.from_pretrained(model_type)
        self.model = self.model.to(self.device)
        if num_added_tokens > 0:
            new_num_tokens = orig_num_tokens + num_added_tokens
            self.model.resize_token_embeddings(new_num_tokens=new_num_tokens)

        # Set Maximum Length & Maximum History:
        self.max_length = min(max_length, self.model.config.n_ctx)
        self.max_history = max_history

        # Load Optimizer:
        logging.info("Loading the optimizer...")
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate
        )

        # Load Train & Validation Datasets:
        logging.info("Loading train & valid data...")
        self.dataset_name = dataset_name
        train_set = ResponseGenerationDataset(dataset_name,
                                              "train",
                                              self.tokenizer,
                                              max_history=self.max_history,
                                              max_length=self.max_length)
        valid_set = ResponseGenerationDataset(dataset_name,
                                              "validation",
                                              self.tokenizer,
                                              max_history=self.max_history,
                                              max_length=self.max_length)
        vocab = self.tokenizer.get_vocab()
        eos_id = vocab[SPECIAL_TOKENS["eos_token"]]
        self.data_collator = PadCollate(eos_id)
        self.batch_size = batch_size
        self.train_loader = DataLoader(train_set,
                                       collate_fn=self.data_collator,
                                       shuffle=True,
                                       batch_size=self.batch_size,
                                       pin_memory=True)
        self.valid_loader = DataLoader(valid_set,
                                       collate_fn=self.data_collator,
                                       batch_size=self.batch_size,
                                       pin_memory=True)

        # Scheduler:
        num_batches = len(self.train_loader)
        self.warmup_ratio = warmup_ratio
        self.num_epochs = num_epochs
        total_train_steps = self.num_epochs * num_batches
        warmup_steps = int(self.warmup_ratio * total_train_steps)
        self.scheduler = get_polynomial_decay_schedule_with_warmup(self.optimizer,
                                                                   num_warmup_steps=warmup_steps,
                                                                   num_training_steps=total_train_steps,
                                                                   power=2)

        # Summary Writer:
        self.writer = SummaryWriter()

        # Set Up Training or Inferring States:
        self.best_loss = sys.float_info.max
        self.last_epoch = 0

        # Settings Completed:
        logging.info("Setting finished.")

    def train(self) -> None:
        logging.info("Training starts.")
        start_epoch = self.last_epoch + 1
        for epoch in range(start_epoch, start_epoch + self.num_epochs):
            self.model.train()
            logging.info(f"#"*50 + f"Epoch: {epoch}" + "#"*50)
            print(f"#"*50 + f"Epoch: {epoch}" + "#"*50)
            train_losses = []
            train_ppls = []
            for _, batch in enumerate(tqdm(self.train_loader)):
                input_ids, token_type_ids, labels = batch
                input_ids = input_ids.to(self.device)
                token_type_ids = token_type_ids.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(input_ids=input_ids,
                                     token_type_ids=token_type_ids,
                                     labels=labels)
                loss, logits = outputs[0], outputs[1]
                self.optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(self.model.parameters(), 1)
                self.optimizer.step()
                self.scheduler.step()
                train_losses.append(loss.detach())
                ppl = torch.exp(loss.detach())
                train_ppls.append(ppl)
            train_losses = [loss.item() for loss in train_losses]
            train_ppls = [
                ppl.item() if not math.isinf(ppl.item()) else 1e+8 for ppl in train_ppls
            ]
            train_loss = np.mean(train_losses)
            train_ppl = np.mean(train_ppls)
            logging.info(
                f"Train loss: {train_loss} || Train perplexity: {train_ppl}"
            )
            self.writer.add_scalar("Loss/train", train_loss, epoch)
            self.writer.add_scalar("PPL/train", train_ppl, epoch)
            self.last_epoch += 1
            valid_loss, valid_ppl = self.validation()
            if valid_loss < self.best_loss:
                self.best_loss = valid_loss
                state_dict = {
                    "model_state_dict": self.model.state_dict(),
                    "optim_state_dict": self.optimizer.state_dict(),
                    "sched_state_dict": self.scheduler.state_dict(),
                    "loss": self.best_loss,
                    "epoch": self.last_epoch
                }
                self.model.save_pretrained(
                    f"{self.output_dir}/{self.dataset_name}_epoch={epoch}"
                )
                logging.info(
                    "*"*10 + "Current best checkpoint is saved." + "*"*10
                )
            logging.info(f"Best valid loss: {self.best_loss}")
            logging.info(
                f"Valid loss: {valid_loss} || Valid perplexity: {valid_ppl}"
            )
            self.writer.add_scalar("Loss/valid", valid_loss, epoch)
            self.writer.add_scalar("PPL/valid", valid_ppl, epoch)
            self.writer.add_scalars("Losses",
                                    {
                                        "train": train_loss,
                                        "valid": valid_loss,
                                    },
                                    epoch)
            self.writer.add_scalars("PPLs",
                                    {
                                        "train": train_ppl,
                                        "valid": valid_ppl,
                                    },
                                    epoch)
        logging.info("Training finished!")

    def validation(self) -> Tuple[list, list]:
        logging.info("Validation processing...")
        self.model.eval()
        valid_losses = []
        valid_ppls = []
        with torch.no_grad():
            for i, batch in enumerate(tqdm(self.valid_loader)):
                input_ids, token_type_ids, labels = batch
                input_ids = input_ids.to(self.device)
                token_type_ids = token_type_ids.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(
                    input_ids=input_ids,
                    token_type_ids=token_type_ids,
                    labels=labels
                )
                loss, logits = outputs[0], outputs[1]
                valid_losses.append(loss.detach())
                ppl = torch.exp(loss.detach())
                valid_ppls.append(ppl)
            valid_losses = [loss.item() for loss in valid_losses]
            valid_ppls = [
                ppl.item() if not math.isinf(ppl.item()) else 1e+8 for ppl in valid_ppls
            ]
            valid_loss = np.mean(valid_losses)
            valid_ppl = np.mean(valid_ppls)
        return valid_loss, valid_ppl


if __name__ == "__main__":
    trainer = Trainer(output_dir="response_generation_outputs",
                      dataset_name="daily_dialog",
                      mode="train")
    trainer.train()
