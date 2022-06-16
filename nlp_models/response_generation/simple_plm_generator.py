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
from typing import List, Optional, Tuple

from dataset import ResponseGenerationDataset
from dataset import PadCollate


logging.basicConfig(filename="response_generation.log",
                    filemode="w",
                    level=logging.INFO,
                    format="%(name)s - %(levelname)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")

SPECIAL_TOKENS = {
    "bos_token": "<bos>",
    "eos_token": "<eos>",
    "additional_special_tokens": ["<speaker1>", "<speaker2>"]
}

SPECIAL_TOKEN_NAMES = ["<bos>", "<eos>", "<speaker1>", "<speaker2>"]


class Trainer():
    def __init__(self,
                 output_dir: Optional[str] = None,
                 mode: Optional[str] = "train",
                 dataset_names: List[str] = ["empathetic_dialogues"],
                 model_type: Optional[str] = "gpt2",
                 checkpoint_path: Optional[str] = None,
                 seed: Optional[float] = 0,
                 max_length: Optional[int] = 512,
                 max_history: Optional[int] = 5,
                 learning_rate: Optional[float] = 1e-5,
                 warmup_ratio: Optional[float] = 0.1,
                 batch_size: Optional[int] = 8,
                 num_epochs: Optional[int] = 10,
                 top_p: Optional[float] = 0.80):
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
                top_p: Nucleus sampling parameter.

        Class fields:
            self.output_dir: Output directory.
            self.mode: Mode.
            self.device: Device for training (cpu or cuda).
            self.seed: Seed.
            self.tokenizer: Transformer's tokenizer.
            self.model: Transformer model.
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
                self.top_p: Nucleus sampling parameter.
        """
        # Set Mode:
        self.mode = mode

        # Set Up Device:
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        logging.info(f"Using {self.device}")

        # Tokenizer & Vocabulary:
        logging.info("Loading the tokenizer...")
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_type)
        orig_num_tokens = len(self.tokenizer.encoder)
        num_added_tokens = self.tokenizer.add_special_tokens(SPECIAL_TOKENS)

        # Load Model:
        logging.info("Loading the model...")
        if not checkpoint_path:
            self.model = GPT2LMHeadModel.from_pretrained(model_type)
        else:
            self.model = GPT2LMHeadModel.from_pretrained(checkpoint_path)
        self.model = self.model.to(self.device)
        if num_added_tokens > 0:
            new_num_tokens = orig_num_tokens + num_added_tokens
            self.model.resize_token_embeddings(new_num_tokens=new_num_tokens)

        # Set Maximum Length & Maximum History:
        self.max_length = min(max_length, self.model.config.n_ctx)
        self.max_history = max_history

        if self.mode == "train":
            # Fix Seed for Random Generations:
            if use_cuda:
                torch.backends.cudnn.deterministic = True
            torch.manual_seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            random.seed(seed)
            self.seed = seed

            # Set Output Directory:
            if not output_dir:
                logging.error("Output directory is not specified.")
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            self.output_dir = output_dir

            # Load Optimizer:
            logging.info("Loading the optimizer...")
            self.learning_rate = learning_rate
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=learning_rate
            )

            # Load Train & Validation Datasets:
            logging.info("Loading train & valid data...")
            self.dataset_names = dataset_names
            train_set = ResponseGenerationDataset(dataset_names,
                                                  "train",
                                                  self.tokenizer,
                                                  max_history=self.max_history,
                                                  max_length=self.max_length)
            valid_set = ResponseGenerationDataset(dataset_names,
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

        # Set Top p:
        self.top_p = top_p

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
                self.model.save_pretrained(
                    f"{self.output_dir}/{self.dataset_names}_epoch={epoch}"
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

    def infer(self) -> None:
        logging.info("Start inferring...")
        print("Let's start!")
        print(
            "If you want to quit the conversation, please type \"Abort!\"."
        )
        self.model.eval()
        with torch.no_grad():
            bos_id, eos_id, speaker1_id, speaker2_id = self.tokenizer.convert_tokens_to_ids(
                SPECIAL_TOKEN_NAMES
            )
            input_history = []
            while True:
                utterance = input("You: ")
                if utterance == "Abort!":
                    print("Bot: Good bye.")
                    break

                # Construct Inputs:
                input_ids = [speaker1_id] + self.tokenizer.encode(utterance)
                input_history.append(input_ids)
                if len(input_history) >= self.max_history:
                    num_exceeded = len(input_history) - self.max_history + 1
                    input_history = input_history[num_exceeded:]
                input_ids = [bos_id] + \
                    list(chain.from_iterable(input_history)) + \
                    [speaker2_id]
                start_sp_id = input_history[0][0]
                next_sp_id = speaker1_id if start_sp_id == speaker2_id else speaker2_id
                if start_sp_id == next_sp_id:
                    logging.error("Repeated speaker id in inference.")
                token_type_ids = [
                    [start_sp_id] * len(history)
                    if h % 2 == 0
                    else [next_sp_id] * len(history)
                    for h, history in enumerate(input_history)
                ]
                if len(token_type_ids) != len(input_history):
                    logging.error("Mismatching input ids and token type ids.")
                token_type_ids = [start_sp_id] + \
                    list(chain.from_iterable(token_type_ids)) + \
                    [speaker2_id]
                if len(input_ids) != len(token_type_ids):
                    logging.error("Mismatching input ids and token type ids.")
                input_len = len(input_ids)
                input_ids = torch.LongTensor(
                    input_ids).unsqueeze(0).to(self.device)
                token_type_ids = torch.LongTensor(
                    token_type_ids).unsqueeze(0).to(self.device)

                # Sampling:
                logging.info("Sampling...")
                output_ids = self.nucleus_sampling(input_ids,
                                                   token_type_ids,
                                                   input_len)

                # Decoding:
                logging.info("Decoding...")
                res = self.tokenizer.decode(
                    output_ids, skip_special_tokens=True
                )
                print(f"Bot: {res}")
                input_history.append(
                    [speaker2_id] + self.tokenizer.encode(res)
                )

    def nucleus_sampling(self,
                         input_ids: list,
                         token_type_ids: list,
                         input_len: int) -> list:
        output_ids = []
        for pos in tqdm(range(input_len, self.max_length)):
            output = self.model(
                input_ids=input_ids, token_type_ids=token_type_ids
            )[0][:, pos-1]  # (1, V)
            output = F.softmax(output, dim=-1)  # (1, V)
            sorted_probs, sorted_idxs = torch.sort(output, descending=True)
            cumsum_probs = torch.cumsum(sorted_probs, dim=-1)  # (1, V)
            idx_remove = cumsum_probs > self.top_p
            idx_remove[:, 1:] = idx_remove[:, :-1].clone()
            idx_remove[:, 0] = False
            sorted_probs[idx_remove] = 0.0
            sorted_probs /= torch.sum(sorted_probs,
                                      dim=-1,
                                      keepdim=True)  # (1, V)
            probs = torch.zeros(output.shape,
                                device=self.device).scatter_(-1, sorted_idxs, sorted_probs)  # (1, V)
            idx = torch.multinomial(probs, 1)  # (1, 1)
            idx_item = idx.squeeze(-1).squeeze(-1).item()
            output_ids.append(idx_item)
            bos_id, eos_id, speaker1_id, speaker2_id = self.tokenizer.convert_tokens_to_ids(
                SPECIAL_TOKEN_NAMES
            )
            if idx_item == eos_id:
                break
            input_ids = torch.cat((input_ids, idx), dim=-1)
            next_type_id = torch.LongTensor(
                [[speaker2_id]]
            ).to(self.device)
            token_type_ids = torch.cat((token_type_ids, next_type_id), dim=-1)
            assert input_ids.shape == token_type_ids.shape
        return output_ids


if __name__ == "__main__":
    trainer = Trainer(output_dir="response_generation_outputs",
                      mode="train",
                      dataset_names=["anno_mi"])
    trainer.train()
