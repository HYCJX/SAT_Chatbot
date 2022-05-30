import logging
from tkinter import dialog
import torch

from datasets import load_dataset
from itertools import chain
from tokenizers import Tokenizer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from tqdm import tqdm
from typing import List, Optional

from transformers import GPT2Tokenizer

SPACE = 'Ġ'
END_MARKS = ['.', ',', '?', '!', '...']
QUOTES = ['"', '\'']
ABBREVIATIONS = ['s', 'd', 't', 'm', 're', 'll', 've',
                 'S', 'D', 'T', 'M', 'Re', 'Ll', 'Ve']
EXCLUDE_SYMBOL = "_conv"
COMMA_SYMBOL = "_comma_"

SPECIAL_TOKENS_NAMES = ["<bos>", "<eos>", "<speaker1>", "<speaker2>"]


class EMDialogResponseGenerationDataset(Dataset):
    def __init__(self,
                 split: str,
                 tokenizer: Tokenizer,
                 max_history: Optional[int] = 5,
                 max_length: Optional[int] = 1024):
        self.input_ids = []  # (N, L)
        self.token_type_ids = []  # (N, L)
        self.labels = []  # (N, L)
        logging.info(f"Processing {split} data...")
        dialogue_ids = load_empathetic_dialogues(split, tokenizer)
        bos_id, eos_id, speaker1_id, speaker2_id = tokenizer.convert_tokens_to_ids(
            SPECIAL_TOKENS_NAMES
        )
        for dialogue_id in tqdm(dialogue_ids):
            history = []
            for u, utterance in enumerate(dialogue_id):
                if u % 2 == 0:
                    history.append([speaker1_id] + utterance)
                else:
                    history.append([speaker2_id] + utterance)

            for h in range(len(history)):
                if history[h][0] == speaker2_id:
                    start = max(0, h - max_history + 1)
                    for s in range(start, h):
                        contexts = history[s:h+1]
                        input_ids = [bos_id] + \
                            list(chain.from_iterable(contexts)) + \
                            [eos_id]
                        if len(input_ids) <= max_length:
                            start_sp_id, next_sp_id = contexts[0][0], contexts[1][0]
                            token_type_ids = [[start_sp_id] * len(ctx)
                                              if c % 2 == 0
                                              else [next_sp_id] * len(ctx)
                                              for c, ctx in enumerate(contexts)]
                            if not token_type_ids[-1][0] == speaker2_id:
                                logging.error("Token type ids wrong speaker.")
                            token_type_ids = [start_sp_id] + \
                                list(chain.from_iterable(token_type_ids)) + \
                                [speaker2_id]
                            if not len(input_ids) == len(token_type_ids):
                                logging.error("Token type ids wrong length.")
                            labels = [[-100] * len(ctx)
                                      if c < len(contexts)-1
                                      else [-100] + ctx[1:]
                                      for c, ctx in enumerate(contexts)]
                            if not labels[-1][1:] == contexts[-1][1:]:
                                logging.error("Labels are incorrect.")
                            labels = [-100] + \
                                list(chain.from_iterable(labels)) + \
                                [eos_id]
                            if not len(input_ids) == len(labels):
                                logging.error("Labels wrong length.")
                            self.input_ids.append(input_ids)
                            self.token_type_ids.append(token_type_ids)
                            self.labels.append(labels)
                            break

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.token_type_ids[idx], self.labels[idx]


class PadCollate():
    def __init__(self, eos_id: str):
        self.eos_id = eos_id

    def __call__(self, batch):
        input_ids, token_type_ids, labels = [], [], []
        for _, sequence in enumerate(batch):
            input_ids.append(torch.LongTensor(sequence[0]))
            token_type_ids.append(torch.LongTensor(sequence[1]))
            labels.append(torch.LongTensor(sequence[2]))
        input_ids = pad_sequence(input_ids,
                                 batch_first=True,
                                 padding_value=self.eos_id)
        token_type_ids = pad_sequence(token_type_ids,
                                      batch_first=True,
                                      padding_value=self.eos_id)
        labels = pad_sequence(labels,
                              batch_first=True,
                              padding_value=-100)

        return input_ids, token_type_ids, labels


def load_empathetic_dialogues(split: str, tokenizer: Tokenizer) -> list:
    dataset = load_dataset("empathetic_dialogues")
    if not (split == "train" or split == "validation" or split == "test"):
        logging.error(
            f"Split {split} is invalid while loading empathetic dialogues.")
    raw_data = dataset[split]
    return collect_utterances(raw_data, tokenizer)


def collect_utterances(dataset: dict, tokenizer: Tokenizer, process_tokens: Optional[bool] = True) -> list:
    # Extract fields:
    utterances = dataset["utterance"]
    conv_ids = dataset["conv_id"]
    speaker_ids = dataset["speaker_idx"]
    # Construct a dictionary of conversations/dialogues:
    conv_dict = {}
    current_speaker_idx = -1
    for i, utterance in enumerate(tqdm(utterances)):
        conv_id = conv_ids[i]
        speaker_idx = speaker_ids[i]
        utterance_modified = utterance.strip().replace(COMMA_SYMBOL, ",")
        token_list = tokenizer.tokenize(utterance_modified)
        if process_tokens:
            token_list = process_token_list(token_list)
        text = tokenizer.convert_tokens_to_string(token_list)
        if EXCLUDE_SYMBOL in utterance:
            continue
        if conv_id not in conv_dict:
            conv_dict[conv_id] = []
            current_speaker_idx = -1
        if current_speaker_idx != speaker_idx:
            conv_dict[conv_id].append(text)
            current_speaker_idx = speaker_idx
        else:
            conv_dict[conv_id][-1] += f" {text}"
    dialogue_ids = []
    for i, (conv_id, utter_list) in enumerate(conv_dict.items()):
        ids = []
        for utterance in utter_list:
            tokens = tokenizer.tokenize(utterance)
            token_ids = tokenizer.convert_tokens_to_ids(tokens)
            ids.append(token_ids)
        dialogue_ids.append(ids)
    return dialogue_ids


def process_token_list(token_list: List[str]) -> List[str]:
    token_list[0] = token_list[0].capitalize()
    quote_count = 0
    for i, token in enumerate(token_list):
        if SPACE in token:
            if token[1:] in END_MARKS or token[1:] in ABBREVIATIONS:
                token_list[i] = token[1:]

            if token[1:] == QUOTES[1]:
                if i < len(token_list)-1:
                    if token_list[i+1] in ABBREVIATIONS or (token_list[i+1][0] == SPACE and token_list[i+1][1:] in ABBREVIATIONS):
                        token_list[i] = token[1:]
        if token[0] == SPACE and token[1:] in QUOTES:
            if quote_count % 2 == 1:
                token_list[i] = token[1:]
                quote_count = 0
            else:
                if i < len(token_list)-1 and token_list[i+1][0] == SPACE:
                    token_list[i+1] = token_list[i+1][1:]
                quote_count += 1
        if token in END_MARKS or token[1:] in END_MARKS:
            if i < len(token_list)-1:
                if token_list[i+1][0] != SPACE:
                    token_list[i+1] = SPACE + token_list[i+1].capitalize()
                else:
                    token_list[i+1] = SPACE + token_list[i+1][1:].capitalize()
    new_token_list = [
        token for token in token_list if token != SPACE and len(token) > 0
    ]
    if new_token_list[-1] not in END_MARKS:
        new_token_list.append(END_MARKS[0])
    return new_token_list
