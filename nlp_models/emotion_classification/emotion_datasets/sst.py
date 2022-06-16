from cProfile import label
import logging
import torch

from datasets import load_dataset
from tokenizers import Tokenizer
from torch.utils.data import Dataset
from typing import List, Optional

logging.basicConfig(
    filename="sentiment_analysis.log",
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)


class StanfordSentimentTreebank(Dataset):
    def __init__(self,
                 split: str) -> None:
        super().__init__()
        logging.info("Loading SST dataset.")
        self.dataset = load_dataset("sst")[split]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]


class SSTDataCollator:
    def __init__(self,
                 tokenizer: Tokenizer,
                 max_len: Optional[int] = 128,
                 is_classification: Optional[bool] = False) -> None:
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.is_classification = is_classification

    def __call__(self, examples: List[dict]) -> dict:
        sentences = [example["sentence"] for example in examples]
        if not self.is_classification:
            labels = [example["label"] for example in examples]
        else:
            labels = []
            for example in examples:
                if example["label"] > 0.6:
                    labels.append(0)
                elif example["label"] < 0.4:
                    labels.append(2)
                else:
                    labels.append(1)
        tokens = self.tokenizer(sentences,
                                return_tensors="pt",
                                padding=True,
                                truncation=True,
                                max_length=self.max_len)
        return dict(labels=torch.tensor(labels), input_ids=tokens["input_ids"], attention_mask=tokens["attention_mask"])


class StanfordSentimentTreebankV2(Dataset):
    def __init__(self,
                 split: str) -> None:
        super().__init__()
        logging.info("Loading SSTv2 dataset.")
        self.dataset = load_dataset("gpt3mix/sst2")[split]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]


class SST2DataCollator:
    def __init__(self,
                 tokenizer: Tokenizer,
                 max_len: Optional[int] = 128) -> None:
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __call__(self, examples: List[dict]) -> dict:
        sentences = [example["text"] for example in examples]
        labels = [example["label"] for example in examples]
        tokens = self.tokenizer(sentences,
                                return_tensors="pt",
                                padding=True,
                                truncation=True,
                                max_length=self.max_len)
        return dict(labels=torch.tensor(labels), input_ids=tokens["input_ids"], attention_mask=tokens["attention_mask"])
