import logging
import torch

from datasets import load_dataset
from tokenizers import Tokenizer
from torch.utils.data import Dataset
from typing import List, Optional

logging.basicConfig(
    filename='sentiment_analysis.log',
    filemode='w',
    level=logging.DEBUG,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class StanfordSentimentTreebank(Dataset):
    def __init__(self,
                 split: str) -> None:
        logging.info("Loading SST dataset.")
        self.dataset = load_dataset("sst")[split].select([*range(0, 1000, 1)])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]


class DataCollator:
    def __init__(self,
                 tokenizer: Tokenizer,
                 max_len: Optional[int] = 128) -> None:
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __call__(self, examples: List[dict]) -> dict:
        sentences = [example["sentence"] for example in examples]
        labels = [example["label"] for example in examples]
        tokens = self.tokenizer(sentences,
                                return_tensors="pt",
                                padding=True,
                                truncation=True,
                                max_length=self.max_len)
        return dict(labels=torch.tensor(labels), input_ids=tokens["input_ids"], attention_mask=tokens["attention_mask"])


StanfordSentimentTreebank("train")
