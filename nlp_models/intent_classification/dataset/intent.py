import pandas as pd
from tokenizers import Tokenizer
import torch

from torch.utils.data import Dataset
from typing import List, Optional

EMOTION_CAUSES = [
    "maybe",
    "no",
    "death",
    "self-blame",
    "injustice",
    "abuse",
    "missing",
    "work",
    "jealousy",
    "partner",
    "loneliness",
    "trauma",
    "goodbye",
    "health",
    "assessment",
    "greeting",
    "yes"
]
EMOCAUSE2ID = {emotion: idx for idx, emotion in enumerate(EMOTION_CAUSES)}
ID2EMOCAUSE = {val: key for key, val in EMOCAUSE2ID.items()}

NUM_LABELS = len(EMOTION_CAUSES)


class EmocauseClDataset(Dataset):
    def __init__(self,
                 split: str) -> None:
        super().__init__()
        self.dataset = pd.read_csv(
            f"nlp_models/intent_classification/dataset/intent/{split}.csv"
        ).values

    def __len__(self) -> None:
        return len(self.dataset)

    def __getitem__(self, index) -> dict:
        return self.dataset[index]


class EmocauseDataCollator:
    def __init__(self, tokenizer: Tokenizer, max_len: Optional[int] = 128) -> None:
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __call__(self, examples: List[dict]) -> dict:
        utterances = [example[2] for example in examples]
        labels = [EMOCAUSE2ID[example[3]] for example in examples]
        tokens = self.tokenizer(utterances,
                                return_tensors="pt",
                                padding=True,
                                truncation=True,
                                max_length=self.max_len)
        return dict(labels=torch.tensor(labels), input_ids=tokens["input_ids"], attention_mask=tokens["attention_mask"])


if __name__ == "__main__":
    e = EmocauseClDataset("train")
    print(e.__getitem__(0)[2])
