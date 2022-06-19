import logging
import pandas as pd
import torch

from torch.utils.data import Dataset
from typing import List

EMOTION_CAUSES = [
    "abuse",
    "assessment_pressure",
    "break-up",
    "change_in_life",
    "disloyalty",
    "embarrassed",
    "family_death",
    "family_health",
    "financial_pressure",
    "friend_death",
    "health",
    "injustice",
    "irritated",
    "jealousy",
    "loneliness",
    "missed expectation",
    "owe",
    "pet_loss",
    "societal_relationship",
    "stress",
    "trauma",
    "work_pressure"
]
EMOCAUSE2ID = {emotion: idx for idx, emotion in enumerate(EMOTION_CAUSES)}
ID2EMOCAUSE = {val: key for key, val in EMOCAUSE2ID.items()}

NUM_LABELS = len(EMOTION_CAUSES)


class EmocauseClDataset(Dataset):
    def __init__(self,
                 split: str) -> None:
        super().__init__()
        df = pd.read_csv(
            "nlp_models/emotion_cause_extraction/emotion_cause_datasets/emocause_cl.csv"
        )
        # random state is a seed value
        train = df.sample(frac=0.8, random_state=0)
        test = df.drop(train.index)
        valid = test.sample(frac=0.5, random_state=0)
        test = test.dropv(valid.index)
        if split == "train":
            self.dataset = train
        elif split == "valid":
            self.dataset = valid
        elif split == "test":
            self.dataset = test
        else:
            logging.error(f"Split {split} is invalid.")

    def __len__(self):
        return self.dataset.size

    def __getitem__(self, index):
        return self.dataset.iloc[index]


class EmocauseDataCollator:
    def __init__(self, tokenizer, max_len=128):
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __call__(self, examples: List[dict]):
        utterances = [example["Utterance"] for example in examples]
        labels = [EMOCAUSE2ID(example["Cause"]) for example in examples]
        tokens = self.tokenizer(utterances,
                                return_tensors="pt",
                                padding=True,
                                truncation=True,
                                max_length=self.max_len)
        return dict(labels=torch.tensor(labels), input_ids=tokens["input_ids"], attention_mask=tokens["attention_mask"])
