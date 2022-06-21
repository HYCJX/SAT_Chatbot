import logging
import pandas as pd
import torch

from torch.utils.data import Dataset
from typing import List

EMOTION_CAUSES = [
    'maybe', 'no', 'death', 'self-blame', 'injustice', 'abuse', 'missing', 'work', 'jealousy', 'partner', 'loneliness', 'trauma', 'goodbye', 'health', 'assessment', 'greeting', 'yes']
EMOCAUSE2ID = {emotion: idx for idx, emotion in enumerate(EMOTION_CAUSES)}
ID2EMOCAUSE = {val: key for key, val in EMOCAUSE2ID.items()}

NUM_LABELS = len(EMOTION_CAUSES)


class EmocauseClDataset(Dataset):
    def __init__(self,
                 split: str) -> None:
        super().__init__()
        self.dataset = pd.read_csv(
            f"{split}.csv"
        ).values
        # # random state is a seed value
        # train = df.sample(frac=0.8, random_state=0)
        # test = df.drop(train.index)
        # valid = test.sample(frac=0.5, random_state=0)
        # test = test.drop(valid.index)
        # if split == "train":
        #     self.dataset = train.values
        # elif split == "valid":
        #     self.dataset = valid.values
        # elif split == "test":
        #     self.dataset = test.values
        # else:
        #     logging.error(f"Split {split} is invalid.")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]


class EmocauseDataCollator:
    def __init__(self, tokenizer, max_len=128):
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __call__(self, examples: List[dict]):
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
