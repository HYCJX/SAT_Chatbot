import pandas as pd
import torch

from torch.utils.data import Dataset
from typing import List

EMOTION_CAUSES = [

]
EMOCAUSE2ID = {emotion: idx for idx, emotion in enumerate(EMOTION_CAUSES)}
ID2EMOCAUSE = {val: key for key, val in EMOCAUSE2ID.items()}


class EmocauseClDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        df = pd.read_csv(
            "nlp_models/emotion_cause_extraction/emotion_cause_datasets/emocause_cl.csv"
        )
        df = df[df["cl1"].notnull()]
        self.dataset = df.loc[:, ["Situation", "cl2"]]
        print(self.dataset)

    def __len__(self):
        return self.dataset.size

    def __getitem__(self, index):
        return self.dataset.iloc[index]


class EmocauseDataCollator:
    def __init__(self, tokenizer, max_len=128):
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __call__(self, examples: List[dict]):
        contexts = [example["situation"] for example in examples]
        labels = [self.emotion2id[context] for context in contexts]
        prompts = [example["prompt"] for example in examples]
        tokens = self.tokenizer(prompts,
                                return_tensors="pt",
                                padding=True,
                                truncation=True,
                                max_length=self.max_len)
        return dict(labels=torch.tensor(labels), input_ids=tokens["input_ids"], attention_mask=tokens["attention_mask"])
