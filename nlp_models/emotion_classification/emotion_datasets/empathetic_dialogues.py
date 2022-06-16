import torch

from datasets import load_dataset
from tokenizers import Tokenizer
from torch.utils.data import Dataset
from typing import List, Tuple

NUM_CLASSES = 32


def get_emotionid_dict() -> Tuple[dict, dict]:
    """
    Get a dict that converts string class to numbers.
    """
    emotions = [
        "afraid",
        "angry",
        "annoyed",
        "anticipating",
        "anxious",
        "apprehensive",
        "ashamed",
        "caring",
        "confident",
        "content",
        "devastated",
        "disappointed",
        "disgusted",
        "embarrassed",
        "excited",
        "faithful",
        "furious",
        "grateful",
        "guilty",
        "hopeful",
        "impressed",
        "jealous",
        "joyful",
        "lonely",
        "nostalgic",
        "prepared",
        "proud",
        "sad",
        "sentimental",
        "surprised",
        "terrified",
        "trusting",
    ]
    emotion2id = {emotion: idx for idx, emotion in enumerate(emotions)}
    id2emotion = {val: key for key, val in emotion2id.items()}

    return emotion2id, id2emotion


class EmpatheticDialoguesDataset(Dataset):
    def __init__(self, split):
        super().__init__()
        self.raw_dataset = load_dataset("empathetic_dialogues")[split]

    def __len__(self):
        return len(self.raw_dataset)

    def __getitem__(self, index):
        return self.raw_dataset[index]


class DataCollator:
    def __init__(self, tokenizer, max_len=128):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.emotion2id, self.id2emotion = get_emotionid_dict()

    def __call__(self, examples: List[dict]):
        contexts = [example["context"] for example in examples]
        labels = [self.emotion2id[context] for context in contexts]
        prompts = [example["prompt"] for example in examples]
        tokens = self.tokenizer(prompts,
                                return_tensors="pt",
                                padding=True,
                                truncation=True,
                                max_length=self.max_len)
        return dict(labels=torch.tensor(labels), input_ids=tokens["input_ids"], attention_mask=tokens["attention_mask"])


def get_data_collator(tokenizer: Tokenizer):
    return DataCollator(tokenizer)
