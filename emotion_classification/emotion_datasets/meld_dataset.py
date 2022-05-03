import json
import logging
import numpy as np
import os
import random
import torch

from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from typing import Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

DATASET = "MELD"
NUM_CLASSES = 7
ROOT_DIR="multimodal-datasets/"
SEED=0

def set_seed(seed: int) -> None:
    """
    Set random seed to a fixed value.
    Set everything to be deterministic
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_emotion2id() -> Tuple[dict, dict]:
    """Get a dict that converts string class to numbers."""
    emotions = [
        "neutral",
        "joy",
        "surprise",
        "anger",
        "sadness",
        "disgust",
        "fear",
    ]
    emotion2id = {emotion: idx for idx, emotion in enumerate(emotions)}
    id2emotion = {val: key for key, val in emotion2id.items()}
    return emotion2id, id2emotion


class Meld_Dataset(Dataset):
    def __init__(
        self,
        data_split,
        model_checkpoint="roberta-base",
        num_future_utterances=0,
        num_past_utterances=0,
        speaker_mode=None,
    ):
        self.data_split = data_split
        self.model_checkpoint = model_checkpoint
        self.emotion2id, self.id2emotion = get_emotion2id()
        self.num_future_utterances = num_future_utterances
        self.num_past_utterances = num_past_utterances
        self.speaker_mode = speaker_mode

        self._load_emotions()
        self._load_utterance_ordered()
        self._string2tokens()

    def _load_emotions(self):
        with open(os.path.join(ROOT_DIR, DATASET, "emotions.json"), "r") as stream:
            self.emotions = json.load(stream)[self.data_split]

    def _load_utterance_ordered(self):
        """
        Load the ids of the utterances in order.
        """
        path = os.path.join(ROOT_DIR, DATASET, "utterance-ordered.json")
        with open(path, "r") as stream:
            self.utterance_ordered = json.load(stream)[self.data_split]

    def _string2tokens(self):
        """
        Convert string to (BPE) tokens.
        """
        logging.info(f"Converting utterances into (BPE) tokens ...")

        diaids = sorted(list(self.utterance_ordered.keys()))

        set_seed(SEED)
        random.shuffle(diaids)

        logging.info(f"Creating input utterances ... ")
        self.inputs_ = self._create_input(
            diaids,
            num_future_utterances=self.num_future_utterances,
            num_past_utterances=self.num_past_utterances,
            speaker_mode=self.speaker_mode,
        )

    def _create_input(
        self,
        diaids,
        num_future_utterances=0,
        num_past_utterances=0,
        speaker_mode=None,
    ):
        args = {
            "diaids": diaids,
            "num_future_utterances": num_future_utterances,
            "num_past_utterances": num_past_utterances,
            "speaker_mode": speaker_mode,
        }
        logging.debug(f"Arguments given: {args}")

        tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint, use_fast=True)
        max_model_input_size = tokenizer.max_model_input_sizes[self.model_checkpoint]
        num_truncated = 0

        inputs = []
        for diaid in tqdm(diaids):
            ues = [
                self._load_utterance_speaker_emotion(uttid, speaker_mode)
                for uttid in self.utterance_ordered[diaid]
            ]

            num_tokens = [len(tokenizer(ue["Utterance"])["input_ids"]) for ue in ues]

            for idx, ue in enumerate(ues):
                if ue["Emotion"] not in list(self.emotion2id.keys()):
                    continue

                label = self.emotion2id[ue["Emotion"]]

                indexes = [idx]
                indexes_past = [
                    i for i in range(idx - 1, idx - num_past_utterances - 1, -1)
                ]
                indexes_future = [
                    i for i in range(idx + 1, idx + num_future_utterances + 1, 1)
                ]

                offset = 0
                if len(indexes_past) < len(indexes_future):
                    for _ in range(len(indexes_future) - len(indexes_past)):
                        indexes_past.append(None)
                elif len(indexes_past) > len(indexes_future):
                    for _ in range(len(indexes_past) - len(indexes_future)):
                        indexes_future.append(None)

                for i, j in zip(indexes_past, indexes_future):
                    if i is not None and i >= 0:
                        indexes.insert(0, i)
                        offset += 1
                        if (sum([num_tokens[idx_] for idx_ in indexes]) > max_model_input_size):
                            del indexes[0]
                            offset -= 1
                            num_truncated += 1
                            break
                    if j is not None and j < len(ues):
                        indexes.append(j)
                        if (sum([num_tokens[idx_] for idx_ in indexes]) > max_model_input_size):
                            del indexes[-1]
                            num_truncated += 1
                            break

                utterances = [ues[idx_]["Utterance"] for idx_ in indexes]

                if num_past_utterances == 0 and num_future_utterances == 0:
                    assert len(utterances) == 1
                    final_utterance = utterances[0]

                elif num_past_utterances > 0 and num_future_utterances == 0:
                    if len(utterances) == 1:
                        final_utterance = "</s></s>" + utterances[-1]
                    else:
                        final_utterance = (
                            " ".join(utterances[:-1]) + "</s></s>" + utterances[-1]
                        )

                elif num_past_utterances == 0 and num_future_utterances > 0:
                    if len(utterances) == 1:
                        final_utterance = utterances[0] + "</s></s>"
                    else:
                        final_utterance = (
                            utterances[0] + "</s></s>" + " ".join(utterances[1:])
                        )

                elif num_past_utterances > 0 and num_future_utterances > 0:
                    if len(utterances) == 1:
                        final_utterance = "</s></s>" + utterances[0] + "</s></s>"
                    else:
                        final_utterance = (
                            " ".join(utterances[:offset])
                            + "</s></s>"
                            + utterances[offset]
                            + "</s></s>"
                            + " ".join(utterances[offset + 1 :])
                        )
                else:
                    raise ValueError

                input_ids_attention_mask = tokenizer(final_utterance)
                input_ids = input_ids_attention_mask["input_ids"]
                attention_mask = input_ids_attention_mask["attention_mask"]

                input_ = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "label": label,
                }

                inputs.append(input_)

        logging.info(f"Number of truncated utterances: {num_truncated}")
        return inputs
    
    def _load_utterance_speaker_emotion(self, uttid, speaker_mode=None) -> dict:
        """
        Load an speaker-name prepended utterance and emotion label
        """
        text_path = os.path.join(
            ROOT_DIR, DATASET, "raw-texts", self.data_split, uttid + ".json"
        )
        with open(text_path, "r") as stream:
            text = json.load(stream)
        utterance = text["Utterance"].strip()
        emotion = text["Emotion"]
        speaker = text["Speaker"]
       
        if speaker_mode is not None and speaker_mode.lower() == "upper":
            utterance = speaker.upper() + ": " + utterance
        elif speaker_mode is not None and speaker_mode.lower() == "title":
            utterance = speaker.title() + ": " + utterance

        return {"Utterance": utterance, "Emotion": emotion}
    
    def __len__(self):
        return len(self.inputs_)

    def __getitem__(self, index):
        return self.inputs_[index]
