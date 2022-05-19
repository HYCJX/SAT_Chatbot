import json
import logging

from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class EmoCauseDataset(Dataset):
    def __init__(
        self,
        data_split,
    ):
        self.data_split = data_split
        self.inputs = self._create_inputs()

    def _create_inputs(self):
        logging.info("Creating inputs...")
        # Load Data:
        raw_data = None
        if self.data_split == "train":
            raw_data_path = "emocause-datasets/train.json"
        elif self.data_split == "test":
            raw_data_path = "emocause-datasets/test.json"
        else:
            logging.error(
                "The DATA_SPLIT parameter of EmoCauseDataset constructor is invalid.")
        with open(raw_data_path, "r") as stream:
            raw_data = json.load(stream)
        # Define Tokenizer:
        tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        # Create Inputs:
        inputs = []
        for data in tqdm(raw_data[5:]):
            # Tokenize Inputs:
            context = data["original_situation"]
            cause = set([cause[1] for cause in data["annotation"]])
            emotion = data["emotion"]
            emotion_clause = "The emotion of the context is: " + \
                emotion[2: (len(emotion) - 2)]
            tokenized_input = tokenizer(emotion_clause,
                                        context,
                                        max_length=128,
                                        truncation="only_second",
                                        stride=50,
                                        return_offsets_mapping=True)
            # Extract Sequence Labels:
            word_ids = tokenized_input.word_ids()
            sequence_ids = tokenized_input.sequence_ids()
            labels = []
            assert len(word_ids) == len(sequence_ids)
            for i in range(len(word_ids)):
                sequence_id = sequence_ids[i]
                word_id = word_ids[i]
                if sequence_id is None:
                    labels.append(-100)
                elif sequence_id == 0:
                    labels.append(0)
                elif sequence_id == 1:
                    if word_id in cause:
                        labels.append(1)
                    else:
                        labels.append(0)
                else:
                    logging.error(f"Sequence ID {sequence_id} is invalid.")
            input = {
                "input_ids": tokenized_input["input_ids"],
                "attention_mask": tokenized_input["attention_mask"],
                "labels": labels,
            }
            inputs.append(input)
        logging.info("Completed...")
        return inputs

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        return self.inputs[index]
