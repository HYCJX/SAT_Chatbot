import logging
import torch
import torch.nn.functional as F

from itertools import chain
from pydantic import BaseModel
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from typing import List, Optional


logging.basicConfig(filename="response_generation.log",
                    filemode="w",
                    level=logging.INFO,
                    format="%(name)s - %(levelname)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")

SPECIAL_TOKENS = {
    "bos_token": "<bos>",
    "eos_token": "<eos>",
    "additional_special_tokens": ["<speaker1>", "<speaker2>"]
}

SPECIAL_TOKEN_NAMES = ["<bos>", "<eos>", "<speaker1>", "<speaker2>"]


class UserResponse(BaseModel):
    utterance: str


class BotResponse(BaseModel):
    utterance: str
    options: Optional[List[str]]


class ResponseGenerator():
    def __init__(self,
                 model: str,
                 max_length: Optional[int] = 512,
                 max_history: Optional[int] = 5,
                 top_p: Optional[float] = 0.80
                 ):
        """
        Arguments:
            model: Transformer model.
            max_length: Maximum length of transformer inputs.
            max_history: Maximum number of turns of conversations used as history.
            top_p: Nucleus Sampling.

        Class fields:
            self.device: Device for training (cpu or cuda).
            self.tokenizer: Transformer's tokenizer.
            self.model: Transformer Model.
            self.max_length: Maximum length of transformer inputs.
            self.max_history: Maximum number of turns of conversations used as history.
            self.top_p: Nucleus sampling
        """
        # Set Top p:
        self.top_p = top_p

        # Set Up Device:
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        logging.info(f"Using {self.device}")

        # Tokenizer & Vocabulary:
        logging.info("Loading the tokenizer...")
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        orig_num_tokens = len(self.tokenizer.encoder)
        num_added_tokens = self.tokenizer.add_special_tokens(SPECIAL_TOKENS)

        # Load Model:
        logging.info("Loading the model...")
        self.model = GPT2LMHeadModel.from_pretrained(model)
        self.model = self.model.to(self.device)
        if num_added_tokens > 0:
            new_num_tokens = orig_num_tokens + num_added_tokens
            self.model.resize_token_embeddings(new_num_tokens=new_num_tokens)

        # Set Maximum Length & Maximum History:
        self.max_length = min(max_length, self.model.config.n_ctx)
        self.max_history = max_history

        # Settings Completed:
        logging.info("Setting finished.")

    def infer(self) -> None:
        logging.info("Start inferring...")
        print("Let's start!")
        print(
            "If you want to quit the conversation, please type \"Abort!\"."
        )
        self.model.eval()
        with torch.no_grad():
            bos_id, eos_id, speaker1_id, speaker2_id = self.tokenizer.convert_tokens_to_ids(
                SPECIAL_TOKEN_NAMES
            )
            input_history = []
            while True:
                utterance = input("You: ")
                if utterance == "Abort!":
                    print("Bot: Good bye.")
                    break

                # Construct Inputs:
                input_ids = [speaker1_id] + self.tokenizer.encode(utterance)
                input_history.append(input_ids)
                if len(input_history) >= self.max_history:
                    num_exceeded = len(input_history) - self.max_history + 1
                    input_history = input_history[num_exceeded:]
                input_ids = [bos_id] + \
                    list(chain.from_iterable(input_history)) + \
                    [speaker2_id]
                start_sp_id = input_history[0][0]
                next_sp_id = speaker1_id if start_sp_id == speaker2_id else speaker2_id
                if start_sp_id == next_sp_id:
                    logging.error("Repeated speaker id in inference.")
                token_type_ids = [
                    [start_sp_id] * len(history)
                    if h % 2 == 0
                    else [next_sp_id] * len(history)
                    for h, history in enumerate(input_history)
                ]
                if len(token_type_ids) != len(input_history):
                    logging.error("Mismatching input ids and token type ids.")
                token_type_ids = [start_sp_id] + \
                    list(chain.from_iterable(token_type_ids)) + \
                    [speaker2_id]
                if len(input_ids) != len(token_type_ids):
                    logging.error("Mismatching input ids and token type ids.")
                input_len = len(input_ids)
                input_ids = torch.LongTensor(
                    input_ids).unsqueeze(0).to(self.device)
                token_type_ids = torch.LongTensor(
                    token_type_ids).unsqueeze(0).to(self.device)

                # Sampling:
                logging.info("Sampling...")
                output_ids = self.nucleus_sampling(input_ids,
                                                   token_type_ids,
                                                   input_len)

                # Decoding:
                logging.info("Decoding...")
                res = self.tokenizer.decode(
                    output_ids, skip_special_tokens=True
                )
                print(f"Bot: {res}")
                input_history.append(
                    [speaker2_id] + self.tokenizer.encode(res)
                )

    def nucleus_sampling(self,
                         input_ids: list,
                         token_type_ids: list,
                         input_len: int) -> list:
        output_ids = []
        for pos in tqdm(range(input_len, self.max_length)):
            output = self.model(
                input_ids=input_ids, token_type_ids=token_type_ids
            )[0][:, pos-1]  # (1, V)
            output = F.softmax(output, dim=-1)  # (1, V)
            sorted_probs, sorted_idxs = torch.sort(output, descending=True)
            cumsum_probs = torch.cumsum(sorted_probs, dim=-1)  # (1, V)
            idx_remove = cumsum_probs > self.top_p
            idx_remove[:, 1:] = idx_remove[:, :-1].clone()
            idx_remove[:, 0] = False
            sorted_probs[idx_remove] = 0.0
            sorted_probs /= torch.sum(sorted_probs,
                                      dim=-1,
                                      keepdim=True)  # (1, V)
            probs = torch.zeros(output.shape,
                                device=self.device).scatter_(-1, sorted_idxs, sorted_probs)  # (1, V)
            idx = torch.multinomial(probs, 1)  # (1, 1)
            idx_item = idx.squeeze(-1).squeeze(-1).item()
            output_ids.append(idx_item)
            bos_id, eos_id, speaker1_id, speaker2_id = self.tokenizer.convert_tokens_to_ids(
                SPECIAL_TOKEN_NAMES
            )
            if idx_item == eos_id:
                break
            input_ids = torch.cat((input_ids, idx), dim=-1)
            next_type_id = torch.LongTensor(
                [[speaker2_id]]
            ).to(self.device)
            token_type_ids = torch.cat((token_type_ids, next_type_id), dim=-1)
            assert input_ids.shape == token_type_ids.shape
        return output_ids


if __name__ == "__main__":
    r = ResponseGenerator(
        "response_generation_outputs/daily_dialog_epoch=7")
    r.infer()
