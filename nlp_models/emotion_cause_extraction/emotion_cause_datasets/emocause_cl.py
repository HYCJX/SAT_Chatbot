import pandas as pd

from torch.utils.data import Dataset


class EmocauseClDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
