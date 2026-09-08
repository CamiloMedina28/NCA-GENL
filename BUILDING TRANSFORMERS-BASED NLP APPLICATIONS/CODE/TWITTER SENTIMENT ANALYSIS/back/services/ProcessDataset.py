import torch 
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pandas as pd

class TwitterDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, word_idx, tokenize):
        self.texts = dataframe['text'].to_list()
        self.labels = dataframe['target'].to_list()

        self.word_idx = word_idx
        self.tokenize = tokenize

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = self.texts[index]
        label = self.labels[index]

        tokens = self.tokenize(text)

        token_ids = [
            self.word_idx.get(
                token,
                self.word_idx["<unk>"]
            )
            for token in tokens
        ]

        return (
            torch.tensor(token_ids, dtype=torch.long),
            torch.tensor(label, dtype=torch.float32)
        )
