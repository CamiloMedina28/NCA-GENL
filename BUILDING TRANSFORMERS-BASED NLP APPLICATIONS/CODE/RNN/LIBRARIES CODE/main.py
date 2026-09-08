import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from collections import Counter


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'\033[35m Using device: {device} \033[0m')

# ===============================================================
# 1. LOAD DATASET & BUILD VOCABULARY
# ===============================================================
print("Loading IMDB dataset...")
raw_dataset = load_dataset("imdb")

MAX_VOCAB_SIZE = 10000
UNK_TOKEN = "<unk>"
PAD_TOKEN = "<pad>"

def build_vocab(texts, max_size):
    counter = Counter()
    for text in texts:
        counter.update(text.lower().split())
    
    most_common = counter.most_common(max_size - 2)
    
    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for word, _ in most_common:
        vocab[word] = len(vocab)
    return vocab