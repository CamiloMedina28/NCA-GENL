from fastapi import FastAPI
from services.obtain_datasets import get_datasets
from services.vocab import build_vocab
from services.embeddings_pt import TwitterRNNSentimentAnalysis
from services.training import Training
import torch

app = FastAPI(
    title="Twitter sentiment analysis"
)

def tokenize(text):
    text = text.lower()
    return text.split()

train_df, validation_df, test_df = get_datasets()
word_idx, idx_word = build_vocab(train_df['text'], min_freq=5, max_vocab_size=50_000)
VOCAB_SIZE = len(word_idx)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
modelo = TwitterRNNSentimentAnalysis(VOCAB_SIZE, 128, 256).to(device)

entrenador = Training(modelo, train_df, word_idx, tokenize, device)

entrenador.train()

