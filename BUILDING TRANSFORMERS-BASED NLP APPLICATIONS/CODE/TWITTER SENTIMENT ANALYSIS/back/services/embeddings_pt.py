import torch 
import torch.nn as nn 
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import re
from collections import Counter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device} for embeddings generation.')

class TwitterRNNSentimentAnalysis(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super(TwitterRNNSentimentAnalysis, self).__init__()
        """
        num_embeddings: Vocabulary total size (ej. 10,000 words)
        embedding_dim: Dimension of the dense vector by word (ej. 32 o 300)
        padding_idx=0: Filling vector <pad>, always a zero vector
        """
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=0
        )

        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        embedded = self.embedding(x)
        
        lstm_out, (h_n, _) = self.lstm(embedded)
        logits = self.fc(h_n.squeeze(0))
        return logits

if __name__ == '__main__':
    VOCAB_SIZE = 100  
    EMBED_DIM = 5    

    embedding_layer = nn.Embedding(num_embeddings=VOCAB_SIZE, embedding_dim=EMBED_DIM, padding_idx=0)

    input_ids = torch.tensor([[12, 45, 0]]) 

    embedded_output = embedding_layer(input_ids)

    print("Entry tensor (IDs):", input_ids.shape)

    print("\nExit tensor (Embeddings):", embedded_output.shape)

    print("\nGenerated tensor:\n", embedded_output)