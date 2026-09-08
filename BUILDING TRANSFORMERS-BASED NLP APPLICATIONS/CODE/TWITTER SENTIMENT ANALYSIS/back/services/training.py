from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from services.ProcessDataset import TwitterDataset

import torch
import torch.nn as nn
import torch.optim as optim

from pathlib import Path

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

print(MODEL_DIR)


BATCH_SIZE = 128
EPOCHS = 7


def collate_batch(batch):

    sequences, labels = zip(*batch)

    sequences = pad_sequence(
        sequences,
        batch_first=True,
        padding_value=0
    )

    labels = torch.stack(labels)

    return sequences, labels


class Training:

    def __init__(
        self,
        model,
        train_df,
        word_idx,
        tokenize,
        device
    ):
        self.model = model
        self.device = device

        # Pandas DataFrame -> PyTorch Dataset
        self.train_dataset = TwitterDataset(
            train_df,
            word_idx,
            tokenize
        )

        # Dataset -> batches
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            collate_fn=collate_batch
        )

        # Binary classification loss
        self.criterion = nn.BCEWithLogitsLoss()

        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=0.001
        )

    def train(self):

        print("BEGIN TRAINING")

        for epoch in range(EPOCHS):

            self.model.train()

            total_loss = 0

            for sequences, labels in self.train_loader:

                sequences = sequences.to(self.device)
                labels = labels.to(self.device)

                # Clean gradients from previous iteration
                self.optimizer.zero_grad()

                # Forward pass
                logits = self.model(sequences)

                # Model returns (batch_size, 1)
                # Labels are (batch_size,)
                logits = logits.squeeze(1)

                # Calculate loss
                loss = self.criterion(
                    logits,
                    labels
                )

                # Backpropagation
                loss.backward()

                # Update model parameters
                self.optimizer.step()

                total_loss += loss.item()

            average_loss = (
                total_loss / len(self.train_loader)
            )

            print(
                f"Epoch {epoch + 1}/{EPOCHS} "
                f"- Loss: {average_loss:.4f}"
            )
        try: 
            torch.save(
            self.model.state_dict(),
            MODEL_DIR / "twitter_sentiment_lstm.pt"
            )
        except:
            torch.save(
                        self.model.state_dict(),
                        "twitter_sentiment_lstm.pt"
                        )

        print("MODEL SAVED")