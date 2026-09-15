"""Validation utilities for the Twitter sentiment LSTM."""

from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    ConfusionMatrixDisplay, accuracy_score, average_precision_score,
    balanced_accuracy_score, brier_score_loss, classification_report,
    cohen_kappa_score, confusion_matrix, f1_score, log_loss,
    matthews_corrcoef, precision_recall_curve, precision_score, recall_score,
    roc_auc_score, roc_curve,
)
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from services.ProcessDataset import TwitterDataset


DEFAULT_MEDIA_DIR = Path(__file__).resolve().parents[1] / "media"
DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "twitter_sentiment_lstm.pt"


def collate_batch(batch):
    sequences, labels = zip(*batch)
    return (
        pad_sequence(sequences, batch_first=True, padding_value=0),
        torch.stack(labels),
    )


class Validator:
    """Run validation and persist metrics and plots in the media directory."""

    def __init__(
        self, device, model, validation_df, word_idx, tokenize,
        threshold=0.5, criterion=None, batch_size=128,
        model_path=DEFAULT_MODEL_PATH, media_dir=DEFAULT_MEDIA_DIR,
    ):
        self.device = device
        self.model = model.to(device)
        self.threshold = threshold
        self.criterion = criterion or nn.BCEWithLogitsLoss()
        self.batch_size = batch_size
        self.media_dir = Path(media_dir)
        self.media_dir.mkdir(parents=True, exist_ok=True)
        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(f"No se encontró el modelo entrenado: {self.model_path}")
        state_dict = torch.load(self.model_path, map_location=device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        self.dataset = TwitterDataset(validation_df, word_idx, tokenize)
        self.loader = DataLoader(
            self.dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch
        )
        self.y_true = np.array([], dtype=int)
        self.y_pred = np.array([], dtype=int)
        self.y_prob = np.array([], dtype=float)
        self.metrics = {}

    def evaluate(self):
        """Evaluate the model, save all applicable artifacts, and return metrics."""
        y_true, y_prob, total_loss = [], [], 0.0
        self.model.eval()
        with torch.inference_mode():
            for sequences, labels in self.loader:
                sequences, labels = sequences.to(self.device), labels.to(self.device)
                logits = self.model(sequences).squeeze(-1)
                total_loss += self.criterion(logits, labels).item() * labels.size(0)
                y_true.extend(labels.cpu().numpy().astype(int).tolist())
                y_prob.extend(torch.sigmoid(logits).cpu().numpy().tolist())

        if not y_true:
            raise ValueError("El conjunto de validación está vacío.")
        self.y_true = np.asarray(y_true, dtype=int)
        self.y_prob = np.asarray(y_prob, dtype=float)
        self.y_pred = (self.y_prob >= self.threshold).astype(int)
        cm = confusion_matrix(self.y_true, self.y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        has_two_classes = len(np.unique(self.y_true)) == 2

        self.metrics = {
            "samples": int(len(self.y_true)), "threshold": float(self.threshold),
            "loss": float(total_loss / len(self.y_true)),
            "accuracy": float(accuracy_score(self.y_true, self.y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(self.y_true, self.y_pred)),
            "precision": float(precision_score(self.y_true, self.y_pred, zero_division=0)),
            "recall": float(recall_score(self.y_true, self.y_pred, zero_division=0)),
            "f1": float(f1_score(self.y_true, self.y_pred, zero_division=0)),
            "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
            "roc_auc": float(roc_auc_score(self.y_true, self.y_prob)) if has_two_classes else None,
            "average_precision": float(average_precision_score(self.y_true, self.y_prob)) if has_two_classes else None,
            "brier_score": float(brier_score_loss(self.y_true, self.y_prob)),
            "log_loss": float(log_loss(self.y_true, self.y_prob, labels=[0, 1])),
            "cohen_kappa": float(cohen_kappa_score(self.y_true, self.y_pred)),
            "matthews_correlation": float(matthews_corrcoef(self.y_true, self.y_pred)),
            "confusion_matrix": cm.tolist(),
            "classification_report": classification_report(
                self.y_true, self.y_pred, labels=[0, 1],
                target_names=["negative", "positive"], zero_division=0, output_dict=True,
            ),
        }

        self._save_metrics()
        self._save_confusion_matrix(cm)
        if has_two_classes:
            self._save_roc_curve()
            self._save_precision_recall_curve()
        self._save_calibration_curve()
        self._save_probability_distribution()
        return self.metrics

    def _save_metrics(self):
        (self.media_dir / "lstm_validation_metrics.json").write_text(
            json.dumps(self.metrics, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def _save_confusion_matrix(self, cm):
        fig, ax = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay(cm, display_labels=["Negativo", "Positivo"]).plot(
            ax=ax, cmap="Blues", values_format="d", colorbar=False
        )
        ax.set_title("Matriz de confusión - validación LSTM")
        fig.tight_layout(); fig.savefig(self.media_dir / "lstm_confusion_matrix.png", dpi=150); plt.close(fig)

    def _save_roc_curve(self):
        fpr, tpr, _ = roc_curve(self.y_true, self.y_prob)
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(fpr, tpr, label=f"ROC-AUC = {self.metrics['roc_auc']:.4f}")
        ax.plot([0, 1], [0, 1], "--", color="gray", label="Azar")
        ax.set(xlabel="Tasa de falsos positivos", ylabel="Tasa de verdaderos positivos", title="Curva ROC")
        ax.legend(loc="lower right"); ax.grid(alpha=0.25)
        fig.tight_layout(); fig.savefig(self.media_dir / "lstm_roc_curve.png", dpi=150); plt.close(fig)

    def _save_precision_recall_curve(self):
        precision, recall, _ = precision_recall_curve(self.y_true, self.y_prob)
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(recall, precision, label=f"AP = {self.metrics['average_precision']:.4f}")
        ax.set(xlabel="Recall", ylabel="Precision", title="Curva Precision-Recall")
        ax.legend(loc="lower left"); ax.grid(alpha=0.25)
        fig.tight_layout(); fig.savefig(self.media_dir / "lstm_precision_recall_curve.png", dpi=150); plt.close(fig)

    def _save_calibration_curve(self):
        fraction_pos, mean_predicted = calibration_curve(
            self.y_true, self.y_prob, n_bins=10, strategy="uniform"
        )
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(mean_predicted, fraction_pos, "o-", label="LSTM")
        ax.plot([0, 1], [0, 1], "--", color="gray", label="Calibración perfecta")
        ax.set(xlabel="Probabilidad predicha", ylabel="Fracción positiva", title="Curva de calibración")
        ax.legend(loc="upper left"); ax.grid(alpha=0.25)
        fig.tight_layout(); fig.savefig(self.media_dir / "lstm_calibration_curve.png", dpi=150); plt.close(fig)

    def _save_probability_distribution(self):
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.hist(self.y_prob[self.y_true == 0], bins=20, alpha=0.65, label="Negativo")
        ax.hist(self.y_prob[self.y_true == 1], bins=20, alpha=0.65, label="Positivo")
        ax.axvline(self.threshold, color="black", linestyle="--", label=f"Umbral = {self.threshold:g}")
        ax.set(xlabel="Probabilidad de sentimiento positivo", ylabel="Cantidad", title="Distribución de probabilidades")
        ax.legend(); ax.grid(alpha=0.25)
        fig.tight_layout(); fig.savefig(self.media_dir / "lstm_probability_distribution.png", dpi=150); plt.close(fig)
