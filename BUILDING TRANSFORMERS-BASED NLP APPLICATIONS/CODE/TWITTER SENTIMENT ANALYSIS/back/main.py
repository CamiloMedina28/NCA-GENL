from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import json
from services.obtain_datasets import get_datasets
from services.vocab import build_vocab
from services.embeddings_pt import TwitterRNNSentimentAnalysis
from services.training import Training
from services.validations import Validator
import torch

app = FastAPI(
    title="Twitter sentiment analysis"
)

MEDIA_DIR = Path(__file__).resolve().parent / "media"
MEDIA_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/media", StaticFiles(directory=MEDIA_DIR), name="media")

def tokenize(text):
    text = text.lower()
    return text.split()

train_df, validation_df, test_df = get_datasets()
word_idx, idx_word = build_vocab(train_df['text'], min_freq=5, max_vocab_size=50_000)
VOCAB_SIZE = len(word_idx)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# No cambiar los parámetros de configuración - el modelo ya se entrenó así
modelo = TwitterRNNSentimentAnalysis(VOCAB_SIZE, 128, 256).to(device)

# ----------- ENTRENAMIENTO -----------

entrenador = Training(modelo, train_df, word_idx, tokenize, device)
entrenador.train()

# ----------- VALIDACIONES -----------

# validador = Validator(device, modelo, validation_df, word_idx, tokenize)


# @app.post("/validate")
# def validate_model():
#     """Evalúa el LSTM y genera métricas y gráficas en back/media."""
#     metrics = validador.evaluate()
#     return {
#         "metrics": metrics,
#         "media_dir": str(MEDIA_DIR),
#         "plots": [
#             "/media/lstm_confusion_matrix.png",
#             "/media/lstm_roc_curve.png",
#             "/media/lstm_precision_recall_curve.png",
#             "/media/lstm_calibration_curve.png",
#             "/media/lstm_probability_distribution.png",
#         ],
#     }


# @app.get("/validation/metrics")
# def validation_metrics():
#     """Devuelve las últimas métricas guardadas, si ya se ejecutó la validación."""
#     metrics_file = MEDIA_DIR / "lstm_validation_metrics.json"
#     if not metrics_file.exists():
#         return {"detail": "Ejecuta POST /validate primero."}
#     return json.loads(metrics_file.read_text(encoding="utf-8"))

