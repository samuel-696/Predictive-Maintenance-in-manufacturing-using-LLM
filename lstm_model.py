# lstm_model.py
# PyTorch LSTM that ingests a window of sensor readings and predicts
# a continuous health score in [0, 1] (1 = perfect, 0 = failed).

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os
from tqdm import tqdm

from config import (
    NUM_SENSORS, SEQUENCE_LENGTH,
    LSTM_HIDDEN_SIZE, LSTM_NUM_LAYERS, LSTM_DROPOUT,
    LSTM_LEARNING_RATE, LSTM_EPOCHS, LSTM_BATCH_SIZE,
    LSTM_MODEL_PATH, SCALER_PATH,
)


# ─────────────────────────────────────────────────────────────────────────────
# Model architecture
# ─────────────────────────────────────────────────────────────────────────────
class HealthLSTM(nn.Module):
    """
    Stacked bi-directional LSTM → dense head → health score.

    Architecture:
      Input  : (batch, seq_len, num_sensors)
      LSTM   : bidirectional, num_layers stacked
      Head   : LayerNorm → Linear → GELU → Dropout → Linear → Sigmoid
      Output : (batch,)  health ∈ [0, 1]

    The final hidden state (concatenated forward + backward) feeds the head,
    giving the model a global view of the entire sequence.
    """

    def __init__(
        self,
        input_size:  int = NUM_SENSORS,
        hidden_size: int = LSTM_HIDDEN_SIZE,
        num_layers:  int = LSTM_NUM_LAYERS,
        dropout:     float = LSTM_DROPOUT,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        self.lstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            dropout     = dropout if num_layers > 1 else 0.0,
            batch_first = True,
            bidirectional = True,
        )

        lstm_out_size = hidden_size * 2   # bidirectional

        self.head = nn.Sequential(
            nn.LayerNorm(lstm_out_size),
            nn.Linear(lstm_out_size, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        health_score : (batch,)
        feature_vec  : (batch, hidden_size * 2)  — useful as RL state
        """
        lstm_out, _ = self.lstm(x)       # (batch, seq, hidden*2)
        last_out = lstm_out[:, -1, :]    # last time-step
        health = self.head(last_out).squeeze(-1)
        return health, last_out


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────
class LSTMTrainer:

    def __init__(self, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model:   HealthLSTM | None = None
        self.scaler:  StandardScaler | None = None

    # ── preprocessing ────────────────────────────────────────────────────────
    def fit_scaler(self, X: np.ndarray) -> np.ndarray:
        """Fit StandardScaler on sensor features and return scaled array."""
        flat = X.reshape(-1, X.shape[-1])
        self.scaler = StandardScaler()
        scaled_flat = self.scaler.fit_transform(flat)
        return scaled_flat.reshape(X.shape).astype(np.float32)

    def transform(self, X: np.ndarray) -> np.ndarray:
        flat = X.reshape(-1, X.shape[-1])
        scaled = self.scaler.transform(flat)
        return scaled.reshape(X.shape).astype(np.float32)

    # ── training loop ─────────────────────────────────────────────────────────
    def train(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Train the LSTM from raw (unscaled) sequences.
        Returns a history dict with 'train_loss' and 'val_loss' lists.
        """
        # Scale
        X_scaled = self.fit_scaler(X)

        # Train / val split
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_scaled, y, test_size=0.15, random_state=42
        )

        # Torch tensors
        def to_loader(Xa, ya, shuffle=True):
            ds = TensorDataset(
                torch.tensor(Xa, dtype=torch.float32),
                torch.tensor(ya, dtype=torch.float32),
            )
            return DataLoader(ds, batch_size=LSTM_BATCH_SIZE, shuffle=shuffle)

        train_loader = to_loader(X_tr, y_tr)
        val_loader   = to_loader(X_val, y_val, shuffle=False)

        # Model
        self.model = HealthLSTM().to(self.device)
        optimiser  = torch.optim.Adam(self.model.parameters(), lr=LSTM_LEARNING_RATE)
        scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimiser, patience=5, factor=0.5
        )
        criterion = nn.MSELoss()

        history = {"train_loss": [], "val_loss": []}
        best_val = float("inf")
        best_state = None

        print(f"Training LSTM on {self.device}  |  {LSTM_EPOCHS} epochs")
        for epoch in tqdm(range(1, LSTM_EPOCHS + 1), desc="LSTM Training"):
            # ── train ──
            self.model.train()
            tr_loss = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimiser.zero_grad()
                preds, _ = self.model(xb)
                loss = criterion(preds, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimiser.step()
                tr_loss += loss.item() * len(xb)
            tr_loss /= len(X_tr)

            # ── validate ──
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    preds, _ = self.model(xb)
                    val_loss += criterion(preds, yb).item() * len(xb)
            val_loss /= len(X_val)

            scheduler.step(val_loss)
            history["train_loss"].append(tr_loss)
            history["val_loss"].append(val_loss)

            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}

        # Restore best weights
        self.model.load_state_dict(best_state)
        print(f"Best val MSE: {best_val:.5f}")
        return history

    # ── inference ─────────────────────────────────────────────────────────────
    @torch.no_grad()
    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Parameters
        ----------
        X : (N, seq_len, num_sensors) raw (unscaled) array

        Returns
        -------
        health_scores : (N,)
        feature_vecs  : (N, hidden*2)
        """
        X_scaled = self.transform(X)
        tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
        self.model.eval()
        health, features = self.model(tensor)
        return health.cpu().numpy(), features.cpu().numpy()

    # ── persistence ──────────────────────────────────────────────────────────
    def save(self, model_path: str = LSTM_MODEL_PATH, scaler_path: str = SCALER_PATH):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(self.model.state_dict(), model_path)
        joblib.dump(self.scaler, scaler_path)
        print(f"Saved model → {model_path}  |  scaler → {scaler_path}")

    def load(self, model_path: str = LSTM_MODEL_PATH, scaler_path: str = SCALER_PATH):
        self.scaler = joblib.load(scaler_path)
        self.model  = HealthLSTM().to(self.device)
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device, weights_only=True)
        )
        self.model.eval()
        print(f"Loaded model from {model_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Quick smoke-test:  python models/lstm_model.py
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from data_generator import generate_dataset, create_sequences

    df = generate_dataset()
    X, y = create_sequences(df)
    print(f"Sequences: X={X.shape}  y={y.shape}")

    trainer = LSTMTrainer()
    trainer.train(X, y)
    trainer.save()

    scores, feats = trainer.predict(X[:5])
    print("Sample predictions:", scores)
    print("True labels:       ", y[:5])
