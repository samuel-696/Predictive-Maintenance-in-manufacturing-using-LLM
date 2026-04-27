# pipeline.py
# End-to-end pipeline: generate data → train LSTM → train PPO → save artefacts.
# Run this once before launching the Streamlit dashboard.

import os
import sys
import numpy as np
import joblib
from pathlib import Path

# Ensure project root is on path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config import (
    LSTM_MODEL_PATH, SCALER_PATH, PPO_MODEL_PATH,
    NUM_MACHINES,
)
from data_generator import generate_dataset, create_sequences
from lstm_model import LSTMTrainer


def run_pipeline(skip_ppo: bool = False):
    os.makedirs("assets", exist_ok=True)

    # ── 1. Data ───────────────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  STEP 1 — Loading C-MAPSS sensor dataset")
    print("═" * 60)
    try:
        df = generate_dataset(dataset_type="cmapss")
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        print("Please download the NASA C-MAPSS dataset and extract it to data/cmapss/")
        sys.exit(1)

    print(f"  Total rows: {len(df):,}  |  Engines (Units): {df['machine_id'].nunique()}")

    # ── 2. LSTM ───────────────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  STEP 2 — Training LSTM health predictor")
    print("═" * 60)
    X, y = create_sequences(df)
    print(f"  Sequences: {X.shape[0]:,}  |  Seq length: {X.shape[1]}  |  Features: {X.shape[2]}")

    lstm_trainer = LSTMTrainer()
    history = lstm_trainer.train(X, y)
    lstm_trainer.save()

    final_val_mse = history["val_loss"][-1]
    print(f"  Final val MSE: {final_val_mse:.5f}  (RMSE ≈ {final_val_mse**0.5:.4f})")

    # Quick sanity-check
    preds, feats = lstm_trainer.predict(X[:10])
    mae = np.mean(np.abs(preds - y[:10]))
    print(f"  Sample MAE on 10 points: {mae:.4f}")

    # ── 3. PPO ────────────────────────────────────────────────────────────────
    if not skip_ppo:
        print("\n" + "═" * 60)
        print("  STEP 3 — Training PPO maintenance agent")
        print("═" * 60)
        from train_ppo import train_ppo
        train_ppo(lstm_trainer, df)
    else:
        print("\n  [Skipping PPO training — skip_ppo=True]")

    print("\n" + "═" * 60)
    print("  ✅  Pipeline complete!  Artefacts saved to ./assets/")
    print("  👉  Launch dashboard:  streamlit run app.py")
    print("═" * 60 + "\n")

    return lstm_trainer, df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Predictive Maintenance Pipeline")
    parser.add_argument("--skip-ppo", action="store_true",
                        help="Skip PPO training (much faster; uses heuristic fallback in dashboard)")
    args = parser.parse_args()
    run_pipeline(skip_ppo=args.skip_ppo)
