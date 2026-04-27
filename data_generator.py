# data_generator.py
# Generates realistic synthetic sensor time-series data for industrial machines.
# Each "run" simulates one machine life-cycle from healthy → degraded → failure.

import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import (
    SENSOR_NAMES, NUM_SENSORS, NUM_MACHINES,
    RANDOM_SEED, SEQUENCE_LENGTH,
    CMAPSS_DATA_DIR, CMAPSS_SUBSET, DROPPED_SENSORS, OPERATIONAL_SETTINGS, MAX_RUL_CAP
)


def load_cmapss(subset: str = CMAPSS_SUBSET) -> pd.DataFrame:
    """
    Loads C-MAPSS dataset directly from `CMAPSSData.zip`.
    Calculates RUL based on machine max steps and cap, then normalizes to health score.
    """
    import zipfile
    
    zip_path = Path(__file__).resolve().parent / "CMAPSSData.zip"
    
    if not zip_path.exists():
        raise FileNotFoundError(f"C-MAPSS dataset zip not found at {zip_path}. Please download it.")

    # NASA CMAPSS original 21 sensors
    SENSOR_NAMES_ALL = [
        "T2", "T24", "T30", "T50", "P2", "P15", "P30", "Nf", "Nc", "epr",
        "Ps30", "phi", "NRf", "NRc", "BPR", "farB", "htBleed", "Nf_dmd",
        "PCNfR_dmd", "W31", "W32"
    ]
    
    # All 26 columns in C-MAPSS
    columns = ["machine_id", "step"] + OPERATIONAL_SETTINGS + SENSOR_NAMES_ALL

    with zipfile.ZipFile(zip_path, 'r') as z:
        with z.open(f"train_{subset}.txt") as f:
            df = pd.read_csv(f, sep=r'\s+', header=None, names=columns)
    
    # Drop near-constant/uninformative sensors
    df.drop(columns=DROPPED_SENSORS, inplace=True, errors="ignore")
    # Drop operational settings for now
    df.drop(columns=OPERATIONAL_SETTINGS, inplace=True, errors="ignore")

    # Calculate RUL
    rul = pd.DataFrame(df.groupby("machine_id")["step"].max()).reset_index()
    rul.columns = ["machine_id", "max_step"]
    df = df.merge(rul, on=["machine_id"], how="left")
    df["rul"] = df["max_step"] - df["step"]
    df.drop(columns=["max_step"], inplace=True)

    # Convert RUL to piece-wise target Health score
    # Cap RUL at MAX_RUL_CAP (e.g. 125) and normalize to [0, 1]
    df["health"] = np.minimum(df["rul"], MAX_RUL_CAP) / MAX_RUL_CAP

    return df


def generate_dataset(
    dataset_type: str = "cmapss",
    num_machines: int = NUM_MACHINES,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Wrapper to load either C-MAPSS data or fallback to synthetic data.
    """
    if dataset_type == "cmapss":
        return load_cmapss(CMAPSS_SUBSET)
    else:
        # Fallback to old synthetic logic if ever defined (omitted for brevity)
        raise NotImplementedError("Synthetic dataset generation disabled. Use cmapss.")


def create_sequences(
    df: pd.DataFrame,
    seq_len: int = SEQUENCE_LENGTH,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Converts per-machine sensor readings into (X, y) arrays for LSTM training.

    X shape: (N, seq_len, num_sensors)
    y shape: (N,)  — health score at the end of each window
    """
    X_list, y_list = [], []

    for _, machine_df in df.groupby("machine_id"):
        sensor_vals = machine_df[SENSOR_NAMES].values.astype(np.float32)
        health_vals = machine_df["health"].values.astype(np.float32)

        for i in range(len(machine_df) - seq_len):
            X_list.append(sensor_vals[i : i + seq_len])
            y_list.append(health_vals[i + seq_len])  # predict health at next step

    return np.stack(X_list), np.array(y_list, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# CLI usage: python data_generator.py
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating synthetic sensor dataset …")
    dataset = generate_dataset()
    print(dataset.head(10).to_string())
    print(f"\nShape: {dataset.shape}")
    print(f"Health range: [{dataset['health'].min():.3f}, {dataset['health'].max():.3f}]")
    print(f"Sensors: {SENSOR_NAMES}")
