# config.py — Central configuration for the Predictive Maintenance System

import os
from dataclasses import dataclass, field
from typing import List

# ─────────────────────────────────────────────
# Sensor configuration
# ─────────────────────────────────────────────
# 21 sensors in total in C-MAPSS. We keep 14 informative ones.
SENSOR_NAMES = [
    "T24", "T30", "T50", "P30", "Nf", "Nc", "Ps30", 
    "phi", "NRf", "NRc", "BPR", "htBleed", "W31", "W32"
]
DROPPED_SENSORS = [
    "T2", "P2", "P15", "epr", "farB", "Nf_dmd", "PCNfR_dmd"
]
OPERATIONAL_SETTINGS = ["setting_1", "setting_2", "setting_3"]

NUM_SENSORS     = len(SENSOR_NAMES)
SEQUENCE_LENGTH = 30          # time-steps fed to LSTM
HEALTH_THRESHOLD_WARN  = 0.40  # below → "degraded"
HEALTH_THRESHOLD_CRIT  = 0.20  # below → "critical"

MAX_RUL_CAP = 125  # for C-MAPSS health score derivation

# ─────────────────────────────────────────────
# Data generation / Loading
# ─────────────────────────────────────────────
CMAPSS_DATA_DIR = "data/cmapss"
CMAPSS_SUBSET   = "FD001"

# ─────────────────────────────────────────────
# Data configuration
# ─────────────────────────────────────────────
NUM_MACHINES  = 5
RANDOM_SEED   = 42

# ─────────────────────────────────────────────
# LSTM model
# ─────────────────────────────────────────────
LSTM_HIDDEN_SIZE  = 64
LSTM_NUM_LAYERS   = 2
LSTM_DROPOUT      = 0.2
LSTM_LEARNING_RATE = 1e-3
LSTM_EPOCHS       = 30
LSTM_BATCH_SIZE   = 32
LSTM_MODEL_PATH   = "assets/lstm_model.pt"
SCALER_PATH       = "assets/scaler.pkl"

# ─────────────────────────────────────────────
# Reinforcement learning (PPO)
# ─────────────────────────────────────────────
ACTION_NAMES = {
    0: "Do Nothing",
    1: "Inspect",
    2: "Repair",
    3: "Replace",
}
NUM_ACTIONS = len(ACTION_NAMES)

# Cost matrix (arbitrary units)
ACTION_COSTS = {
    0: 0,      # do nothing — no cost
    1: 50,     # inspection cost
    2: 300,    # repair cost
    3: 800,    # replacement cost
}
FAILURE_PENALTY   = 5000   # penalty when machine fails unattended
DOWNTIME_PENALTY  = 200    # penalty per step machine is down

PPO_TOTAL_TIMESTEPS = 100_000
PPO_MODEL_PATH      = "assets/ppo_model"

# ─────────────────────────────────────────────
# LLM
# ─────────────────────────────────────────────
LLM_PROVIDER = "gemini"               # "anthropic" | "openai" | "gemini"
LLM_MODEL_ANTHROPIC = "claude-sonnet-4-20250514"
LLM_MODEL_OPENAI    = "gpt-4o-mini"
LLM_MODEL_GEMINI    = "gemini-2.5-flash"
LLM_MAX_TOKENS      = 2048

# ─────────────────────────────────────────────
# Streamlit dashboard
# ─────────────────────────────────────────────
PAGE_TITLE  = "🔧 Predictive Maintenance AI"
PAGE_LAYOUT = "wide"
THEME_PRIMARY = "#0066CC"
