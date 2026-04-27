<div align="center">

# 🔧 Predictive Maintenance in manufacturing using LLM

### AI-Powered Machine Health Monitoring & Predictive Maintenance Scheduling

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.36+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)

**LSTM Health Prediction** · **PPO Reinforcement Learning** · **LLM-Powered Explanations**

A production-grade predictive maintenance system that combines deep learning, reinforcement learning, and large language models to predict equipment failures, schedule optimal maintenance actions, and explain decisions in plain English.

<br/>

[Getting Started](#-getting-started) · [Architecture](#-architecture) · [Features](#-features) · [Configuration](#-configuration) · [Extending](#-extending-the-system)

</div>

<br/>

---

## ✨ Features

<table>
<tr>
<td width="50%">

### 🧠 LSTM Health Predictor
Bi-directional LSTM neural network that analyzes 14 sensor channels over sliding windows of 30 time-steps to produce a real-time health score ∈ [0, 1].

### 🎯 PPO Decision Agent
Proximal Policy Optimization agent trained via reinforcement learning to select optimal maintenance actions — balancing repair costs against failure risks.

</td>
<td width="50%">

### 💬 LLM Explainer
AI-powered natural language explanations of every maintenance decision. Supports **Google Gemini**, **OpenAI**, and **Anthropic**, with automatic rule-based fallback.

### 📊 Real-Time Dashboard
Dark-themed Streamlit dashboard with glassmorphism UI, interactive Plotly charts, KPI cards, and a custom CSV upload feature for external datasets.

</td>
</tr>
</table>

---

## 🏗 Architecture

```
                    ┌─────────────────────────────────────────────┐
                    │           NASA C-MAPSS Dataset              │
                    │   100 engines · 14 sensors · FD001 subset   │
                    └──────────────────┬──────────────────────────┘
                                       │
                    ┌──────────────────▼──────────────────────────┐
                    │            DATA PIPELINE                    │
                    │                                             │
                    │  ► Sliding windows (30 time-steps)          │
                    │  ► StandardScaler normalization             │
                    │  ► RUL → Health score mapping               │
                    └──────────────────┬──────────────────────────┘
                                       │
                    ┌──────────────────▼──────────────────────────┐
                    │         LSTM HEALTH PREDICTOR               │
                    │                                             │
                    │  Bi-LSTM (2 layers, 64 hidden, dropout 0.2) │
                    │  Output: health ∈ [0,1] + 128-dim features  │
                    └──────────────────┬──────────────────────────┘
                                       │
                  ┌────────────────────┴────────────────────┐
                  ▼                                         ▼
    ┌─────────────────────────┐           ┌──────────────────────────┐
    │    PPO DECISION AGENT   │           │      LLM EXPLAINER       │
    │                         │           │                          │
    │  Actions:               │           │  Gemini / GPT-4o / Claude│
    │   0: Do Nothing ($0)    │           │  Rule-based fallback     │
    │   1: Inspect    ($50)   │──────────►│                          │
    │   2: Repair     ($300)  │           │  "Why this action?"      │
    │   3: Replace    ($800)  │           │  + urgency + risk        │
    └────────────┬────────────┘           └──────────┬───────────────┘
                 │                                    │
                 └──────────────┬─────────────────────┘
                                ▼
                 ┌──────────────────────────────┐
                 │     STREAMLIT DASHBOARD      │
                 │                              │
                 │  ► Health & decision timeline │
                 │  ► Sensor trend subplots     │
                 │  ► Action distribution & cost│
                 │  ► AI explanation panel      │
                 │  ► Custom CSV data upload    │
                 └──────────────────────────────┘
```

---

## 📁 Project Structure

```
PdM/
├── app.py                  # Streamlit dashboard (main entry point)
├── pipeline.py             # End-to-end training pipeline
├── config.py               # Hyperparameters & constants
├── requirements.txt        # Python dependencies
├── CMAPSSData.zip           # NASA C-MAPSS dataset (FD001)
│
├── data_generator.py       # C-MAPSS data loader & preprocessor
├── lstm_model.py           # Bi-LSTM health predictor (PyTorch)
├── maintenance_env.py      # Custom Gymnasium RL environment
├── train_ppo.py            # PPO agent training
├── explainer.py            # Multi-provider LLM explanation engine
│
└── assets/                 # Saved model weights (auto-created)
    ├── lstm_model.pt       # Trained LSTM weights
    ├── scaler.pkl          # Feature scaler
    └── ppo_model.zip       # Trained PPO policy
```

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.10+**
- **pip** package manager
- *(Optional)* CUDA-capable GPU for faster training

### 1. Clone & Install

```bash
# Clone the repository
git clone <repo-url>
cd PdM

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows

# Install PyTorch (choose one):
pip install torch torchvision torchaudio                                    # CPU
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121  # CUDA 12.1

# Install remaining dependencies
pip install -r requirements.txt
```

### 2. Set API Key *(Optional)*

```bash
# For AI-powered explanations (pick one):
export GEMINI_API_KEY=your_key_here        # Google Gemini (default)
export OPENAI_API_KEY=your_key_here        # OpenAI GPT-4o-mini
export ANTHROPIC_API_KEY=your_key_here     # Anthropic Claude
```

> [!NOTE]
> **No API key?** The system automatically falls back to deterministic rule-based explanations. All other features work without any API key.

### 3. Train Models

```bash
# Quick start — train only the LSTM (~1 min on CPU)
python pipeline.py --skip-ppo

# Full pipeline — train LSTM + PPO agent (~5-10 min on CPU)
python pipeline.py
```

### 4. Launch Dashboard

```bash
streamlit run app.py
```

The dashboard opens at **http://localhost:8501** with the full interactive interface.

---

## 🧠 Model Details

### LSTM Health Predictor

| Parameter | Value |
|:---|:---|
| Architecture | Bi-directional LSTM |
| Layers | 2 stacked |
| Hidden size | 64 (× 2 bidirectional = 128) |
| Sequence length | 30 time-steps |
| Input features | 14 sensor channels |
| Output | Health score ∈ \[0, 1\] + 128-dim feature vector |
| Loss function | MSE |
| Optimizer | Adam + ReduceLROnPlateau |
| Dropout | 0.2 |

### PPO Decision Agent

| Parameter | Value |
|:---|:---|
| Framework | Stable-Baselines3 |
| Policy network | MLP (128 → 128) |
| Learning rate | 3 × 10⁻⁴ |
| Rollout steps | 256 |
| Batch size | 64 |
| Discount (γ) | 0.99 |
| Entropy coefficient | 0.01 |
| Total timesteps | 100,000 |

**Reward shaping:**

| Component | Formula | Purpose |
|:---|:---|:---|
| Action cost | `−cost / 100` | Penalize expensive actions |
| Downtime | `−downtime_penalty` | Penalize machine downtime |
| Health bonus | `+health × 2` | Reward maintaining health |
| Proactive bonus | `+1.5` | Reward acting before critical |
| Failure penalty | `−5000 / 1000` | Catastrophic failure cost |
| Waste penalty | `−2.0` | Penalize replacing healthy equipment |

### LLM Explainer

Supports **Google Gemini** (default), **OpenAI GPT-4o-mini**, and **Anthropic Claude** with automatic fallback chain. Prompt engineering ensures explanations include:

- Specific sensor anomalies and their significance
- Failure risk assessment if action is delayed
- Urgency classification (immediate / 24h / scheduled)
- Concise "bottom line" summary

---

## 📊 Dataset

This project uses the **NASA C-MAPSS** (Commercial Modular Aero-Propulsion System Simulation) turbofan engine degradation dataset.

| Property | Value |
|:---|:---|
| Subset | FD001 |
| Engines | 100 run-to-failure trajectories |
| Sensors | 14 informative (of 21 total) |
| Dropped | 7 near-constant/uninformative sensors |
| Health mapping | `health = min(RUL, 125) / 125` |

The dashboard also supports **custom CSV uploads** — bring your own sensor data with automatic column detection and health score generation.

---

## 📈 Expected Results

| Metric | Value |
|:---|:---|
| LSTM Validation RMSE | 0.04 – 0.08 |
| PPO Inspect threshold | ~55% health |
| PPO Repair threshold | ~35% health |
| PPO Replace threshold | ~20% health |
| Cost savings vs. random | 25–40% |

---

## ⚙ Configuration

All hyperparameters are centralized in [`config.py`](config.py). Key settings:

```python
# Sensor & sequence
SEQUENCE_LENGTH = 30
HEALTH_THRESHOLD_WARN = 0.40     # Below → "degraded"
HEALTH_THRESHOLD_CRIT = 0.20     # Below → "critical"

# LSTM
LSTM_HIDDEN_SIZE = 64
LSTM_NUM_LAYERS = 2
LSTM_EPOCHS = 30

# PPO
PPO_TOTAL_TIMESTEPS = 100_000

# LLM
LLM_PROVIDER = "gemini"          # "gemini" | "openai" | "anthropic"
```

---

## 🔧 Extending the System

<details>
<summary><b>📄 Use custom sensor data</b></summary>

Upload any CSV directly through the dashboard sidebar, or modify the data pipeline:

```python
# In data_generator.py, add a new loader:
def load_my_data():
    df = pd.read_csv("my_sensors.csv")
    # Ensure columns: machine_id, step, health, sensor_1, sensor_2, ...
    return df
```

</details>

<details>
<summary><b>🎯 Add more maintenance actions</b></summary>

```python
# In config.py:
ACTION_NAMES = {
    0: "Do Nothing",
    1: "Inspect",
    2: "Lubricate",    # new
    3: "Repair",
    4: "Replace",
}
ACTION_COSTS = {0: 0, 1: 50, 2: 100, 3: 300, 4: 800}
```

</details>

<details>
<summary><b>💬 Switch LLM provider</b></summary>

```python
# In config.py:
LLM_PROVIDER = "openai"     # or "anthropic" or "gemini"
LLM_MODEL_OPENAI = "gpt-4o-mini"
LLM_MODEL_ANTHROPIC = "claude-sonnet-4-20250514"
LLM_MODEL_GEMINI = "gemini-2.5-flash"
```

</details>

---

## 🛠 Troubleshooting

| Issue | Solution |
|:---|:---|
| `No module named 'gymnasium'` | `pip install gymnasium stable-baselines3` |
| LSTM model not found | Run `python pipeline.py --skip-ppo` |
| PPO warning in sidebar | Normal if `--skip-ppo` used; heuristic fallback is active |
| Slow training | Reduce `LSTM_EPOCHS` in `config.py` or use `--skip-ppo` |
| LLM returns generic text | Set `GEMINI_API_KEY` (or OpenAI/Anthropic) env variable |
| PyTorch CUDA error | Ensure CUDA toolkit matches your `torch` install version |
| Dashboard port conflict | `streamlit run app.py --server.port 8503` |

---

<div align="center">

### Tech Stack

**PyTorch** · **Stable-Baselines3** · **Gymnasium** · **Streamlit** · **Plotly** · **Google Gemini**

---

*Demonstrating the synergy of deep learning, reinforcement learning, and large language models in industrial AI systems.*

</div>
