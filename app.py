# app.py — Streamlit dashboard for Predictive Maintenance AI
# Run with:  streamlit run app.py

from __future__ import annotations
import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config import (
    SENSOR_NAMES, SEQUENCE_LENGTH, ACTION_NAMES, ACTION_COSTS,
    LSTM_MODEL_PATH, SCALER_PATH, PPO_MODEL_PATH,
    HEALTH_THRESHOLD_WARN, HEALTH_THRESHOLD_CRIT, PAGE_TITLE,
)

# ─────────────────────────────────────────────────────────────────────────────
# Page setup
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Premium CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Google Font ─────────────────────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── Hide default Streamlit branding ─────────────────────────────────── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* ── Main background ─────────────────────────────────────────────────── */
    .stApp {
        background: linear-gradient(135deg, #0a0e1a 0%, #0d1525 40%, #101c30 100%);
    }

    /* ── Sidebar ─────────────────────────────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0c1220 0%, #0a0f1c 100%);
        border-right: 1px solid rgba(56, 189, 248, 0.08);
    }
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown li {
        color: #94a3b8;
        font-size: 0.88rem;
    }
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stMultiSelect label,
    section[data-testid="stSidebar"] .stRadio label,
    section[data-testid="stSidebar"] .stCheckbox label,
    section[data-testid="stSidebar"] .stTextInput label {
        color: #cbd5e1 !important;
        font-weight: 500;
    }

    /* ── Glass cards ─────────────────────────────────────────────────────── */
    .glass-card {
        background: rgba(15, 23, 42, 0.6);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(56, 189, 248, 0.1);
        border-radius: 16px;
        padding: 24px;
        transition: all 0.3s ease;
    }
    .glass-card:hover {
        border-color: rgba(56, 189, 248, 0.25);
        box-shadow: 0 8px 32px rgba(56, 189, 248, 0.08);
        transform: translateY(-2px);
    }

    /* ── KPI metric cards ────────────────────────────────────────────────── */
    .kpi-card {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.8) 0%, rgba(15, 23, 42, 0.4) 100%);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(56, 189, 248, 0.12);
        border-radius: 16px;
        padding: 20px 24px;
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    .kpi-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        border-radius: 16px 16px 0 0;
    }
    .kpi-card.blue::before   { background: linear-gradient(90deg, #0ea5e9, #38bdf8); }
    .kpi-card.green::before  { background: linear-gradient(90deg, #10b981, #34d399); }
    .kpi-card.amber::before  { background: linear-gradient(90deg, #f59e0b, #fbbf24); }
    .kpi-card.purple::before { background: linear-gradient(90deg, #8b5cf6, #a78bfa); }
    .kpi-card.red::before    { background: linear-gradient(90deg, #ef4444, #f87171); }
    .kpi-label {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #64748b;
        margin-bottom: 8px;
    }
    .kpi-value {
        font-size: 1.85rem;
        font-weight: 700;
        color: #e2e8f0;
        line-height: 1.1;
    }
    .kpi-delta {
        font-size: 0.8rem;
        font-weight: 500;
        margin-top: 6px;
    }
    .kpi-delta.positive { color: #34d399; }
    .kpi-delta.negative { color: #f87171; }
    .kpi-delta.neutral  { color: #64748b; }

    /* ── Header ──────────────────────────────────────────────────────────── */
    .dashboard-header {
        display: flex;
        align-items: center;
        gap: 16px;
        padding: 8px 0 4px 0;
    }
    .dashboard-title {
        font-size: 1.75rem;
        font-weight: 800;
        background: linear-gradient(135deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.02em;
    }
    .dashboard-subtitle {
        font-size: 0.85rem;
        color: #64748b;
        font-weight: 400;
        letter-spacing: 0.01em;
    }
    .dashboard-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(16, 185, 129, 0.1);
        border: 1px solid rgba(16, 185, 129, 0.2);
        color: #34d399;
        font-size: 0.72rem;
        font-weight: 600;
        padding: 4px 12px;
        border-radius: 100px;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }

    /* ── Dividers ────────────────────────────────────────────────────────── */
    .divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(56,189,248,0.15), transparent);
        margin: 20px 0;
        border: none;
    }

    /* ── Explainer card ──────────────────────────────────────────────────── */
    .info-card {
        background: linear-gradient(135deg, rgba(15,23,42,0.7) 0%, rgba(30,41,59,0.5) 100%);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(56, 189, 248, 0.1);
        border-radius: 14px;
        padding: 20px 24px;
    }
    .info-card-label {
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #64748b;
        margin-bottom: 8px;
    }
    .info-card-value {
        font-size: 1.6rem;
        font-weight: 700;
        line-height: 1.15;
    }
    .info-card-sub {
        font-size: 0.82rem;
        color: #94a3b8;
        margin-top: 4px;
    }

    /* ── Tab styling ─────────────────────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(15,23,42,0.5);
        border-radius: 12px;
        padding: 4px;
        border: 1px solid rgba(56,189,248,0.08);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        color: #94a3b8;
        font-weight: 500;
        font-size: 0.88rem;
        padding: 8px 20px;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(56,189,248,0.1) !important;
        color: #38bdf8 !important;
    }
    .stTabs [data-baseweb="tab-highlight"] {
        background-color: transparent !important;
    }
    .stTabs [data-baseweb="tab-border"] {
        display: none;
    }

    /* ── Status badges ───────────────────────────────────────────────────── */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-size: 0.78rem;
        font-weight: 600;
        padding: 5px 14px;
        border-radius: 100px;
    }
    .status-healthy {
        background: rgba(16,185,129,0.12);
        border: 1px solid rgba(16,185,129,0.2);
        color: #34d399;
    }
    .status-degraded {
        background: rgba(245,158,11,0.12);
        border: 1px solid rgba(245,158,11,0.2);
        color: #fbbf24;
    }
    .status-critical {
        background: rgba(239,68,68,0.12);
        border: 1px solid rgba(239,68,68,0.2);
        color: #f87171;
    }

    /* ── Sidebar section headers ─────────────────────────────────────────── */
    .sidebar-section {
        font-size: 0.7rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #475569;
        margin: 20px 0 10px 0;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .sidebar-section::after {
        content: '';
        flex: 1;
        height: 1px;
        background: rgba(71,85,105,0.3);
    }

    /* ── Button overrides ────────────────────────────────────────────────── */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #0ea5e9, #6366f1) !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        letter-spacing: 0.02em;
        transition: all 0.3s ease !important;
    }
    .stButton > button[kind="primary"]:hover {
        box-shadow: 0 4px 20px rgba(14,165,233,0.3) !important;
        transform: translateY(-1px);
    }

    /* ── Data table ──────────────────────────────────────────────────────── */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
    }

    /* ── Streamlit metric override ───────────────────────────────────────── */
    [data-testid="stMetric"] {
        background: rgba(15,23,42,0.5);
        border: 1px solid rgba(56,189,248,0.08);
        border-radius: 12px;
        padding: 14px 16px;
    }
    [data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
        font-size: 0.78rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    [data-testid="stMetricValue"] {
        color: #e2e8f0 !important;
        font-weight: 700 !important;
    }

    /* ── Upload area ─────────────────────────────────────────────────────── */
    .csv-requirements {
        background: linear-gradient(135deg, rgba(15,23,42,0.8), rgba(30,41,59,0.6));
        border: 1px solid rgba(56,189,248,0.12);
        border-radius: 12px;
        padding: 14px 16px;
        font-size: 0.82rem;
        line-height: 1.7;
        color: #94a3b8;
        margin-bottom: 12px;
    }
    .csv-requirements b { color: #cbd5e1; }
    .csv-requirements code {
        background: rgba(56,189,248,0.1);
        color: #38bdf8;
        padding: 1px 6px;
        border-radius: 4px;
        font-size: 0.78rem;
    }

    /* ── Footer ──────────────────────────────────────────────────────────── */
    .app-footer {
        text-align: center;
        padding: 24px 0 8px 0;
        color: #334155;
        font-size: 0.78rem;
        font-weight: 500;
        letter-spacing: 0.03em;
    }
    .app-footer a { color: #475569; text-decoration: none; }

    /* ── Scrollbar ────────────────────────────────────────────────────────── */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: rgba(56,189,248,0.15); border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(56,189,248,0.3); }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Colour helpers
# ─────────────────────────────────────────────────────────────────────────────
ACTION_COLOURS = {0: "#475569", 1: "#0ea5e9", 2: "#f59e0b", 3: "#ef4444"}
ACTION_ICONS   = {0: "💤", 1: "🔍", 2: "🔧", 3: "🔄"}

# Plotly theme overrides
PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(10,15,30,0.6)",
    font=dict(family="Inter, sans-serif", color="#94a3b8"),
    margin=dict(l=24, r=24, t=48, b=24),
)


def health_colour(h: float) -> str:
    if h > HEALTH_THRESHOLD_WARN: return "#34d399"
    if h > HEALTH_THRESHOLD_CRIT: return "#fbbf24"
    return "#f87171"


def health_label(h: float) -> str:
    if h > HEALTH_THRESHOLD_WARN: return "Healthy"
    if h > HEALTH_THRESHOLD_CRIT: return "Degraded"
    return "Critical"


def health_badge(h: float) -> str:
    label = health_label(h)
    cls = {"Healthy": "status-healthy", "Degraded": "status-degraded", "Critical": "status-critical"}[label]
    dot = {"Healthy": "🟢", "Degraded": "🟡", "Critical": "🔴"}[label]
    return f'<span class="status-badge {cls}">{dot} {label}</span>'


# ─────────────────────────────────────────────────────────────────────────────
# Cached resource loaders
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_lstm():
    from lstm_model import LSTMTrainer
    trainer = LSTMTrainer()
    if os.path.exists(LSTM_MODEL_PATH) and os.path.exists(SCALER_PATH):
        trainer.load()
        return trainer
    return None


@st.cache_resource
def load_ppo():
    if os.path.exists(PPO_MODEL_PATH + ".zip"):
        from stable_baselines3 import PPO
        return PPO.load(PPO_MODEL_PATH)
    return None


@st.cache_data
def load_data():
    from data_generator import generate_dataset
    return generate_dataset()


# ─────────────────────────────────────────────────────────────────────────────
# Custom CSV parser
# ─────────────────────────────────────────────────────────────────────────────
def parse_uploaded_csv(uploaded_file, machine_col, health_col, rul_col, sensor_cols):
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        return None, [], f"Failed to read CSV: {e}"

    if len(df) < SEQUENCE_LENGTH + 10:
        return None, [], f"Dataset too small — need at least {SEQUENCE_LENGTH + 10} rows."

    if machine_col and machine_col in df.columns:
        df = df.rename(columns={machine_col: "machine_id"})
    elif "machine_id" not in df.columns:
        df["machine_id"] = 1

    if "step" not in df.columns:
        df["step"] = df.groupby("machine_id").cumcount() + 1

    if health_col and health_col in df.columns:
        df = df.rename(columns={health_col: "health"})
        h_max = df["health"].max()
        if h_max > 1.5:
            df["health"] = df["health"] / h_max
    elif rul_col and rul_col in df.columns:
        df = df.rename(columns={rul_col: "rul"})
        cap = df["rul"].max()
        df["health"] = np.minimum(df["rul"], cap) / cap
    elif "health" not in df.columns:
        parts = []
        for _, grp in df.groupby("machine_id"):
            n = len(grp)
            grp = grp.copy()
            grp["health"] = np.linspace(1.0, 0.0, n)
            parts.append(grp)
        df = pd.concat(parts, ignore_index=True)

    if sensor_cols:
        detected = [c for c in sensor_cols if c in df.columns]
    else:
        skip = {"machine_id", "step", "health", "rul"}
        detected = [c for c in df.columns if c not in skip and pd.api.types.is_numeric_dtype(df[c])]

    if not detected:
        return None, [], "No numeric sensor columns detected in the uploaded file."

    return df, detected, None


# ─────────────────────────────────────────────────────────────────────────────
# Heuristic fallback
# ─────────────────────────────────────────────────────────────────────────────
def heuristic_action(health: float, time_since_maint: int) -> int:
    if health < HEALTH_THRESHOLD_CRIT:         return 3
    if health < HEALTH_THRESHOLD_WARN:         return 2
    if time_since_maint > 60 or health < 0.55: return 1
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# Run full simulation
# ─────────────────────────────────────────────────────────────────────────────
def run_simulation(lstm_trainer, ppo_model, df_machine: pd.DataFrame,
                   sensor_names: list[str] | None = None) -> list[dict]:
    from data_generator import create_sequences

    if sensor_names is None:
        sensor_names = SENSOR_NAMES

    results = []
    time_since_maint = 0
    df_m = df_machine.reset_index(drop=True)
    n = len(df_m) - SEQUENCE_LENGTH - 1
    can_predict = lstm_trainer is not None and set(SENSOR_NAMES).issubset(df_m.columns)

    for i in range(0, n, 2):
        if can_predict:
            window = df_m[SENSOR_NAMES].iloc[i : i + SEQUENCE_LENGTH].values[np.newaxis]
            health_pred, feat_vec = lstm_trainer.predict(window.astype("float32"))
            health = float(health_pred[0])
        else:
            health = float(df_m["health"].iloc[i + SEQUENCE_LENGTH])
            feat_vec = None

        if ppo_model is not None and can_predict and feat_vec is not None:
            true_health = float(df_m["health"].iloc[i + SEQUENCE_LENGTH])
            deg_rate    = 0.0 if i == 0 else (results[-1]["health"] - true_health)
            obs = np.concatenate([
                feat_vec[0],
                np.array([true_health, time_since_maint / 100.0, deg_rate], dtype="float32"),
            ])
            action, _ = ppo_model.predict(obs, deterministic=True)
            action = int(action)
        else:
            action = heuristic_action(health, time_since_maint)

        if action >= 2:
            boost  = 0.30 if action == 2 else (1.0 - health)
            health = min(1.0, health + boost)
            time_since_maint = 0
        elif action == 1:
            time_since_maint = 0
        else:
            time_since_maint += 2

        sensor_row = {s: float(df_m[s].iloc[i + SEQUENCE_LENGTH]) for s in sensor_names}
        results.append({
            "step": i + SEQUENCE_LENGTH,
            "health": health,
            "action": action,
            "action_name": ACTION_NAMES[action],
            "cost": ACTION_COSTS[action],
            "time_since_maint": time_since_maint,
            **sensor_row,
        })

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────────────────────
if "custom_df" not in st.session_state:
    st.session_state.custom_df      = None
    st.session_state.custom_sensors = []

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    # Sidebar header
    st.markdown("""
    <div style="text-align:center;padding:12px 0 4px 0;">
        <div style="font-size:1.4rem;font-weight:800;
                    background:linear-gradient(135deg,#38bdf8,#818cf8);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
            PdM AI
        </div>
        <div style="font-size:0.72rem;color:#475569;font-weight:500;letter-spacing:0.06em;
                    text-transform:uppercase;margin-top:2px;">
            Predictive Maintenance Suite
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── Model status ─────────────────────────────────────────────────────
    st.markdown('<div class="sidebar-section">System Status</div>', unsafe_allow_html=True)
    lstm_ok = os.path.exists(LSTM_MODEL_PATH) and os.path.exists(SCALER_PATH)
    ppo_ok  = os.path.exists(PPO_MODEL_PATH + ".zip")

    status_html = ""
    for label, ok, fallback in [
        ("LSTM Health Predictor", lstm_ok, ""),
        ("PPO Decision Agent", ppo_ok, "heuristic fallback"),
    ]:
        icon = "✅" if ok else "⚠️"
        color = "#34d399" if ok else "#fbbf24"
        note = f' <span style="color:#64748b;font-size:0.72rem;">({fallback})</span>' if not ok and fallback else ""
        status_html += f'<div style="display:flex;align-items:center;gap:8px;padding:4px 0;font-size:0.82rem;color:{color};">{icon} {label}{note}</div>'
    st.markdown(status_html, unsafe_allow_html=True)

    if not lstm_ok:
        st.warning("Run `python pipeline.py --skip-ppo` first to train the LSTM.")
        if st.button("🚀 Train LSTM now", type="primary"):
            with st.spinner("Training LSTM (~1 min)…"):
                from pipeline import run_pipeline
                run_pipeline(skip_ppo=True)
            st.success("Done!"); st.rerun()

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── Data Source ──────────────────────────────────────────────────────
    st.markdown('<div class="sidebar-section">Data Source</div>', unsafe_allow_html=True)
    data_source = st.radio(
        "Choose data source",
        ["🛰️ NASA C-MAPSS (built-in)", "📄 Upload custom CSV"],
        index=0,
        label_visibility="collapsed",
        help="Use the built-in C-MAPSS turbofan data or upload your own sensor CSV.",
    )

    use_custom = data_source.startswith("📄")

    if use_custom:
        st.markdown("""
        <div class="csv-requirements">
        <b>CSV requirements</b><br/>
        &bull; One row per time-step per machine<br/>
        &bull; Numeric sensor columns<br/>
        &bull; Optionally include <code>machine_id</code>,
          <code>health</code> or <code>RUL</code> columns
        </div>
        """, unsafe_allow_html=True)

        uploaded = st.file_uploader(
            "Upload sensor CSV", type=["csv"], label_visibility="collapsed",
            help="Upload a CSV with sensor readings. The app will auto-detect columns.",
        )

        if uploaded is not None:
            peek_df = pd.read_csv(uploaded)
            uploaded.seek(0)
            all_cols = peek_df.columns.tolist()
            none_opt = ["— auto-detect —"]

            machine_col = st.selectbox("Machine / unit ID column", none_opt + all_cols, index=0)
            health_col  = st.selectbox("Health score column", none_opt + all_cols, index=0)
            rul_col     = st.selectbox("RUL column (if no health)", none_opt + all_cols, index=0)

            numeric_cols = [c for c in all_cols if pd.api.types.is_numeric_dtype(peek_df[c])]
            sensor_cols  = st.multiselect("Sensor columns (blank = auto)", numeric_cols, default=[])

            if st.button("✅ Load uploaded data", type="primary"):
                _mc = machine_col if machine_col != none_opt[0] else None
                _hc = health_col  if health_col  != none_opt[0] else None
                _rc = rul_col     if rul_col     != none_opt[0] else None
                df_custom, det_sensors, err = parse_uploaded_csv(uploaded, _mc, _hc, _rc, sensor_cols or None)
                if err:
                    st.error(err)
                else:
                    st.session_state.custom_df      = df_custom
                    st.session_state.custom_sensors = det_sensors
                    st.success(f"✅ {len(df_custom):,} rows · {df_custom['machine_id'].nunique()} machine(s) · {len(det_sensors)} sensors")
                    st.rerun()

        if st.session_state.custom_df is not None:
            st.caption(f"📊 Custom data active — {len(st.session_state.custom_df):,} rows")
            if st.button("🔄 Reset to C-MAPSS"):
                st.session_state.custom_df = None
                st.session_state.custom_sensors = []
                st.rerun()

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── Simulation Settings ──────────────────────────────────────────────
    st.markdown('<div class="sidebar-section">Simulation</div>', unsafe_allow_html=True)

    if use_custom and st.session_state.custom_df is not None:
        df_all         = st.session_state.custom_df
        active_sensors = st.session_state.custom_sensors
    else:
        df_all         = load_data()
        active_sensors = SENSOR_NAMES

    machine_ids = sorted(df_all["machine_id"].unique().tolist())
    selected_machine = st.selectbox(
        "Engine", machine_ids,
        format_func=lambda x: f"Engine {int(x)}"
    )

    show_sensors = st.multiselect(
        "Display sensors",
        active_sensors,
        default=active_sensors[:3],
    )

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── API Key ──────────────────────────────────────────────────────────
    st.markdown('<div class="sidebar-section">AI Explainer</div>', unsafe_allow_html=True)
    api_key = st.text_input("Gemini API Key", type="password",
                            help="Used for AI-powered explanations. Leave blank for rule-based fallback.",
                            label_visibility="collapsed",
                            placeholder="Paste your Gemini API Key…")
    if api_key:
        os.environ["GEMINI_API_KEY"] = api_key

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── Footer info ──────────────────────────────────────────────────────
    st.markdown("""
    <div style="font-size:0.72rem;color:#334155;line-height:1.7;text-align:center;padding:8px 0;">
        LSTM · PPO · Gemini<br/>
        NASA C-MAPSS Dataset<br/>
        PyTorch &amp; Streamlit
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="dashboard-header">
    <div>
        <span class="dashboard-title">🔧 Predictive Maintenance AI</span><br/>
        <span class="dashboard-subtitle">
            Real-time health monitoring &amp; intelligent maintenance scheduling
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

# Badge row
col_badge1, col_badge2, col_badge3, _ = st.columns([1, 1, 1, 4])
with col_badge1:
    src_label = "Custom CSV" if (use_custom and st.session_state.custom_df is not None) else "C-MAPSS"
    st.markdown(f'<span class="dashboard-badge">📡 {src_label}</span>', unsafe_allow_html=True)
with col_badge2:
    model_label = "LSTM + PPO" if (os.path.exists(LSTM_MODEL_PATH) and os.path.exists(PPO_MODEL_PATH + ".zip")) else "LSTM + Heuristic"
    st.markdown(f'<span class="dashboard-badge">🧠 {model_label}</span>', unsafe_allow_html=True)
with col_badge3:
    eng_count = df_all["machine_id"].nunique()
    st.markdown(f'<span class="dashboard-badge">⚙️ {eng_count} Engines</span>', unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ── Load models ──────────────────────────────────────────────────────────────
lstm_trainer = load_lstm()
ppo_model    = load_ppo()

if lstm_trainer is None and not use_custom:
    st.error("⚠️ LSTM model not found. Train first: `python pipeline.py --skip-ppo`")
    st.stop()
elif lstm_trainer is None and use_custom:
    st.info("ℹ️ LSTM not trained — using health scores from uploaded data.")

# ── Run simulation ────────────────────────────────────────────────────────────
df_machine = df_all[df_all["machine_id"] == selected_machine].copy()

with st.spinner("Running simulation…"):
    simulation = run_simulation(lstm_trainer, ppo_model, df_machine, sensor_names=active_sensors)

sim_df = pd.DataFrame(simulation)

# ─────────────────────────────────────────────────────────────────────────────
# KPI cards — top row
# ─────────────────────────────────────────────────────────────────────────────
current     = sim_df.iloc[-1]
avg_health  = sim_df["health"].mean()
total_cost  = sim_df["cost"].sum()
min_health  = sim_df["health"].min()
h_now       = float(current["health"])
h_delta     = (h_now - sim_df["health"].iloc[max(0, len(sim_df) - 10)]) * 100

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    delta_cls = "positive" if h_delta >= 0 else "negative"
    delta_arrow = "↑" if h_delta >= 0 else "↓"
    st.markdown(f"""
    <div class="kpi-card blue">
        <div class="kpi-label">Current Health</div>
        <div class="kpi-value">{h_now*100:.1f}%</div>
        <div class="kpi-delta {delta_cls}">{delta_arrow} {abs(h_delta):.1f}pp</div>
    </div>
    """, unsafe_allow_html=True)
with c2:
    st.markdown(f"""
    <div class="kpi-card green">
        <div class="kpi-label">Avg Health</div>
        <div class="kpi-value">{avg_health*100:.1f}%</div>
        <div class="kpi-delta neutral">{health_badge(avg_health)}</div>
    </div>
    """, unsafe_allow_html=True)
with c3:
    st.markdown(f"""
    <div class="kpi-card amber">
        <div class="kpi-label">Total Maint. Cost</div>
        <div class="kpi-value">${total_cost:,.0f}</div>
        <div class="kpi-delta neutral">{len(sim_df)} cycles</div>
    </div>
    """, unsafe_allow_html=True)
with c4:
    action_now = int(current["action"])
    st.markdown(f"""
    <div class="kpi-card purple">
        <div class="kpi-label">Last Decision</div>
        <div class="kpi-value">{ACTION_ICONS[action_now]}</div>
        <div class="kpi-delta neutral">{current["action_name"]}</div>
    </div>
    """, unsafe_allow_html=True)
with c5:
    st.markdown(f"""
    <div class="kpi-card red">
        <div class="kpi-label">Min Health</div>
        <div class="kpi-value">{min_health*100:.1f}%</div>
        <div class="kpi-delta neutral">{health_badge(min_health)}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Main charts
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📈  Health & Decisions",
    "🌡️  Sensor Trends",
    "📊  Action Analytics",
    "🤖  AI Explainer",
])

# ── Tab 1 ─────────────────────────────────────────────────────────────────────
with tab1:
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Machine Health Score", "Maintenance Actions Timeline"),
        row_heights=[0.65, 0.35],
    )

    # Health curve with gradient fill
    fig.add_trace(go.Scatter(
        x=sim_df["step"], y=sim_df["health"],
        mode="lines", name="Health",
        line=dict(color="#38bdf8", width=2.5),
        fill="tozeroy",
        fillcolor="rgba(56,189,248,0.06)",
    ), row=1, col=1)

    # Threshold bands
    fig.add_hrect(y0=0, y1=HEALTH_THRESHOLD_CRIT, fillcolor="rgba(239,68,68,0.06)",
                  line_width=0, row=1, col=1)
    fig.add_hrect(y0=HEALTH_THRESHOLD_CRIT, y1=HEALTH_THRESHOLD_WARN,
                  fillcolor="rgba(245,158,11,0.05)", line_width=0, row=1, col=1)
    fig.add_hline(y=HEALTH_THRESHOLD_WARN, line=dict(color="rgba(251,191,36,0.4)", dash="dot", width=1), row=1, col=1)
    fig.add_hline(y=HEALTH_THRESHOLD_CRIT, line=dict(color="rgba(248,113,113,0.4)", dash="dot", width=1), row=1, col=1)

    # Action markers
    marker_sizes  = {0: 5, 1: 7, 2: 9, 3: 11}
    marker_shapes = {0: "circle", 1: "diamond", 2: "square", 3: "star"}
    for action_id, icon in ACTION_ICONS.items():
        mask = sim_df["action"] == action_id
        if mask.any():
            sub = sim_df[mask]
            fig.add_trace(go.Scatter(
                x=sub["step"], y=sub["health"],
                mode="markers",
                name=f"{icon} {ACTION_NAMES[action_id]}",
                marker=dict(
                    color=ACTION_COLOURS[action_id],
                    size=marker_sizes[action_id],
                    symbol=marker_shapes[action_id],
                    line=dict(width=1, color="rgba(255,255,255,0.2)"),
                ),
            ), row=1, col=1)

    # Action timeline
    for action_id in range(4):
        mask = sim_df["action"] == action_id
        if mask.any():
            fig.add_trace(go.Bar(
                x=sim_df[mask]["step"], y=[1] * mask.sum(),
                name=ACTION_NAMES[action_id],
                marker_color=ACTION_COLOURS[action_id],
                marker_line_width=0,
                showlegend=False,
            ), row=2, col=1)

    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=540,
        legend=dict(orientation="h", yanchor="bottom", y=1.04, bgcolor="rgba(0,0,0,0)"),
        barmode="stack",
    )
    fig.update_yaxes(title_text="Health [0–1]", range=[0, 1.05], row=1, col=1,
                     gridcolor="rgba(56,189,248,0.05)")
    fig.update_yaxes(showticklabels=False, row=2, col=1)
    fig.update_xaxes(title_text="Time Step", row=2, col=1,
                     gridcolor="rgba(56,189,248,0.05)")
    st.plotly_chart(fig, use_container_width=True)


# ── Tab 2 ─────────────────────────────────────────────────────────────────────
with tab2:
    if not show_sensors:
        st.info("Select sensors in the sidebar to display.")
    else:
        rows = (len(show_sensors) + 1) // 2
        cols = 2
        fig2 = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=[s.replace("_", " ").upper() for s in show_sensors],
            vertical_spacing=0.14,
            horizontal_spacing=0.08,
        )
        palette = ["#38bdf8", "#818cf8", "#34d399", "#fbbf24", "#f87171", "#c084fc", "#fb923c"]
        for idx, sensor in enumerate(show_sensors):
            r, c = divmod(idx, cols)
            fig2.add_trace(go.Scatter(
                x=sim_df["step"], y=sim_df[sensor],
                mode="lines",
                name=sensor.replace("_", " ").title(),
                line=dict(color=palette[idx % len(palette)], width=1.8),
                fill="tozeroy",
                fillcolor=f"rgba({','.join(str(int(palette[idx % len(palette)][i:i+2], 16)) for i in (1,3,5))},0.04)",
            ), row=r + 1, col=c + 1)

            repair_mask = sim_df["action"] >= 2
            if repair_mask.any():
                fig2.add_trace(go.Scatter(
                    x=sim_df[repair_mask]["step"],
                    y=sim_df[repair_mask][sensor],
                    mode="markers", showlegend=False,
                    marker=dict(color="#f59e0b", size=6, symbol="diamond",
                                line=dict(width=1, color="rgba(255,255,255,0.2)")),
                ), row=r + 1, col=c + 1)

        fig2.update_layout(
            **PLOTLY_LAYOUT,
            height=max(340, rows * 230),
            showlegend=False,
        )
        st.plotly_chart(fig2, use_container_width=True)
        st.caption("🔶 Orange diamonds mark repair / replace events")


# ── Tab 3 ─────────────────────────────────────────────────────────────────────
with tab3:
    col_a, col_b = st.columns(2)

    with col_a:
        action_counts = sim_df["action"].value_counts().reset_index()
        action_counts.columns = ["action_id", "count"]
        action_counts["label"] = action_counts["action_id"].map(
            lambda x: f"{ACTION_ICONS[x]} {ACTION_NAMES[x]}"
        )
        action_counts["colour"] = action_counts["action_id"].map(ACTION_COLOURS)

        fig3a = go.Figure(go.Pie(
            labels=action_counts["label"],
            values=action_counts["count"],
            marker=dict(colors=action_counts["colour"].tolist(),
                        line=dict(color="rgba(10,15,30,0.8)", width=2)),
            hole=0.55,
            textinfo="percent",
            textfont=dict(size=12, color="#e2e8f0"),
            hoverinfo="label+value+percent",
        ))
        fig3a.update_layout(
            **PLOTLY_LAYOUT,
            title=dict(text="Action Distribution", font=dict(size=14, color="#cbd5e1")),
            height=360,
        )
        st.plotly_chart(fig3a, use_container_width=True)

    with col_b:
        cost_by_action = sim_df.groupby("action")["cost"].sum().reset_index()
        cost_by_action["label"] = cost_by_action["action"].map(ACTION_NAMES)
        cost_by_action["colour"] = cost_by_action["action"].map(ACTION_COLOURS)

        fig3b = go.Figure(go.Bar(
            x=cost_by_action["label"],
            y=cost_by_action["cost"],
            marker_color=cost_by_action["colour"].tolist(),
            marker_line_width=0,
            text=cost_by_action["cost"].apply(lambda v: f"${v:,.0f}"),
            textposition="auto",
            textfont=dict(size=11, color="#e2e8f0"),
        ))
        fig3b.update_layout(
            **PLOTLY_LAYOUT,
            title=dict(text="Cost by Action Type", font=dict(size=14, color="#cbd5e1")),
            yaxis_title="Cost ($)", height=360,
        )
        st.plotly_chart(fig3b, use_container_width=True)

    # Health coloured by action
    fig3c = go.Figure()
    fig3c.add_trace(go.Scatter(
        x=sim_df["step"], y=sim_df["health"],
        mode="lines", line=dict(color="rgba(148,163,184,0.15)", width=1),
        name="Health", showlegend=False,
    ))
    for action_id in range(4):
        m = sim_df["action"] == action_id
        if m.any():
            fig3c.add_trace(go.Scatter(
                x=sim_df[m]["step"], y=sim_df[m]["health"],
                mode="markers",
                name=f"{ACTION_ICONS[action_id]} {ACTION_NAMES[action_id]}",
                marker=dict(color=ACTION_COLOURS[action_id], size=5,
                            line=dict(width=0.5, color="rgba(255,255,255,0.1)")),
            ))
    fig3c.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="Health Score Coloured by Action", font=dict(size=14, color="#cbd5e1")),
        height=300,
        xaxis_title="Step", yaxis_title="Health",
    )
    st.plotly_chart(fig3c, use_container_width=True)


# ── Tab 4: AI Explainer ───────────────────────────────────────────────────────
with tab4:
    st.markdown("""
    <div style="margin-bottom:8px;">
        <span style="font-size:1.1rem;font-weight:700;color:#e2e8f0;">🤖 Decision Explainer</span>
        <span style="font-size:0.82rem;color:#64748b;margin-left:8px;">
            Select a step to understand the AI's reasoning
        </span>
    </div>
    """, unsafe_allow_html=True)

    step_options = sim_df["step"].tolist()
    selected_step = st.select_slider(
        "Select step to explain",
        options=step_options,
        value=step_options[len(step_options) // 2],
        label_visibility="collapsed",
    )

    row = sim_df[sim_df["step"] == selected_step].iloc[0]
    action  = int(row["action"])
    health  = float(row["health"])

    col_x, col_y, col_z = st.columns(3)
    with col_x:
        h_color = health_colour(health)
        st.markdown(f"""
        <div class="info-card">
            <div class="info-card-label">Health Score</div>
            <div class="info-card-value" style="color:{h_color};">{health*100:.1f}%</div>
            <div class="info-card-sub">{health_badge(health)}</div>
        </div>
        """, unsafe_allow_html=True)
    with col_y:
        a_color = ACTION_COLOURS[action]
        st.markdown(f"""
        <div class="info-card">
            <div class="info-card-label">AI Decision</div>
            <div class="info-card-value" style="color:{a_color};">
                {ACTION_ICONS[action]} {ACTION_NAMES[action]}
            </div>
            <div class="info-card-sub">Cost: ${ACTION_COSTS[action]:,}</div>
        </div>
        """, unsafe_allow_html=True)
    with col_z:
        st.markdown(f"""
        <div class="info-card">
            <div class="info-card-label">Time Step</div>
            <div class="info-card-value" style="color:#e2e8f0;">{selected_step}</div>
            <div class="info-card-sub">of {max(step_options)} total</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")

    # Sensor snapshot
    st.markdown("""
    <div style="font-size:0.75rem;font-weight:600;text-transform:uppercase;
                letter-spacing:0.08em;color:#64748b;margin-bottom:8px;">
        📡 Sensor Readings
    </div>
    """, unsafe_allow_html=True)

    sensor_vals = {s: float(row[s]) for s in active_sensors if s in row.index}
    sens_display = list(sensor_vals.items())
    for chunk_start in range(0, len(sens_display), 7):
        chunk = sens_display[chunk_start:chunk_start + 7]
        sens_cols = st.columns(len(chunk))
        for col, (s, v) in zip(sens_cols, chunk):
            col.metric(s.replace("_", " ").upper(), f"{v:.2f}")

    st.markdown("")

    if st.button("💬 Generate AI Explanation", type="primary", use_container_width=True):
        from explainer import MaintenanceExplainer
        explainer = MaintenanceExplainer()
        with st.spinner("Generating explanation…"):
            explanation = explainer.explain(
                sensor_readings=sensor_vals,
                health_score=health,
                action=action,
                step=selected_step,
            )

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="font-size:0.75rem;font-weight:600;text-transform:uppercase;
                    letter-spacing:0.08em;color:#64748b;margin-bottom:8px;">
            🤖 AI Analysis
        </div>
        """, unsafe_allow_html=True)
        st.info(explanation)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="font-size:0.75rem;font-weight:600;text-transform:uppercase;
                letter-spacing:0.08em;color:#64748b;margin-bottom:8px;">
        📋 Recent Decisions
    </div>
    """, unsafe_allow_html=True)
    last_10 = sim_df.tail(10)[["step", "health", "action_name", "cost"]].copy()
    last_10["health"] = (last_10["health"] * 100).round(1).astype(str) + "%"
    last_10["cost"]   = last_10["cost"].apply(lambda c: f"${c:,}")
    last_10 = last_10.rename(columns={
        "step": "Step", "health": "Health", "action_name": "Action", "cost": "Cost"
    })
    st.dataframe(last_10, use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown("""
<div class="app-footer">
    Built with PyTorch · Stable-Baselines3 · Streamlit<br/>
    Powered by NASA C-MAPSS Turbofan Engine Data
</div>
""", unsafe_allow_html=True)
