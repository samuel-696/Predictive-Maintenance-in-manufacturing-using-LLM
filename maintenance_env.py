# maintenance_env.py
# Custom Gymnasium environment for the maintenance decision agent.
#
# State  : LSTM feature vector + scalar health + time-since-last-maintenance
# Actions: 0=DoNothing | 1=Inspect | 2=Repair | 3=Replace
# Reward : balances action cost, failure penalty, and proactive care bonus

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import (
    SENSOR_NAMES, SEQUENCE_LENGTH, NUM_ACTIONS,
    ACTION_COSTS, FAILURE_PENALTY, DOWNTIME_PENALTY,
    LSTM_HIDDEN_SIZE, HEALTH_THRESHOLD_WARN, HEALTH_THRESHOLD_CRIT,
)


# ─────────────────────────────────────────────────────────────────────────────
# State dimensionality
# ─────────────────────────────────────────────────────────────────────────────
FEATURE_DIM       = LSTM_HIDDEN_SIZE * 2  # bi-LSTM output
EXTRA_DIMS        = 3                      # health, time_since_maint, degradation_rate
STATE_DIM         = FEATURE_DIM + EXTRA_DIMS


class MaintenanceEnv(gym.Env):
    """
    Single-machine predictive maintenance environment.

    At each step the agent observes the current machine state (an LSTM-derived
    feature vector enriched with scalar health signals) and selects a
    maintenance action.  The environment advances the machine's health by one
    step and returns a reward.

    Episode ends when:
      • health < 0.05  (machine failure), OR
      • max_steps reached

    Parameters
    ----------
    lstm_trainer : LSTMTrainer
        Trained LSTM used to encode raw sensor windows.
    df_machine : pd.DataFrame
        Sensor data for one machine run  (shape: steps × sensors+meta).
    render_mode : str | None
        Only "human" is supported (prints to stdout).
    """

    metadata = {"render_modes": ["human"], "render_fps": 1}

    def __init__(self, lstm_trainer, df_machine, render_mode=None):
        super().__init__()

        self.lstm_trainer = lstm_trainer
        self.df           = df_machine.reset_index(drop=True)
        self.max_steps    = len(self.df) - SEQUENCE_LENGTH - 1
        self.render_mode  = render_mode

        # ── spaces ──────────────────────────────────────────────────────────
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(STATE_DIM,), dtype=np.float32,
        )
        self.action_space = spaces.Discrete(NUM_ACTIONS)

        # Internal state
        self._step_idx         = 0
        self._health           = 1.0
        self._time_since_maint = 0
        self._prev_health      = 1.0
        self._is_failed        = False
        self._episode_log: list[dict] = []

    # ── helpers ───────────────────────────────────────────────────────────────
    def _get_sensor_window(self) -> np.ndarray:
        """Return the current LSTM input window (1, seq_len, num_sensors)."""
        start = self._step_idx
        end   = start + SEQUENCE_LENGTH
        window = self.df[SENSOR_NAMES].iloc[start:end].values.astype(np.float32)
        return window[np.newaxis]   # (1, seq_len, features)

    def _encode_state(self) -> np.ndarray:
        """Run LSTM on current window; append scalar extras."""
        window = self._get_sensor_window()
        health_pred, feat_vec = self.lstm_trainer.predict(window)

        # Use true health (available in simulation); in prod use LSTM prediction
        true_health = float(self.df["health"].iloc[self._step_idx + SEQUENCE_LENGTH])
        degradation_rate = float(self._prev_health - true_health)

        extras = np.array(
            [true_health, self._time_since_maint / 100.0, degradation_rate],
            dtype=np.float32,
        )
        return np.concatenate([feat_vec[0], extras])

    # ── Gymnasium API ─────────────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._step_idx         = 0
        self._health           = 1.0
        self._prev_health      = 1.0
        self._time_since_maint = 0
        self._is_failed        = False
        self._episode_log      = []
        obs = self._encode_state()
        return obs, {}

    def step(self, action: int):
        current_health = float(
            self.df["health"].iloc[self._step_idx + SEQUENCE_LENGTH]
        )
        self._health = current_health

        # ── apply action effect ───────────────────────────────────────────────
        action_cost = ACTION_COSTS[action]
        health_boost = 0.0
        downtime     = 0

        if action == 0:   # do nothing
            self._time_since_maint += 1

        elif action == 1:  # inspect  — no health change, small cost
            self._time_since_maint = 0
            downtime = 1   # 1 step downtime

        elif action == 2:  # repair   — partial health restoration
            health_boost = min(0.30, 1.0 - current_health)
            self._time_since_maint = 0
            downtime = 3

        elif action == 3:  # replace  — full restoration
            health_boost = 1.0 - current_health
            self._time_since_maint = 0
            downtime = 8

        effective_health = np.clip(current_health + health_boost, 0.0, 1.0)

        # ── reward shaping ────────────────────────────────────────────────────
        #  1. Penalise action cost
        reward = -action_cost / 100.0

        #  2. Penalise downtime
        reward -= downtime * DOWNTIME_PENALTY / 1000.0

        #  3. Big bonus for staying healthy
        reward += effective_health * 2.0

        #  4. Proactive bonus: acting before critical threshold
        if action >= 1 and current_health < HEALTH_THRESHOLD_WARN:
            reward += 1.5   # rewarded for intervening while there's still time

        #  5. Failure penalty
        if effective_health < 0.05:
            reward -= FAILURE_PENALTY / 1000.0
            self._is_failed = True

        #  6. Unnecessary replacement penalty (wasteful if nearly healthy)
        if action == 3 and current_health > 0.70:
            reward -= 2.0

        # ── advance ───────────────────────────────────────────────────────────
        self._prev_health  = current_health
        self._step_idx    += 1
        done = self._is_failed or (self._step_idx >= self.max_steps)

        # Collect next observation
        if done:
            obs = np.zeros(STATE_DIM, dtype=np.float32)
        else:
            obs = self._encode_state()

        info = {
            "health":           effective_health,
            "action_name":      ["Do Nothing", "Inspect", "Repair", "Replace"][action],
            "action_cost":      action_cost,
            "downtime":         downtime,
            "failed":           self._is_failed,
        }
        self._episode_log.append(info)

        if self.render_mode == "human":
            self.render()

        return obs, float(reward), done, False, info

    def render(self):
        if self._episode_log:
            last = self._episode_log[-1]
            print(
                f"Step {self._step_idx:4d} | Health: {last['health']:.3f} | "
                f"Action: {last['action_name']:<12} | Reward components logged"
            )

    def get_episode_log(self) -> list[dict]:
        return self._episode_log
