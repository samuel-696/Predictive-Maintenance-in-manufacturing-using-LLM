# train_ppo.py
# Trains a PPO agent on the maintenance environment using Stable-Baselines3.

import sys
import os
import numpy as np

from config import PPO_TOTAL_TIMESTEPS, PPO_MODEL_PATH, RANDOM_SEED


def train_ppo(lstm_trainer, df_all):
    """
    Train PPO agent and return the trained model.

    Parameters
    ----------
    lstm_trainer : LSTMTrainer  (already trained + loaded)
    df_all       : DataFrame from generate_dataset()
    """
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
    from maintenance_env import MaintenanceEnv

    # Pick one machine for training, another for eval
    machine_ids = df_all["machine_id"].unique().tolist()
    train_id    = machine_ids[0]
    eval_id     = machine_ids[1] if len(machine_ids) > 1 else machine_ids[0]

    df_train = df_all[df_all["machine_id"] == train_id].copy()
    df_eval  = df_all[df_all["machine_id"] == eval_id].copy()

    def make_env(df):
        def _init():
            return MaintenanceEnv(lstm_trainer, df)
        return _init

    train_env = make_vec_env(make_env(df_train), n_envs=1, seed=RANDOM_SEED)
    eval_env  = make_vec_env(make_env(df_eval),  n_envs=1, seed=RANDOM_SEED + 1)

    # ── PPO hyper-parameters ─────────────────────────────────────────────────
    model = PPO(
        policy          = "MlpPolicy",
        env             = train_env,
        learning_rate   = 3e-4,
        n_steps         = 256,
        batch_size      = 64,
        n_epochs        = 10,
        gamma           = 0.99,
        gae_lambda      = 0.95,
        clip_range      = 0.2,
        ent_coef        = 0.01,   # encourages exploration
        vf_coef         = 0.5,
        max_grad_norm   = 0.5,
        verbose         = 1,
        seed            = RANDOM_SEED,
        policy_kwargs   = dict(net_arch=[128, 128]),
    )

    # ── callbacks ─────────────────────────────────────────────────────────────
    stop_cb = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=10, min_evals=20, verbose=1
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=os.path.dirname(PPO_MODEL_PATH),
        log_path=os.path.dirname(PPO_MODEL_PATH),
        eval_freq=2000,
        deterministic=True,
        callback_after_eval=stop_cb,
        verbose=1,
    )

    print(f"\nTraining PPO for up to {PPO_TOTAL_TIMESTEPS:,} timesteps …")
    model.learn(total_timesteps=PPO_TOTAL_TIMESTEPS, callback=eval_cb, progress_bar=True)

    os.makedirs(os.path.dirname(PPO_MODEL_PATH), exist_ok=True)
    # The EvalCallback saves the best model as best_model.zip. 
    # Let's copy/rename it to match PPO_MODEL_PATH.
    import shutil
    best_model_file = os.path.join(os.path.dirname(PPO_MODEL_PATH), "best_model.zip")
    if os.path.exists(best_model_file):
        shutil.copy(best_model_file, PPO_MODEL_PATH + ".zip")
        print(f"PPO best model copied → {PPO_MODEL_PATH}.zip")
    else:
        model.save(PPO_MODEL_PATH)
        print(f"PPO final model saved → {PPO_MODEL_PATH}.zip")
        
    return model


def load_ppo(path: str = PPO_MODEL_PATH):
    from stable_baselines3 import PPO
    model = PPO.load(path)
    print(f"PPO model loaded from {path}")
    return model


def run_episode(ppo_model, lstm_trainer, df_machine):
    """
    Run a single episode and return the episode log.

    Returns list of dicts with keys:
        step, health, action, action_name, reward, sensor readings
    """
    from maintenance_env import MaintenanceEnv
    from config import SENSOR_NAMES, SEQUENCE_LENGTH

    env = MaintenanceEnv(lstm_trainer, df_machine, render_mode=None)
    obs, _ = env.reset()
    done = False
    episode = []
    step = 0

    while not done:
        action, _ = ppo_model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env.step(int(action))

        row = {
            "step":   step + SEQUENCE_LENGTH,
            "action": int(action),
            **info,
            "reward": reward,
        }
        # Append raw sensor readings for the current step
        raw_idx = min(step + SEQUENCE_LENGTH, len(df_machine) - 1)
        for s in SENSOR_NAMES:
            row[s] = float(df_machine[s].iloc[raw_idx])
        episode.append(row)
        step += 1

    return episode


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from data_generator import generate_dataset
    from lstm_model import LSTMTrainer
    from data_generator import create_sequences

    print("=== Step 1: Generate data ===")
    df = generate_dataset()

    print("=== Step 2: Train LSTM ===")
    X, y = create_sequences(df)
    trainer = LSTMTrainer()
    trainer.train(X, y)
    trainer.save()

    print("=== Step 3: Train PPO ===")
    train_ppo(trainer, df)
