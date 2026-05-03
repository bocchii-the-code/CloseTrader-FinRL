#%%
"""
Stock NeurIPS2018 Part 2. Train

This series is a reproduction of paper "Deep reinforcement learning for
automated stock trading: An ensemble strategy".

Introduce how to use FinRL to make data into the gym form environment, and train DRL agents on it.
"""

from __future__ import annotations

import os
import pandas as pd

# Suppress Pandas4Warning from yfinance/pandas: monkey-patch because
# regular warnings.filterwarnings() does not catch C-level deprecations.
_pd_utcnow = getattr(pd.Timestamp, "utcnow", None)
if _pd_utcnow is not None:
    pd.Timestamp.utcnow = lambda: pd.Timestamp.now("UTC")
from stable_baselines3.common.logger import configure

from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.config import INDICATORS
from finrl.config import RESULTS_DIR
from finrl.config import TRAINED_MODEL_DIR
from finrl.main import check_and_make_directories
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv

VALID_MODELS = {"a2c", "ddpg", "ppo", "td3", "sac"}
VALID_MODELS_LIST = sorted(VALID_MODELS)

BUILTIN_MODEL_PARAMS: dict[str, dict] = {
    "ppo": {
        "n_steps": 1024, "batch_size": 256, "n_epochs": 10,
        "learning_rate": 3e-4, "gamma": 0.995, "gae_lambda": 0.99,
        "clip_range": 0.2, "ent_coef": 0.02, "vf_coef": 0.5,
        "max_grad_norm": 0.5, "normalize_advantage": True,
        "policy_kwargs": {"net_arch": [256, 128], "ortho_init": True},
    },
    "ddpg": {"batch_size": 128, "buffer_size": 50_000, "learning_rate": 1e-3},
    "td3": {"batch_size": 100, "buffer_size": 1_000_000, "learning_rate": 1e-3},
    "sac": {
        "batch_size": 128, "buffer_size": 100_000, "learning_rate": 1e-4,
        "learning_starts": 100, "ent_coef": "auto_0.1",
    },
}

# %% Part 1. Prepare directories
def make_trained_model_dir(save_path = TRAINED_MODEL_DIR) -> str:

    check_and_make_directories([TRAINED_MODEL_DIR])
    
    return save_path

# %% Part 2. Build environment
def build_environment(train_data_path: str = None) -> StockTradingEnv:
# Use absolute path to read train data
    '''
    Build the StockTradingEnv environment using the training data.
    Args:
        train_data_path (str): The path to the training data CSV file. If None, defaults to "train_data.csv" in the parent directory.
    Returns:
        StockTradingEnv: The initialized StockTradingEnv environment.
    '''
    parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if train_data_path is None else os.path.dirname(os.path.abspath(train_data_path))
    train = pd.read_csv(os.path.join(parent_path, "train_data.csv"))
    train = train.set_index(train.columns[0])
    train.index.names = [""]

    stock_dimension = len(train.tic.unique())
    state_space = 1 + 2 * stock_dimension + len(INDICATORS) * stock_dimension
    print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

    buy_cost_list = sell_cost_list = [0.001] * stock_dimension
    num_stock_shares = [0] * stock_dimension

    env_kwargs = {
        "hmax": 100,
        "initial_amount": 1000000,
        "num_stock_shares": num_stock_shares,
        "buy_cost_pct": buy_cost_list,
        "sell_cost_pct": sell_cost_list,
        "state_space": state_space,
        "stock_dim": stock_dimension,
        "tech_indicator_list": INDICATORS,
        "action_space": stock_dimension,
        "reward_scaling": 1e-4,
    }

    e_train_gym = StockTradingEnv(df=train, **env_kwargs)
    env_train, _ = e_train_gym.get_sb_env()
    print(type(env_train))
    return env_train

# %% Part 3. Train DRL Agents
def train_drl_agents(
        env_train: StockTradingEnv,
        total_timesteps: int = 30000,
        save_path: str = TRAINED_MODEL_DIR,
        models: list[str] | None = None,
        model_params: dict[str, dict] | None = None,
        ) -> dict[str, DRLAgent]:
    '''Train DRL agents and save trained models.

    Args:
        env_train: The training environment.
        total_timesteps: Training steps per agent.
        save_path: Directory for saving model files.
        models: Algorithms to train. Default all five: ["a2c", ..., "sac"].
        model_params: Per-model hyperparameter overrides, merged with BUILTIN_MODEL_PARAMS.
    Returns:
        Dict mapping algorithm name -> trained model (or None if skipped).
    '''
    if models is None:
        models = VALID_MODELS_LIST

    # Validate model names
    invalid = set(models) - VALID_MODELS
    if invalid:
        raise ValueError(f"Invalid model name(s): {invalid}. Valid: {VALID_MODELS}")

    # Merge built-in defaults with caller overrides (caller wins)
    merged_params = {k: dict(v) for k, v in BUILTIN_MODEL_PARAMS.items()}
    if model_params is not None:
        for name in model_params:
            if name in merged_params:
                merged_params[name] = {**merged_params[name], **model_params[name]}

    trained = {}
    for name in ["a2c", "ddpg", "ppo", "td3", "sac"]:
        if name not in models:
            trained[name] = None
            continue

        agent = DRLAgent(env=env_train)
        kwargs = merged_params.get(name)
        model = agent.get_model(name, model_kwargs=kwargs)

        tmp_path = f"{RESULTS_DIR}/{name}"
        model.set_logger(configure(tmp_path, ["stdout", "csv", "tensorboard"]))

        trained[name] = agent.train_model(
            model=model, tb_log_name=name, total_timesteps=total_timesteps
        )
        trained[name].save(os.path.join(save_path, f"agent_{name}"))
        print(f"  [✓] {name.upper()} trained and saved to {os.path.join(save_path, f'agent_{name}')}")

    return trained

#%% Main function to run the training steps
def main():
    # Prepare directories
    save_dir = make_trained_model_dir()
    print(f"Trained models will be saved to: {save_dir}")

    # Build environment
    env_train = build_environment()

    # Train DRL agents
    trained_agents = train_drl_agents(env_train, total_timesteps=30000, save_path=TRAINED_MODEL_DIR, models=["ppo"])


if __name__ == "__main__":
    main()