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
from stable_baselines3.common.logger import configure

from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.config import INDICATORS
from finrl.config import RESULTS_DIR
from finrl.config import TRAINED_MODEL_DIR
from finrl.main import check_and_make_directories
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv

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
        models: list[str] = ["a2c", "ddpg", "ppo", "td3", "sac"]
        ) -> dict[str, DRLAgent]:
    '''
    Train DRL agents (A2C, DDPG, PPO, TD3, SAC) on the given environment and save the trained models.
    Args:
        env_train (StockTradingEnv): The environment to train the agents on.
        total_timesteps (int): The total number of timesteps to train each agent. Default is 300,000.
        save_path (str): The directory path where the trained models will be saved. If None, defaults to TRAINED_MODEL_DIR.
        models (list[str]): The models to train. If None, all agents will be trained. Options are ["a2c", "ddpg", "ppo", "td3", "sac"].
    Returns:
        dict[str, DRLAgent]: A dictionary containing the trained agents with keys as agent names
    '''

    if_using_a2c = False
    if_using_ddpg = False
    if_using_ppo = False
    if_using_td3 = False
    if_using_sac = False

    # Set models used for training based on user input
    for model in models:
        if model not in ["a2c", "ddpg", "ppo", "td3", "sac"]:
            raise ValueError(f"Invalid model name: {model}. Valid options are ['a2c', 'ddpg', 'ppo', 'td3', 'sac']")
        
        match model:
            case "a2c":
                if_using_a2c = True
            case "ddpg":
                if_using_ddpg = True
            case "ppo":
                if_using_ppo = True
            case "td3":
                if_using_td3 = True
            case "sac":
                if_using_sac = True

    # Hyperparameter overrides per algorithm (defaults from config.py used otherwise)
    model_params = {
        "ppo": {
            "n_steps": 3014,
            "batch_size": 256,
            "n_epochs": 10,
            "learning_rate": 3e-4,
            "gamma": 0.995,
            "gae_lambda": 0.99,
            "clip_range": 0.2,
            "ent_coef": 0.02,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "normalize_advantage": True,
            "policy_kwargs": {"net_arch": [256, 128], "ortho_init": True},
        },
        "ddpg": {
            "batch_size": 128,
            "buffer_size": 50000,
            "learning_rate": 1e-3,
        },
        "td3": {
            "batch_size": 100,
            "buffer_size": 1000000,
            "learning_rate": 1e-3,
        },
        "sac": {
            "batch_size": 128,
            "buffer_size": 100000,
            "learning_rate": 1e-4,
            "learning_starts": 100,
            "ent_coef": "auto_0.1",
        },
    }


    # Start training loop for each enabled agent
    enabled = {"a2c": if_using_a2c, 
               "ddpg": if_using_ddpg, 
               "ppo": if_using_ppo,
               "td3": if_using_td3, 
               "sac": if_using_sac}

    trained = {}
    for name in ["a2c", "ddpg", "ppo", "td3", "sac"]:
        if not enabled[name]:
            trained[name] = None
            continue

        agent = DRLAgent(env=env_train)
        kwargs = model_params.get(name, None)
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
    make_trained_model_dir()

    # Build environment
    env_train = build_environment()

    # Train DRL agents
    trained_agents = train_drl_agents(env_train, total_timesteps=30000, save_path=TRAINED_MODEL_DIR, models=["ppo"])


if __name__ == "__main__":
    main()