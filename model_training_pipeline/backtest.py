"""
Stock NeurIPS2018 Part 3. Backtest

This series is a reproduction of paper "Deep reinforcement learning for
automated stock trading: An ensemble strategy".

Introducing how to use the agents we trained to do backtest, and compare with
baselines such as buy-and-hold and DJIA index.
"""
#%%
from __future__ import annotations

import os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

# Suppress Pandas4Warning from yfinance/pandas: monkey-patch because
# regular warnings.filterwarnings() does not catch C-level deprecations.
_pd_utcnow = getattr(pd.Timestamp, "utcnow", None)
if _pd_utcnow is not None:
    pd.Timestamp.utcnow = lambda: pd.Timestamp.now("UTC")

from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3

from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.config import (
    INDICATORS,
    TRAINED_MODEL_DIR,
    TRADE_END_DATE,
    TRADE_START_DATE,
)
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv

MODEL_CLASSES = {"a2c": A2C, "ddpg": DDPG, "ppo": PPO, "td3": TD3, "sac": SAC}
MODEL_FILENAMES = {name: f"agent_{name}" for name in MODEL_CLASSES}


# =============================================================================
# Part 1. Load Data
# =============================================================================

def load_data(data_dir: str = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load training and trading data from CSV files.

    Args:
        data_dir: Directory containing train_data.csv and trade_data.csv.
                  If None, defaults to the repo root (parent of model_training_pipeline/).

    Returns:
        (train, trade) DataFrames.
    """
    parent_path = (
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if data_dir is None
        else os.path.abspath(data_dir)
    )

    train = pd.read_csv(os.path.join(parent_path, "train_data.csv"))
    trade = pd.read_csv(os.path.join(parent_path, "trade_data.csv"))

    train = train.set_index(train.columns[0])
    train.index.names = [""]
    trade = trade.set_index(trade.columns[0])
    trade.index.names = [""]
    return train, trade


# =============================================================================
# Part 2. Load Trained Models
# =============================================================================

def load_trained_models(
    model_dir: str = TRAINED_MODEL_DIR,
    model_names: list[str] | None = None,
) -> dict[str, object]:
    """Load trained SB3 models from disk.

    Args:
        model_dir: Directory where .zip model files are stored.
        model_names: Which algorithms to load (e.g. ["ppo", "a2c"]).
                     If None, attempts all five.

    Returns:
        Dict mapping algorithm name → loaded model, or None if not found.
    """
    if model_names is None:
        model_names = list(MODEL_CLASSES.keys())

    models = {}
    for name in model_names:
        model_path = os.path.join(model_dir, MODEL_FILENAMES[name] + ".zip")
        if os.path.exists(model_path):
            models[name] = MODEL_CLASSES[name].load(model_path)
            print(f"  [OK] Loaded {name.upper()} from {model_path}")
        else:
            models[name] = None
            print(f"  [MISSING] {name.upper()} not found at {model_path}")
    return models


# =============================================================================
# Performance Metrics
# =============================================================================

def _print_performance_report(result: pd.DataFrame, initial_amount: float) -> None:
    """Print formatted performance table for each strategy column."""
    final_values = result.iloc[-1]
    total_return = (final_values / initial_amount - 1) * 100

    print("\n" + "=" * 60)
    print("PERFORMANCE REPORT")
    print("=" * 60)

    for col in result.columns:
        daily = result[col].pct_change().dropna()
        sharpe = (
            (np.sqrt(252) * daily.mean() / daily.std())
            if daily.std() > 0
            else 0.0
        )
        rolling_max = result[col].cummax()
        max_dd = (result[col] - rolling_max).div(rolling_max).min() * 100
        print(
            f"  {col:>12s} | "
            f"Final: ${final_values[col]:>12,.2f} | "
            f"Return: {total_return[col]:>7.2f}% | "
            f"Sharpe: {sharpe:>6.2f} | "
            f"MaxDD: {max_dd:>6.2f}%"
        )
    print("=" * 60 + "\n")


# =============================================================================
# Part 3. Baselines
# =============================================================================

def _compute_buy_and_hold(trade: pd.DataFrame, initial_amount: float = 1_000_000) -> pd.Series:
    """Buy-and-hold: invest all capital into the single stock on day 1."""
    ticker = trade["tic"].unique()[0]
    prices = trade[trade["tic"] == ticker].set_index("date")["close"]
    fst_price = prices.iloc[0]
    return prices.div(fst_price).mul(initial_amount)


def _compute_djia(
    start_date: str = TRADE_START_DATE,
    end_date: str = TRADE_END_DATE,
    initial_amount: float = 1_000_000,
) -> pd.Series:
    """Download DJIA index and normalize to start at initial_amount."""
    df_dji = yf.download("^DJI", start=start_date, end=end_date)
    if df_dji.empty:
        raise RuntimeError("Failed to download DJIA data")
    dji = df_dji[["Close"]].reset_index()
    dji.columns = ["date", "close"]
    dji["date"] = dji["date"].dt.strftime("%Y-%m-%d")
    fst_close = dji["close"].iloc[0]
    dji = dji.set_index("date")
    return dji["close"].div(fst_close).mul(initial_amount)


# =============================================================================
# Part 4. Build Environment
# =============================================================================

def build_trade_env(trade: pd.DataFrame, **env_overrides) -> StockTradingEnv:
    """Build the trading environment for backtesting.

    Args:
        trade: Trading data DataFrame.
        env_overrides: Overrides for env_kwargs (e.g. reward_scaling).

    Returns:
        Initialized StockTradingEnv in test mode.
    """
    stock_dimension = len(trade["tic"].unique())
    state_space = 1 + 2 * stock_dimension + len(INDICATORS) * stock_dimension
    print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

    buy_cost_list = sell_cost_list = [0.001] * stock_dimension
    num_stock_shares = [0] * stock_dimension

    env_kwargs = {
        "hmax": 100,
        "initial_amount": 1_000_000,
        "num_stock_shares": num_stock_shares,
        "buy_cost_pct": buy_cost_list,
        "sell_cost_pct": sell_cost_list,
        "state_space": state_space,
        "stock_dim": stock_dimension,
        "tech_indicator_list": INDICATORS,
        "action_space": stock_dimension,
        "reward_scaling": 1e-4,
        **env_overrides,
    }

    return StockTradingEnv(
        df=trade, turbulence_threshold=70, risk_indicator_col="vix", **env_kwargs
    )


# =============================================================================
# Part 5. Run Backtest
# =============================================================================

def test_trained_model(
    trade: pd.DataFrame,
    train: pd.DataFrame,
    trained_models: dict[str, object],
    *,
    output_dir: str = "results",
    plot_filename: str = "backtest_result.png",
    turbulence_threshold: float = 70,
    initial_amount: float = 1000000,
) -> pd.DataFrame:
    """Run backtest with loaded DRL models and compute baselines.

    Args:
        trade: Trading data DataFrame (the test period).
        train: Training data DataFrame (used for buy-and-hold baseline).
        trained_models: Dict of {name: loaded_model} from load_trained_models().
        output_dir: Directory for saving plot and results CSV.
        plot_filename: Name of the output plot file (relative to output_dir).
        turbulence_threshold: VIX threshold for risk-off mode.
        initial_amount: Starting portfolio value.

    Returns:
        DataFrame with one column per strategy, indexed by date.
    """
    os.makedirs(output_dir, exist_ok=True)

    # --- Build environment ---
    e_trade_gym = build_trade_env(trade)

    # --- Run DRL predictions ---
    drl_results: dict[str, pd.Series] = {}
    for name, model in trained_models.items():
        if model is None:
            continue
        print(f"\nRunning {name.upper()} backtest ...")
        try:
            account_value, _actions = DRLAgent.DRL_prediction(
                model=model, environment=e_trade_gym
            )
            df = account_value.set_index(account_value.columns[0])
            drl_results[name] = df["account_value"]
        except (ValueError, RuntimeError) as exc:
            print(f"  [SKIP] {name.upper()} model incompatible: {exc}")

    # --- Baselines ---
    num_stocks = len(trade["tic"].unique())
    print(f"\nComputing baselines for {num_stocks} stock(s) ...")

    if num_stocks == 1:
        drl_results["buy_hold"] = _compute_buy_and_hold(trade, initial_amount)
    else:
        from pypfopt.efficient_frontier import EfficientFrontier

        train_prices = train.pivot(index="date", columns="tic", values="close")
        trade_prices = trade.pivot(index="date", columns="tic", values="close")

        returns = train_prices.pct_change().dropna()
        mean_ret = returns.mean().values
        cov_ret = returns.cov().values

        ef = EfficientFrontier(mean_ret, cov_ret, weight_bounds=(0, 0.5))
        weights = np.array(list(ef.max_sharpe().values()))
        # convert weights to shares
        last_prices = 1 / train_prices.tail(1).values[0]
        initial_shares = initial_amount * weights * last_prices
        drl_results["mvo"] = trade_prices @ initial_shares

    drl_results["dji"] = _compute_djia(
        TRADE_START_DATE, TRADE_END_DATE, initial_amount
    )

    # --- Combine results ---
    result = pd.DataFrame(drl_results).dropna()

    # --- Performance report ---
    _print_performance_report(result, initial_amount)

    # --- Plot ---
    plt.rcParams["figure.figsize"] = (15, 5)
    plt.figure()
    result.plot()
    plt.title("Portfolio Value Over Time")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value ($)")
    plot_path = os.path.join(output_dir, plot_filename)
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {plot_path}")

    return result


# =============================================================================
# Main
# =============================================================================
def main():
    # Load data
    train, trade = load_data()

    # Load trained models
    trained_models = load_trained_models()

    # Run backtest and get results
    results = test_trained_model(
        trade=trade,
        train=train,
        trained_models=trained_models,
        output_dir="results",
        plot_filename="backtest_result.png",
        turbulence_threshold=70,
        initial_amount=1000000,
    )

if __name__ == "__main__":
    main()
