"""
Stock NeurIPS2018 Part 3. Backtest (Enhanced)

This series is a reproduction of paper "Deep reinforcement learning for
automated stock trading: An ensemble strategy".

Introducing how to use the agents we trained to do backtest, and compare with baselines such as
Mean Variance Optimization and DJIA index.

Enhancement: Added risk-adjusted performance metrics —
  Sharpe Ratio, Sortino Ratio, Max Drawdown, Calmar Ratio
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3

from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.config import INDICATORS, TRAINED_MODEL_DIR, TRADE_START_DATE, TRADE_END_DATE
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv

# ============================================================
# Part 0. Risk Metric Utilities
# ============================================================

RISK_FREE_RATE_ANNUAL = 0.05      # 5% annual risk-free rate (adjust as needed)
TRADING_DAYS_PER_YEAR = 252


def compute_metrics(portfolio_values: pd.Series, risk_free_rate_annual: float = RISK_FREE_RATE_ANNUAL) -> dict:
    """
    Compute Sharpe Ratio, Sortino Ratio, Max Drawdown, and Calmar Ratio
    from a time series of portfolio values (daily frequency assumed).

    Parameters
    ----------
    portfolio_values : pd.Series
        Daily portfolio value (e.g. account_value column).
    risk_free_rate_annual : float
        Annualised risk-free rate, default 5%.

    Returns
    -------
    dict with keys: sharpe, sortino, max_drawdown, calmar
    """
    pv = portfolio_values.dropna()
    if len(pv) < 2:
        return {"sharpe": np.nan, "sortino": np.nan, "max_drawdown": np.nan, "calmar": np.nan}

    # Daily returns
    daily_returns = pv.pct_change().dropna()

    # Daily risk-free rate
    rf_daily = (1 + risk_free_rate_annual) ** (1 / TRADING_DAYS_PER_YEAR) - 1

    # --- Sharpe Ratio ---
    excess_returns = daily_returns - rf_daily
    sharpe = (excess_returns.mean() / excess_returns.std()) * np.sqrt(TRADING_DAYS_PER_YEAR) \
        if excess_returns.std() != 0 else np.nan

    # --- Sortino Ratio ---
    # Uses only downside deviation (negative excess returns)
    downside = excess_returns[excess_returns < 0]
    downside_std = np.sqrt((downside ** 2).mean()) if len(downside) > 0 else np.nan
    sortino = (excess_returns.mean() / downside_std) * np.sqrt(TRADING_DAYS_PER_YEAR) \
        if downside_std and downside_std != 0 else np.nan

    # --- Max Drawdown ---
    # Rolling peak and trough
    rolling_max = pv.cummax()
    drawdown = (pv - rolling_max) / rolling_max
    max_drawdown = drawdown.min()   # most negative value, e.g. -0.25 = -25%

    # --- Calmar Ratio ---
    # Annualised return / abs(Max Drawdown)
    total_return = (pv.iloc[-1] / pv.iloc[0]) - 1
    n_days = len(pv)
    annualised_return = (1 + total_return) ** (TRADING_DAYS_PER_YEAR / n_days) - 1
    calmar = annualised_return / abs(max_drawdown) if max_drawdown != 0 else np.nan

    return {
        "sharpe": round(sharpe, 4),
        "sortino": round(sortino, 4),
        "max_drawdown": round(max_drawdown * 100, 4),   # expressed as %
        "calmar": round(calmar, 4),
    }


def print_metrics_table(metrics_dict: dict) -> None:
    """Pretty-print a comparison table of risk metrics for all strategies."""
    strategies = list(metrics_dict.keys())
    header = f"{'Strategy':<12} {'Sharpe':>10} {'Sortino':>10} {'MaxDD (%)':>12} {'Calmar':>10}"
    separator = "-" * len(header)
    print("\n=== Risk-Adjusted Performance Metrics ===")
    print(separator)
    print(header)
    print(separator)
    for strat in strategies:
        m = metrics_dict[strat]
        sharpe   = f"{m['sharpe']:.4f}"   if not np.isnan(m['sharpe'])       else "  N/A"
        sortino  = f"{m['sortino']:.4f}"  if not np.isnan(m['sortino'])      else "  N/A"
        max_dd   = f"{m['max_drawdown']:.4f}" if not np.isnan(m['max_drawdown']) else "  N/A"
        calmar   = f"{m['calmar']:.4f}"   if not np.isnan(m['calmar'])       else "  N/A"
        print(f"{strat:<12} {sharpe:>10} {sortino:>10} {max_dd:>12} {calmar:>10}")
    print(separator)


def save_metrics_csv(metrics_dict: dict, filepath: str = "backtest_metrics.csv") -> None:
    """Save the metrics table to a CSV file."""
    rows = []
    for strat, m in metrics_dict.items():
        rows.append({
            "strategy": strat,
            "sharpe_ratio": m["sharpe"],
            "sortino_ratio": m["sortino"],
            "max_drawdown_pct": m["max_drawdown"],
            "calmar_ratio": m["calmar"],
        })
    pd.DataFrame(rows).to_csv(filepath, index=False)
    print(f"Metrics saved to {filepath}")


# ============================================================
# Part 1. Load data
# ============================================================

parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

train = pd.read_csv(os.path.join(parent_path, "train_data.csv"))
trade = pd.read_csv(os.path.join(parent_path, "trade_data.csv"))

train = train.set_index(train.columns[0])
train.index.names = [""]
trade = trade.set_index(trade.columns[0])
trade.index.names = [""]

# ============================================================
# Part 2. Load trained agents
# ============================================================

if_using_a2c  = True
if_using_ddpg = True
if_using_ppo  = True
if_using_td3  = True
if_using_sac  = True

trained_a2c  = A2C.load(TRAINED_MODEL_DIR + "/agent_a2c")   if if_using_a2c  else None
trained_ddpg = DDPG.load(TRAINED_MODEL_DIR + "/agent_ddpg") if if_using_ddpg else None
trained_ppo  = PPO.load(TRAINED_MODEL_DIR + "/agent_ppo")   if if_using_ppo  else None
trained_td3  = TD3.load(TRAINED_MODEL_DIR + "/agent_td3")   if if_using_td3  else None
trained_sac  = SAC.load(TRAINED_MODEL_DIR + "/agent_sac")   if if_using_sac  else None

# ============================================================
# Part 3. Backtesting — DRL agents (fresh env per agent)
# ============================================================

stock_dimension = len(trade.tic.unique())
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

# NOTE: Each agent gets its own fresh environment instance to avoid
# internal state contamination between sequential DRL_prediction calls.
def make_env():
    return StockTradingEnv(df=trade, turbulence_threshold=70, risk_indicator_col="vix", **env_kwargs)


df_account_value_a2c, df_actions_a2c = (
    DRLAgent.DRL_prediction(model=trained_a2c, environment=make_env())
    if if_using_a2c else (None, None)
)

df_account_value_ddpg, df_actions_ddpg = (
    DRLAgent.DRL_prediction(model=trained_ddpg, environment=make_env())
    if if_using_ddpg else (None, None)
)

df_account_value_ppo, df_actions_ppo = (
    DRLAgent.DRL_prediction(model=trained_ppo, environment=make_env())
    if if_using_ppo else (None, None)
)

df_account_value_td3, df_actions_td3 = (
    DRLAgent.DRL_prediction(model=trained_td3, environment=make_env())
    if if_using_td3 else (None, None)
)

df_account_value_sac, df_actions_sac = (
    DRLAgent.DRL_prediction(model=trained_sac, environment=make_env())
    if if_using_sac else (None, None)
)

# ============================================================
# Part 4. Mean Variance Optimization baseline
# ============================================================

def process_df_for_mvo(df):
    return df.pivot(index="date", columns="tic", values="close")


def StockReturnsComputing(StockPrice, Rows, Columns):
    StockReturn = np.zeros([Rows - 1, Columns])
    for j in range(Columns):
        for i in range(Rows - 1):
            StockReturn[i, j] = (
                (StockPrice[i + 1, j] - StockPrice[i, j]) / StockPrice[i, j]
            ) * 100
    return StockReturn


StockData = process_df_for_mvo(train)
TradeData = process_df_for_mvo(trade)

arStockPrices = np.asarray(StockData)
[Rows, Cols] = arStockPrices.shape
arReturns = StockReturnsComputing(arStockPrices, Rows, Cols)

meanReturns = np.mean(arReturns, axis=0)
covReturns  = np.cov(arReturns, rowvar=False)

np.set_printoptions(precision=3, suppress=True)
print("Mean returns of assets in portfolio\n", meanReturns)

from pypfopt.efficient_frontier import EfficientFrontier

ef_mean = EfficientFrontier(meanReturns, covReturns, weight_bounds=(0, 0.5))
raw_weights_mean    = ef_mean.max_sharpe()
cleaned_weights_mean = ef_mean.clean_weights()
mvo_weights = np.array(
    [1000000 * cleaned_weights_mean[i] for i in range(len(cleaned_weights_mean))]
)

LastPrice = np.array([1 / p for p in StockData.tail(1).to_numpy()[0]])
Initial_Portfolio = np.multiply(mvo_weights, LastPrice)

Portfolio_Assets = TradeData @ Initial_Portfolio
MVO_result = pd.DataFrame(Portfolio_Assets, columns=["Mean Var"])

# ============================================================
# Part 5. DJIA index baseline
# ============================================================

import yfinance as yf

df_dji = yf.download("^DJI", start=TRADE_START_DATE, end=TRADE_END_DATE)
df_dji = df_dji[["Close"]].reset_index()
df_dji.columns = ["date", "close"]
df_dji["date"] = df_dji["date"].astype(str)
fst_day = df_dji["close"].iloc[0]
dji = pd.merge(
    df_dji["date"],
    df_dji["close"].div(fst_day).mul(1000000),
    how="outer",
    left_index=True,
    right_index=True,
).set_index("date")

# ============================================================
# Part 6. Assemble results DataFrame
# ============================================================

df_result_a2c  = df_account_value_a2c.set_index(df_account_value_a2c.columns[0])   if if_using_a2c  else None
df_result_ddpg = df_account_value_ddpg.set_index(df_account_value_ddpg.columns[0]) if if_using_ddpg else None
df_result_ppo  = df_account_value_ppo.set_index(df_account_value_ppo.columns[0])   if if_using_ppo  else None
df_result_td3  = df_account_value_td3.set_index(df_account_value_td3.columns[0])   if if_using_td3  else None
df_result_sac  = df_account_value_sac.set_index(df_account_value_sac.columns[0])   if if_using_sac  else None

result = pd.DataFrame(
    {
        "a2c":  df_result_a2c["account_value"]  if if_using_a2c  else None,
        "ddpg": df_result_ddpg["account_value"] if if_using_ddpg else None,
        "ppo":  df_result_ppo["account_value"]  if if_using_ppo  else None,
        "td3":  df_result_td3["account_value"]  if if_using_td3  else None,
        "sac":  df_result_sac["account_value"]  if if_using_sac  else None,
        "mvo":  MVO_result["Mean Var"],
        "dji":  dji["close"],
    }
)

print("\n=== Backtest Results (Portfolio Values) ===")
print(result)

# ============================================================
# Part 7. Risk-Adjusted Metrics  ← NEW SECTION
# ============================================================

metrics = {}
for col in result.columns:
    series = result[col].dropna()
    if len(series) >= 2:
        metrics[col] = compute_metrics(series)

# Print formatted table to console
print_metrics_table(metrics)

# Persist metrics to CSV for downstream analysis
save_metrics_csv(metrics, filepath="backtest_metrics.csv")

# ============================================================
# Part 8. Plot
# ============================================================

# --- 8a. Portfolio value over time (original plot) ---
plt.rcParams["figure.figsize"] = (15, 5)
fig, ax = plt.subplots()
result.plot(ax=ax)
ax.set_title("Portfolio Value Over Time")
ax.set_xlabel("Date")
ax.set_ylabel("Portfolio Value ($)")
plt.tight_layout()
plt.savefig("backtest_result.png", dpi=150, bbox_inches="tight")
print("\nPlot saved to backtest_result.png")
plt.close()

# --- 8b. Risk metrics bar charts (NEW) ---
metrics_df = pd.DataFrame(metrics).T   # rows = strategies, cols = metrics
strategy_labels = metrics_df.index.tolist()
x = np.arange(len(strategy_labels))
bar_width = 0.6

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle("Risk-Adjusted Performance Metrics", fontsize=16, fontweight="bold")

metric_configs = [
    ("sharpe",        "Sharpe Ratio",         "tab:blue",   axes[0, 0]),
    ("sortino",       "Sortino Ratio",         "tab:green",  axes[0, 1]),
    ("max_drawdown",  "Max Drawdown (%)",      "tab:red",    axes[1, 0]),
    ("calmar",        "Calmar Ratio",          "tab:orange", axes[1, 1]),
]

for metric_key, title, color, ax in metric_configs:
    values = metrics_df[metric_key].values.astype(float)
    bars = ax.bar(x, values, width=bar_width, color=color, alpha=0.8, edgecolor="white", linewidth=0.8)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([s.upper() for s in strategy_labels], fontsize=10)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    # Annotate bar values
    for bar, val in zip(bars, values):
        if not np.isnan(val):
            va = "bottom" if val >= 0 else "top"
            offset = 0.01 * abs(max(values) - min(values)) if (max(values) - min(values)) != 0 else 0.01
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (offset if val >= 0 else -offset),
                f"{val:.2f}",
                ha="center", va=va, fontsize=9, fontweight="bold"
            )

plt.tight_layout()
plt.savefig("backtest_metrics.png", dpi=150, bbox_inches="tight")
print("Risk metrics chart saved to backtest_metrics.png")
plt.close()
