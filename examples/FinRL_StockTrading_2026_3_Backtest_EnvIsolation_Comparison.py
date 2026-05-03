"""
Environment Isolation Comparison
=================================
Quantifies and visualises the effect of sharing a single StockTradingEnv
across all DRL agents (BUGGY) vs. giving each agent its own fresh instance
(CORRECT).

Outputs
-------
  env_isolation_portfolio_values.png  – side-by-side portfolio trajectories
  env_isolation_drift.png             – per-agent absolute drift (|shared - isolated|)
  env_isolation_metrics.png           – grouped bar chart of Sharpe / MaxDD / Calmar
  env_isolation_comparison.csv        – full numeric results table
"""

from __future__ import annotations

import copy
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3

from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.config import INDICATORS, TRAINED_MODEL_DIR, TRADE_START_DATE, TRADE_END_DATE
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv

# ── Metric helpers ────────────────────────────────────────────────────────────

TRADING_DAYS = 252
RF_ANNUAL    = 0.05


def compute_metrics(pv: pd.Series, rf_annual: float = RF_ANNUAL) -> dict:
    pv = pv.dropna()
    if len(pv) < 2:
        return dict(sharpe=np.nan, sortino=np.nan, max_drawdown=np.nan, calmar=np.nan,
                    total_return=np.nan)
    rf_d = (1 + rf_annual) ** (1 / TRADING_DAYS) - 1
    er   = pv.pct_change().dropna() - rf_d

    sharpe  = (er.mean() / er.std()) * np.sqrt(TRADING_DAYS) if er.std() else np.nan
    neg     = er[er < 0]
    ds      = np.sqrt((neg ** 2).mean()) if len(neg) else np.nan
    sortino = (er.mean() / ds) * np.sqrt(TRADING_DAYS) if ds else np.nan

    roll_max = pv.cummax()
    max_dd   = ((pv - roll_max) / roll_max).min() * 100

    tot_ret  = (pv.iloc[-1] / pv.iloc[0]) - 1
    ann_ret  = (1 + tot_ret) ** (TRADING_DAYS / len(pv)) - 1
    calmar   = ann_ret / abs(max_dd / 100) if max_dd != 0 else np.nan

    return dict(sharpe=round(sharpe, 4), sortino=round(sortino, 4),
                max_drawdown=round(max_dd, 4), calmar=round(calmar, 4),
                total_return=round(tot_ret * 100, 4))


# ── Data & agent loading ──────────────────────────────────────────────────────

parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
train = pd.read_csv(os.path.join(parent_path, "train_data.csv"))
trade = pd.read_csv(os.path.join(parent_path, "trade_data.csv"))
train = train.set_index(train.columns[0]); train.index.names = [""]
trade = trade.set_index(trade.columns[0]); trade.index.names = [""]

AGENTS = {
    "a2c":  ("A2C",  A2C),
    "ddpg": ("DDPG", DDPG),
    "ppo":  ("PPO",  PPO),
    "td3":  ("TD3",  TD3),
    "sac":  ("SAC",  SAC),
}

trained = {}
for key, (label, cls) in AGENTS.items():
    path = os.path.join(TRAINED_MODEL_DIR, f"agent_{key}")
    trained[key] = cls.load(path)
    print(f"Loaded {label}")

stock_dim   = len(trade.tic.unique())
state_space = 1 + 2 * stock_dim + len(INDICATORS) * stock_dim

env_kwargs = {
    "hmax":                100,
    "initial_amount":      1_000_000,
    "num_stock_shares":    [0] * stock_dim,
    "buy_cost_pct":        [0.001] * stock_dim,
    "sell_cost_pct":       [0.001] * stock_dim,
    "state_space":         state_space,
    "stock_dim":           stock_dim,
    "tech_indicator_list": INDICATORS,
    "action_space":        stock_dim,
    "reward_scaling":      1e-4,
}

ENV_COMMON = dict(df=trade, turbulence_threshold=70, risk_indicator_col="vix")


def make_env() -> StockTradingEnv:
    """Fresh environment — deep-copies all mutable kwargs."""
    return StockTradingEnv(**ENV_COMMON, **copy.deepcopy(env_kwargs))


# ── Condition A: SHARED environment (buggy baseline) ─────────────────────────
# One object is constructed ONCE and reused for every agent.
# After each DRL_prediction call the env's internal day pointer, cash balance,
# and num_stock_shares list are in an unknown post-episode state.

print("\n── Running SHARED env condition ──")
shared_env = make_env()          # created once, never reset between agents
shared_results = {}

for key, (label, _) in AGENTS.items():
    df_val, _ = DRLAgent.DRL_prediction(model=trained[key], environment=shared_env)
    df_val = df_val.set_index(df_val.columns[0])
    shared_results[key] = df_val["account_value"]
    print(f"  {label} done  |  final value: ${df_val['account_value'].iloc[-1]:,.0f}")


# ── Condition B: ISOLATED environments (corrected) ───────────────────────────
# Each agent receives a brand-new StockTradingEnv constructed from scratch.

print("\n── Running ISOLATED env condition ──")
isolated_results = {}

for key, (label, _) in AGENTS.items():
    df_val, _ = DRLAgent.DRL_prediction(model=trained[key], environment=make_env())
    df_val = df_val.set_index(df_val.columns[0])
    isolated_results[key] = df_val["account_value"]
    print(f"  {label} done  |  final value: ${df_val['account_value'].iloc[-1]:,.0f}")


# ── Build comparison DataFrames ───────────────────────────────────────────────

shared_df   = pd.DataFrame(shared_results)
isolated_df = pd.DataFrame(isolated_results)

# Align indices (both should share the same date range)
shared_df, isolated_df = shared_df.align(isolated_df, join="inner")

# Absolute drift per agent per day
drift_df = (shared_df - isolated_df).abs()

# Per-agent metrics for both conditions
metric_rows = []
for key, (label, _) in AGENTS.items():
    for condition, df in [("shared", shared_df), ("isolated", isolated_df)]:
        m = compute_metrics(df[key])
        metric_rows.append({"agent": label, "condition": condition, **m})

metrics_df = pd.DataFrame(metric_rows)

# Numeric summary: final portfolio value difference
summary_rows = []
for key, (label, _) in AGENTS.items():
    sv = shared_df[key].iloc[-1]
    iv = isolated_df[key].iloc[-1]
    summary_rows.append({
        "agent":              label,
        "shared_final ($)":   round(sv, 2),
        "isolated_final ($)": round(iv, 2),
        "diff ($)":           round(iv - sv, 2),
        "diff (%)":           round((iv - sv) / sv * 100, 4),
        "max_drift ($)":      round(drift_df[key].max(), 2),
        "mean_drift ($)":     round(drift_df[key].mean(), 2),
    })
summary_df = pd.DataFrame(summary_rows)

print("\n=== Portfolio Value Divergence Summary ===")
print(summary_df.to_string(index=False))

# Save full data to CSV
out_csv = pd.concat([
    shared_df.add_suffix("_shared"),
    isolated_df.add_suffix("_isolated"),
    drift_df.add_suffix("_drift"),
], axis=1)
out_csv.to_csv("env_isolation_comparison.csv")
print("\nFull time-series saved → env_isolation_comparison.csv")


# ── Plot 1: Portfolio trajectories, shared vs isolated (per agent) ────────────

agent_keys   = list(AGENTS.keys())
agent_labels = [v[0] for v in AGENTS.values()]
n            = len(agent_keys)
COLORS       = {"shared": "#e05c5c", "isolated": "#3a7dc9"}

fig, axes = plt.subplots(n, 1, figsize=(14, 3.5 * n), sharex=True)
fig.suptitle("Portfolio Value: Shared Env vs Isolated Env",
             fontsize=15, fontweight="bold", y=1.01)

for ax, key, label in zip(axes, agent_keys, agent_labels):
    ax.plot(shared_df.index,   shared_df[key].values,
            color=COLORS["shared"],   linewidth=1.5, label="Shared env (buggy)", linestyle="--")
    ax.plot(isolated_df.index, isolated_df[key].values,
            color=COLORS["isolated"], linewidth=1.8, label="Isolated env (correct)")
    ax.fill_between(shared_df.index,
                    shared_df[key].values, isolated_df[key].values,
                    alpha=0.15, color="gray", label="Divergence area")
    ax.set_ylabel("Portfolio Value ($)", fontsize=9)
    ax.set_title(label, fontsize=11, fontweight="bold", loc="left")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"${x/1e6:.2f}M")
    )

axes[-1].set_xlabel("Date")
plt.tight_layout()
plt.savefig("env_isolation_portfolio_values.png", dpi=150, bbox_inches="tight")
plt.close()
print("Plot saved → env_isolation_portfolio_values.png")


# ── Plot 2: Absolute drift |shared - isolated| per agent over time ────────────

fig, ax = plt.subplots(figsize=(14, 5))
for key, label in zip(agent_keys, agent_labels):
    ax.plot(drift_df.index, drift_df[key].values, linewidth=1.5, label=label)

ax.set_title("Absolute Divergence |Shared − Isolated| Portfolio Value",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Date")
ax.set_ylabel("Absolute Drift ($)")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
ax.legend(fontsize=9)
ax.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("env_isolation_drift.png", dpi=150, bbox_inches="tight")
plt.close()
print("Plot saved → env_isolation_drift.png")


# ── Plot 3: Grouped bar chart — Sharpe / MaxDD / Calmar ──────────────────────

metric_cols   = ["sharpe", "max_drawdown", "calmar"]
metric_titles = ["Sharpe Ratio", "Max Drawdown (%)", "Calmar Ratio"]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Risk Metrics: Shared vs Isolated Environment",
             fontsize=14, fontweight="bold")

x      = np.arange(len(agent_labels))
width  = 0.35

for ax, col, title in zip(axes, metric_cols, metric_titles):
    vals_shared   = metrics_df[metrics_df["condition"] == "shared"][col].values.astype(float)
    vals_isolated = metrics_df[metrics_df["condition"] == "isolated"][col].values.astype(float)

    bars_s = ax.bar(x - width/2, vals_shared,   width, label="Shared",   color=COLORS["shared"],   alpha=0.85, edgecolor="white")
    bars_i = ax.bar(x + width/2, vals_isolated, width, label="Isolated", color=COLORS["isolated"], alpha=0.85, edgecolor="white")

    # Annotate bars
    for bar in list(bars_s) + list(bars_i):
        h = bar.get_height()
        if not np.isnan(h):
            ax.text(bar.get_x() + bar.get_width()/2, h,
                    f"{h:.2f}", ha="center",
                    va="bottom" if h >= 0 else "top",
                    fontsize=7.5, fontweight="bold")

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(agent_labels, fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

plt.tight_layout()
plt.savefig("env_isolation_metrics.png", dpi=150, bbox_inches="tight")
plt.close()
print("Plot saved → env_isolation_metrics.png")

print("\n=== All outputs ===")
print("  env_isolation_portfolio_values.png  – trajectory comparison per agent")
print("  env_isolation_drift.png             – |shared − isolated| drift over time")
print("  env_isolation_metrics.png           – Sharpe / MaxDD / Calmar grouped bars")
print("  env_isolation_comparison.csv        – full numeric time-series + metrics")
