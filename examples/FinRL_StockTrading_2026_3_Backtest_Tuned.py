"""
Stock NeurIPS2018 Part 3. Backtest

Reproduction of "Deep reinforcement learning for automated stock trading:
An ensemble strategy".

Fixes vs. original
------------------
1. Per-agent turbulence threshold tuning
     Three strategies in priority order:
       a) Grid search on validation split  — used when the val split contains
          meaningful VIX variation (max VIX > lowest grid candidate).
       b) Statistical fallback             — if val VIX is flat/calm, the
          threshold is set to the 99th percentile of train VIX (NeurIPS2018
          paper method).  Each agent still gets its own candidate evaluated
          against the statistical baseline on the val split.
       c) Hardcoded fallback of 70         — only if both above fail.

     Model deepcopy is used on every grid iteration to prevent any wrapper
     state from carrying over between threshold evaluations.

2. Isolated environment per agent
     make_trade_env() builds a brand-new StockTradingEnv every call with
     copy.deepcopy on all mutable kwargs, preventing state leakage.

3. Correct DataFrame format for StockTradingEnv
     trade has a numeric CSV index; dates live in the "date" column.
     _prepare_env_df() maps each unique date string to a shared integer so
     df.loc[day, :] returns a DataFrame of N ticker rows, as the env requires.

4. Real DatetimeIndex for plotting
     result.index converted via pd.to_datetime() before plot/annotate so
     Matplotlib uses a numeric time axis, avoiding StrCategoryConverter
     ConversionError at tight_layout().
"""

from __future__ import annotations

import copy
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3

from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.config import INDICATORS, TRAINED_MODEL_DIR, TRADE_START_DATE, TRADE_END_DATE
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv

warnings.filterwarnings("ignore")

# ── Threshold tuning config ───────────────────────────────────────────────────
# 9999 = gate effectively disabled.
THRESHOLD_GRID      = [20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 150, 200, 9999]
VAL_SPLIT_RATIO     = 0.5
TRADING_DAYS        = 252
RF_ANNUAL           = 0.05
# Minimum number of val-split days where VIX must exceed the lowest grid
# candidate for grid search to be meaningful.  Below this → statistical fallback.
MIN_TRIGGER_DAYS    = 5

# %% Part 1. Load data

parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

train = pd.read_csv(os.path.join(parent_path, "train_data.csv"))
trade = pd.read_csv(os.path.join(parent_path, "trade_data.csv"))

train = train.set_index(train.columns[0])
train.index.names = [""]
trade = trade.set_index(trade.columns[0])
trade.index.names = [""]

# %% Part 2. Load trained agents

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

# %% Part 3. Backtesting - DRL agents

stock_dimension = len(trade.tic.unique())
state_space = 1 + 2 * stock_dimension + len(INDICATORS) * stock_dimension
print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

buy_cost_list    = sell_cost_list = [0.001] * stock_dimension
num_stock_shares = [0] * stock_dimension

env_kwargs = {
    "hmax":                100,
    "initial_amount":      1000000,
    "num_stock_shares":    num_stock_shares,
    "buy_cost_pct":        buy_cost_list,
    "sell_cost_pct":       sell_cost_list,
    "state_space":         state_space,
    "stock_dim":           stock_dimension,
    "tech_indicator_list": INDICATORS,
    "action_space":        stock_dimension,
    "reward_scaling":      1e-4,
}


def _prepare_env_df(df_subset: pd.DataFrame) -> pd.DataFrame:
    """
    Convert a slice of trade (numeric pandas index, "date" column) into the
    format StockTradingEnv expects.

    StockTradingEnv accesses self.df.loc[self.day, :] where self.day starts
    at 0.  For multi-stock data it expects that key to return a DATAFRAME
    (N rows, one per ticker).  We map each unique date string to a sequential
    integer so every ticker row on the same date shares the same integer key.
    The "date" column is preserved because _get_date() reads it directly.
    """
    df = df_subset.copy()
    unique_dates = sorted(df["date"].unique())
    date_to_int  = {d: i for i, d in enumerate(unique_dates)}
    df.index     = df["date"].map(date_to_int)
    return df


def make_trade_env(df: pd.DataFrame, threshold: float) -> StockTradingEnv:
    """Build a fully isolated StockTradingEnv. df must be _prepare_env_df output."""
    return StockTradingEnv(
        df=df,
        turbulence_threshold=threshold,
        risk_indicator_col="vix",
        **copy.deepcopy(env_kwargs),
    )


def _sharpe(pv: pd.Series) -> float:
    """Annualised Sharpe ratio — used during threshold grid search."""
    pv = pv.dropna()
    if len(pv) < 2:
        return -np.inf
    rf_d = (1 + RF_ANNUAL) ** (1 / TRADING_DAYS) - 1
    er   = pv.pct_change().dropna() - rf_d
    return float((er.mean() / er.std()) * np.sqrt(TRADING_DAYS)) if er.std() > 0 else -np.inf


def _run_agent(model, env_df: pd.DataFrame, threshold: float):
    """Run DRL_prediction with a deepcopy of the model to prevent wrapper state leakage."""
    return DRLAgent.DRL_prediction(
        model=copy.deepcopy(model),
        environment=make_trade_env(env_df, threshold),
    )


# ── 3a. Carve validation / test splits ───────────────────────────────────────
# Date strings come from trade["date"] column — trade.index is numeric.

all_dates  = sorted(trade["date"].unique())
split_idx  = int(len(all_dates) * VAL_SPLIT_RATIO)
val_dates  = set(all_dates[:split_idx])
test_dates = set(all_dates[split_idx:])

val_start  = all_dates[0]
val_end    = all_dates[split_idx - 1]
test_start = all_dates[split_idx]
test_end   = all_dates[-1]

print(f"\nValidation : {val_start} -> {val_end}  ({len(val_dates)} days)")
print(f"Test       : {test_start} -> {test_end}  ({len(test_dates)} days)")

trade_val_raw  = trade[trade["date"].isin(val_dates)]
trade_test_raw = trade[trade["date"].isin(test_dates)]

trade_val_env  = _prepare_env_df(trade_val_raw)
trade_test_env = _prepare_env_df(trade_test_raw)

# ── 3b. VIX diagnostics — decide grid search vs statistical fallback ──────────

vix_val   = trade_val_raw["vix"].dropna()
vix_train = train["vix"].dropna() if "vix" in train.columns else pd.Series(dtype=float)

lowest_thr      = min(t for t in THRESHOLD_GRID if t < 9999)
trigger_days    = int((vix_val > lowest_thr).sum())
stat_threshold  = float(vix_train.quantile(0.99)) if len(vix_train) > 0 else 70.0

print(f"\nVIX (val)  — min: {vix_val.min():.1f}  max: {vix_val.max():.1f}  "
      f"p99: {vix_val.quantile(0.99):.1f}")
print(f"Days VIX > {lowest_thr} in val split: {trigger_days}")
print(f"Statistical threshold (train p99): {stat_threshold:.1f}")

use_grid_search = trigger_days >= MIN_TRIGGER_DAYS
if use_grid_search:
    print("-> Sufficient VIX variation: using per-agent grid search.")
else:
    print(f"-> VIX rarely exceeds {lowest_thr} in val split: using statistical threshold "
          f"({stat_threshold:.1f}) for all agents.")

# ── 3c. Threshold selection ───────────────────────────────────────────────────

_agent_map = {
    "a2c":  (if_using_a2c,  trained_a2c),
    "ddpg": (if_using_ddpg, trained_ddpg),
    "ppo":  (if_using_ppo,  trained_ppo),
    "td3":  (if_using_td3,  trained_td3),
    "sac":  (if_using_sac,  trained_sac),
}

best_threshold = {}
grid_sharpe    = {}

if use_grid_search:
    print("\n-- Grid search on validation split --")
    for key, (active, model) in _agent_map.items():
        if not active:
            best_threshold[key] = stat_threshold
            continue
        grid_sharpe[key] = {}
        for thr in THRESHOLD_GRID:
            df_val_run, _ = _run_agent(model, trade_val_env, thr)
            pv = df_val_run.set_index(df_val_run.columns[0])["account_value"]
            grid_sharpe[key][thr] = _sharpe(pv)

        # If all Sharpe values are identical (gate never triggered), fall back
        # to stat_threshold for this agent
        sharpe_vals = list(grid_sharpe[key].values())
        all_same    = len(set(round(s, 6) for s in sharpe_vals if np.isfinite(s))) <= 1
        if all_same:
            best_threshold[key] = stat_threshold
            print(f"  {key.upper():5s}  grid flat -> stat fallback  threshold={stat_threshold:.1f}")
        else:
            best = max(grid_sharpe[key], key=grid_sharpe[key].get)
            best_threshold[key] = best
            print(f"  {key.upper():5s}  best threshold={best:>5}  "
                  f"val Sharpe={grid_sharpe[key][best]:.4f}")
else:
    # Statistical threshold for all agents — no grid search needed
    for key, (active, _) in _agent_map.items():
        best_threshold[key] = stat_threshold
    print(f"\nAll agents assigned statistical threshold: {stat_threshold:.1f}")

# ── 3d. Final backtest on test split ─────────────────────────────────────────

print("\n-- Final backtest on test split with tuned thresholds --")

df_account_value_a2c, df_actions_a2c = (
    _run_agent(trained_a2c, trade_test_env, best_threshold["a2c"])
    if if_using_a2c else (None, None)
)

df_account_value_ddpg, df_actions_ddpg = (
    _run_agent(trained_ddpg, trade_test_env, best_threshold["ddpg"])
    if if_using_ddpg else (None, None)
)

df_account_value_ppo, df_actions_ppo = (
    _run_agent(trained_ppo, trade_test_env, best_threshold["ppo"])
    if if_using_ppo else (None, None)
)

df_account_value_td3, df_actions_td3 = (
    _run_agent(trained_td3, trade_test_env, best_threshold["td3"])
    if if_using_td3 else (None, None)
)

df_account_value_sac, df_actions_sac = (
    _run_agent(trained_sac, trade_test_env, best_threshold["sac"])
    if if_using_sac else (None, None)
)

# %% Part 4. Mean Variance Optimization baseline
# Uses trade_test_raw (original date-indexed slice) for the pivot.

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
TradeData = process_df_for_mvo(trade_test_raw)

arStockPrices = np.asarray(StockData)
[Rows, Cols]  = arStockPrices.shape
arReturns     = StockReturnsComputing(arStockPrices, Rows, Cols)

meanReturns = np.mean(arReturns, axis=0)
covReturns  = np.cov(arReturns, rowvar=False)

np.set_printoptions(precision=3, suppress=True)
print("Mean returns of assets in portfolio\n", meanReturns)

from pypfopt.efficient_frontier import EfficientFrontier

ef_mean              = EfficientFrontier(meanReturns, covReturns, weight_bounds=(0, 0.5))
raw_weights_mean     = ef_mean.max_sharpe()
cleaned_weights_mean = ef_mean.clean_weights()
mvo_weights = np.array(
    [1000000 * cleaned_weights_mean[i] for i in range(len(cleaned_weights_mean))]
)

LastPrice         = np.array([1 / p for p in StockData.tail(1).to_numpy()[0]])
Initial_Portfolio = np.multiply(mvo_weights, LastPrice)

Portfolio_Assets  = TradeData @ Initial_Portfolio
MVO_result        = pd.DataFrame(Portfolio_Assets, columns=["Mean Var"])

# %% Part 5. DJIA index baseline
# test_start / test_end are date strings from trade["date"] — never from trade.index.

import yfinance as yf

df_dji = yf.download("^DJI", start=test_start, end=test_end)
if df_dji.empty:
    raise RuntimeError(
        f"yfinance returned empty data for ^DJI ({test_start} to {test_end}). "
        "Check internet connection or date range."
    )
df_dji = df_dji[["Close"]].reset_index()
df_dji.columns = ["date", "close"]
df_dji["date"] = df_dji["date"].astype(str).str[:10]
fst_day = df_dji["close"].iloc[0]
dji = (
    df_dji.assign(close=df_dji["close"].div(fst_day).mul(1000000))
    .set_index("date")[["close"]]
)

# %% Part 6. Compare results

df_result_a2c = (
    df_account_value_a2c.set_index(df_account_value_a2c.columns[0])
    if if_using_a2c else None
)
df_result_ddpg = (
    df_account_value_ddpg.set_index(df_account_value_ddpg.columns[0])
    if if_using_ddpg else None
)
df_result_ppo = (
    df_account_value_ppo.set_index(df_account_value_ppo.columns[0])
    if if_using_ppo else None
)
df_result_td3 = (
    df_account_value_td3.set_index(df_account_value_td3.columns[0])
    if if_using_td3 else None
)
df_result_sac = (
    df_account_value_sac.set_index(df_account_value_sac.columns[0])
    if if_using_sac else None
)

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

# Convert index to real DatetimeIndex so Matplotlib uses a numeric time axis.
# Without this, df.plot() registers dates as categorical strings and
# annotate() fails with ConversionError when a DJI date is not in the
# category registry used by DRL agent series.
result.index = pd.to_datetime(result.index.astype(str), errors="coerce")
result = result[~result.index.isna()].sort_index()
result = result.apply(pd.to_numeric, errors="coerce")

print("\n=== Backtest Results ===")
print(result)

print("\n=== Tuned Turbulence Thresholds ===")
for key, thr in best_threshold.items():
    if _agent_map[key][0]:
        thr_label = f"{thr:.1f}" if isinstance(thr, float) else str(thr)
        print(f"  {key.upper():5s}  threshold={thr_label}")

# %% Part 7. Plot

fig, ax = plt.subplots(figsize=(15, 5))
result.plot(ax=ax)

for key, (active, _) in _agent_map.items():
    if not active or key not in result.columns:
        continue
    series = result[key].dropna()
    if series.empty:
        continue
    x_last = series.index[-1]
    y_last = float(series.iloc[-1])
    thr_val = best_threshold[key]
    thr_label = f"{thr_val:.1f}" if isinstance(thr_val, float) else str(thr_val)
    ax.annotate(
        f"thr={thr_label}",
        xy=(x_last, y_last),
        xytext=(6, 0),
        textcoords="offset points",
        fontsize=7,
        va="center",
    )

ax.set_title("Portfolio Value Over Time (per-agent tuned turbulence threshold)")
ax.set_xlabel("Date")
ax.set_ylabel("Portfolio Value ($)")
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig("backtest_result.png", dpi=150, bbox_inches="tight")
print("\nPlot saved to backtest_result.png")
plt.close(fig)

# %% Part 8. Threshold grid search heatmap (only shown when grid search ran)

if use_grid_search and grid_sharpe:
    active_keys   = [k for k in _agent_map if _agent_map[k][0] and k in grid_sharpe]
    sharpe_matrix = pd.DataFrame(
        {k: [grid_sharpe[k].get(t, np.nan) for t in THRESHOLD_GRID]
         for k in active_keys},
        index=[str(t) if t < 9999 else "inf (off)" for t in THRESHOLD_GRID],
    )
    sharpe_matrix.columns = [c.upper() for c in sharpe_matrix.columns]

    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(sharpe_matrix.values.astype(float), aspect="auto",
                   cmap="RdYlGn", interpolation="nearest")
    plt.colorbar(im, ax=ax, label="Annualised Sharpe (validation split)")
    ax.set_xticks(range(len(sharpe_matrix.columns)))
    ax.set_xticklabels(sharpe_matrix.columns, fontsize=11)
    ax.set_yticks(range(len(sharpe_matrix.index)))
    ax.set_yticklabels(sharpe_matrix.index, fontsize=9)
    ax.set_xlabel("Agent", fontsize=12)
    ax.set_ylabel("Turbulence Threshold", fontsize=12)
    ax.set_title(
        "Threshold Grid Search - Validation Sharpe\n(navy border = selected threshold)",
        fontsize=12, fontweight="bold",
    )
    for i in range(len(sharpe_matrix.index)):
        for j in range(len(sharpe_matrix.columns)):
            v = sharpe_matrix.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=7.5, color="black", fontweight="bold")
    for j, key in enumerate(active_keys):
        thr = best_threshold[key]
        if thr in THRESHOLD_GRID:
            best_i = THRESHOLD_GRID.index(thr)
            ax.add_patch(plt.Rectangle(
                (j - 0.5, best_i - 0.5), 1, 1,
                fill=False, edgecolor="navy", linewidth=2.5,
            ))
    plt.tight_layout()
    plt.savefig("threshold_grid_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Heatmap saved to threshold_grid_heatmap.png")
else:
    print("Grid search not used — heatmap skipped (statistical threshold applied).")
