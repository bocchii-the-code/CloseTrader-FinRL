# Model Training Pipeline

A modular, configurable RL pipeline for stock trading — from data download through training to backtest.

## Quick Start

Full pipeline (download data, train, backtest):

```bash
uv run python model_training_pipeline/ModelPipeline.py --ticker AAPL --models ppo
```

Re-run backtest only (skip download and training):

```bash
uv run python model_training_pipeline/ModelPipeline.py --ticker AAPL --models ppo --skip-data --skip-train
```

Train multiple models on multiple stocks:

```bash
uv run python model_training_pipeline/ModelPipeline.py --ticker AAPL MSFT --models ppo a2c sac
```

> **Note:** Models trained on different state dimensions (e.g. DOW 30 vs single stock) are
> automatically skipped with a `[SKIP]` message. Re-train on the same ticker list for full evaluation.

---

## Architecture

```
ModelPipeline.run_pipeline()
    ├─ Stage 1: Data     (data.py)
    │   └─ YahooDownloader → FeatureEngineer → split_and_save_data()
    │       Outputs: train_data.csv, trade_data.csv
    │
    ├─ Stage 2: Train    (train.py)
    │   └─ build_environment() → train_drl_agents()
    │       Outputs: trained_models/agent_*.zip
    │
    └─ Stage 3: Backtest (backtest.py)
        └─ test_trained_model()
            ├─ DRL predictions
            ├─ Baselines (buy-hold or MVO + DJIA)
            ├─ Performance report (Sharpe, MaxDD, return)
            └─ Plot saved as backtest_result.png
```

---

## CLI Options

| Flag | Default | Description |
|---|---|---|
| `--ticker` | `AAPL` | Stock ticker(s), space-separated (e.g. `AAPL MSFT`) |
| `--models` | `ppo` | Algorithms to train: `a2c ddpg ppo td3 sac` |
| `--total-timesteps` | `100000` | Training steps per agent |
| `--train-start` | `2014-01-06` | Training data start date |
| `--train-end` | `2025-12-31` | Training data end date |
| `--trade-start` | `2026-01-01` | Backtest start date |
| `--trade-end` | `2026-03-20` | Backtest end date |
| `--initial-amount` | `1000000` | Starting portfolio value ($) |
| `--output-dir` | `results` | Output directory for plots and logs |
| `--plot` | `backtest_result.png` | Plot filename |
| `--skip-data` | `false` | Reuse existing CSV files |
| `--skip-train` | `false` | Skip training, only backtest |
| `--plot-live` | `false` | Show live matplotlib window with episode rewards during training |

---

## API Usage

```python
from model_training_pipeline.ModelPipeline import run_pipeline

results = run_pipeline(
    ticker_list=["AAPL"],
    models=["ppo", "a2c"],
    total_timesteps=100_000,
    model_params={
        "ppo": {"n_steps": 2048, "learning_rate": 2.5e-4},
    },
    skip_data=True,    # reuse existing CSVs
)
```

---

## Live Reward Plotting

Enable a real-time matplotlib window that tracks episode rewards during training:

### Via the unified pipeline (CLI)

```bash
uv run python model_training_pipeline/ModelPipeline.py --ticker AAPL --models ppo --plot-live
```

### Via the unified pipeline (API)

```python
from model_training_pipeline.ModelPipeline import run_pipeline

results = run_pipeline(
    ticker_list=["AAPL"],
    models=["ppo"],
    plot_live=True,
)
```

### Via the training stage directly

```python
from model_training_pipeline.train import build_environment, train_drl_agents

env = build_environment()
train_drl_agents(env, total_timesteps=100_000, models=["ppo"], plot_live=True)
```

- **Blue line** = raw episode reward  
- **Red line** = rolling mean (window = 20 episodes)

If no interactive display is available (e.g. remote server with `Agg` backend), the callback still collects rewards silently and skips the GUI redraw.

---

## Individual Stage Scripts

Each stage can also be run independently:

### Data only

```bash
uv run python model_training_pipeline/data.py
```

Fetches raw data and saves `train_data.csv` + `trade_data.csv` to the repo root.

### Train only

```bash
uv run python model_training_pipeline/train.py
```

Builds environment from CSV, trains models, saves to `trained_models/`.

### Backtest only

```bash
uv run python model_training_pipeline/backtest.py
```

Loads trained models and evaluates against baselines.

---

## Hyperparameter Overrides

Pass per-model hyperparameter overrides via `model_params`:

```python
run_pipeline(
    ticker_list=["AAPL"],
    models=["ppo", "sac"],
    model_params={
        "ppo": {"n_steps": 2048, "n_epochs": 5},
        "sac": {"learning_rate": 5e-4, "batch_size": 256},
    },
)
```

Default hyperparameters are defined in `train.py::BUILTIN_MODEL_PARAMS`. User overrides are merged on top — you only need to specify what you want to change.

---

## Supported Algorithms

| Algorithm | Type | Description |
|---|---|---|
| **PPO** | On-policy | Proximal Policy Optimization — stable, widely used |
| **A2C** | On-policy | Advantage Actor-Critic — simpler but effective |
| **DDPG** | Off-policy | Deep Deterministic Policy Gradient — continuous action space |
| **TD3** | Off-policy | Twin Delayed DDPG — improved stability over DDPG |
| **SAC** | Off-policy | Soft Actor-Critic — entropy-regularized, good exploration |

---

## File Structure

```
model_training_pipeline/
├── ModelPipeline.py     # Unified entry point (CLI + API)
├── data.py              # Data download & preprocessing
├── train.py             # Environment building & DRL agent training
├── backtest.py          # Backtesting, baselines, performance report
├── results/             # TensorBoard logs & backtest plots
├── trained_models/      # Saved model files (.zip)
└── README.md
```

## Data Sources

- **Yahoo Finance** via `yfinance` — daily OHLCV data
- **Technical indicators** via `stockstats` — MACD, RSI, Bollinger Bands, etc.
- **VIX** from Yahoo Finance — market volatility index
- **Turbulence index** — computed from stock price covariance
