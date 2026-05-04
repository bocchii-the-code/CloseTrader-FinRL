# CloseTrader-FinRL

<div align="center">
<img align="center" src=figs/logo_transparent_background.png width="55%"/>
</div>

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
![License](https://img.shields.io/github/license/AI4Finance-Foundation/finrl.svg?color=brightgreen)

**CloseTrader-FinRL** is a streamlined financial reinforcement learning framework built on top of FinRL, optimized for practical quantitative trading research and education.

This project preserves the core FinRL workflow while providing modern tooling via `uv` for dependency management and virtual environments.

## Quick Start

### Prerequisites

- Python 3.11 (recommended)
- [uv](https://docs.astral.sh/uv/) installed

### Installation with uv

```bash
# Install uv if not already installed
pip install uv

# Clone the repository
git clone https://github.com/bocchii-the-code/CloseTrader-FinRL.git
cd FinRL

# Pin Python version and sync dependencies
uv python pin 3.11
uv sync --locked
```

### Optional Extras

```bash
# ElegantRL adapter (may pull native deps)
uv sync --extra elegantrl

# Box2D environments
uv sync --extra box2d
```

## FinRL Stock Trading 2026 Tutorial

This tutorial demonstrates the original FinRL workflow for educational and research purposes using the modern `uv` setup.

### Step 1: Environment Setup (with uv)

```bash
# Clone and enter repository
git clone https://github.com/bocchii-the-code/CloseTrader-FinRL.git
cd FinRL

# Pin Python 3.11 and create virtual environment
uv python pin 3.11
uv sync --locked
```

### Step 2: Run the Tutorial Scripts

**1. Data Download & Preprocessing**

```bash
uv run python examples/FinRL_StockTrading_2026_1_data.py
```

This script downloads DOW 30 stock data from Yahoo Finance, adds technical indicators (MACD, RSI, etc.), VIX, and turbulence index, then splits the data into training set (2014–2025) and trading set (2026-01-01 to 2026-03-20), saving them as `train_data.csv` and `trade_data.csv`.

**2. Train DRL Agents**

```bash
# You may need to tweak the original hyperparameter and model selected to accelerate the training example
uv run python examples/FinRL_StockTrading_2026_2_train.py
```

This script trains 5 DRL agents (A2C, DDPG, PPO, TD3, SAC) using Stable Baselines 3 on the training data. Trained models are saved to the `trained_models/` directory.

**3. Backtest**

```bash
uv run python examples/FinRL_StockTrading_2026_3_Backtest.py
```

This script loads the trained agents, runs them on the trading data, and compares their performance against two baselines: Mean Variance Optimization (MVO) and the DJIA index. Results are printed to the console and a plot is saved as `backtest_result.png`.

## Quick Start Training Pipeline

The `model_training_pipeline/` module provides a unified, configurable, one-line user friendly RL pipeline that chains data download, training, and backtesting into a single command.

### Basic Usage

```bash
# Train PPO on AAPL and backtest (full pipeline)
uv run -m model_training_pipeline.ModelPipeline --ticker AAPL --models ppo

# Use custom date ranges
uv run -m model_training_pipeline.ModelPipeline \
    --ticker AAPL MSFT \
    --models ppo a2c \
    --total-timesteps 100000 \
    --train-end 2024-12-31 \
    --trade-start 2025-01-01 \
    --trade-end 2026-05-31

# Skip data download (reuse existing CSVs)
uv run -m model_training_pipeline.ModelPipeline --ticker AAPL --models ppo --skip-data

# Live plot of training rewards per iteration
uv run -m model_training_pipeline.ModelPipeline --ticker AAPL --models ppo --plot-live
```

Full CLI reference → `model_training_pipeline/README.md`.

### Guide for Customizing Training Pipeline

The pipeline is split into three files — modify them independently:

#### `model_training_pipeline/data.py` — Data Source & Preprocessing

| What to change       | Where                                  | Example                                                                              |
| -------------------- | -------------------------------------- | ------------------------------------------------------------------------------------ |
| Ticker universe      | `main()` / `fetch_dow30_data`      | Replace `["AMZN"]` with `config_tickers.DOW_30_TICKER`                           |
| Data provider        | `preprocess_data()` caller           | Swap `YahooDownloader` for `AlpacaDownloader` (see `finrl/meta/preprocessor/`) |
| Technical indicators | `INDICATORS` in `finrl.config`     | Add/remove indicators like `"macd"`, `"rsi"`, `"boll_ub"`                      |
| VIX / turbulence     | `use_vix`, `use_turbulence` params | Set to `False` to exclude macro features                                           |

#### `model_training_pipeline/train.py` — Model Selection & Hyperparameters

| What to change       | Where                                                | Example                                                      |
| -------------------- | ---------------------------------------------------- | ------------------------------------------------------------ |
| Algorithms to train  | `--models` CLI / `models` param                  | `--models ppo sac td3`                                     |
| Hyperparameters      | `BUILTIN_MODEL_PARAMS` dict                        | Change `"ppo": {"n_steps": 2048, "learning_rate": 2.5e-4}` |
| Per-model overrides  | `model_params` in API / CLI (via `run_pipeline`) | `model_params={"ppo": {"n_epochs": 5}}`                    |
| Training timesteps   | `--total-timesteps` CLI                            | `--total-timesteps 100000`                                 |
| Live reward plotting | `--plot-live` CLI                                  | Opens a matplotlib window tracking mean reward per iteration |

#### `model_training_pipeline/backtest.py` — Evaluation & Baselines

| What to change       | Where                                    | Example                                          |
| -------------------- | ---------------------------------------- | ------------------------------------------------ |
| Performance metrics  | `_print_performance_report()`          | Add Sortino ratio, Calmar ratio, win rate        |
| Baselines            | `test_trained_model()` (lines 256-280) | Add custom strategy series to `drl_results`    |
| Initial capital      | `--initial-amount` CLI                 | `--initial-amount 500000`                      |
| Turbulence threshold | `turbulence_threshold` param           | Lower for more conservative risk-off behavior    |
| Plot appearance      | `test_trained_model()` (lines 289-296) | Change figure size, title, save format (PNG/PDF) |

### Quick Customization Example

To train SAC with custom params on TSLA, using data from 2018-2024 and testing 2025:

```bash
uv run -m model_training_pipeline.ModelPipeline \
    --ticker TSLA \
    --models sac \
    --total-timesteps 50000 \
    --train-start 2018-01-01 \
    --train-end 2024-12-31 \
    --trade-start 2025-01-01 \
    --trade-end 2025-12-31
```

Or programmatically:

```python
from model_training_pipeline.ModelPipeline import run_pipeline

results = run_pipeline(
    ticker_list=["TSLA"],
    models=["sac"],
    total_timesteps=50_000,
    model_params={"sac": {"learning_rate": 5e-4, "buffer_size": 200_000}},
    train_start="2018-01-01",
    train_end="2024-12-31",
    trade_start="2025-01-01",
    trade_end="2025-12-31",
)
print(results.head())
```

---

## File Structure

```
FinRL
├── finrl (main folder)
│   ├── applications
│   │   ├── Stock_NeurIPS2018
│   │   ├── imitation_learning
│   │   ├── cryptocurrency_trading
│   │   ├── high_frequency_trading
│   │   ├── portfolio_allocation
│   │   └── stock_trading
│   ├── agents
│   │   ├── elegantrl
│   │   ├── rllib
│   │   └── stablebaseline3
│   ├── meta
│   │   ├── data_processors
│   │   ├── env_cryptocurrency_trading
│   │   ├── env_portfolio_allocation
│   │   ├── env_stock_trading
│   │   ├── preprocessor
│   │   ├── data_processor.py
│   │   ├── meta_config_tickers.py
│   │   └── meta_config.py
│   ├── config.py
│   ├── config_tickers.py
│   ├── main.py
│   ├── plot.py
│   ├── train.py
│   ├── test.py
│   └── trade.py
├── examples
├── model_training_pipeline
│   ├── ModelPipeline.py     # Unified entry point (CLI + API)
│   ├── data.py              # Data download & preprocessing
│   ├── train.py             # Environment building & DRL agent training
│   ├── backtest.py          # Backtesting, baselines, performance report
│   ├── results/             # TensorBoard logs & backtest plots
│   ├── trained_models/      # Saved model files (.zip)
│   └── README.md
├── unit_tests
│   ├── environments
│   │   └── test_env_cashpenalty.py
│   └── downloaders
│       ├── test_yahoodownload.py
│       └── test_alpaca_downloader.py
├── setup.py
├── requirements.txt
├── pyproject.toml
├── README.md
└── README_FINRL.md
```

## Key Features

- **Market Environments**: Gym-style financial market environments
- **DRL Agents**: Integration with Stable Baselines 3, ElegantRL, and RLlib
- **Data Processors**: Support for multiple data sources (Yahoo Finance, Alpaca, etc.)
- **Train-Test-Trade Pipeline**: Complete workflow for strategy development

## Running Code Unit Tests

```bash
# Run all tests
uv run pytest unit_tests -v

# Exclude Alpaca tests (requires credentials)
uv run pytest unit_tests -k "not alpaca_downloader" -v
```

## Original FinRL Documentation

For the complete original documentation and ecosystem information, see [README_FINRL.md](README_FINRL.md).

For the next-generation production-oriented stack, visit [FinRL-X / FinRL-Trading](https://github.com/AI4Finance-Foundation/FinRL-Trading).

## Citation

If you use this framework, please cite the original FinRL paper:

```bibtex
@article{finrl2020,
    author  = {Liu, Xiao-Yang and Yang, Hongyang and Chen, Qian and Zhang, Runjia and Yang, Liuqing and Xiao, Bowen and Wang, Christina Dan},
    title   = {{FinRL}: A deep reinforcement learning library for automated stock trading in quantitative finance},
    journal = {Deep RL Workshop, NeurIPS 2020},
    year    = {2020}
}
```

## License

MIT License

**Disclaimer**: This software is for academic and research purposes only. Nothing herein constitutes financial advice or a recommendation to trade real money.
