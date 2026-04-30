

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
git clone https://github.com/AI4Finance-Foundation/FinRL.git
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
git clone https://github.com/AI4Finance-Foundation/FinRL.git
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
uv run python examples/FinRL_StockTrading_2026_2_train.py
```

This script trains 5 DRL agents (A2C, DDPG, PPO, TD3, SAC) using Stable Baselines 3 on the training data. Trained models are saved to the `trained_models/` directory.

**3. Backtest**

```bash
uv run python examples/FinRL_StockTrading_2026_3_Backtest.py
```

This script loads the trained agents, runs them on the trading data, and compares their performance against two baselines: Mean Variance Optimization (MVO) and the DJIA index. Results are printed to the console and a plot is saved as `backtest_result.png`.

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

## Running Tests

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
