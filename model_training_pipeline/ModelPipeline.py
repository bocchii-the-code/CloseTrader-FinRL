"""
ModelPipeline -- Unified RL pipeline for single/multi-stock trading.

Entry point that orchestrates the three stages:
  1. Data -- fetch & preprocess
  2. Train -- train DRL agents with configurable hyperparameters
  3. Backtest -- evaluate against baselines and report performance

CLI usage:
    uv run python model_training_pipeline/ModelPipeline.py --ticker AAPL --models ppo --total-timesteps 50000

API usage:
    from model_training_pipeline.ModelPipeline import run_pipeline
    results = run_pipeline(ticker_list=["AAPL"], models=["ppo"], total_timesteps=50000)
"""

from __future__ import annotations

import argparse
import os
import sys

# Ensure repo root is on sys.path for both `python -m` and `python file.py`
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import pandas as pd

# Suppress Pandas4Warning from yfinance/pandas: monkey-patch because
# regular warnings.filterwarnings() does not catch C-level deprecations.
_pd_utcnow = getattr(pd.Timestamp, "utcnow", None)
if _pd_utcnow is not None:
    pd.Timestamp.utcnow = lambda: pd.Timestamp.now("UTC")

from model_training_pipeline.backtest import load_data as _load_data
from model_training_pipeline.backtest import load_trained_models as _load_trained_models
from model_training_pipeline.backtest import test_trained_model as _test_trained_model
from model_training_pipeline.data import preprocess_data
from model_training_pipeline.data import split_and_save_data
from model_training_pipeline.train import BUILTIN_MODEL_PARAMS
from model_training_pipeline.train import VALID_MODELS_LIST as VALID_MODELS
from model_training_pipeline.train import build_environment
from model_training_pipeline.train import make_trained_model_dir
from model_training_pipeline.train import train_drl_agents

from finrl.config import INDICATORS, TRAINED_MODEL_DIR
from finrl.config import TRAIN_END_DATE, TRAIN_START_DATE
from finrl.config import TRADE_END_DATE, TRADE_START_DATE


# =============================================================================
# Pipeline
# =============================================================================

def _banner(stage: str, detail: str = "") -> None:
    """Print a stage header with consistent formatting."""
    print("\n" + "=" * 60)
    print(f"{stage}: {detail}" if detail else stage)
    print("=" * 60)


def run_pipeline(
    *, # Must use keyword args to avoid confusion with so many parameters
    # Data
    ticker_list: list[str],
    train_start: str = TRAIN_START_DATE,
    train_end: str = TRAIN_END_DATE,
    trade_start: str = TRADE_START_DATE,
    trade_end: str = TRADE_END_DATE,
    indicator_list: list[str] | None = None,
    use_vix: bool = True,
    use_turbulence: bool = True,
    # Train
    models: list[str] | None = None,
    total_timesteps: int = 100_000,
    model_params: dict[str, dict] | None = None,
    trained_model_dir: str = TRAINED_MODEL_DIR,
    plot_live: bool = True,
    # Backtest
    initial_amount: float = 1000000,
    turbulence_threshold: float = 70,
    output_dir: str = "results",
    plot_filename: str = "backtest_result.png",
    skip_data: bool = False,
    skip_train: bool = False,
) -> pd.DataFrame:
    """Run the complete RL trading pipeline: data -> train -> backtest.

    Parameters
    ----------
    ticker_list: Stock tickers, e.g. ["AAPL"] or ["AAPL", "MSFT"].
    indicator_list: Technical indicators (default: config.INDICATORS).
    models: DRL algorithms to train (default: ["ppo"]).
        Valid: ["a2c", "ddpg", "ppo", "td3", "sac"].
    model_params: Per-model hyperparameter overrides.
        Merged with BUILTIN_MODEL_PARAMS. Example:
        {"ppo": {"n_steps": 2048, "learning_rate": 2.5e-4}}.
    skip_data: Reuse existing CSV files instead of downloading.
    skip_train: Only backtest (models must already exist).

    Returns
    -------
    pd.DataFrame
        One column per strategy, indexed by trade date.
    """
    if models is None:
        models = ["ppo"]
    if indicator_list is None:
        indicator_list = INDICATORS

    # Merge user hyperparam overrides with defaults
    merged_params: dict[str, dict] = {
        k: dict(v) for k, v in BUILTIN_MODEL_PARAMS.items()
    }
    if model_params:
        for name in model_params:
            if name in merged_params:
                merged_params[name] = {**merged_params[name], **model_params[name]}

    # ====== Stage 1: Data ================================================
    if not skip_data:
        _banner("STAGE 1: Data", f"Downloading {ticker_list}")

        from finrl.meta.preprocessor.yahoodownloader import YahooDownloader

        df_raw = YahooDownloader(
            start_date=train_start,
            end_date=trade_end,
            ticker_list=ticker_list,
        ).fetch_data()

        processed_full = preprocess_data(
            df_raw,
            tech_indicator_list=indicator_list,
            use_vix=use_vix,
            use_turbulence=use_turbulence,
        )

        train_df, trade_df = split_and_save_data(
            processed_full,
            save_path=_repo_root,
            train_start=train_start,
            train_end=train_end,
            trade_start=trade_start,
            trade_end=trade_end,
        )
    else:
        print("Skipping data stage -- using existing CSV files")
        train_df, trade_df = _load_data()

    # ====== Stage 2: Train ===============================================
    if not skip_train:
        _banner("STAGE 2: Train", f"Models: {models}")

        make_trained_model_dir(trained_model_dir)

        env_train = build_environment()

        trained_models = train_drl_agents(
            env_train,
            total_timesteps=total_timesteps,
            save_path=trained_model_dir,
            models=models,
            model_params=merged_params,
            plot_live=plot_live,
        )
    else:
        print("\nSkipping training stage -- loading existing models")
        trained_models = _load_trained_models(
            model_dir=trained_model_dir, model_names=models
        )

    # ====== Stage 3: Backtest ============================================
    printable = [k for k, v in trained_models.items() if v is not None]
    _banner("STAGE 3: Backtest", f"Evaluating {printable}")

    results = _test_trained_model(
        trade=trade_df,
        train=train_df,
        trained_models=trained_models,
        output_dir=output_dir,
        plot_filename=plot_filename,
        turbulence_threshold=turbulence_threshold,
        initial_amount=initial_amount,
    )

    return results


# =============================================================================
# CLI
# =============================================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ModelPipeline -- RL pipeline for stock trading"
    )
    p.add_argument(
        "--ticker", nargs="+", default=["AAPL"],
        help="Stock ticker(s), e.g. AAPL or AAPL MSFT",
    )
    p.add_argument(
        "--models", nargs="+", default=["ppo"],
        choices=VALID_MODELS,
        help="DRL algorithm(s) to train",
    )
    p.add_argument(
        "--total-timesteps", type=int, default=20000,
        help="Training steps per agent",
    )
    p.add_argument("--train-start", default=TRAIN_START_DATE)
    p.add_argument("--train-end", default=TRAIN_END_DATE)
    p.add_argument("--trade-start", default=TRADE_START_DATE)
    p.add_argument("--trade-end", default=TRADE_END_DATE)
    p.add_argument("--initial-amount", type=float, default=1000000)
    p.add_argument("--output-dir", default="results")
    p.add_argument("--plot", default="backtest_result.png")
    p.add_argument(
        "--skip-data", action="store_true",
        help="Reuse existing CSV files",
    )
    p.add_argument(
        "--skip-train", action="store_true",
        help="Only backtest (models must already exist)",
    )
    p.add_argument(
        "--plot-live", action="store_true",
        help="Open a live matplotlib window tracking reward per episode during training",
    )
    return p.parse_args()


if __name__ == "__main__":
    model_params = {    
        "ppo": {
        "n_steps": 512, 
        "batch_size": 256, 
        "n_epochs": 10,
        "learning_rate": 0.001, 
        "gamma": 0.995, 
        "gae_lambda": 0.99,
        "clip_range": 0.2, 
        "ent_coef": 0.02, 
        "vf_coef": 0.5,
        "max_grad_norm": 0.5, 
        "normalize_advantage": True,
        "policy_kwargs": {"net_arch": [256, 256], "ortho_init": True},
    },}
    args = _parse_args()
    run_pipeline(
        model_params= model_params,
        ticker_list=args.ticker,
        models=args.models,
        total_timesteps=args.total_timesteps,
        train_start=args.train_start,
        train_end=args.train_end,
        trade_start=args.trade_start,
        trade_end=args.trade_end,
        initial_amount=args.initial_amount,
        output_dir=args.output_dir,
        plot_filename=args.plot,
        skip_data=args.skip_data,
        skip_train=args.skip_train,
        plot_live=args.plot_live,
    )
