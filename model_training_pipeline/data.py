#%%
"""
Stock NeurIPS2018 Part 1. Data

This series is a reproduction of paper "Deep reinforcement learning for automated stock trading: An ensemble strategy".

Introduce how to use FinRL to fetch and process data that we need for ML/RL trading.
"""

from __future__ import annotations

import itertools
import os

import pandas as pd
import yfinance as yf

# Suppress Pandas4Warning from yfinance/pandas: monkey-patch because
# regular warnings.filterwarnings() does not catch C-level deprecations.
_pd_utcnow = getattr(pd.Timestamp, "utcnow", None)
if _pd_utcnow is not None:
    pd.Timestamp.utcnow = lambda: pd.Timestamp.now("UTC")

from finrl import config_tickers
from finrl.config import INDICATORS
from finrl.config import TRADE_END_DATE
from finrl.config import TRADE_START_DATE
from finrl.config import TRAIN_END_DATE
from finrl.config import TRAIN_START_DATE
from finrl.meta.preprocessor.preprocessors import data_split
from finrl.meta.preprocessor.preprocessors import FeatureEngineer
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader



# %% Part 1. Fetch data - Single ticker
def fetch_single_stock_data(ticker: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """
    Fetch historical stock data for a single ticker using yfinance.

    Args:
        ticker (str): The stock ticker symbol (e.g., "AAPL").
        start_date (str): The start date for fetching data (format: "YYYY-MM-DD"). If None, defaults to TRAIN_START_DATE.
        end_date (str): The end date for fetching data (format: "YYYY-MM-DD"). If None, defaults to TRADE_END_DATE.

    Returns:
        pd.DataFrame: A DataFrame containing the historical stock data.
    """
    # Using yfinance directly
    df_yf = yf.download(tickers=ticker, start=start_date, end=end_date)
    print("=== yfinance download ===")
    print(df_yf.head())

    # Using FinRL's YahooDownloader
    df_finrl = YahooDownloader(
        start_date=start_date if start_date else TRAIN_START_DATE,
        end_date=end_date if end_date else TRADE_END_DATE,
        ticker_list=[ticker],
    ).fetch_data()
    print("\n=== FinRL YahooDownloader ===")
    print(df_finrl.head())
    return df_finrl

# %% Part 2. Fetch data - DOW 30 tickers
def fetch_dow30_data(ticker_list: list = config_tickers.DOW_30_TICKER, start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """
    Fetch historical stock data for DOW 30 tickers using FinRL's YahooDownloader.

    Args:
        ticker_list (list): The list of ticker symbols. If None, defaults to config_tickers.DOW_30_TICKER. Available tickers are in finrl.config_tickers.DOW_30_TICKER: 
            ['AAPL', 'AXP', 'BA', 'CAT', 'CSCO', 'CVX', 'DIS', 'DOW', 'GS', 'HD',
             'IBM', 'INTC', 'JNJ', 'JPM', 'KO', 'MCD', 'MMM', 'MRK', 'MSFT',
             'NKE', 'PFE', 'PG', 'TRV', 'UNH', 'VZ', 'WBA', 'WMT', 'XOM']
        
        start_date (str): The start date for fetching data (format: "YYYY-MM-DD"). If None, defaults to TRAIN_START_DATE.
        end_date (str): The end date for fetching data (format: "YYYY-MM-DD"). If None, defaults to TRADE_END_DATE.

    Returns:
        pd.DataFrame: A DataFrame containing the historical stock data for DOW 30 tickers.
    """
    print("\n=== DOW 30 Tickers ===")
    print(ticker_list)

    df_raw = YahooDownloader(
        start_date=start_date if start_date else TRAIN_START_DATE,
        end_date=end_date if end_date else TRADE_END_DATE,
        ticker_list=ticker_list
    ).fetch_data()
    print("\n=== Raw data ===")
    print(df_raw.head())
    return df_raw

# %% Part 3. Preprocess data
def preprocess_data(df_raw: pd.DataFrame,
                    use_technical_indicator: bool = True,
                    tech_indicator_list: list = INDICATORS,
                    use_vix: bool = True,
                    use_turbulence: bool = True,
                    user_defined_feature: bool = False) -> pd.DataFrame:
    """
    Preprocess the raw stock data using FinRL's FeatureEngineer.

    Args:
        df_raw (pd.DataFrame): The raw stock data DataFrame.
        use_technical_indicator (bool): Whether to use technical indicators.
        tech_indicator_list (list): List of technical indicators to use.
        use_vix (bool): Whether to use VIX data.
        use_turbulence (bool): Whether to use turbulence data.
        user_defined_feature (bool): Whether to use user-defined features.

    Returns:
        pd.DataFrame: A preprocessed DataFrame ready for training/testing.
    """
    fe = FeatureEngineer(
        use_technical_indicator=use_technical_indicator,
        tech_indicator_list=tech_indicator_list,
        use_vix=use_vix,
        use_turbulence=use_turbulence,
        user_defined_feature=user_defined_feature,
    )

    processed = fe.preprocess_data(df_raw)

    list_ticker = processed["tic"].unique().tolist()
    list_date = list(
        pd.date_range(processed["date"].min(), processed["date"].max()).astype(str)
    )
    combination = list(itertools.product(list_date, list_ticker))

    processed_full = pd.DataFrame(combination, columns=["date", "tic"]).merge(
        processed, on=["date", "tic"], how="left"
    )
    processed_full = processed_full[processed_full["date"].isin(processed["date"])]
    processed_full = processed_full.sort_values(["date", "tic"])
    processed_full = processed_full.fillna(0)

    print("\n=== Processed data ===")
    print(processed_full.head())

    return processed_full


# %% Part 4. Split and save data
def split_and_save_data(
    processed_full: pd.DataFrame,
    save_path: str = None,
    *,
    train_start: str = TRAIN_START_DATE,
    train_end: str = TRAIN_END_DATE,
    trade_start: str = TRADE_START_DATE,
    trade_end: str = TRADE_END_DATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split processed data into train/trade sets and save as CSV.
    """
    parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if save_path is None else save_path

    train = data_split(processed_full, train_start, train_end)
    trade = data_split(processed_full, trade_start, trade_end)
    print(f"\nTrain data length: {len(train)}")
    print(f"Trade data length: {len(trade)}")

    train.to_csv(os.path.join(parent_path, "train_data.csv"))
    trade.to_csv(os.path.join(parent_path, "trade_data.csv"))
    print("Data saved to train_data.csv and trade_data.csv")
    return train, trade


#%% Main function to run the preprocessing steps
def main():
    # Fetch data for DOW 30 tickers
    df_raw = fetch_dow30_data(ticker_list=["AMZN"])

    # Preprocess the data
    processed_full = preprocess_data(df_raw)

    # Save the processed data
    train_df, trade_df = split_and_save_data(processed_full)


if __name__ == "__main__":
    main()
