from dataclasses import dataclass
from typing import Tuple, Optional, Dict
import pandas as pd
import numpy as np
import logging
from pandas_market_calendars import get_calendar
from datetime import datetime
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[3]))

from bin.main import get_path
from main import Manager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class StockData:
    """Data container for stock price, option, and returns data."""
    ohlcv: pd.DataFrame
    option_df: pd.DataFrame
    returns: pd.DataFrame
    stock: str
    backtest_date: Optional[datetime] = None

    def __post_init__(self):
        """Validate and preprocess the data."""
        if not isinstance(self.ohlcv, pd.DataFrame) or not isinstance(self.option_df, pd.DataFrame) or not isinstance(self.returns, pd.DataFrame):
            raise TypeError("ohlcv, option_df, and returns must be pandas DataFrames")
        if self.ohlcv.empty or self.option_df.empty:
            raise ValueError(f"ohlcv or option_df DataFrame is empty for {self.stock}")
        if not self.ohlcv.index.equals(self.option_df.index):
            logger.error(f"Index misalignment between ohlcv and option_df for {self.stock}. OHLCV index: {self.ohlcv.index[:5]}, Option index: {self.option_df.index[:5]}")
            raise ValueError(f"Index misalignment for {self.stock}")
        if self.ohlcv.index.has_duplicates or self.option_df.index.has_duplicates:
            logger.error(f"Duplicate indices detected in ohlcv or option_df for {self.stock}")
            raise ValueError(f"Duplicate indices in data for {self.stock}")

@dataclass
class DataUtility:
    def __init__(self, connections: Dict | str):
        """
        Initialize the data utility for fetching and formatting stock and option data.

        Args:
            connections (Dict | str): Database connection details.
        """
        self.data_manager = Manager(connections)
        self.stocks = self.data_manager.Pricedb.stocks['mag8']
        self.calendar = get_calendar('NYSE')
        self.event_dates = self.calendar.holidays().holidays
        self.returns_df = pd.DataFrame()

    def validate_backtest_date(self, backtest_date: str | datetime, ohlcv: pd.DataFrame) -> datetime:
        """
        Validate and adjust the backtest date to the last valid trading day.

        Args:
            backtest_date (str | datetime): The target backtest date.
            ohlcv (pd.DataFrame): OHLCV data with date index.

        Returns:
            datetime: Validated backtest date.

        Raises:
            ValueError: If no valid trading day is found.
        """
        try:
            backtest_date = pd.to_datetime(backtest_date)
            if ohlcv.empty:
                raise ValueError("OHLCV data is empty, cannot validate backtest date")
            if backtest_date < ohlcv.index.min():
                raise ValueError(f"Backtest date {backtest_date} is before available data starting {ohlcv.index.min()}")
            if backtest_date not in ohlcv.index:
                valid_date = ohlcv[ohlcv.index <= backtest_date].index[-1] if not ohlcv[ohlcv.index <= backtest_date].empty else None
                if valid_date is None:
                    raise ValueError(f"No valid trading day found before {backtest_date}")
                backtest_date = valid_date
            return backtest_date
        except Exception as e:
            logger.error(f"Error validating backtest date: {str(e)}")
            raise ValueError(f"Invalid backtest date: {str(e)}")

    def format_price_data(self, stock: str) -> pd.DataFrame:
        """
        Format price data for a given stock.

        Args:
            stock (str): Stock ticker symbol.

        Returns:
            pd.DataFrame: DataFrame with 'Close', 'Volume', 'returns' columns.

        Raises:
            ValueError: If no data is available or required columns are missing.
        """
        try:
            ohlcv = self.data_manager.Pricedb.ohlc(stock).sort_index()
            if ohlcv.empty:
                raise ValueError(f"No price data available for {stock}")
            required_cols = ['Close', 'Volume']
            if not all(col in ohlcv.columns for col in required_cols):
                raise ValueError(f"Missing required columns {required_cols} in OHLCV data for {stock}")

            # Remove duplicate index entries, keeping the last
            if ohlcv.index.has_duplicates:
                ohlcv = ohlcv[~ohlcv.index.duplicated(keep='last')]

            ohlcv = ohlcv[required_cols].copy()
            ohlcv['Returns'] = ohlcv['Close'].pct_change()
            ohlcv = ohlcv[~ohlcv.index.isin(self.event_dates)]
            return ohlcv.sort_index()
        except Exception as e:
            logger.error(f"Error formatting price data for {stock}: {str(e)}")
            raise

    def format_option_data(self, stock: str) -> pd.DataFrame:
        """
        Format option data for a given stock.

        Args:
            stock (str): Stock ticker symbol.

        Returns:
            pd.DataFrame: DataFrame with option metrics and PCR ratios.

        Raises:
            ValueError: If no data is available or required columns are missing.
        """
        try:
            option_df = self.data_manager.Optionsdb.get_daily_option_stats(stock).sort_index()
            if option_df.empty:
                raise ValueError(f"No option data available for {stock}")
            required_cols = ['total_vol', 'total_oi', 'call_vol', 'put_vol', 'call_oi', 'put_oi', 'atm_iv']
            if not all(col in option_df.columns for col in required_cols):
                raise ValueError(f"Missing required columns {required_cols} in option data for {stock}")

            # Remove duplicate index entries, keeping the last
            if option_df.index.has_duplicates:
                option_df = option_df[~option_df.index.duplicated(keep='last')]

            option_df = option_df.copy()
            option_df['total_vol_oi'] = option_df['total_vol'] + option_df['total_oi']
            option_df['pcr_volume'] = option_df['put_vol'] / option_df['call_vol']
            option_df['pcr_oi'] = option_df['put_oi'] / option_df['call_oi']
            option_df = option_df[~option_df.index.isin(self.event_dates)]
            return option_df.sort_index()
        except Exception as e:
            logger.error(f"Error formatting option data for {stock}: {str(e)}")
            raise

    def get_aligned_data(self, stock: str, backtest_date: Optional[str | datetime] = None) -> StockData:
        try:
            ohlcv = self.format_price_data(stock)
            option_df = self.format_option_data(stock)

            # Resample option data to daily frequency with sum aggregation
            option_df = option_df.resample('1D').sum()
            option_df = option_df[~option_df.index.isin(self.event_dates)]

            # Align price data to option data's index range
            start_date = option_df.index.min()
            ohlcv = ohlcv.loc[start_date:].copy()
            ohlcv = ohlcv[~ohlcv.index.isin(self.event_dates)]

            # Replace zeros with NaN in option data
            option_df = option_df.replace(0, np.nan)

            # Align to common dates
            common_dates = ohlcv.index.intersection(option_df.index)
            if common_dates.empty:
                logger.error(f"No overlapping dates after alignment for {stock}")
                raise ValueError(f"No overlapping dates after alignment for {stock}")

            ohlcv = ohlcv.loc[common_dates].sort_index()
            option_df = option_df.loc[common_dates].sort_index()
            ohlcv.index = pd.DatetimeIndex(ohlcv.index, tz=None)
            option_df.index = pd.DatetimeIndex(option_df.index, tz=None)

            returns = self.returns_df
            if backtest_date is not None:
                backtest_date = self.validate_backtest_date(backtest_date, ohlcv)
                historical_ohlcv = ohlcv[ohlcv.index <= backtest_date]
                historical_option_df = option_df[option_df.index <= backtest_date]
                returns = self.get_n_day_returns(stock, ohlcv, backtest_date)

                if historical_ohlcv.empty or historical_option_df.empty:
                    logger.error(f"No data available after applying backtest filters for {stock}")
                    raise ValueError(f"No data available after applying backtest filters for {stock}")

                return StockData(
                    ohlcv=historical_ohlcv,
                    option_df=historical_option_df,
                    returns=returns,
                    stock=stock,
                    backtest_date=backtest_date
                )

            return StockData(
                ohlcv=ohlcv,
                option_df=option_df,
                returns=returns,
                stock=stock,
                backtest_date=None
            )
        except Exception as e:
            logger.error(f"Error getting aligned data for {stock}: {str(e)}")
            raise

    def get_n_day_returns(self, stock: str, price_df: pd.DataFrame, backtest_date: datetime) -> pd.DataFrame:
        """
        Calculate cumulative n-day future returns for a given stock and date, using trading days.

        Args:
            stock (str): Stock ticker symbol.
            price_df (pd.DataFrame): OHLCV data with 'Close' column.
            backtest_date (datetime): Date to calculate returns from.

        Returns:
            pd.DataFrame: DataFrame with stock, date, returns, and available 1d, 2d, 3d cumulative returns.

        Raises:
            ValueError: If no future data is available.
        """
        try:
            if 'returns' not in price_df.columns:
                price_df['returns'] = price_df['Close'].pct_change()

            backtest_date = pd.to_datetime(backtest_date)
            trading_days = self.calendar.valid_days(
                start_date=backtest_date - pd.Timedelta(days=1),
                end_date=backtest_date + pd.Timedelta(days=10)
            ).tz_localize(None)

            close_prices = price_df[['Close', 'returns']].reindex(trading_days).ffill().dropna()

            if len(close_prices) < 2:
                raise ValueError(f"Insufficient future data for {stock} starting {backtest_date}: {len(close_prices)} trading days available")

            current_price = close_prices['Close'].iloc[0] if backtest_date in close_prices.index else np.nan
            current_returns = close_prices['returns'].iloc[1] if len(close_prices) > 1 else np.nan

            res = pd.DataFrame({
                'date': [backtest_date],
                'stock': [stock],
                'current_price': [current_price],
                'current_returns': [current_returns],
                '1d': [np.nan],
                '2d': [np.nan],
                '3d': [np.nan]
            })

            future_indices = close_prices.index[close_prices.index > backtest_date]
            for n in range(1, min(4, len(future_indices) + 1)):
                if n <= len(future_indices):
                    price_n_days = close_prices['Close'].loc[future_indices[n-1]]
                    res[f'{n}d'] = (price_n_days - current_price) / current_price
                else:
                    logger.warning(f"Insufficient data for {n}-day returns for {stock} at {backtest_date}")

            if res[['current_returns', '1d', '2d', '3d']].isna().all().all():
                raise ValueError(f"No valid returns calculated for {stock} at {backtest_date}")

            self.returns_df = pd.concat([self.returns_df, res], ignore_index=True)
            return res
        except Exception as e:
            logger.error(f"Error calculating returns for {stock}: {str(e)}")
            raise

if __name__ == "__main__":
    connections = get_path()
    data_util = DataUtility(connections)
    stocks = data_util.data_manager.Optionsdb.all_stocks
    backtest_date = '2025-03-01'


    try:
        stock = 'gld'
        # for stock in stocks:
        #     try:
        #         result = data_util.get_aligned_data(stock, backtest_date=backtest_date)
        #         result_full = data_util.get_aligned_data(stock)
        #     except Exception as e:
        #         logger.error(f"Error processing {stock}: {str(e)}")
        result_bt = data_util.get_aligned_data(stock, backtest_date=backtest_date)
        result = data_util.get_aligned_data(stock)

        logger.info(f"Aligned data for {stock} with backtest date {backtest_date}:\n{result.ohlcv.head()}")
        logger.info(f"Option data for {stock}:\n{result.option_df}")

    except ValueError as e:
        logger.error(f"Error in main execution: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error in main execution: {str(e)}")
        raise
    finally:
        data_util.data_manager.close_connection()