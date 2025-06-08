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
                logger.info(f"Adjusted backtest date from {backtest_date} to {valid_date}")
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
            logger.info(f"Fetched {len(ohlcv)} rows of OHLCV data for {stock}")
            if ohlcv.empty:
                raise ValueError(f"No price data available for {stock}")
            required_cols = ['Close', 'Volume']
            if not all(col in ohlcv.columns for col in required_cols):
                raise ValueError(f"Missing required columns {required_cols} in OHLCV data for {stock}")

            ohlcv = ohlcv[required_cols].copy()
            ohlcv['returns'] = ohlcv['Close'].pct_change()
            ohlcv = ohlcv[~ohlcv.index.isin(self.event_dates)].dropna()
            logger.info(f"After filtering holidays and NaNs, {len(ohlcv)} rows remain for {stock}")
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
            option_db = self.data_manager.Optionsdb.get_daily_option_stats(stock).sort_index()
            logger.info(f"Fetched {len(option_db)} rows of option data for {stock}")
            if option_db.empty:
                raise ValueError(f"No option data available for {stock}")
            required_cols = ['total_vol', 'total_oi', 'call_vol', 'put_vol', 'call_oi', 'put_oi', 'atm_iv']
            if not all(col in option_db.columns for col in required_cols):
                raise ValueError(f"Missing required columns {required_cols} in option data for {stock}")

            option_db = option_db[required_cols].copy()
            option_db['pcr_vol'] = option_db['put_vol'] / option_db['call_vol'].replace(0, np.nan).fillna(0.0)
            option_db['pcr_oi'] = option_db['put_oi'] / option_db['call_oi'].replace(0, np.nan).fillna(0.0)
            option_db = option_db[~option_db.index.isin(self.event_dates)].dropna()
            logger.info(f"After filtering holidays and NaNs, {len(option_db)} rows remain for {stock}")
            return option_db.sort_index()
        except Exception as e:
            logger.error(f"Error formatting option data for {stock}: {str(e)}")
            raise

    def get_aligned_data(self, stock: str, backtest_date: str | datetime = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Fetch and align price and option data for a stock, optionally truncated to a backtest date for historical data.

        Args:
            stock (str): Stock ticker symbol.
            backtest_date (str | datetime, optional): Backtest date for truncation.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Aligned OHLCV and option data.

        Raises:
            ValueError: If data cannot be aligned or is insufficient.
        """
        try:
            ohlcv = self.format_price_data(stock)
            option_db = self.format_option_data(stock)

            # Align date ranges
            common_dates = ohlcv.index.intersection(option_db.index)
            if common_dates.empty:
                raise ValueError(f"No overlapping dates between price and option data for {stock}")
            logger.info(f"Aligned data for {stock}: {len(common_dates)} common dates")

            ohlcv = ohlcv.loc[common_dates]
            option_db = option_db.loc[common_dates]

            if backtest_date is not None:
                backtest_date = self.validate_backtest_date(backtest_date, ohlcv)
                # Truncate historical data for analysis, but pass full ohlcv for returns
                historical_ohlcv = ohlcv[ohlcv.index <= backtest_date]
                historical_option_db = option_db[option_db.index <= backtest_date]
                self.get_n_day_returns(stock, ohlcv, backtest_date)  # Use full ohlcv for future returns

                if historical_ohlcv.empty or historical_option_db.empty:
                    raise ValueError(f"No data available after applying backtest filters for {stock}")

                return historical_ohlcv, historical_option_db

            return ohlcv, option_db
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
            if 'Returns' not in price_df.columns:
                price_df['Returns'] = price_df['Close'].pct_change()

            backtest_date = pd.to_datetime(backtest_date)
            # Get trading days starting from backtest_date
            trading_days = self.calendar.valid_days(
                start_date=backtest_date,
                end_date=backtest_date + pd.Timedelta(days=10)  # Extend to ensure 3 trading days
            ).tz_localize(None)

            close_prices = price_df[['Close','Returns']].reindex(trading_days).dropna()
            logger.info(f"Found {len(close_prices)} trading days for {stock} starting {backtest_date}")

            if len(close_prices) < 2:
                raise ValueError(f"Insufficient future data for {stock} starting {backtest_date}: {len(close_prices)} trading days available")

            current_price = close_prices['Close'].iloc[0] if backtest_date in close_prices.index else np.nan
            current_returns = close_prices['Returns'].iloc[0] if backtest_date in close_prices.index else np.nan

            # Initialize result DataFrame
            res = pd.DataFrame({
                'date': [backtest_date],
                'stock': [stock],
                'current_price': [current_price],
                'current_returns': [current_returns],
                '1d': [np.nan],
                '2d': [np.nan],
                '3d': [np.nan]
            })

            # Calculate buy-and-hold returns for available trading days
            for n in range(1, min(4, len(close_prices))):
                price_n_days = close_prices['Close'].iloc[n] if n < len(close_prices) else np.nan
                if not np.isnan(price_n_days):
                    res[f'{n}d'] = (price_n_days - current_price) / current_price
                else:
                    logger.warning(f"Insufficient data for {n}-day returns for {stock} at {backtest_date}")

            if res[['current_returns', '1d', '2d', '3d']].isna().all().all():
                raise ValueError(f"No valid returns calculated for {stock} at {backtest_date}")

            self.returns_df = pd.concat([self.returns_df, res], ignore_index=True)
            return res
        except Exception as e:
            logger.error(f"Error calculating returns for {stock} at {backtest_date}: {str(e)}")
            raise


if __name__ == "__main__":
    connections = get_path()
    data_util = DataUtility(connections)
    stock = 'AAPL'
    try:
        ohlcv, option_db = data_util.get_aligned_data(stock, backtest_date='2025-03-01')
        print(f"OHLCV Data for {stock}:\n{ohlcv.head()}\n")
        print(f"Option Data for {stock}:\n{option_db.head()}\n")
        returns = data_util.returns_df
        print(f"Returns Data for {stock}:\n{returns}\n")

        # Ensure that current data is working
        ohcv, option_db = data_util.get_aligned_data(stock)
        print(f"Aligned OHLCV Data for {stock}:\n{ohcv.tail()}\n")
        print(f"Aligned Option Data for {stock}:\n{option_db.tail()}\n")



    except ValueError as e:
        logger.error(f"Error in main execution: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error in main execution: {str(e)}")
        raise
    finally:
        data_util.data_manager.close_connection()