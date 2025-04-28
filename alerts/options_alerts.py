import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import datetime as dt 
from tqdm.contrib.logging import logging_redirect_tqdm
from tqdm import tqdm 
import scipy.stats as st 
import time
import json 
import logging
from typing import List, Optional, Dict
import sqlite3 as sql
from scipy import stats


import sys
sys.path.append('/Users/jerald/Documents/Dir/Python/Stocks')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Notifications:
    def __init__(self, connections: Dict[str, str]):
        try:
            self.vol_db = sql.connect(connections['vol_db'])
            self.stocks = json.load(open(connections['ticker_path'], 'r'))
            logger.info("Notifications instance initialized.")
        except KeyError as e:
            logger.error(f"Key error in connections dict: {e}")
            raise
        except FileNotFoundError:
            logger.error("Ticker file not found.")
            raise

    def stock_data(self, stock: str, n: int = 5, date: Optional[str] = None) -> pd.DataFrame:
        """ 
        Retrieve the daily Option Statistics for a given stock.

        Args:
            stock (str): The stock symbol.
            n (int): Lookback period for change calculations (not used in this method).
            date (Optional[str]): Specific date for data retrieval. If None, use latest data.

        Returns:
            pd.DataFrame: DataFrame with stock option data.

        Raises:
            Exception: If SQL query fails.
        """
        try:
            if date is None:
                df = pd.read_sql(f'select * from {stock} where date(gatherdate) > date("2023-10-01")', self.vol_db, parse_dates=['gatherdate'], index_col=['gatherdate'])
            else:
                df = pd.read_sql(f'select * from {stock} where date(gatherdate) <= date("{date}")', self.vol_db, parse_dates=['gatherdate'], index_col=['gatherdate'])
                
            # dropCols = list(df.filter(regex='pct|spread|total|atm|otm|iv_chng'))
            dropCols = list(df.filter(regex='pct|spread|iv_chng|prem'))
            df = df.drop(columns=dropCols)
            return df
        except Exception as e:
            logger.error(f"Error retrieving stock data for {stock}: {e}")
            return pd.DataFrame()

    def col_map(self, col: str) -> str:
        """ 
        Map column names to human-readable descriptions.

        Args:
            col (str): Column name to map.

        Returns:
            str: Mapped description.

        Raises:
            KeyError: If column name is not in the mapping dictionary.
        """
        cmap = {
                    "call_vol": "Call Volume",
                    "put_vol": "Put Volume",
                    "total_vol": "Total Volume",
                    "call_oi": "Call OI",
                    "put_oi": "Put OI",
                    "total_oi": "Total OI",
                    "call_prem": "Call Premium",
                    "put_prem": "Put Premium",
                    "total_prem": "Total Premium",
                    "call_iv": "Call IV",
                    "put_iv": "Put IV",
                    "atm_iv": "ATM IV",
                    "otm_iv": "OTM IV",
                    "call_vol_chng": "Call Volume Chng",
                    "put_vol_chng": "Put Volume Chng",
                    "total_vol_chng": "Total Volume Chng",
                    "call_oi_chng": "Call OI Chng",
                    "put_oi_chng": "Put OI Chng",
                    "total_oi_chng": "Total OI Chng",
                    "call_prem_chng": "Call Prem. Chng",
                    "put_prem_chng": "Put Prem. Chng",
                    "total_prem_chng": "Total Prem. Chng",
                    "call_iv_chng": "Call IV Chng",
                    "put_iv_chng": "Put IV Chng",
                    "atm_iv_chng": "ATM IV Chng",
                    "otm_iv_chng": "OTM IV Chng",
                    "call_oi_chng5d": "Call OI Chng (5d)",
                    "put_oi_chng5d": "Put OI Chng (5d)",
                    "call_vol_chng5d": "Call Vol Chng (5d)",
                    "put_vol_chng5d": "Put Vol Chng (5d)",
                }
        try:
            return cmap[col]
        except KeyError:
            logger.warning(f"Column {col} not found in col_map")
            return col

    def _colors(self, color: Optional[str] = None, word: Optional[str] = None) -> str:
        """ 
        Generate ANSI color codes or apply color to text.

        Args:
            color (Optional[str]): Name of color or None for all colors.
            word (Optional[str]): Text to colorize.

        Returns:
            str: Color code or colored text.

        Raises:
            KeyError: If an invalid color name is provided.
        """
        colors = {
            'header': '\033[95m', 
            # Basic Colors 
            'red': '\033[031m', 'green': '\033[032m', 'yellow': '\033[033m','blue': '\033[34m',
            'cyan': '\033[36m', 'white': '\033[37m', 'grey': '\033[30m','purple': '\033[035m', 
            # Bright colors
            'bright-red': '\033[91m', 'bright-green': '\033[92m', 'bright-yellow': '\033[93m',
            'bright-blue': '\033[94m', 'bright-cyan': '\033[96m', 'bright-white': '\033[97m',
            # Bold colors
            'bold-red': '\033[1;31m', 'bold-green': '\033[1;32m', 'bold-yellow': '\033[1;33m',
            'bold-blue': '\033[1;34m', 'bold-cyan': '\033[1;36m', 'bold-white': '\033[1;37m',
            # End color
            'end': '\033[0m',
        }
        
        if color is None:
            return colors
        elif word is None:
            return colors.get(color, '')
        else:
            return colors.get(color, '') + word + colors['end']
        
    ##############################################  Metrics  ########################################################
    def _percentile_score(self, df: pd.DataFrame, col: str) -> float:
        """ 
        Calculate percentile score for a column in a DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame containing the data.
            col (str): The column name to analyze.

        Returns:
            float: Percentile score where 0 indicates unusually high values.

        Raises:
            ValueError: If column does not exist in DataFrame.
        """
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame")
        return st.percentileofscore(df[col], df[col].iloc[-1], kind='strict', nan_policy='omit')
    
    def calculate_ivr(self, df, col):
        """
        Calculate Implied Volatility Rank (IVR).

        Args:
            df (pd.DataFrame): DataFrame containing historical data.
            col (str): Column name for IV data.

        Returns:
            float: IVR between 0 and 100.
        """

        iv = df[col].iloc[-1]
        iv_52w_high = df[col].rolling(window=252).max().iloc[-1]
        iv_52w_low = df[col].rolling(window=252,).min().iloc[-1]
        
        if iv_52w_high == iv_52w_low:
            return None
        
        ivr = (iv - iv_52w_low) / (iv_52w_high - iv_52w_low) * 100
        if ivr is np.nan:
            return None
        else:
            return ivr

    def calculate_ivp(self, df, col):
        """
        Calculate Implied Volatility Percentile (IVP).

        Args:
            df (pd.DataFrame): DataFrame containing historical data.
            col (str): Column name for IV data.

        Returns:
            float: IVP between 0 and 100.
        """
        current_iv = df[col].iloc[-1]
        historical_iv = df[col].rolling(window=252, min_periods=10).mean()
        percentile = stats.percentileofscore(historical_iv, current_iv)
        return percentile

    def open_interest_change_rate(self, df, col, period=1):
        """
        Calculate the rate of change in open interest.

        Args:
            df (pd.DataFrame): DataFrame with open interest data.
            col (str): Column name for OI.
            period (int): Number of periods to look back.

        Returns:
            float: Change rate as a percentage.
        """
        current_oi = df[col].iloc[-1]
        previous_oi = df[col].iloc[-period]
        if previous_oi == 0:
            return 0  # Avoid division by zero
        change_rate = (current_oi - previous_oi) / previous_oi * 100
        return change_rate

    def z_score(self, df, col):
        """
        Calculate the Z-score for a given column.

        Args:
            df (pd.DataFrame): DataFrame with the data.
            col (str): Column name.

        Returns:
            float: Z-score of the last entry in the column.
        """
        last_value = df[col].iloc[-1]
        mean = df[col].mean()
        std_dev = df[col].std()
        if std_dev == 0:
            return 0  # Avoid division by zero
        return (last_value - mean) / std_dev      
    
    ############################################################################################################################
    ##############################################  NOTIFICATIONS TEXT  ########################################################
    ############################################################################################################################


    def __check_iv_metrics(self, stock: str, df: pd.DataFrame, col: str, col_name: str, current: float) -> Optional[str]:
        """
        Generate notifications based on implied volatility metrics.
        """
        try:
            ivr = self.calculate_ivr(df, col)
            ivp = self.calculate_ivp(df, col)
            
            if df.shape[0] <= 252 or ivr is None or ivp is None:
                return None
                
            # Check IVP (Implied Volatility Percentile)
            if ivp > 90:
                return self._colors('yellow', f"${stock.upper()} {col_name} is in the {ivp:.2f}th percentile of historical volatility")
            elif ivp < 10:
                return self._colors('yellow', f"${stock.upper()} {col_name} is in the {ivp:.2f}th percentile, indicating low volatility")
            
            # Check IVR (Implied Volatility Rank)
            if ivr > 80 or ivr < 20:
                option_type = 'Call' if 'call' in col.lower() else 'Put' if 'put' in col.lower() else ''
                expense = 'expensive' if ivr > 80 else 'cheap'
                color = 'bright-red' if ivr > 80 else 'bright-green'
                msg = f"${stock.upper()} {col_name} Rank is at {ivr:.2f}%, suggesting"
                msg += f" {option_type} Options are {expense}" if option_type else f" Options are {expense}"
                logger.info(self._colors(color, msg))
                return self._colors(color, msg)
                
        except Exception as e:
            logger.error(f"Error in IV metrics check for {stock}, column {col}: {e}")
        return None

    def __check_oi_change(self, stock: str, df: pd.DataFrame, col: str, col_name: str) -> Optional[str]:
        """
        Generate notifications based on open interest changes.
        """
        oi_change_rate = self.open_interest_change_rate(df, col) if 'oi' in col else None
        if oi_change_rate is not None and abs(oi_change_rate) > 5:
            direction = "increased" if oi_change_rate > 0 else "decreased"
            color = 'green' if oi_change_rate > 0 else 'red'
            msg = self._colors(color, f"${stock.upper()} {col_name} has {direction} by {abs(oi_change_rate):.2f}%")
            logger.info(msg)
            return msg
        return None

    def __check_volume_oi(self, stock: str, df: pd.DataFrame, col: str, col_name: str, current: float, previous: Optional[float]) -> Optional[str]:
        """
        Generate notifications based on volume or open interest levels.
        """
        if ('vol' in col or 'oi' in col) and df.shape[0] > 5:
            five_day_avg = df[col].rolling(window=5).mean().iloc[-1]
            twenty_day_avg = df[col].rolling(window=20).mean().iloc[-1]
            if current > five_day_avg * 5 and previous is not None and current > 1000 and five_day_avg > 0:
                direction = "surged higher" if current > previous else "pulled back"
                msg = self._colors('bright-yellow',
                    f"${stock.upper()} {col_name} has {direction} to {current:,.2f}, 5x higher than 5-day average ({five_day_avg:,.2f})")
                logger.info(msg)
                return msg
        return None

    def __check_zscore(self, stock: str, col: str, col_name: str, current: float, z_score_val: float) -> Optional[str]:
        """
        Generate notifications based on z-score analysis.
        """
        if abs(z_score_val) > 2:
            msg = f"${stock.upper()} {col_name} ({current:,.2f}) with a z-score of {z_score_val:.2f}, indicating a significant deviation"
            if 'call' in col.lower():
                msg = self._colors('bright-red', msg)
            elif 'put' in col.lower():
                msg = self._colors('bright-green', msg)
            
            if msg:
                logger.info(msg)
                return msg

        return None

    def __check_percent_change(self, stock: str, col_name: str, current: float, previous: float) -> Optional[str]:
        """
        Generate notifications based on percentage changes.
        """
        if previous is not None:
            pct_change = (current - previous) / previous * 100
            if abs(pct_change) > 1000:
                color = 'green' if pct_change > 2 else 'red'
                msg = self._colors(color,
                    f"${stock.upper()} {col_name} changed by {pct_change:,.2f}% from {previous:,.2f} to {current:,.2f}")
                logger.info(msg)
                return msg
        return None

    def __check_ma_deviation(self, stock: str, col_name: str, current: float, ma_20: float) -> Optional[str]:
        """
        Generate notifications based on moving average deviations.
        """
        if ma_20 is not None:
            deviation = (current - ma_20) / ma_20 * 100
            if abs(deviation) < 0.5:
                msg = self._colors('white',
                    f"${stock.upper()} {col_name} is {deviation:.2f}% within 1% of the 20-day moving average")
                logger.info(msg)
                return msg
        return None

    def __check_historical_extremes(self, stock: str, col_name: str, current: float, historical_max: float, historical_min: float) -> Optional[str]:
        """
        Generate notifications based on historical extremes.
        """
        if current >= historical_max or (current >= historical_max * 0.99):
            return self._colors('bold-blue',
                f"${stock.upper()} {col_name} at new all-time high or within 1% of it: {current:,.2f}")
        elif current <= historical_min or (current <= historical_min * 1.01):
            msg = self._colors('bold-yellow',
                f"${stock.upper()} {col_name} plummeted to a new all-time low or within 1% of it: {current:,.2f}")
            logger.info(msg)
            return msg
        return None

    def __generate_text(self, stock: str, df: pd.DataFrame, col: str) -> Optional[str]:
        """
        Generate text notifications based on changes in option metrics.

        Args:
            stock (str): Stock symbol.
            df (pd.DataFrame): DataFrame with stock data.
            col (str): Column to analyze.

        Returns:
            Optional[str]: Notification text or None if not applicable.
        """
        if col not in df.columns:
            logger.warning(f"Column {col} not found in DataFrame for {stock}")
            return None

        try:
            col_name = self.col_map(col)
            current = df[col].iloc[-1]
            previous = df[col].iloc[-2] if df.shape[0] > 1 else None
            ma_20 = df[col].rolling(window=20).mean().iloc[-1] if df.shape[0] > 20 else None
            historical_max = df[col].max()
            historical_min = df[col].min()
            z_score_val = self.z_score(df, col)

            # Check metrics in order of priority
            notification = (
                # IV metrics
                (self.__check_iv_metrics(stock, df, col, col_name, current) if 'iv' in col else None)
                # Open Interest changes
                or self.__check_oi_change(stock, df, col, col_name)
                # Volume/OI levels
                or self.__check_volume_oi(stock, df, col, col_name, current, previous)
                # Z-score analysis
                or self.__check_zscore(stock, col, col_name, current, z_score_val)
                # Percent changes
                or (self.__check_percent_change(stock, col_name, current, previous) if previous is not None else None)
                # MA deviation
                # or (self.__check_ma_deviation(stock, col_name, current, ma_20) if ma_20 is not None else None)
                # Historical extremes
                or self.__check_historical_extremes(stock, col_name, current, historical_max, historical_min)
            )


            if notification:
                return notification

        except IndexError:
            logger.error(f"Index error in column {col} for {stock}")
        except Exception as e:
            logger.error(f"Error in __generate_text for {stock}, column {col}: {e}")

        return None
    
    def __remove_colors(self, txt: str) -> str:
        """ 
        Remove ANSI color codes from text.

        Args:
            txt (str): Text containing color codes.

        Returns:
            str: Text without color codes.
        """
        colors = [v for v in self._colors().values()]
        for color in colors:
            txt = txt.replace(color, '')
        return txt

    def notifications(self, stock: str, n: int = 5, date: Optional[str] = None) -> List[str]:
        """
        Generate notifications for a specific stock.

        Args:
            stock (str): Stock symbol.
            n (int): Lookback period.
            date (Optional[str]): Specific date for data.

        Returns:
            List[str]: List of notification texts.

        Raises:
            ValueError: If stock is not in the known stocks list.
        """
        if stock not in self.stocks['all_stocks']:
            logger.warning(f"Stock {stock} not found in known stocks.")
            return []
        
        df = self.stock_data(stock, n, date)
        if df.empty or df.shape[0] <= 10:
            return []
        
        notifications = []
        for i in df.columns:
            txt = self.__generate_text(stock, df, i)
            if txt is not None:
                # logger.info(f"Notification for {stock}: {txt}")
                notifications.append(self.__remove_colors(txt))
        return notifications

    def iterator(self, n: int = 5, date: Optional[str] = None) -> List[str]:
        """ 
        Iterate through all stocks to generate notifications.

        Args:
            n (int): Number of days for data analysis.
            date (Optional[str]): Specific date for analysis.

        Returns:
            List[str]: All generated notifications.

        Raises:
            Exception: If there's an issue iterating over stocks.
        """
        out = []
        
        # for stock in tqdm(self.stocks['all_stocks'], desc="Generating Notifications"):
        lot = sorted(self.stocks['etf'])
        with logging_redirect_tqdm():
            pbar = tqdm(lot, desc="Stock Notifications")
            for stock in pbar:
                pbar.set_description(f"Processing {stock}")
                try:
                    notifications = self.notifications(stock, n, date)
                    if notifications:
                        out.extend(notifications)
                    # time.sleep(np.random.normal(3, 0.2))  # Consider if this can be optimized or removed
                except Exception as e:
                    logger.error(f"Error processing notifications for {stock}: {e}")
            pbar.close()
        if len(out) > 0: 
            return out

if __name__ == '__main__':
    from bin.main import get_path
    connections = get_path()
    notif = Notifications(connections)
    try:
        out = notif.iterator(n=900, date=None)
    except Exception as e:
        logger.error(f"Main execution failed: {e}")
    
    if out is not None:
        with open('alerts.txt', 'w') as f:
            f.write(f"Option Alerts for Date: {dt.datetime.now().strftime('%Y-%m-%d')}\n")
            for line in out:
                f.write(f"{line}\n")
