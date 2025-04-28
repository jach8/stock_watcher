"""
Momentum Price Signals.

This module implements technical analysis tools for calculating and analyzing
the momentum of a financial instrument over time. Momentum is a measure of the
rate of change in the price of a security or market index. It is often used to
identify the strength of a trend and to determine the optimal entry and exit
points for a given trade.


Key Features:
- Rate of Change (ROC)
- Relative Strength Index (RSI)
- Stochastic Oscillator
"""


from typing import Union, Optional, Dict, List
import pandas as pd 
import numpy as np 
import sqlite3 as sql 
from logging import getLogger
# from .utils import combine_timeframes, derive_timeframe
#from utils import combine_timeframes, derive_timeframe

logger = getLogger(__name__)


class momentum:
    def __init__(self) -> None:
        """
        Initialize the Momentum class
        """
        self.windows = np.array([6, 10, 28])

    def _validate_dataframe(self, df):
        df.columns = [str(x).lower() for x in df.columns]
        if not isinstance(df, pd.DataFrame):
            raise ValueError('Data must be a DataFrame')
        if not all(str(col).lower() in df.columns for col in ['open', 'high', 'low', 'close']):
            raise ValueError(f'Data must contain open, high, low, and close columns found: {df.columns}')
        if not df.index.is_monotonic_increasing:
            raise ValueError('Data must be sorted by date')
        if not len(df) > 0:
            raise ValueError('Data must contain at least one row')
        return df

    def roc(self, df: pd.DataFrame, window: int = 10) -> pd.Series:
        ''' MOMENTUM:  Rate of Change. '''
        try:
            out = np.array((df.close - df.close.shift(window)) / df.close.shift(window))
            return pd.Series(out, index=df.index, name = f'ROC_{window}')
        except Exception as e:
            logger.error(f"Error calculating Momentum: {str(e)}")
            raise


    def rsi(self, df: pd.DataFrame, window: int = 14) -> pd.Series:
        ''' Relative Strength Index. '''
        try:
            delta = df.close.diff()
            up_days = delta.copy()
            up_days[delta<=0]=0.0
            down_days = abs(delta.copy())
            down_days[delta>0]=0.0
            RS_up = up_days.rolling(window).mean()
            RS_down = down_days.rolling(window).mean()
            out = (100-100/(1+RS_up/RS_down))
            return pd.Series(out, index=df.index, name = f'RSI_{window}')
        except Exception as e:
            logger.error(f"Error calculating RSI: {str(e)}")
            raise

    def stochastic(self, df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
        ''' MOMENTUM: Stochastic Oscillator. '''
        try:
            h14 = df.close.rolling(window=window).max()
            l14 = df.close.rolling(window=window).min()
            stoch = np.array((df.close - l14) / (h14 - l14))
            stoch = pd.Series(stoch, index=df.index, name = f'STOCH_{window}')
            stoch_k = stoch.rolling(window=3).mean()
            stoch_k.name = f'STOCHK_{window}'
            stoch_d = stoch_k.rolling(window=3).mean()
            stoch_d.name = f'STOCHD_{window}'
            return pd.concat([stoch_k, stoch_d], axis=1)
        except Exception as e:
            logger.error(f"Error calculating Stochastic Oscillator: {str(e)}")
            raise    

    def mom_indicators(self, df: pd.DataFrame, windows: Optional[List[int]] = None) -> pd.DataFrame:
        ''' Calculate Momentum Indicators. '''
        try:
            df = self._validate_dataframe(df)
            if windows is None:
                windows = self.windows
            out = pd.DataFrame()
            # Rate of Change
            for window in windows: out = pd.concat([out, self.roc(df, window)], axis=1)
            # Stochastic Oscillator
            for window in windows: out = pd.concat([out, self.stochastic(df, window)], axis=1)
            # Relative Strength Index
            for window in windows: out = pd.concat([out, self.rsi(df, window)], axis=1)
            
            # Cast as float 
            for col in out.columns: out[col] = out[col].astype(float)
            return out
        except Exception as e:
            logger.error(f"Error calculating Momentum Indicators: {str(e)}")
            raise


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from main import Manager, get_path 

    connections = get_path()
    m = Manager(connections)
    df = m.Pricedb.ohlc('spy', daily=True).resample('3D').last()

    mom = momentum()
    df = mom._validate_dataframe(df)
    mom_df = mom.mom_indicators(df)

    print(mom_df)