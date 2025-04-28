"""
Other Indicators: 

1. Recent Highs and Lows 
2. All time Highs and Lows 
3. Mean Reversion 

"""


from typing import Union, Optional, Dict, List
import pandas as pd 
import numpy as np 
import sqlite3 as sql 
from logging import getLogger
from .utils import combine_timeframes, derive_timeframe
# from utils import combine_timeframes, derive_timeframe

logger = getLogger(__name__)


class descriptive_indicators:

    def __init__(self) -> None:
        """
        Initialize the Volatility class
        
        *Needs OHLC data to estimate volatility ***
        
        
        """
        # self.windows = np.array([6, 10, 20, 28, 50, 96, 108, 200, 496])
        self.windows = np.array([10, 20])

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
    
    def all_time_highs(self, df: pd.DataFrame, window: int = 30) -> pd.Series:
        """
        Calculate all time highs for a given window size.
        
        Historical volatility is a measure of the dispersion of returns for a given
        size for a given trade.
        
        """
        out = df['high'].expanding().max()
        out.name = f'ATH'
        return out

    def all_time_lows(self, df: pd.DataFrame, window: int = 30) -> pd.Series:
        """
        Calculate all time lows for a given window size.
        
        Historical volatility is a measure of the dispersion of returns for a given
        size for a given trade.
        
        """
        out = df['low'].expanding().min()
        out.name = f'ATL'
        return out
    
    def recent_highs(self, df: pd.DataFrame, window: int = 30) -> pd.Series:
        """
        Calculate recent highs for a given window size.
        
        Historical volatility is a measure of the dispersion of returns for a given
        size for a given trade.
        
        """
        out = df['high'].rolling(window, min_periods=1).max()
        out.name = f'RH_{window}'
        return out
    
    def recent_lows(self, df: pd.DataFrame, window: int = 30) -> pd.Series:
        """
        Calculate recent lows for a given window size.
        
        Historical volatility is a measure of the dispersion of returns for a given
        size for a given trade.
        
        """
        out = df['low'].rolling(window, min_periods=1).min()
        out.name = f'RL_{window}'
        return out
    

    def mean_reversion(self, df: pd.DataFrame, window: int = 30) -> pd.Series:
        """
        Calculate mean reversion for a given window size.
        
        Historical volatility is a measure of the dispersion of returns for a given
        size for a given trade.

        Values over 1.5 are considered overbought, values under -1.5 are considered oversold.
        
        """
        z = (df['close'] - df['close'].rolling(window).mean()) / df['close'].rolling(window).std()
        z.name = f'MR_{window}'
        return z

    
    def descriptive_indicators(self, df: pd.DataFrame, windows: List[int] = None) -> pd.DataFrame:
        ''' Calculate multiple volatility indicators. '''
        if windows is None:
            windows = self.windows
        try:
            df = self._validate_dataframe(df)
            tf = derive_timeframe(df)
            out = pd.DataFrame()
            out['ATH'] = self.all_time_highs(df)
            out['ATL'] = self.all_time_lows(df)
            for window in windows: out[f'highs_{window}{tf}'] = self.recent_highs(df, window)
            for window in windows: out[f'lows_{window}{tf}'] = self.recent_lows(df, window)
            for window in windows: out[f'reversion_{window}{tf}'] = self.mean_reversion(df, window)
            return out
        except Exception as e:
            logger.error(f"Error calculating volatility indicators: {str(e)}")
            raise
    



if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from main import Manager, get_path 

    connections = get_path()
    m = Manager(connections)
    df = m.Pricedb.ohlc('spy', daily=False).resample('3T').last()

    v = descriptive_indicators()
    df = v._validate_dataframe(df)
    vol_df = v.descriptive_indicators(df)
    print(vol_df)


