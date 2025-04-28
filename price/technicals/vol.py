"""
Volatility Signals.

This module implements technical analysis tools for calculating and analyzing
the volatility of a financial instrument over time. Volatility is a measure of
the dispersion of returns for a given security or market index. It is often
used to assess the risk of an investment and to determine the optimal position
size for a given trade.

Key Features:
- Historical volatility calculation
- Estimated volatility calculation
- Bollinger Bands 
- Average True Range (ATR)
- Keltner Channels
- Automatic timeframe detection

Example:
    To calculate the historical volatility of a stock over a 30-day period:

        >>> from technicals.volatility import Volatility
        >>> vol = Volatility()
        >>> vol.historical_volatility('AAPL', 30)

    This will return the historical volatility of Apple stock over the past 30
    trading days.
"""

import pandas as pd 
import numpy as np 
import sqlite3 as sql 
from logging import getLogger
from typing import Union, Optional, Dict, List
from .utils import combine_timeframes, derive_timeframe
# from utils import combine_timeframes, derive_timeframe

logger = getLogger(__name__)


class volatility:

    def __init__(self) -> None:
        """
        Initialize the Volatility class
        
        *Needs OHLC data to estimate volatility ***
        
        
        """
        # self.windows = np.array([6, 10, 20, 28, 50, 96, 108, 200, 496])
        self.windows = np.array([6, 10, 20, 28])

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
    
    def historical_volatility(self, df: pd.DataFrame, window: int = 30) -> pd.Series:
        """
        Calculate historical volatility for a given window size.
        
        Historical volatility is a measure of the dispersion of returns for a given
        security or market index over a specified period. It is often used to assess
        the risk of an investment and to determine the optimal position size for a
        given trade.    

        Args:
            df (pd.DataFrame): DataFrame with OHLC data
            window (int): Number of trading days to use for calculation
        
        Returns:
            pd.Series: Historical volatility values
        """
        returns = df['close'].pct_change().dropna()
        vol = returns.rolling(window=window).std() * np.sqrt(window)
        tf = derive_timeframe(df)
        vol.name = f'HV_{window}{tf}'
        return vol

    def est_vol(self, df: pd.DataFrame, lookback: int = 10) -> pd.Series:
        """ 
        This is the Yang-Zheng (2005) Estimator for Volatility; 
            Yang Zhang is a historical volatility estimator that handles 
                1. opening jumps
                2. the drift and has a minimum estimation error 
        """
        try:
            o = df.open.values
            h = df.high.values
            l = df.low.values
            c = df.close.values
            k = 0.34 / (1.34 + (lookback+1)/(lookback-1))
            cc = np.log(c/c.shift(1))
            ho = np.log(h/o)
            lo = np.log(l/o)
            co = np.log(c/o)
            oc = np.log(o/c.shift(1))
            oc_sq = oc**2
            cc_sq = cc**2
            rs = ho*(ho-co)+lo*(lo-co)
            close_vol = cc_sq.rolling(window=lookback).sum() * (1.0 / (lookback - 1.0))
            open_vol = oc_sq.rolling(window=lookback).sum() * (1.0 / (lookback - 1.0))
            window_rs = rs.rolling(window=lookback).sum() * (1.0 / (lookback - 1.0))
            result = (open_vol + k * close_vol + (1-k) * window_rs).apply(np.sqrt) * np.sqrt(252)
            result[:lookback-1] = np.nan
            tf = derive_timeframe(df)
            out = pd.Series(result, index = df.index, name = f'EVT_{lookback}{tf}')
            return out
        except Exception as e:
            logger.error(f"Error calculating volatility: {str(e)}")
            raise

    def BB(self,df:pd.DataFrame, window: int = 20, m: float = 2) -> pd.Series:
        ''' Bollinger Bands. '''
        try:
            sma = df.close.rolling(window=window).mean()
            sigma = df.close.rolling(window=window).std()
            out = (df.close - sma) / (m * sigma)
            tf = derive_timeframe(df)
            out.name = f'BB_{window}{tf}'
            return out
        except Exception as e:
            logger.error(f"Error calculating Bollinger Bands: {str(e)}")
            raise
    

    def KC(self, df:pd.DataFrame,  window: int = 20, m: float = 2) -> pd.Series:
        ''' Keltner Channels. Indicator '''
        try:
            close = df.close.values
            ema = df.close.ewm(span=window).mean().values
            atr = self.ATR(df, window).values
            kc = (close - ema ) / (m * atr)
            tf = derive_timeframe(df)
            out = pd.Series(kc, index = df.index, name = f'KC_{window}{tf}')
            return out
        except Exception as e:
            logger.error(f"Error calculating Keltner Channels: {str(e)}")
            raise
    

    def ATR(self, df:pd.DataFrame, window: int) -> pd.Series:
        ''' Average True Range.: Need three columns, High, Low, and Close. '''
        try:
            hi = df.high.values
            lo = df.low.values
            c = df.close.values
            tr = np.vstack([np.abs(hi[1:]-c[:-1]),np.abs(lo[1:]-c[:-1]),(hi-lo)[1:]]).max(axis=0)
            tr = np.concatenate([[np.nan], tr])
            tf = derive_timeframe(df)
            return pd.Series(tr, index = df.index, name = f'ATR_{window}{tf}').ewm(alpha = 1/window, min_periods = window).mean()
        except Exception as e:
            logger.error(f"Error calculating ATR: {str(e)}")
            raise

    def ADX(self, df:pd.DataFrame, window: int) -> pd.DataFrame:
        ''' Average Directional Index. '''
        def EMA(x, window):
            return pd.Series(x).ewm(alpha=1/window).mean()
        try:
            hi = df.high.rolling(window=window).min().values 
            lo = df.low.rolling(window=window).max().values
            c = df.close.values 
            up = hi[1:] - hi[:-1]
            down = lo[:-1] - lo[1:]
            up_ind = up > down
            down_ind = down > up
            dmup = np.zeros(len(up))
            dmdown = np.zeros(len(down))
            dmup = np.where(up_ind, up, 0)
            dmdown = np.where(down_ind, down, 0)
            atr = self.ATR(df, window).values[1:]
            diplus = 100 * EMA(dmup, window) / atr
            diminus = 100 * EMA(dmdown, window) / atr
            diminus[(diplus + diminus) == 0] = 1e-5
            dx = 100 * np.abs(diplus - diminus) / (diplus + diminus)
            out = dx.ewm(alpha = 1/window).mean()
            tf = derive_timeframe(df)
            out.name = f'ADX_{window}{tf}'
            out.index = df.index[1:]
            return out
        except Exception as e:
            logger.error(f"Error calculating ADX: {str(e)}")
            raise
    
    
    def vol_indicators(self, df: pd.DataFrame, windows: List[int] = None) -> pd.DataFrame:
        ''' Calculate multiple volatility indicators. '''
        if windows is None:
            windows = self.windows
        try:
            tf = derive_timeframe(df)
            out = pd.DataFrame()
            for window in windows: out[f'ATR_{window}{tf}'] = self.ATR(df, window)
            for window in windows: out[f'BB_{window}{tf}'] = self.BB(df, window)
            for window in windows: out[f'KC_{window}{tf}'] = self.KC(df, window)
            for window in windows: out[f'ADX_{window}{tf}'] = self.ADX(df, window)
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

    v = volatility()
    df = v._validate_dataframe(df) 
    vol_df = v.vol_indicators(df)
    print(vol_df)


