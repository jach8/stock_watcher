"""Multi-timeframe Moving Average Analysis

This module extends the base moving_avg class to handle combined analysis of
intraday and daily timeframes while preserving the original implementation's design.
"""

import pandas as pd
import numpy as np

class MultiTimeframeMA:
    """A class for analyzing moving averages across multiple timeframes.
    
    This class extends the base moving_avg class to handle both intraday and
    daily data simultaneously, preserving the original implementation's approach
    while adding capabilities for cross-timeframe analysis.
    """
    
    def __init__(self):
        """Initialize with parent class windows."""
        super().__init__()
    
    def combine_timeframes(self, min_df, daily_df):
        """Combine intraday and daily moving averages.
        
        An enhanced version of the original concatenate_min_daily function that
        preserves column names and properly aligns timeframes.
        
        Args:
            min_df (pandas.DataFrame): DataFrame with intraday data and MA columns
            daily_df (pandas.DataFrame): DataFrame with daily data and MA columns
            
        Returns:
            pandas.DataFrame: Combined DataFrame with both timeframes' MAs
        """
        min_df = min_df.copy()
        daily_df = daily_df.copy()
        
        # Add day column for merging
        min_df['day'] = min_df.index.date
        daily_df['day'] = daily_df.index.date
        
        # Prefix daily columns to distinguish them
        daily_cols = {col: f'daily_{col}' for col in daily_df.columns if col not in ['day']}
        daily_df.rename(columns=daily_cols, inplace=True)
        
        # Merge and clean up
        combined = pd.merge(
            min_df, daily_df,
            on='day',
            how='inner'
        ).set_index(min_df.index)
        
        return combined.drop(columns=['day'])
    
    def generate_signals(self, min_df, daily_df, params=None):
        """Generate trading signals using both timeframes.
        
        This method applies moving averages to both timeframes and generates
        signals based on MA crossovers and inter-timeframe relationships.
        
        Args:
            min_df (pandas.DataFrame): Intraday OHLCV data
            daily_df (pandas.DataFrame): Daily OHLCV data
            params (dict, optional): Parameters for signal generation:
                - min_ma: Type of MA for intraday ('sma', 'ema', etc.)
                - daily_ma: Type of MA for daily
                - threshold: Distance threshold for consolidation (default: 0.02)
                
        Returns:
            pandas.DataFrame: Combined DataFrame with signals:
                - All original MA columns
                - ma_cross_signal: Crossover signals
                - ma_distance: Distance between MAs
                - consolidation: Boolean consolidation indicator
        """
        params = params or {}
        min_ma = params.get('min_ma', 'ema')
        daily_ma = params.get('daily_ma', 'ema')
        threshold = params.get('threshold', 0.02)
        
        # Generate MA ribbons for both timeframes
        min_ribbon = self.ribbon(min_df, ma=min_ma)
        daily_ribbon = self.ribbon(daily_df, ma=daily_ma)
        
        # Combine timeframes
        combined = self.combine_timeframes(min_ribbon, daily_ribbon)
        
        # Calculate signals using shortest and longest MAs from each timeframe
        min_fast = f"{min_ma.upper()}{self.windows[0]}{self.derive_timeframe(min_df)}"
        min_slow = f"{min_ma.upper()}{self.windows[-1]}{self.derive_timeframe(min_df)}"
        daily_fast = f"daily_{daily_ma.upper()}{self.windows[0]}{self.derive_timeframe(daily_df)}"
        daily_slow = f"daily_{daily_ma.upper()}{self.windows[-1]}{self.derive_timeframe(daily_df)}"
        
        # Calculate MA distances
        combined['intraday_ma_distance'] = (
            (combined[min_fast] - combined[min_slow]) / combined[min_slow]
        )
        combined['daily_ma_distance'] = (
            (combined[daily_fast] - combined[daily_slow]) / combined[daily_slow]
        )
        
        # Generate crossover signals
        combined['intraday_signal'] = 0
        combined.loc[combined[min_fast] > combined[min_slow], 'intraday_signal'] = 1
        combined.loc[combined[min_fast] < combined[min_slow], 'intraday_signal'] = -1
        
        combined['daily_signal'] = 0
        combined.loc[combined[daily_fast] > combined[daily_slow], 'daily_signal'] = 1
        combined.loc[combined[daily_fast] < combined[daily_slow], 'daily_signal'] = -1
        
        # Identify consolidation periods
        combined['intraday_consolidation'] = abs(combined['intraday_ma_distance']) <= threshold
        combined['daily_consolidation'] = abs(combined['daily_ma_distance']) <= threshold
        
        return combined

if __name__ == "__main__":
    import sys
    sys.path.append('/Users/jerald/Documents/Dir/Python/Stocks')
    from main import Pipeline as Manager
    from bin.main import get_path
    
    # Initialize objects
    connections = get_path()
    m = Manager(connections)
    mtf = MultiTimeframeMA()
    
    # Get sample data
    min_df = m.Pricedb.ohlc('aapl', daily=False, start="2025-02-01")
    daily_df = m.Pricedb.ohlc('aapl', daily=True, start="2025-02-01")
    
    # Generate signals using both timeframes
    signals = mtf.generate_signals(
        min_df,
        daily_df,
        params={
            'min_ma': 'ema',
            'daily_ma': 'ema',
            'threshold': 0.02
        }
    )
    
    # Display results
    print("\nMulti-timeframe Analysis Results:")
    print("=================================")
    cols_to_show = ['close', 'intraday_signal', 'daily_signal', 
                    'intraday_consolidation', 'daily_consolidation']
    print(signals[cols_to_show].tail())