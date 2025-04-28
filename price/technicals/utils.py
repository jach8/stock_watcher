import pandas as pd
import datetime as dt
import sqlite3 as sql
import json
import os
import pickle
import logging
from typing import Optional, List, Dict, Union


def combine_timeframes(min_df, daily_df):
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
    # daily_cols = {col: f'daily_{col}' for col in daily_df.columns 
    #                 if col not in ['day'] and str(col).lower() in ['open', 'high', 'low', 'close']}
    daily_cols = {}
    for col in daily_df.columns:
        if col not in ['day']:
            if str(col).lower() in ['date','open', 'high', 'low', 'close','volume']:
                daily_cols[col] = f'daily_{col.lower()}'
            
    daily_df.rename(columns=daily_cols, inplace=True)
    
    # Merge and clean up
    combined = pd.merge(
        min_df, daily_df,
        on='day',
        how='inner'
    )
    if 'daily_Date' in combined.columns:
        combined.drop(columns=['daily_Date'], inplace=True)
    
    if 'Date' in combined.columns:
        combined.drop(columns=['Date'], inplace=True)

    if combined.shape[0] == min_df.shape[0]:
        return combined.drop(columns=['day']).set_index(min_df.index)
    
    else:
        return combined.drop(columns=['day'])


def get_time_difference(df: pd.DataFrame) -> str:
    """Determine time difference between consecutive rows.

    Args:
        df: DataFrame with datetime index

    Returns:
        Single character timeframe indicator:
            'T': Minutes
            'H': Hours
            'D': Days
            'W': Weeks
            'M': Months

    Raises:
        TimeframeError: If time difference cannot be determined
    """
    try:
        diff = df.index[-1] - df.index[-2]
        if diff.days >= 28:
            return 'M'
        elif diff.days >= 7:
            return 'W'
        elif diff.days >= 1:
            return 'D'
        elif diff.seconds >= 3600:
            return 'H'
        else:
            return 'T'
    except Exception as e:
        raise TimeframeError(f"Failed to determine timeframe: {str(e)}")

def derive_timeframe(df: pd.DataFrame) -> str:
    """Get DataFrame frequency indicator.

    Args:
        df: DataFrame with datetime index

    Returns:
        Single character timeframe indicator

    Raises:
        TimeframeError: If frequency cannot be determined
    """
    try:
        freq = pd.infer_freq(df.index)
        if freq is None:
            freq = get_time_difference(df)
        return freq
    except Exception as e:
        raise TimeframeError(f"Failed to derive timeframe: {str(e)}")
    