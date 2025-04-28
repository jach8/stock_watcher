import pandas as pd
import numpy as np
import sqlite3 as sql
import statsmodels.api as sm
from typing import Dict, List, Tuple, Optional, Union
import json

class OpeningTrends:
    """Utility class for analyzing options trading trends and patterns."""
    
    def __init__(self, 
                 connections: Dict[str, str],
                 cp_df: pd.DataFrame = None, 
                 stock_pics: Dict[str, List[Tuple[str, float]]] = {}
                 ) -> None:
        """
        Initialize the OpeningTrends analyzer.
        
        Args:
            connections (Dict[str, str]): Dictionary containing database and file paths
        """
        self.verbose = False
        self.stock_dict = json.load(open(connections['ticker_path'], 'r'))
        self.stats_db = sql.connect(connections['stats_db'])
        self.vol_db = sql.connect(connections['vol_db'])
        self.daily_db = sql.connect(connections['daily_db'])
        self.cp_df = cp_df if cp_df is not None else self._all_cp()
        self.stock_dict = json.load(open(connections['ticker_path'], 'r'))
        self.stock_pics = stock_pics if stock_pics else {}

    def get_historical_cp(self, stock: str) -> pd.DataFrame:
        """
        Fetch historical call/put data for a given stock.
        
        Args:
            stock (str): Stock symbol
            
        Returns:
            pd.DataFrame: DataFrame containing historical options data
        """
        max_date = self.cp_df.gatherdate.max()
        query = f"SELECT * FROM {stock} where date(gatherdate) <= '{max_date}'"
        return pd.read_sql(query, self.vol_db, parse_dates=['gatherdate'])

    def _all_cp(self, thresh: int = 20) -> pd.DataFrame:
        """
        Get the latest call/put data for all stocks.
        
        Args:
            thresh (int): Price threshold for filtering stocks
            
        Returns:
            pd.DataFrame: DataFrame containing latest options data
        """
        query = "SELECT * FROM daily_option_stats where date(gatherdate) = (select max(date(gatherdate)) from daily_option_stats)"
        out = pd.read_sql(query, self.stats_db, parse_dates=['gatherdate'])
        stocks = self.stock_dict['all_stocks']
        c = out['stock'].isin(stocks)
        return out[c]


    def estimate_open_close_transactions(self, stock: str, option_type: str = 'call') -> Optional[pd.DataFrame]:
        """
        Estimate opening and closing transactions for options.
        
        Args:
            stock (str): Stock symbol
            option_type (str): Type of option ('call' or 'put')
            
        Returns:
            Optional[pd.DataFrame]: DataFrame with transaction estimates or None
        """
        if option_type not in ['call', 'put']:
            raise ValueError("option_type must be 'call' or 'put'")

        df = self.get_historical_cp(stock)
        if df.empty:
            return None

        vol_col = 'call_vol' if option_type == 'call' else 'put_vol'
        oi_col = 'call_oi' if option_type == 'call' else 'put_oi'

        df = df.sort_values('gatherdate')
        df['delta_oi'] = df[oi_col].diff()
        df['N_open'] = (df[vol_col] + df['delta_oi']) / 2
        df['N_close'] = (df[vol_col] - df['delta_oi']) / 2

        return df[['gatherdate', vol_col, oi_col, 'delta_oi', 'N_open', 'N_close']]

    def _calculate_trend_slope(self, df: pd.DataFrame, value_col: str, date_col: str, 
                             lookback_days: int) -> Tuple[Optional[float], Optional[float]]:
        """
        Calculate normalized slope and p-value for trend analysis.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            value_col (str): Column name for values
            date_col (str): Column name for dates
            lookback_days (int): Number of days to look back
            
        Returns:
            Tuple[Optional[float], Optional[float]]: (normalized_slope, p_value) or (None, None)
        """
        if len(df) < lookback_days + 1:
            return None, None
            
        df = df.sort_values(date_col).tail(lookback_days + 1)
        x = sm.add_constant(np.arange(len(df)))
        y = df[value_col].values
        model = sm.OLS(y, x).fit()
        slope = model.params[1]
        p_value = model.pvalues[1]
        avg_value = y.mean()
        
        if avg_value == 0:
            return None, None
            
        normalized_slope = (slope / avg_value) * 100
        return normalized_slope, p_value

    def analyze_opening_trend(self, group: str = 'equities', lookback_days: int = 5, 
                            top_n: int = 10, p_value_threshold: float = 0.05) -> Dict[str, List[Tuple[str, float]]]:
        """
        Analyze opening trends for options and return results.
        
        Args:
            group (str): Stock group to analyze
            lookback_days (int): Days to look back for trend analysis
            top_n (int): Number of top stocks to return
            p_value_threshold (float): P-value threshold for statistical significance
            
        Returns:
            Dict[str, List[Tuple[str, float]]]: Dictionary with trend analysis results
        """
        out = self.cp_df.copy()
        out = out[out['group'] == group].set_index('stock')
        call_slopes = []
        put_slopes = []
        
        for stock in out.index:
            try:
                df_call = self.estimate_open_close_transactions(stock, 'call')
                if df_call is not None and len(df_call) >= lookback_days + 1:
                    slope_call, p_value_call = self._calculate_trend_slope(
                        df_call, 'N_open', 'gatherdate', lookback_days
                    )
                    if slope_call is not None and p_value_call < p_value_threshold:
                        call_slopes.append((stock, slope_call))
            except Exception as e:
                if self.verbose:
                    print(f"Error processing {stock} for call opening trend: {e}")
                continue

            try:
                df_put = self.estimate_open_close_transactions(stock, 'put')
                if df_put is not None and len(df_put) >= lookback_days + 1:
                    slope_put, p_value_put = self._calculate_trend_slope(
                        df_put, 'N_open', 'gatherdate', lookback_days
                    )
                    if slope_put is not None and p_value_put < p_value_threshold:
                        put_slopes.append((stock, slope_put))
            except Exception as e:
                if self.verbose:
                    print(f"Error processing {stock} for put opening trend: {e}")
                continue

        call_top = sorted(call_slopes, key=lambda x: x[1], reverse=True)[:top_n]
        put_top = sorted(put_slopes, key=lambda x: x[1], reverse=False)[:top_n]

        return {
            'highest_call_opening_trend': call_top,
            'highest_put_opening_trend': put_top
        }
    
    def get_opening_trends(self, group: str = 'equities', lookback_days: int = 5, 
                         top_n: int = 10) -> Dict[str, List[Tuple[str, float]]]:
        """
        Get opening trends for a specific stock group.
        
        Args:
            group (str): Stock group to analyze
            lookback_days (int): Days to look back for trend analysis
            top_n (int): Number of top stocks to return
            
        Returns:
            Dict[str, List[Tuple[str, float]]]: Dictionary with trend analysis results
        """
        results =  self.analyze_opening_trend(group, lookback_days, top_n)
        self.stock_pics.update(results)
        return self.stock_pics


if __name__ == '__main__':
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parents[3]))

    from bin.main import get_path
    connections = get_path()

    opening_trends = OpeningTrends(connections)
    print(opening_trends.analyze_opening_trend(group='equities', lookback_days=5, top_n=10))