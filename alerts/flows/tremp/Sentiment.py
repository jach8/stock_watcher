import sys
from tqdm import tqdm
import numpy as np
import pandas as pd
import sqlite3 as sql
import datetime as dt
from itertools import chain 
from statsmodels.tsa.seasonal import seasonal_decompose
import json 
from typing import Dict, List, Tuple, Optional, Union
from tweets.Options.flows.backtestingUtility import cp_backtesting_utility 
from bin.alerts.flows.tremp.opening_trends import OpeningTrends

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[3]))

class SentimetnalAnalysis:
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

        
    def _price_filter(self, thresh=20):
        df = self.Pricedb.all_stock_Close().tail(1)
        keep_stock = []
        for i in df.columns.to_list():
            if df[i].values[0] > thresh:
                keep_stock.append(i)
        return keep_stock
    
    def get_daily_ohlcv(self, stock):
        """ Returns a dataframe with columns Date, Close, High, Low, Open, and Volume"""
        query = f"SELECT * FROM {stock}"
        out = pd.read_sql(query, self.daily_db, parse_dates=['Date'])
        return out

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
    
    def _all_cp(self, thresh=20): 
        query = "SELECT * FROM daily_option_stats where date(gatherdate) = (select max(date(gatherdate)) from daily_option_stats)"
        out = pd.read_sql(query, self.stats_db, parse_dates=['gatherdate'])
        stocks = self.stock_dict['all_stocks']
        c = out['stock'].isin(stocks)
        return out[c]
    
    def _calculate_trend(self, df, value_col, date_col, lookback_days):
        """ Calculate the trend strength for a given metric over a lookback period.
        
        Args:
            df (pd.DataFrame): DataFrame containing the data.
            value_col (str): Column name of the metric to calculate the trend for.
            date_col (str): Column name of the date column.
            lookback_days (int): Number of days to look back for the trend calculation.
        
        Returns:
            float: Trend strength as a percentage change (positive for upward trend, negative for downward).
            Returns None if insufficient data.
        """
        if len(df) < lookback_days + 1:
            return None
        
        df = df.sort_values(date_col)
        recent_value = df[value_col].iloc[-1]
        past_value = df[value_col].iloc[-lookback_days - 1]
        
        if past_value == 0:  # Avoid division by zero
            return None
        
        # Calculate percentage change as the trend strength
        trend_strength = ((recent_value - past_value) / past_value) * 100
        return trend_strength
    
    def __decompose_timeseries(self, df, value_col, date_col):
        """ Decompose a time series into trend, seasonal, and residual components.
        
        Args:
            df (pd.DataFrame): DataFrame containing the data.
            value_col (str): Column name of the metric to decompose.
            date_col (str): Column name of the date column.
        
        Returns:
            tuple: Decomposed components (trend, seasonal, residual).
        """
        df = df.set_index(date_col)
        decomposition = seasonal_decompose(df[value_col], model='additive')
        return decomposition.trend, decomposition.seasonal, decomposition.resid

    def _calculate_put_call_ratio(self, df, use_volume=True):
        """Calculate the put-to-call ratio.

        Args:
            df (pd.DataFrame): Options data with put/call volume or OI.
            use_volume (bool): Use volume if True, OI if False.

        Returns:
            float: Put-to-call ratio or NaN if invalid.
        """
        if use_volume:
            put = df['put_vol'].sum()
            call = df['call_vol'].sum()
        else:
            put = df['put_oi'].sum()
            call = df['call_oi'].sum()
        return put / call if call > 0 else np.nan

    def _sentiment(self, group='equities', lookback_days=5, top_n=10, trend_threshold=5.0, pcr_threshold=1.0):
        """Analyze market sentiment using price, volume, OI, and put-to-call ratio.

        Args:
            group (str): Filter group (e.g., 'equities').
            lookback_days (int): Days to analyze trends.
            top_n (int): Number of stocks per sentiment category.
            trend_threshold (float): Min % change for a significant trend.
            pcr_threshold (float): Put-to-call ratio threshold for overbought/oversold.

        Returns:
            dict: Sentiment categories with stock details.
        """
        out = self.cp_df.copy()
        out = out[out['group'] == group].set_index('stock')
        sentiment_results = []

        for stock in out.index:
            try:
                # Price trend
                price_data = self.get_daily_ohlcv(stock).sort_values('Date')
                price_trend = self._calculate_trend(price_data, 'Close', 'Date', lookback_days)
                if price_trend is None:
                    continue

                # Options data trends
                options_data = self.get_historical_cp(stock).sort_values('gatherdate')
                vol_trend = self._calculate_trend(options_data, 'total_vol', 'gatherdate', lookback_days)
                oi_trend = self._calculate_trend(options_data, 'total_oi', 'gatherdate', lookback_days)
                if vol_trend is None or oi_trend is None:
                    continue

                # Put-to-call ratio
                latest_options = options_data.tail(1)
                pcr = self._calculate_put_call_ratio(latest_options)

                # Sentiment logic
                price_rising = price_trend > trend_threshold
                price_falling = price_trend < -trend_threshold
                vol_rising = vol_trend > trend_threshold
                vol_falling = vol_trend < -trend_threshold
                oi_rising = oi_trend > trend_threshold
                oi_falling = oi_trend < -trend_threshold

                if price_rising and vol_rising and oi_rising:
                    sentiment = 'strong_bullish'
                    if pcr < pcr_threshold:
                        sentiment += '_confirmed'
                elif price_rising and (vol_falling or oi_falling):
                    sentiment = 'weak_bullish'
                    if vol_falling:
                        sentiment += '_divergence'
                    if pcr > pcr_threshold:
                        sentiment += '_overbought'
                elif price_falling and vol_rising and oi_rising:
                    sentiment = 'strong_bearish'
                    if pcr > pcr_threshold:
                        sentiment += '_confirmed'
                elif price_falling and (vol_falling or oi_falling):
                    sentiment = 'weak_bearish'
                    if vol_falling:
                        sentiment += '_divergence'
                    if pcr < pcr_threshold:
                        sentiment += '_oversold'
                else:
                    sentiment = 'neutral'

                # Composite score for ranking
                composite_score = np.round((price_trend + vol_trend + oi_trend) / 3, 3)
                price_trend = np.round(price_trend, 3)
                vol_trend = np.round(vol_trend, 3)
                oi_trend = np.round(oi_trend, 3)
                pcr = np.round(pcr, 3)
                sentiment_results.append((stock, sentiment, composite_score, price_trend, vol_trend, oi_trend, pcr))

            except Exception as e:
                print(f"Error processing {stock}: {e}")
                continue

        # Convert to DataFrame and sort
        sentiment_df = pd.DataFrame(sentiment_results, columns=['stock', 'sentiment', 'composite_score', 'price_trend', 'vol_trend', 'oi_trend', 'pcr'])
        dout = {
            'strong_bullish': sentiment_df[sentiment_df['sentiment'].str.contains('strong_bullish')]
                .sort_values('composite_score', ascending=False).head(top_n).to_dict('records'),
            'weak_bullish': sentiment_df[sentiment_df['sentiment'].str.contains('weak_bullish')]
                .sort_values('composite_score', ascending=False).head(top_n).to_dict('records'),
            'strong_bearish': sentiment_df[sentiment_df['sentiment'].str.contains('strong_bearish')]
                .sort_values('composite_score', ascending=True).head(top_n).to_dict('records'),
            'weak_bearish': sentiment_df[sentiment_df['sentiment'].str.contains('weak_bearish')]
                .sort_values('composite_score', ascending=True).head(top_n).to_dict('records')
        }
        self.stock_pics.update(dout)
        return dout

    def get_stocks(self):
        # self._volume()
        # self._oi()
        self._sentiment()  # Add sentiment analysis
        return self.stock_pics


if __name__ == "__main__":
    from bin.main import get_path 
    connections = get_path()
    k = cp.get_stocks()
    for key, value in k.items():
        print(key)
        for item in value:
            print(item)
        print('')