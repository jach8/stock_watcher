import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[3]))

from typing import Dict, List, Tuple, Optional, Union
import pandas as pd
import sqlite3 as sql
from tweets.Options.flows.drill.flow import MarketAnalyzer
import json

class MarketAnalyzerUtility:
    """Utility class for market analysis and sentiment calculations."""
    
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

    def get_daily_ohlcv(self, stock: str) -> pd.DataFrame:
        """
        Fetch daily OHLCV data for a given stock.
        
        Args:
            stock (str): Stock symbol
            
        Returns:
            pd.DataFrame: DataFrame containing OHLCV data
        """
        max_date = self.cp_df.gatherdate.max()
        query = f"SELECT * FROM {stock} where date(Date) <= date('{max_date}')"
        return pd.read_sql(query, self.daily_db, parse_dates=['Date'])

    def get_historical_cp(self, stock: str) -> pd.DataFrame:
        """
        Fetch historical call/put data for a given stock.
        
        Args:
            stock (str): Stock symbol
            
        Returns:
            pd.DataFrame: DataFrame containing historical options data
        """
        max_date = self.cp_df.gatherdate.max()
        query = f"SELECT * FROM {stock} where date(gatherdate) <= date('{max_date}')"
        return pd.read_sql(query, self.vol_db, parse_dates=['gatherdate'])

    def prepare_market_data(self, stocks: List[str]) -> Dict[str, pd.DataFrame]:
        """
        Prepare data for MarketAnalyzer by fetching and merging OHLCV and options data.
        
        Args:
            stocks (list): List of stock symbols
            
        Returns:
            Dict[str, pd.DataFrame]: Dictionary with stock symbols as keys and formatted DataFrames as values
        """
        data = {}
        for stock in stocks:
            try:
                # Fetch OHLCV data
                df_ohlcv = self.get_daily_ohlcv(stock)
                df_ohlcv['day'] = df_ohlcv['Date'].dt.date
                # Fetch options data
                df_options = self.get_historical_cp(stock)
                df_options['day'] = df_options['gatherdate'].dt.date

                # Merge on date columns
                df_merged = pd.merge(
                    df_ohlcv,
                    df_options,
                    left_on='day',
                    right_on='day',
                    # left_on='Date',
                    # right_on='gatherdate',
                    how='inner'
                )

                # Select and rename required columns
                df_prepared = df_merged[['Date', 'Close', 'total_vol', 'total_oi']].rename(
                    columns={
                        'Date': 'date',
                        'Close': 'price',
                        'total_vol': 'volume',
                        'total_oi': 'openinterest'
                    }
                )

                data[stock] = df_prepared
            except Exception as e:
                if self.verbose:
                    print(f"Error preparing data for {stock}: {e}")
        return data

    def analyze_market_sentiment(self, group: str = 'equities', lookback_days: int = 30, 
                               top_n: int = 10) -> Dict[str, List[Tuple[str, Union[str, float, bool]]]]:
        """
        Perform comprehensive market sentiment analysis.
        
        Args:
            group (str): Stock group name (e.g., 'equities')
            lookback_days (int): Number of days to analyze
            top_n (int): Number of top stocks to include for intensity metric
            
        Returns:
            Dict[str, List[Tuple[str, Union[str, float, bool]]]]: Dictionary containing analysis results
        """
        stocks = self.stock_dict.get(group, [])
        data = self.prepare_market_data(stocks)
        analyzer = MarketAnalyzer(data)
        results = analyzer.ams(stocks, lookback_days=lookback_days)
        
        # Extract stocks with 'strong' sentiment
        strong_sentiment = [(stock, result['sentiment']) 
                           for stock, result in results.items() 
                           if result['sentiment'] == 'strong']
        
        # Extract top N stocks by intensity
        intensity_scores = [(stock, result['intensity']) 
                           for stock, result in results.items()]
        high_intensity = sorted(intensity_scores, key=lambda x: x[1], reverse=True)[:top_n]

        # Extract various market patterns and indicators
        analysis_results = {
            'strong_sentiment': strong_sentiment,
            'high_intensity': high_intensity,
            'divergence': [(stock, result['divergence']) 
                          for stock, result in results.items() 
                          if result['divergence'] is not None and result['divergence']][:10],
            'reversal': [(stock, result['reversal']) 
                        for stock, result in results.items() 
                        if result['reversal'] is not None and result['reversal']][:10],
            'pressure': [(stock, result['pressure']) 
                        for stock, result in results.items() 
                        if result['pressure']][:10],
            'new_money_flow': [(stock, result['new_money_flow']) 
                              for stock, result in results.items() 
                              if result['new_money_flow'] is not None and result['new_money_flow']][:10],
            'short_covering': [(stock, result['short_covering']) 
                             for stock, result in results.items() 
                             if result['short_covering'] is not None and result['short_covering']][:10],
            'liquidation': [(stock, result['liquidation']) 
                           for stock, result in results.items() 
                           if result['liquidation'] is not None and result['liquidation']][:10],
            'high_oi': [(stock, result['high_oi']) 
                       for stock, result in results.items() 
                       if result['high_oi'] is not None and result['high_oi']][:10],
            'oi_increase': [(stock, result['oi_increase']) 
                           for stock, result in results.items() 
                           if result['oi_increase'] is not None and result['oi_increase']][:10],
            # 'peak': [(stock, result['peaks']) 
            #         for stock, result in results.items() 
            #         if result['peaks'] is not None]
        }
        
        self.stock_pics.update(analysis_results)
        return analysis_results

    def get_market_analysis(self, lookback_days: int = 20, top_n: int = 10) -> Dict[str, List[Tuple[str, Union[str, float, bool]]]]:
        """
        Get comprehensive market analysis including sentiment metrics.
        
        Args:
            lookback_days (int): Days to look back for analysis
            top_n (int): Number of top stocks to include
            
        Returns:
            Dict[str, List[Tuple[str, Union[str, float, bool]]]]: Dictionary containing all analysis results
        """
        self.analyze_market_sentiment(lookback_days=lookback_days, top_n=top_n)
        return self.stock_pics
    
if __name__ == "__main__":
    # Example usage
    from bin.main import get_path 

    connections = get_path()
    analyzer_utility = MarketAnalyzerUtility(connections)
    # results = analyzer_utility.get_market_analysis(lookback_days=30, top_n=5)
    # for key, value in results.items():
    #     print(f"\n{key}:\n \t\t{value}")

    d = analyzer_utility.prepare_market_data(['amzn', 'tsla'])
    print(d)
    print(d['amzn'].sort_values('date', ascending=True))


