import json 
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import datetime as dt
import sqlite3 as sql

# Local imports
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[3]))
# from tweets.Options.flows.backtestingUtility import cp_backtesting_utility 
from backtestingUtility import cp_backtesting_utility
from bin.alerts.flows.tremp.opening_trends import OpeningTrends
from bin.alerts.flows.tremp.Sentiment import SentimetnalAnalysis
from bin.alerts.flows.tremp.market_analyzer_utility import MarketAnalyzerUtility

class cp_data_utility:
    def __init__(self, connections):
        self.verbose = False
        # Connections
        self.stock_dict = json.load(open(connections['ticker_path'], 'r'))
        self.stats_db = sql.connect(connections['stats_db'])
        self.vol_db = sql.connect(connections['vol_db'])
        self.daily_db = sql.connect(connections['daily_db'])
        self.connections = connections
        # Daily Option Stats
        self.cp_df = self._all_cp()

        # Stock Picks from our methods
        self.stock_pics = {}

        # Imports
        self.backtesting = cp_backtesting_utility(connections)
        self.initialize_df_dict(self.cp_df, self.stock_pics)
        

    def initialize_df_dict(self, cp_df, stock_pics):
        """ Initialize the Stock Picks Dictionary and the Daily Option Statistics Dataframe. 

        - Also initialize the Opening Trends, Sentiment Analysis and Market Analyzer classes.
        
        """
        self.cp_df = cp_df
        self.stock_pics = stock_pics
        connections = self.connections
        
        # Opening Trends
        self.opening_trends = OpeningTrends(
            cp_df = self.cp_df, 
            stock_pics = self.stock_pics, 
            connections = connections
        )

        # Sentiment
        self.sentiment = SentimetnalAnalysis(
            cp_df = self.cp_df, 
            stock_pics = self.stock_pics, 
            connections = connections
        )

        # Market Analyzer 
        self.market_analyzer = MarketAnalyzerUtility(
            cp_df = self.cp_df, 
            stock_pics = self.stock_pics, 
            connections = connections
        )

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

    def get_historical_cp(self, stock):
        """ Returns a DataFrame with historical options data for the specified stock.
        Columns include gatherdate, call_vol, put_vol, total_vol, call_oi, put_oi, total_oi, etc.
        """
        query = f"SELECT * FROM {stock}"
        out = pd.read_sql(query, self.vol_db, parse_dates=['gatherdate'])
        return out
    
    def _all_cp(self, thresh=20): 
        query = "SELECT * FROM daily_option_stats where date(gatherdate) = (select max(date(gatherdate)) from daily_option_stats)"
        out = pd.read_sql(query, self.stats_db, parse_dates=['gatherdate'])
        stocks = self.stock_dict['all_stocks']
        c = out['stock'].isin(stocks)
        return out[c]
    
    def _sorting_function(self, df, column, ascending=False, top_n=10):
        out = df.sort_values(column, ascending=ascending).head(top_n)[[column]].reset_index().values
        return [(x[0],x[1]) for x in out]
        
    def _volume(self, group='equities'):
        """ All Volume Columns """
        out = self.cp_df.copy()
        out = out[out['group'] == group].set_index('stock')
        dout = {
            'highest_call_volume': self._sorting_function(out, 'call_vol'),
            'largest_call_volume_increase': self._sorting_function(out, 'call_vol_chng'),
            'largest_call_volume_decrease': self._sorting_function(out, 'call_vol_chng', ascending=True),
            'highest_put_volume': self._sorting_function(out, 'put_vol'),
            'largest_put_volume_increase': self._sorting_function(out, 'put_vol_chng'),
            'largest_put_volume_decrease': self._sorting_function(out, 'put_vol_chng', ascending=True),
        }
        self.stock_pics.update(dout)

    def _oi(self, group='equities'):
        """ All Open Interest Columns """
        out = self.cp_df.copy()
        out = out[out['group'] == group].set_index('stock')
        dout = {
            'highest_call_oi': self._sorting_function(out, 'call_oi'),
            'largest_call_oi_increase': self._sorting_function(out, 'call_oi_chng'),
            'largest_call_oi_decrease': self._sorting_function(out, 'call_oi_chng', ascending=True),
            'highest_put_oi': self._sorting_function(out, 'put_oi'),
            'largest_put_oi_increase': self._sorting_function(out, 'put_oi_chng'),
            'largest_put_oi_decrease': self._sorting_function(out, 'put_oi_chng', ascending=True),
        }
        self.stock_pics.update(dout)
    

    def get_stocks(self, date = None, write = False):
        if date is not None:
            self.cp_df = self.backtesting.build_cp_table(date)
            self.stock_pics = {}
            self.initialize_df_dict(self.cp_df, self.stock_pics)

            self._volume()
            self._oi()

            x = self.opening_trends.get_opening_trends()
            self.stock_pics.update(y)
            
            y = self.sentiment._sentiment()
            self.stock_pics.update(x)
            
            z = self.market_analyzer.get_market_analysis()
            self.stock_pics.update(z)
        

        else:
            self._volume()
            self._oi()
            x = self.opening_trends.get_opening_trends()
            self.stock_pics.update(x)
            y = self.sentiment._sentiment()
            self.stock_pics.update(y)
            z = self.market_analyzer.get_market_analysis()
            self.stock_pics.update(z)
        
        if write:
            file = 'bin/alerts/flows/flow.pkl'
            pickle.dump(self.stock_pics, open(file, 'wb'))
            print(f"Stock Picks saved to {file}")
        
        return self.stock_pics
        



if __name__ == "__main__":
    from bin.main import get_path 
    connections = get_path()
    cp = cp_data_utility(connections)



    
    # k = cp.get_stocks( date = "2025-02-24")
    k = cp.get_stocks()
    for key, value in k.items():
        print(key)
        for item in value:
            print(item)
        print('')