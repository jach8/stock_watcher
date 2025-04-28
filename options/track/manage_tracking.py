"""
Manager for the options data pipeline. 
    1. Get new option chain data
    2. Append data to the option database. 
    3. Update the vol.db after calculating the change variables. 

"""
import sys
sys.path.append('/Users/jerald/Documents/Dir/Python/Stocks')
import pandas as pd 
import numpy as np 
import yfinance as yf 
import datetime as dt 
from tqdm import tqdm
import time 

from bin.options.optgd.db_connect import Connector
from bin.options.track.scanner import Scanner


class Screener(Scanner):
    def __init__(self, connections):
        super().__init__(connections)
        
    def run_screener(self, stock, new_chain):
        try:
            # self.scan(stock)
            self.track(stock)
        except Exception as e:
            pass
        
        
    
if __name__ == "__main__":
    print("\n(47) Your right is to work only and never to the fruit thereof. Do not consider yourself to be the cause of the fruit of action; nor let your attachment to be to inaction.\n")
        
    connections = {
            'backup_db': 'bin/pipe/log/backup.db',
            'tracking_values_db': 'bin/pipe/test_data/tracking_values.db',
            'tracking_db': 'bin/pipe/test_data/tracking.db',
            'stats_db': 'bin/pipe/test_data/stats.db',
            'vol_db': 'bin/pipe/test_data/vol.db',
            'change_db': 'bin//pipe/test_data/option_change.db', 
            'option_db': 'bin/pipe/test_data/test.db', 
            'testing_option_db': 'bin/pipe/test_data/test.db',
            'options_stat': 'bin/pipe/test_data/options_stat.db',
            'ticker_path': 'data/stocks/tickers.json'
            }

    
    s = Stats(connections)
    s._scan('qqq')