import pickle
import pandas as pd 
import sqlite3 
import os 
import json 
import datetime as dt 
import numpy as np 
from bin.utils.add_stocks import add_stock
from bin.utils.delete_stocks import delete_stock

def Initialize():
    """ Create the database of connections, stock names and initialize the program. """
    pre = ''
    connections = {
                ##### Bonds Data ###########################
                'bonds_db': f'{pre}data/bonds/bonds.db', 
                ##### Price Data ###########################
                'daily_db': f'{pre}data/prices/stocks.db', 
                'intraday_db': f'{pre}data/prices/stocks_intraday.db',
                'ticker_path': f'{pre}data/stocks/tickers.json',
                ##### Options Data ###########################
                'inactive_db': f'{pre}data/options/log/inactive.db',
                'backup_db': f'{pre}data/options/log/backup.db',
                'tracking_values_db': f'{pre}data/options/tracking_values.db',
                'tracking_db': f'{pre}data/options/tracking.db',
                'stats_db': f'{pre}data/options/stats.db',
                'vol_db': f'{pre}data/options/vol.db',
                'change_db': f'{pre}data/options/option_change.db', 
                'option_db': f'{pre}data/options/options.db', 
                ##### Earnings + Company Info ###########################
                'earnings_dict': f'{pre}data/earnings/earnings.pkl',
                'stock_names' : f'{pre}data/stocks/stock_names.db',
                'stock_info_dict': f'{pre}data/stocks/stock_info.json',
                'earnings_calendar': f'{pre}data/earnings/earnings_dates_alpha.csv',

        }

    for key, value in connections.items():
        if '.' in value: 
            # This is a file. If it is a db, create the database, else pass
            # First check if the folder exists, if not create it
            folder = value.split('/')[:-1]
            folder = '/'.join(folder)
            if not os.path.exists(folder):
                os.makedirs(folder)
            if 'db' in value: 
                # Connect 
                conn = sqlite3.connect(value)
                conn.close()
            if 'json' in value:
                with open(value, 'w') as f:
                    if 'ticker' in value:
                        json.dump({'all_stocks': []}, f)
                    else:
                        json.dump({}, f)
            if 'pkl' in value:
                pickle.dump({}, open(value, 'wb'))
                
        else:
            # This is a folder
            if not os.path.exists(value):
                os.makedirs(value)
                
                
    # Add the stocks to the database
    print("Initialized Files, Would you like to add stocks to the database? (y/n)")
    add = input()    
    if add == 'y':
        conn = sqlite3.connect(connections['stock_names'])
        stock = input('Enter the stock you would like to add, if adding more than one stock seperate them with a comma: ')
        if ',' in stock:
            stock = stock.split(',')
            for s in stock:
                add_stock(conn=conn, path = connections['ticker_path'], stock=s)
        else:
            add_stock(conn=conn, path = connections['ticker_path'], stock=stock)
    else:
        print('Not adding stocks')
    