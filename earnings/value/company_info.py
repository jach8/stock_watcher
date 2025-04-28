import json 
import numpy as np 
import pandas as pd 
import datetime as dt 
import yfinance as yf 
from tqdm import tqdm 
import sqlite3 as sql
from time import sleep 


stock_path = 'data/stocks/tickers.json'
stocks = json.load(open(stock_path, 'r'))['all_stocks']

# out_path = 'value/stock_info.json'
out_path = 'value/stock_info_all.json'

d = {}
for i in tqdm(stocks, desc = "Loading In Comapny Info"):
    try: 
        data = yf.Ticker(i).info
        data['date'] = dt.datetime.now().strftime('%Y-%m-%d')
        d[i] = data
    except:
        pass
    sleep(5)
    

with open(out_path, 'w') as f:
    json.dump(d, f)
    
print('done')