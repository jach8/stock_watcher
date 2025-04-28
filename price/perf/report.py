import pandas as pd 
import numpy as np 
import sqlite3 as sql 
import datetime as dt 
import matplotlib.pyplot as plt 
import json 
import sys
sys.path.append('/Users/jerald/Documents/Dir/Python/Stocks')
from bin.price.db_connect import Prices

# class perf(performance):
class perf(Prices):
    def __init__(self, connections):
        try:
            super().__init__(connections)
            
        except Exception as e:
            print(f"Error: {e}")
            return None
    
    def get_stock_prices(self):
        """ Return Pandas DF of all stock prices """
        lot = []
        for i in range(len(self.stocks)):
            t = pd.read_sql(f''' select date(Date) as Date, Close as "{self.stocks[i].upper()}" from {self.stocks[i]} order by Date asc ''', self.daily_db, index_col='Date')
            t.index = pd.to_datetime(t.index)
            if len(t) > 30:
                lot.append(t)
        return pd.concat(lot, axis = 1)

    def get_returns(self):
        """ 
            Return A Data Frame: Quarterly Annual, and YTD returns for all stocks in the database.
        """
        sd = (dt.datetime.now().replace(day=1) - dt.timedelta(days=1)).replace(day=1).strftime('%Y-%m-%d')
        sp = self.get_stock_prices()
        self.sp = sp.copy()
        daily = sp.copy()
        weekly = daily.resample('W').last()
        monthly = daily.resample('M').last()
        quarterly = daily.resample('Q').last()
        annual = daily[dt.datetime.today() - dt.timedelta(days = 365):]
        ytd = daily[daily.index.year == dt.datetime.today().year]
        
        daily_returns = daily.ffill().pct_change().dropna().tail(1).T
        weekly_returns = ((1 + weekly.tail(3).ffill().pct_change().dropna()).cumprod() - 1).tail(1).T
        monthly_returns = ((1 + monthly.tail(2).ffill().pct_change().dropna()).cumprod() - 1).tail(1).T
        quarterly_returns = ((1 + quarterly.tail(2).ffill().pct_change().dropna()).cumprod() - 1).tail(1).T
        annual_returns = ((1 + annual.ffill().pct_change().dropna()).cumprod() - 1).tail(1).T
        ytd_returns = ((1 + ytd.ffill().pct_change().dropna()).cumprod() - 1).tail(1).T
        
        ret_ser = [daily_returns, weekly_returns, monthly_returns, quarterly_returns, annual_returns, ytd_returns]
        names = ['Daily', 'Weekly', 'Monthly', 'Quarterly', 'Annual', 'YTD']
        
        for j in range(len(ret_ser)):
            ret_ser[j].columns = [names[j]]
            
        return pd.concat(ret_ser, axis = 1)
    
    def winners(self,df, N = 10, asc = False):
        """
        return the top N winners: for Daily, Weekly, Monthly, Quarterly, Annually, and YTD
        
        """
        top_performance = {}
        for col in df.columns:
            tup = [(x, np.round(100 * df[col][x], 3)) for x in df[col].sort_values(ascending=asc).head(N).index]
            tup = [("$"+ x, str(y) + '%') for x, y in tup]
            top_performance[col] = tup
        return pd.DataFrame(top_performance)
        
    def losers(self,df, N = 10):
        return self.winners(df, N, True)
    
    def show_performance(self, N = 10):
        r = self.get_returns()
        print("---" * 19,"Winners", "---" * 19)
        print(self.winners(r, N = N))
        print("---" * 19,"Losers", "---" * 19)
        print(self.losers(r, N = N))
    
    
    
if __name__ == "__main__":
    print("\n(22) O Arjuna, those pleasures arising from the senses contacting sense objects are indeed the source of misery only; subject to a beginning and end; therefore the spiritually intelligent never take delight in them.\n")
    connections = {
        'daily_db':'data/prices/stocks.db',
        'intraday_db':'data/prices/stocks_intraday.db',
        'ticker_path':'data/stocks/tickers.json'
    }
    
    p = perf(connections)
    p.show_performance(N = 30)