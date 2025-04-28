import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import datetime as dt 
from tqdm import tqdm 
import scipy.stats as st 
import time
import json 
import sys
sys.path.append('/Users/jerald/Documents/Dir/Python/Stocks')
from bin.options.optgd.db_connect import Connector as Manager


class Iterator(Manager):
    def __init__(self, connections):
        super().__init__(connections)
        self.stock_dict = self.stocks.copy()
        
    def get_stocks(self, group = 'all_stocks'):
        return self.stock_dict[group]
        
    def _iterate_function(self, func, group = 'all_stocks'):
        stocks = self.get_stocks(group)
        pbar = tqdm(stocks, desc = 'Iterating')
        out = [func(x) for x in pbar]
        return out
    
    def dataframe_iterator_function(self, func, group = 'all_stocks'):
        lodf = self._iterate_function(func, group = group)
        return pd.concat(lodf)
    
    def query_iteroator(self, query, connection, group = 'etf'):
        """ query must be a function that intakes one parameter: a stock """
        stocks = self.get_stocks(group)
        pbar =  tqdm(stocks, desc = 'Iterating')
        out = []
        for stock in pbar:
            q = connection.cursor()
            g = q.execute(query(stock))
            gr = g.fetchall()
            df = pd.DataFrame(gr, columns = [x[0] for x in g.description])
            out.append(df)
        return pd.concat(out)


if __name__ == '__main__':
    
    from bin.main import Manager, get_path
    
    m = get_path()
    it = Iterator(m)
    
    def test_func(stock, conn = it.vol_db):
        """ Return todays option statistics """
        out = pd.read_sql('select * from {} order by date(gatherdate) desc limit 1'.format(stock), con = conn)
        out.insert(0, 'stock', stock)
        return out
    
    # out = it._iterate_function(test_func, group = 'etf')
    
    # out = it.dataframe_iterator_function(test_func, group = 'all_stocks')
    # print(out)
    

    def test_query(stock):
        out = f'''
        select * from {stock} 
        where datetime(gatherdate) = (select max(datetime(gatherdate)) from {stock}) 
        and volume > 1000 
        and oi_chg > 1000
        and impliedvolatility < iv_avg_30d
        and (volume/openinterest) > 1
        '''
        return out


    out = it.query_iteroator(test_query, it.change_db)
    print(out)
        