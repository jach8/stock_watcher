import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import datetime as dt 
from tqdm import tqdm 
import scipy.stats as st 
import time
import json 
import sys
import warnings
import logging 
sys.path.append('/Users/jerald/Documents/Dir/Python/Stocks')
from bin.options.optgd.db_connect import Connector 


# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)



class Iterator(Connector):
    """
    Iterator class for the Options Data. This class is used to iterate through the stocks in the database.
    
    """
    
    def __init__(self, connections):
        super().__init__(connections)
        self.stock_dict = self.stocks.copy()
        # logger.info('Iterator Module Initialized')
        
    def get_stocks(self, group = 'all_stocks'):
        return self.stock_dict[group]
        
    def _iterate_function(self, func, group = 'all_stocks'):
        """
        Iterate through the stocks and apply a function to each stock. 
        Please make sure that the function handles any queries that need to be made to the database. 
        This function will not handle any queries. 
        
        Args:
            func (function): Function to apply to each stock. 
            group (str): Group of stocks to iterate through.
        
        Returns:
            list: List of the outputs of the function.
        
        """
        # logger.info(f'Iterating through {group}')
        stocks = self.get_stocks(group)
        pbar = tqdm(stocks, desc = 'Iterating')
        out = []
        for stock in pbar:
            try:
                out.append(func(stock))
            except Exception as e:
                logger.warning(f'Error in {stock}: {e}')
                continue
        return out
    
    def dataframe_iterator_function(self, func, group = 'all_stocks'):
        lodf = self._iterate_function(func, group = group)
        return pd.concat(lodf)
    
    def query_iteroator(self, query, connection, group = 'etf',  *args, **kwargs):
        """ query must be a function that intakes one parameter: a stock """
        additional_function = kwargs.get('additional_function', None)
        stocks = self.get_stocks(group)
        pbar =  tqdm(stocks, desc = 'Iterating')
        out = []
        for stock in pbar:
            q = connection.cursor()
            g = q.execute(query(stock))
            gr = g.fetchall()
            df = pd.DataFrame(gr, columns = [x[0] for x in g.description])
            
            if additional_function:
                if len(df) == 0:
                    logger.warning(f'No data for {stock}')
                    continue
                else:
                    try:
                        df = additional_function(df)
                    except Exception as e:
                        print(df)
                        print(f'Error in {stock}: {e}')
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
    from models.sim.test import lsm
    from models.bsm.bs2 import bs_df

    def test_query(stock):
        out = f'''
        select * from {stock} 
        where datetime(gatherdate) = (select max(datetime(gatherdate)) from {stock}) 
        and expiry < date("2025-02-15")
        and volume > 1000 
        and abs(strike/stk_price) between 0.95 and 1.05
        '''
        return out


    out = it.query_iteroator(test_query, it.option_db, additional_function = bs_df)
    print(out)
    print(out.sample(40))
        