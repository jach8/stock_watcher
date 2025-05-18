import pandas as pd 
import numpy as np 
import datetime as dt 
from tqdm import tqdm 
import json 
import sys
import sqlite3 as sql
sys.path.append('/Users/jerald/Documents/Dir/Python/Stocks')
from bin.options.optgd.db_connect import Connector as Manager


class Iterator(Manager):
    def __init__(self, connections):
        super().__init__(connections)
        self.stock_dict = self.stocks.copy()
        self.connection_dict = connections
    
    def get_connection(self, connection):
        assert connection in list(self.connection_dict.keys()), f'Connection not found, Available Connections: {list(self.connection_dict.keys())}'
        return sql.connect(self.connection_dict[connection])
        
        
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
    
    def group_query_iterator(self, query, conn, group = 'etf'):
        """ query must be a function that intakes one parameter: a stock """
        connection = self.get_connection(conn)
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
    
    def cursor_iterator(self, query, cursor, group = 'etf'):
        """ query must be a function that intakes one parameter: a stock """
        stocks = self.get_stocks(group)
        pbar =  tqdm(stocks, desc = 'Iterating')
        q = cursor.cursor()
        out = []
        for stock in pbar:
            g = pd.DataFrame(q.execute(query(stock)).fetchall(), columns = [x[0] for x in q.description])
            out.append(g)
        return out
    
    def run_query(self, query, cursor):
        ''' Run a single query using a cursor '''
        q = cursor.cursor()
        g = pd.DataFrame(q.execute(query).fetchall(), columns = [x[0] for x in q.description])
        return g
    
    def query_iteroator(self, qf, conn, group = 'etf'):
        """
        Run a query for each stock in the group and return the results as a dataframe
    
        Args:
            qf (function): This must be a function that only takes one parameter: stock, it returns a query string
            cursor (_type_): cursor object
            group (str, optional): Stock group. Defaults to 'etf'.

        Returns:
            pd.DataFrame : DataFrame of the results
        """
        stocks = self.get_stocks(group)
        connection = self.get_connection(conn)
        pbar =  tqdm(stocks, desc = 'Iterating')
        out = []
        for stock in pbar:
            df = self.run_query(qf(stock), connection)
            df['flag_name'] = qf.__name__
            out.append(df)
        return pd.concat(out)
    
        
    def list_iterator(self, loqf, conn, group = 'etf', names = None):
        """
        Run multiple queries for each stock in the group and return the results as a dataframe
    
        Args:
            loqf (List of functions): Each Function only takes one parameter: stock, it returns a query string
            cursor (_type_): cursor object
            group (str, optional): Stock group. Defaults to 'etf'.

        Returns:
            pd.DataFrame : DataFrame of the results
        """
        assert type(loqf) == list, 'loqf must be a list of functions'
        assert all([callable(x) for x in loqf]), 'All elements in loqf must be functions'
        assert len(loqf) > 0, 'loqf must have at least one function'
        
        connection = self.get_connection(conn)
        stocks = self.get_stocks(group)
        pbar =  tqdm(stocks, desc = 'Iterating')
        out = {}
        for stock in pbar:
            pbar.set_description(f"{stock}")
            for i, qf in enumerate(loqf):
                if names is not None: 
                    pbar.set_postfix({'Query': names[i]})
                df = self.run_query(qf(stock), connection)
                df['flag_name'] = qf.__name__
                # If key exists, append to the dataframe, else add it 
                if qf.__name__ in out:
                    out[qf.__name__] = pd.concat([out[qf.__name__], df])
                else:
                    out[qf.__name__] = df
        return out 


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
        