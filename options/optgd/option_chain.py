import sys
sys.path.append('/Users/jerald/Documents/Dir/Python/Stocks')

from bin.options.optgd.db_connect import Connector

import pandas as pd 
import numpy as np 
import yfinance as yf 
import datetime as dt 
import sqlite3 as sql
import logging


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class OptionChain(Connector):
    def __init__(self, connections):
        """
        Module for obtaining the option chain from Yahoo Finance. 
        
        Args: 
            connections: Dictionary of the connections.
        
        Methods:
            get_option_chain(stock:str) -> pd.DataFrame: Get the option chain for a stock.
            _check_for_stock_in_option_db(stock:str) -> bool: Check if the stock is in the option database.
            insert_new_chain(stock:str) -> pd.DataFrame: Insert a new chain into the database.
        
        """
        super().__init__(connections)
        self.date_db = sql.connect(connections['dates_db'])
        # logger.info('Option Chain Module Initialized')
        
        
    def get_option_chain(self, stock:str) -> pd.DataFrame:
        """ 
        Gets the option chain from Yahoo Finance for a stock. 
        
        Args: 
            stock (str): Stock symbol.
        
        Returns:
            pd.DataFrame: Option Chain DataFrame.
        
        
        """
        try:
            symbol = stock
            tk = yf.Ticker(symbol)
            last = tk.history(period = '1d').iloc[-1]["Close"]
            
            exps = tk.options
            option_list = []
            for exp in exps: 
                chain = tk.option_chain(exp)
                puts, calls = chain.puts, chain.calls
                puts['type'] = 'Put'
                calls['type'] = 'Call'
                opt = pd.concat([puts, calls])
                opt['expiry'] = pd.to_datetime(exp)
                option_list.append(opt)
                
            options = pd.concat(option_list)
            if len(options) == 0:
                return None
            else:
                options['mid'] = (options['bid'] + options['ask']) / 2
                options['cash'] = abs((last - options['strike'])* options['openInterest'])
                options['stk_price'] = round(last, 2)
                options = options.drop(columns = ['contractSize', 'currency'])
                options.insert(0, 'gatherdate', dt.datetime.today().strftime('%Y-%m-%d %H:%M:%S'))
                options['timeValue'] = (pd.to_datetime(options['expiry']) - pd.to_datetime(options['gatherdate']))/ np.timedelta64(1,'D')/252
                options.columns = [x.lower() for x in options.columns]
                options.expiry = pd.to_datetime(options.expiry)
                # options = bs_df(options)
                return options
        except Exception as e:
            logger.error(f'STOCK: {stock.upper()} Error: {e}')
            return None
        
    def _check_for_stock_in_option_db(self, stock:str) -> bool:
        """ 
        Check if the stock is in the option database. 
        Args:
            stock (str): Stock Symbol
        
        Returns:
            bool: True if the stock is in the database, False if not.
        
        """
        cursor = self.option_db.cursor()
        query = f"""
        select exists(select 1 from sqlite_master where type='table' and name='{stock}')
        """
        valid = cursor.execute(query).fetchone()[0]
        return bool(valid)
    
    def _update_dates_db(self, stock:str, new_date:str) -> None:
        """
        Update the dates database with the new date for the stock. 
        
        Args:
            stock (str): Stock Symbol
            new_date (str): New Date
        
        Returns:
            None
        
        """
        cursor = self.date_db.cursor()
        query = f"""
        select exists(select 1 from sqlite_master where type='table' and name='{stock}')
        """
        valid = cursor.execute(query).fetchone()[0]
        if valid:
            # add the date to the database
            query = f"""
            insert into {stock} (stock, gatherdate) values ('{stock}','{new_date}')
            """
            cursor.execute(query)
            self.date_db.commit()
            # logger.info(f'{stock.upper()} Updated Date Database')
        else:
            self._initialize_date_db(stock, new_date)
            # logger.info(f'{stock.upper()} Initialized Date Database')
            
            
    def _initialize_date_db(self, stock:str, gatherdate:str) -> None:
        """ 
        Initialize the date database with the dates that the data was gathered.
        Create a table with the stock as the name and that has two columns:
            - stock: Stock Symbol
            - date: Date that the data was gathered.
    
        Args:
            stock (str): Stock Symbol
            gatherdate (str): Date that the data was gathered.
        
        returns:
            None
        """
        cursor = self.date_db.cursor()
        query = f"""
        create table {stock} (stock text, gatherdate text)
        """
        cursor.execute(query)
        self.date_db.commit()
        query = f"""
        insert into {stock} (stock, gatherdate) values ('{stock}', '{gatherdate}')
        """
        cursor.execute(query)
        self.date_db.commit()
    
            
    def insert_new_chain(self, stock: str) -> pd.DataFrame:
        """
        Insert a new chain into the database. If the stock is in the database, append the new chain.
        Otherwise the chain will be replaced and added to the database. 
        
        Args:
            stock (str): Stock Symbol
        
        Returns:
            pd.DataFrame: Option Chain DataFrame.    
    
        """
        df = self.get_option_chain(stock)
        if df is None:
            return None
        else:
            #  If length of the new option chain > 0 and the stock is in the option database
            if len(df)> 0 and self._check_for_stock_in_option_db(stock) == True:
                df.to_sql(stock, self.write_option_db, if_exists = 'append', index = False)
                # logger.info(f'{stock.upper()} Appended to Database')
                self._update_dates_db(stock, df.gatherdate.max())
            elif len(df) > 0 and self._check_for_stock_in_option_db(stock) == False:
                # First check to see if there is existing data in the database, for the stock. 
                # logger.info(f'Checking for {stock.upper()} in the database')
                try:
                    oldf_q = f'select * from {stock}'
                    cursor = self.option_db.cursor()
                    oldf = cursor.execute(oldf_q).fetchall()
                    oldf = pd.DataFrame(oldf, columns = [x[0] for x in cursor.description])
                    if oldf.shape[0] > 0:
                        # If there is data in the database, then we need to check if we are overwriting it.
                        raise ValueError(f'{stock.upper()} You are about to overwrite {oldf.shape[0]} rows with {df.shape[0]} rows')
                except:
                    # If there is no data in the database, then we can write the new data.
                    # This is the case when we add a new stock to the database. 
                    df.to_sql(stock, self.write_option_db, if_exists = 'replace', index = False)
                    self._initialize_date_db(stock, df.gatherdate.max())
                    # logger.info(f'{stock.upper()} Initialized in Database')
            else:
                logger.error(f'{stock.upper()} No Data to Insert')
                return None
            
            self.write_option_db.commit()
            return df
        
    
if __name__ == "__main__":
    import time
    from tqdm import tqdm
    from bin.main import Manager, get_path
    connections = get_path()
    start_time = time.time()
    oc = OptionChain(connections)
    oc.insert_new_chain('spy')
    end_time = time.time()
    print(f'\n\nTime: {end_time - start_time}')
    oc.close_connections()
    