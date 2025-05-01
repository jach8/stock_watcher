"""

Do we really need to connect to ALL the databases at once? 
- Should we connect to the databases as needed?


"""


import sqlite3 as sql 
import numpy as np 
import pandas as pd 
import datetime as dt 
import re
import time
import json 
import logging

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Connector: 
    def __init__(self, connections: dict) -> None:
        """ 
        Database Connector for Options Data. 
        
        
        Args:
            connections (dict): Dictionary of the paths to the databases. 
                Example:
                    {
                        'option_db': 'Path to the option database',
                        'change_db': 'Path to the option change database',
                        'vol_db': 'Path to the volume database',
                        'stats_db': 'Path to the statistics database',
                        'tracking_db': 'Path to the tracking database',
                        'tracking_values_db': 'Path to the tracking values database',
                        'backup_db': 'Path to the backup database',
                        'inactive_db': 'Path to the inactive database',
                        'ticker_path': 'Path to the ticker json file
                    }
                    
        Attributes:
            stocks: Dictionary of the stocks.
            all_stocks: List of all the stocks in the database.
            
        To Do: 
            1_ Change the way we connect to the databases. Opting for a Pooling Method, 
                the main class that manages connections should look something like this:
                    Example: 
                    
                    # connection_manager.py
                    import sqlite3
                    from contextlib import contextmanager

                    class ConnectionManager:
                        def __init__(self, db_paths):
                            self.db_paths = db_paths
                            self.connections = {}

                        @contextmanager
                        def get_connection(self, db_name):
                            if db_name not in self.connections:
                                self.connections[db_name] = sqlite3.connect(self.db_paths[db_name])
                            try:
                                yield self.connections[db_name]
                            finally:
                                # Here you might choose to not close the connection if using lazy loading
                                # self.connections[db_name].close()
                                pass

                        def close_all(self):
                            for conn in self.connections.values():
                                conn.close()
                            self.connections.clear()

                    # Usage in other modules:
                    # from connection_manager import ConnectionManager
                    # conn_manager = ConnectionManager({'option_db': 'path_to_db'})
                    # with conn_manager.get_connection('option_db') as conn:
                    #     cursor = conn.cursor()
                    #     # Operations here
                    
            2_ Add method to Ensure that Deletions from the Option DB are correctly Logged in the backup DB or Inactive DB.
                - For any LARGE CHANGES you should log them in the backup db or inactive db.
                (1/31/2025): Error: I mistakenly deleted the entire option_db and change_db for SPY :(
                    - This happened because i used 'replace' instead of 'append' when updating the option_db.
                    
        """
        self.execution_start_time = time.time()
        try:
            # Add stocks, Keys: ['all_stocks', 'bonds','etf', 'equities', 'market', 'mag8']
            self.stocks = json.load(open(connections['ticker_path'], 'r'))     
            self.all_stocks = self.stocks['market'] + self.stocks['bonds'] + self.stocks['mag8'] + self.stocks['etf'] + self.stocks['equities']
            
            # Add Connections 
            self.path_dict = connections
            self.option_db = sql.connect('file:' + connections['option_db'] + '?mode=ro', uri = True)
            self.option_db_cursor = self.option_db.cursor()
            
            self.write_option_db = sql.connect(connections['option_db'])
            self.write_option_db_cursor = self.write_option_db.cursor()
            
            self.change_db = sql.connect(connections['change_db'])
            self.change_db_cursor = self.change_db.cursor()
            
            self.vol_db = sql.connect(connections['vol_db'])
            self.stats_db = sql.connect(connections['stats_db'])
            self.tracking_db = sql.connect(connections['tracking_db'])
            self.tracking_values_db = sql.connect(connections['tracking_values_db'])
            self.backup = sql.connect(connections['backup_db'])
            
            
            self.inactive_db = sql.connect(connections['inactive_db'])
            self.inactive_db_cursor = self.inactive_db.cursor()

            logger.info("Options DB Initialized")
            logger.warning("You currently have 8 connections open.")
        
        except Exception as e:
            print("Connection Failed: ", e,)       
        
    def _check_inactive_db_for_stock(self, stock: str) -> bool: 
        """ 
        Check if the stock is in the inactive database 
        
        Args:
            stock (str): Stock Ticker Symbol
        
        Returns:
            bool: True if the stock is in the database, False if not.
        
        """
        query = f"""
        select exists(select 1 from sqlite_master where type='table' and name='{stock}')
        """
        cursor = self.inactive_db_cursor
        valid = cursor.execute(query).fetchone()[0]
        return bool(valid)
            
    def _purge_inactive(self, stock: str) -> None:
        """
        Purge Inactive Contracts from the Option_db database. 
            - Save them in the inactive_db so that we can use them for tracking. 
        
        Args:
            stock (str): Stock Ticker Symbol
        
        Returns:
            None
        
        """
        exp_q = f''' select * from {stock} where date(expiry) < date('now') '''
        cursor = self.write_option_db_cursor
        exp = cursor.execute(exp_q).fetchall()
        exp = pd.DataFrame(exp, columns = [x[0] for x in cursor.description])
        
        if exp.shape[0] > 0:
            contracts = ','.join([f'"{x}"' for x in exp.contractsymbol])
            change_db_q = f''' select * from {stock} where contractsymbol in ({contracts}) '''
            cdb = pd.read_sql_query(change_db_q, self.change_db)
            
            if self._check_inactive_db_for_stock(stock):
                print("EXISTING TABLE",len(exp), len(cdb))
                exp.to_sql(stock, self.inactive_db, if_exists='append', index=False)
                cdb.to_sql(stock + "_change", self.inactive_db, if_exists='append', index=False)
            else:
                print("NEW TABLE:",len(exp), len(cdb))
                exp.to_sql(stock, self.inactive_db, if_exists='replace', index=False)
                cdb.to_sql(stock + "_change", self.inactive_db, if_exists='replace', index=False)
            
            # Purge from Option DB 
            self.write_option_db_cursor.execute(f'delete from {stock} where date(expiry) < date("now")')
            self.write_option_db.commit()
            
            # Purge from Change DB
            self.change_db_cursor.execute(f'delete from {stock} where contractsymbol in ({contracts})')
            self.change_db.commit()
        
    def close_connections(self) -> None:
        """
        Closes all the connections to the databases. 
        
        Returns:
            None
        
        """
        db_list = [
            self.option_db, 
            self.write_option_db, 
            self.change_db, 
            self.vol_db, 
            self.stats_db, 
            self.tracking_db, 
            self.tracking_values_db, 
            self.backup
        ]
        for i in db_list:
            i.close()
        end_time = time.time()
        runtime_min = (end_time - self.execution_start_time) / 60
        # print("Connections Closed {}\nTotal Runtime: {:.2f} min".format(dt.datetime.now(), runtime_min))
        logger.info("Connections Closed, Total Runtime: {:.2f} min".format(runtime_min))
        print()
        return None
    
if __name__ == "__main__":
    print("True Humility is not thinking less of yourself; It is thinking of yourself less.")
    import sys
    sys.path.append('/Users/jerald/Documents/Dir/Python/Stocks')
    from bin.main import get_path 
    connections = get_path()  
    conn = Connector(connections)
    # conn._purge_inactive('spy')
    # conn.close_connections()        