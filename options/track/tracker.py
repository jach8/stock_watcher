""" 
Historical Data For the Scanner: 
    - This module gets historical prices for contracts previously identified from the scanner. 
    - It will be used to track the change in contract prices 
    - This will help us determine if our strategy is working or not. 
    - By default, we will find the price of each contract on the day of expiration
        : We are also interested in knowing if there was a maximum profit opportunity, where the observed price was higher than the starting price. 
"""

import pandas as pd 
import numpy as np 
import sqlite3 as sql
import datetime as dt
import sys
sys.path.append('/Users/jerald/Documents/Dir/Python/Stocks')
from bin.options.optgd.db_connect import Connector

class Tracker(Connector):
    def __init__(self, connections):
        super().__init__(connections)
        self.verbose = True
        
    def get_max_date(self, stock, date = None):
        if date == None: 
            q = f'select max(datetime(gatherdate)) from {stock}'
        else:
            q = f'select max(datetime(gatherdate)) from {stock} where date(gatherdate) = date("{date}")'
        cursor = self.option_db.cursor()
        return pd.to_datetime([cursor.execute(q).fetchall()[0][0]])[0]
    
    def load_tracking(self, stock):
        """
        Load in Contracts Found from Critera Search. This is used to determine which historcal values needed. 
        """
        if self._check_tracking_chain_for_stock(stock):
            q = f'select * from {stock}'
            return pd.read_sql(q, self.tracking_db)
        else:
            print("No Contracts in the Tracking Database")
    
    def _check_tracking_values_for_stock(self, stock):
        """ Check if the stock has an existing table in the tracking values db and if the table is empty
            Returns:
                - True if the stock is in the tracking values db 
                - False if the stock is not in the tracking values db
        """
        q = f'select name from sqlite_master where type="table" and name="{stock}"'
        cursor = self.tracking_values_db.cursor()
        res = cursor.execute(q).fetchall()
        if res != []:
            return True
        else:
            return False
    
    def load_tracking_values(self, stock):
        """
        Load in the current tracking values. This will include active and inactive contracts. 
        To be used when we need to update the tracking values. 
        """
        if self._check_tracking_values_for_stock(stock):
            q = f'select * from {stock}'
            return pd.read_sql(q, self.tracking_values_db)
        else:
            print("No Contracts in the Tracking Database")
                
    def _load_contract_values(self, stock, contractsymbol, start_date):
        q = f'''
        select 
            contractsymbol, 
            datetime(gatherdate) as last_date,
            lastprice as last_price,
            impliedvolatility as last_iv
            from {stock} where contractsymbol = "{contractsymbol}"
            and datetime(gatherdate) >= datetime("{start_date}")
            order by datetime(gatherdate) desc 
            limit 1
        '''
        return pd.read_sql(q, self.change_db)
    
    def _check_tracking_chain_for_stock(self, stock):
        """ Check if the stock has an existing table in the tracking db and if the table is empty
            Returns:
                - True if the stock is in the tracking db 
                - False if the stock is not in the tracking db
        """
        q = f'select name from sqlite_master where type="table" and name="{stock}"'
        cursor = self.tracking_db.cursor()
        res = cursor.execute(q).fetchall()
        if res != []:
            return True
        else:
            return False
    
    def _extreme_swings(self, df, start_price):
        ap = df.copy()
        ap['mxp'] = np.where(ap.last_price > start_price, 1, 0)
        ap['mxl'] = np.where(ap.last_price < start_price, 1, 0)

        if ap.mxp.sum() > 0:
            mxp_price = ap.last_price.max()
            mxp_date = ap[ap.last_price == ap.last_price.max()].last_date.values[0]
            if mxp_date != ap.last_date.max():
                ap['mxp'] = f'{ap.last_price.max()}, {ap[ap.last_price == ap.last_price.max()].last_date.iloc[0]}'
            # ap['mxp_price'] = ap.last_price.max()
            # ap['mxp_date'] = ap[ap.last_price == ap.last_price.max()].last_date.values[0]
        
        if ap.mxl.sum() > 0:
            mxl_price = ap.last_price.min()
            mxl_date = ap[ap.last_price == ap.last_price.min()].last_date.values[0]
            if mxl_date != ap.last_date.max():
                ap['mxl'] = f'{ap.last_price.min()}, {ap[ap.last_price == ap.last_price.min()].last_date.iloc[0]}'
            # ap['mxl_price'] = ap.last_price.min()
            # ap['mxl_date'] = ap[ap.last_price == ap.last_price.min()].last_date.values[0]
        
        return ap.sort_values('last_date').groupby('contractsymbol').tail(1) 

    def intialize_tracking_values(self, stock):
        """
        Initialize the tracking values for a stock. 
            - This is to be done if:
                1. the stock is not currently in the tracking values database. 
                2. You wish to re-initialize the tracking values. 
        """
        contracts = self.load_tracking(stock)
        out = []
        for row in contracts.iterrows():
            row = row[1]
            cv = self._load_contract_values(stock, row.contractsymbol, row.start_date)
            cv = self._extreme_swings(cv, row.start_price)
            out.append(cv)
            
        df_out = pd.concat(out)
        df_out = contracts.merge(df_out, on = 'contractsymbol', how = 'left')
        if self.verbose:
            print(f"{stock} Tracking Values Initialized")
            ##print(df_out)
        df_out.to_sql(stock, self.tracking_values_db, if_exists = 'replace', index = False)
        return df_out
        
    def update_tracking_values(self, stock):
        """
        Get the latest prices, iv, and other values for the Active contracts in the tracking values database. 
            - Only update the active contracts, if their values changed. 
        """
        tracking_values = self.load_tracking_values(stock)
        ac = tracking_values.copy()
        for row in ac.iterrows():
            row = row[1]
            if row['active'] == 1:
                cv = self._load_contract_values(stock, row.contractsymbol, row.last_date)
                if row.last_price != cv.last_price.values[0]:
                    if self.verbose: print(f"!! Updates Found for {row.contractsymbol}")
                    cv = self._extreme_swings(cv, row.start_price)
                    ac.loc[ac.contractsymbol == row.contractsymbol, 'last_date'] = cv.last_date.values[0]
                    ac.loc[ac.contractsymbol == row.contractsymbol, 'last_price'] = cv.last_price.values[0]
                    ac.loc[ac.contractsymbol == row.contractsymbol, 'last_iv'] = cv.last_iv.values[0]
                    ac.loc[ac.contractsymbol == row.contractsymbol, 'mxp'] = cv.mxp.values[0]
                    ac.loc[ac.contractsymbol == row.contractsymbol, 'mxl'] = cv.mxl.values[0]
                
        ac.to_sql(stock, self.tracking_values_db, if_exists = 'replace', index = False)
        return ac

    def _purge_max_date_from_tracking_values_db(self, stock):
        """ Purge the max date from the tracking values db """
        q = f'''
        delete from {stock} where last_date = (select max(last_date) from {stock})
        '''
        self.tracking_values_db.execute(q)
        self.tracking_values_db.commit()
        return None

    def track(self, stock):
        """
        Run the Tracker for a stock. 
            1. Check if the stock is in the values db 
                : If not, initialize the tracking values
            2. If the stock table exist
                : Update the tracking values
        """
        if self._check_tracking_values_for_stock(stock) == False:
            self.intialize_tracking_values(stock)
        else:
            self.update_tracking_values(stock)
        return None

    
    
                    
if __name__ == "__main__":
    from tqdm import tqdm 
    print(" ")
    
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

    oc = Tracker(connections)
    print(pd.read_sql('select * from spy', oc.tracking_values_db))
    print('\n\n')
    d = oc.update_tracking_values('spy')
    print(d)
    