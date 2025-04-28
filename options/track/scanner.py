""" 
Contract Scanner Module: 
    - This module is responsible for scanning contracts that meet criteria for a stock, and storing them in the tracking db. 
"""
import sys
sys.path.append('/Users/jerald/Documents/Dir/Python/Stocks')

import pandas as pd 
import numpy as np 
import sqlite3 as sql
import datetime as dt
import uuid 
from bin.options.track.tracker import Tracker

class Scanner(Tracker):
    def __init__(self, connections):
        super().__init__(connections)
        self.verbose = True
        
    def _last_dates(self, stock, N = 5):
        """ Get the last N dates for a stock in the vol.db """
        q = f'''
        select 
            distinct
            last_value (datetime(gatherdate)) over 
                (partition by date(gatherdate) rows between 
                unbounded preceding and unbounded following) as gatherdate
        from {stock}
            where date(gatherdate) > date("2022-11-15")
            '''
        cursor = self.option_db.cursor()
        out = cursor.execute(q).fetchall()
        out = [x[0] for x in out]
        return out[-N:]
                
    def max_date(self, stock, date = None):
        """ Return the Most Recent Date in the database. """
        if date is None:
            q = f'select max(datetime(gatherdate)) from {stock}'
        else:
            q = f'select max(datetime(gatherdate)) from {stock} where date(gatherdate) = date("{date}")'
        cursor = self.option_db.cursor()
        return cursor.execute(q).fetchall()[0][0]

    def _construct_id(self, out):
        """ Construct a unique ID for the contract in the passed df. 
                - Columns must include 'contractsymbol' and 'start_date'
            :Returns a List of the IDs
        """
        lot = zip(out.contractsymbol, out.start_date)
        ids = [str(x)+'_'+str(y).split(' ')[0]+str(y).split(' ')[1] for x, y in lot]
        return ids
    
    def _check_ids(self, stock, ids):
        """ Check if the ID is already in the Database """
        tracking_cursor = self.tracking_db.cursor()
        check = tracking_cursor.execute(f'select id from {stock}').fetchall()
        check = [x[0] for x in check]
        id_diff = list(set(ids) - set(check))
        return id_diff
        
    def _chain_query(self, stock, dte = None):
        ''' Get active contracts for a given day 
                - Add Expiry column to the change db, so we can filter from there instead of the regular option chain. 
        
        '''
        md = self.max_date(stock)
        if dte is None:
            dte = f'datetime(gatherdate) = (select max(datetime(gatherdate)) from {stock})'
        else:
            dte = f'datetime(gatherdate) = (select max(datetime(gatherdate)) from {stock} where date(gatherdate) = date("{dte}") )'
        q = f'''
        select 
            contractsymbol,
            datetime(gatherdate) as start_date,
            date(expiry) as expiry,
            lastprice as start_price,
            impliedvolatility as start_iv, 
            case when date(expiry) > date("{md}") then 1 else 0 end as active
        from {stock}
        where 
            {dte}
            and julianday(date(expiry)) - julianday(date(gatherdate)) <= 30
            and julianday(date(expiry)) - julianday(date(gatherdate)) >= 0
            and cast(gamma as float) > 0
            and cast(volume as int) / ifnull(openinterest, 1) > 1
            and volume > 10000
            and bid > 0.05
            and lastprice > 0.10
            -- where contractsymbol in (select contractsymbol from t1 group by contractsymbol having count(*) > 3)

        order by 
            cast(volume as int) / ifnull(openinterest, 1) desc,
            date(expiry) asc,
            cast(gamma as float) desc
        limit 2
        '''
        out = pd.read_sql(q, self.option_db, parse_dates=['start_date', 'expiry'])
        return out
        
    def _todays_chain(self, stock, dte = None):
        """ Get Active Contracts with high volume/open interest ratio, positive gamma and decent volume.
            :Returns a List of the TWO most Active Contracts with the above criteria.        
        """
        out = self._chain_query(stock, dte)
        if len(out) == 0 or out is None:
            return None
        else:
            # Construct the ID 
            ids = self._construct_id(out)
            out.insert(0, 'id', ids)
            return out
        
    def _initialize_scanner(self, stock):
        """ 
        Initialize the Scanner for the first time. 
            Default: scans the past 5 days of data and inserts the contracts into the tracking_db
            - add option to specify the dates: 
                Input should look like this: 
                    str(x)[:10] for x in list(pd.bdate_range(start = "2024-03-22", end = "2024-04-03", freq = 'B'))]
        """
        dates = self._last_dates(stock) # Returns a list of dates 
        out = [self._todays_chain(stock, d) for d in dates]
        out = pd.concat(out)
        out.to_sql(stock, self.tracking_db, if_exists='replace', index=False)    
    
    def _insert_scan(self, stock, dte = None):
        """ 
        Insert the Active Contracts into the Tracking Database 
            - This is to be used for Updating the Tracking Database with new contracts. 
        """
        # Check if the stock is already in the Tracking Database. 
        out = self._todays_chain(stock, dte)
        if out is None or len(out) == 0:
            return None
        else:
            ids = self._construct_id(out)
            name_check = self._check_tracking_chain_for_stock(stock)
            if name_check == True:
                id_diff = self._check_ids(stock, ids)
                if len(id_diff) == 0:
                    if self.verbose:
                        print(f"{stock} Up-to-Date")
                    return None
                else:
                    if len(out) > 0:
                        if self.verbose:
                            print("New Contracts Found")
                            print(out, '\n\n')
                        return out         
        
    def _update_scan(self, stock):
        """ Update the Scanner with the most recent data """
        out = self._insert_scan(stock)
        if out is None:
            return None
        else:
            out.to_sql(stock, self.tracking_db, if_exists='append', index=False)
            return out
    
    def scan(self, stock):
        """ Run the Scanner """
        if self._check_tracking_chain_for_stock(stock) == False:
            self._initialize_scanner(stock)
        else:
            self._update_scan(stock)
    

            
if __name__ == "__main__":
    from tqdm import tqdm 
    print("One should not come under the influence of attraction or aversion. ")
    
    
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
    
    
    connections = {
            'inactive_db': 'data/options/log/inactive.db',
            'backup_db': 'data/options/log/backup.db',
            'tracking_values_db': 'data/options/tracking_values.db',
            'tracking_db': 'data/options/tracking.db',
            'stats_db': 'data/options/stats.db',
            'vol_db': 'data/options/vol.db',
            'change_db': 'data/options/option_change.db', 
            'option_db': 'data/options/options.db', 
            'options_stat': 'data/options/options_stat.db',
            'ticker_path': 'data/stocks/tickers.json'
    }

    oc = Scanner(connections)
    oc.scan('qqq') # Should manually replace the spy table. 
    print(pd.read_sql( 'select * from qqq', oc.tracking_db))
    # print('\n\n')
    # print(oc.run_scan('spy'))
    
    
    

    