"""
Manager for the options data pipeline. 
    1. Get new option chain data
    2. Append data to the option database. 
    3. Update the vol.db after calculating the change variables. 

"""
import sys
sys.path.append('/Users/jerald/Documents/Dir/Python/Stocks')
import pandas as pd 
import numpy as np 
# import yfinance as yf 
import datetime as dt 
from tqdm import tqdm
import re
import time 
import sqlite3 as sql

from bin.options.stat.em import Exp
from bin.options.optgd.option_chain import OptionChain
from bin.options.stat.manage_stats import Stats
# from bin.options.track.manage_tracking import Screener
from models.bsm.bs2 import bs_df 



class Manager(OptionChain, Stats):
    def __init__(self, connections):
        super().__init__(connections)
        self.connections = connections
        #self.stock_list = ['spy', 'qqq' , 'iwm']
        
    def update_options(self):
        for stock in tqdm(self.stocks['all_stocks']):
            new_chain = self.insert_new_chain(stock)
            if new_chain is None:
                continue
            else:
                self.update_stats(stock, new_chain)
                self.run_screener(stock)
                time.sleep(5)  
    
    def get_em(self):
        return pd.concat([self._em(stock) for stock in tqdm(self.stocks['all_stocks'])])
    
    def _manage_db(self):
        """ Manage the Option Database. 
                : Purge Inactive Contracts 
        """
        print('WARNING: THIS WILL PURGE INACTIVE CONTRACTS DO YOU WITH TO CONTINUE?')
        ans = input('Y/N: ')
        if ans.lower() == 'y':
            for stock in tqdm(self.stocks['all_stocks'][:]):
                self._purge_inactive(stock)
        else:
            print('Exiting')
    
    def _delete_option_greeks(self):
        for stock in tqdm(self.stocks['all_stocks'][1:]):
            self._delete_option_greeks_from_option_db(stock)
    
    def _test_import(self):
        print('Imported')  
        print(self.__dict__) 
        
        
    def _join_purged_data(self, stock):
        q = f'''
        select 
        *
        from {stock}
        where date(expiry) > date('now')
        '''
        # df = pd.read_sql_query(q, self.inactive_db, parse_dates = ['gatherdate']) 
        cursor = self.inactive_db.cursor()
        cursor.execute(q)
        df = pd.DataFrame(cursor.fetchall(), columns = [desc[0] for desc in cursor.description])
        
        
    def describe_option(self, y):
        ''' Given an option contract symbol, using regular expressions to return the stock, expiration date, contract type, and strike price. '''
        valid = re.compile(r'(?P<stock>[A-Z]+)(?P<expiry>[0-9]+)(?P<type>[C|P])(?P<strike>[0-9]+)')
        stock = valid.match(y).group('stock')
        expiration = dt.datetime.strptime(valid.match(y).group('expiry'), "%y%m%d")
        conttype = valid.match(y).group('type')
        strike = float(valid.match(y).group('strike')) / 1000
        return ("$"+stock, conttype, float(strike), expiration.strftime('%m/%d/%y'))

    def _parse_change_db(self, stock, today = True, bsdf = True):
        stock = stock.lower()
        if today: 
            q = f'''select * from {stock} where date(gatherdate) = (select max(date(gatherdate)) from {stock}) '''
        else:
            q = f'''select * from {stock}'''
        # q = f'''select * from {stock}'''
        df = pd.read_sql(q, self.change_db, parse_dates=['gatherdate'])
        df['desc'] = [self.describe_option(x) for x in df.contractsymbol]
        names = ['stock', 'type', 'strike', 'expiry']
        df[names] = pd.DataFrame(df['desc'].tolist(), index=df.index)
        df.drop(['desc'], axis=1, inplace=True)
        df = df.set_index(names).reset_index()
        df['expiry'] = pd.to_datetime(df.expiry)
        df.type = df.type.map({'C': 'Call', 'P': 'Put'})
        df['stock'] = [x.replace('$', '') for x in df.stock]
        if bsdf == True:
            days = ((df.expiry+ pd.Timedelta('16:59:59')) - df.gatherdate).dt.days 
            # if days < 1, add 1 to days
            days = days.apply(lambda x: x if x > 0 else x + 0.5)
            df['timevalue'] = days / 252
            df['cash'] = np.abs(df.strike - df.stk_price) * df.openinterest * 100 
        
            return bs_df(df)
        else: 
            return df 
    
    def parse_change_db(self, df):
        """ Parse the output from the change_db or any dataframe where contractsymbol is in the dataframe

            args: 
                -df : pd.DataFrame containing contractsymbol in the columns 
        
        """
        out_desc = {x: self.describe_option(x) for x in df.contractsymbol.to_list()}
        out_desc_df = pd.DataFrame(out_desc).T.reset_index()
        out_desc_df.columns = ['contractsymbol', 'stock', 'type', 'strike', 'expiry']
        out_desc_df.type = out_desc_df.type.map({'C': 'Call', 'P': 'Put'})
        out_desc_df['stock'] = out_desc_df.stock.apply(lambda x: x.replace('$', ''))
        out_df = out_desc_df.merge(df, on = 'contractsymbol')
        out_df.strike = out_df.strike.astype(float)
        out_df.expiry = pd.to_datetime(out_df.expiry)
        return out_df.set_index(['gatherdate', 'contractsymbol']).sort_index().reset_index()

    def option_custom_q(self, q, db = 'option_db'):
        cursor = sql.connect(self.connections[db]).cursor()
        cursor.execute(q)
        df = pd.DataFrame(cursor.fetchall(), columns = [x[0] for x in cursor.description])
        if db != 'change_db':
            return df
        else:
            return self.parse_change_db(df)

    def contract_lookup(self, contractsymbol):
        """Lookup a contract symbol in the option database."""
        stock, _, _, _ = self.describe_option(contractsymbol)
        stock = stock.replace('$', '').lower()
        q = f'''select * from {stock} where contractsymbol = "{contractsymbol}"'''
        df = pd.read_sql(q, self.change_db, parse_dates = ['gatherdate'])
        return self.parse_change_db(df)
    
    def get_daily_option_stats(self, stock):
        """ 
        Get the daily option stats for a given stock. 
        Args: 
            - stock: str = Stock Symbol
            - dropCols: str = REGEX formated string of columns to drop
        
        Returns:
            - df: pd.DataFrame = Daily Option Stats for the given stock. 
        """
        q = f'''select * from {stock} '''
        df = pd.read_sql(q, self.vol_db, parse_dates = ['gatherdate'])
        df['gatherdate'] = pd.to_datetime(df['gatherdate'])
        df.insert(0, 'date', df.gatherdate.dt.date)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        return df.drop(columns = ['gatherdate'])

if __name__ == "__main__":
    print("You Cant go back and change the begining, but you can start right now and change the ending.")
    
    connections = {
            'inactive_db': 'bin/pipe/log/inactive.db',
            'backup_db': 'bin/pipe/log/backup.db',
            'tracking_values_db': 'data/options/bin/pipe/test_data/tracking_values.db',
            'tracking_db': 'data/options/bin/pipe/test_data/tracking.db',
            'stats_db': 'bin/pipe/test_data/stats.db',
            'vol_db': 'bin/pipe/test_data/vol.db',
            'change_db': 'bin//pipe/test_data/option_change.db', 
            'option_db': 'bin/pipe/test_data/test.db', 
            'testing_option_db': 'bin/pipe/test_data/test.db',
            'options_stat': 'bin/pipe/test_data/options_stat.db',
            'ticker_path': 'data/stocks/tickers.json'
            }

    import sys
    sys.path.append('/Users/jerald/Documents/Dir/Python/Stocks')
    from bin.main import get_path
    
    connections = get_path()
    oc = Manager(connections)
    
    # print(oc.pcdb(oc.option_custom_q('select * from aapl', db = 'change_db')))
    # print(oc.option_custom_q('select * from spy', db = 'change_db'))

    df = oc.get_daily_option_stats('amd')

    print(df)
    print(df.index)
    
