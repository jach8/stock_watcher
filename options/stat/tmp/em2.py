"""
Class that calculates The Expected Move based on the option chain data. 
"""

import sys
import pandas as pd 
import numpy as np 
import yfinance as yf 
import datetime as dt 
from tqdm import tqdm
import time 

sys.path.append('/Users/jerald/Documents/Dir/Python/Stocks')
from bin.options.optgd.db_connect import Connector


class Exp(Connector):
    def __init__(self, connections):
        super().__init__(connections)
    
    def __next_friday(self):
        today = dt.datetime.today()
        if today.weekday() == 4:
            return today + dt.timedelta(7)
        else:
            return today + dt.timedelta((4-today.weekday()) % 7)
    
    def _get_em_data(self, stock):
        q = f'''
        select 
            type, 
            datetime(gatherdate) as gatherdate,
            date(expiry) as expiry, 
            strike, 
            stk_price, 
            ask, 
            bid, 
            lastprice
        from {stock}
            where datetime(gatherdate) = (select max(datetime(gatherdate)) from {stock})
            and date(expiry) >= date('now')
            --and strike / stk_price between 0.90 and 1.10
        order by 
            date(expiry) asc
        '''
        df = pd.read_sql_query(q, self.option_db, parse_dates = ['expiry', 'gatherdate'])
        return df
    
    def get_twt(self, em):

        # Flag if the expected move exceeds 1%
        em['flag'] = em['empct'] > 0.05

        # Calculate 'twt' column based on the DataFrame
        em['twt'] = em.apply(lambda x: f'${x["stock"].upper()} ±{x["em"]:.2f} ({x["empct"]:.2%})', axis=1)

        # If flag insert 🔥 at the end
        em.loc[em.flag == True, 'twt'] = em.loc[em.flag == True, 'twt'].astype(str) + ' 🔥'

        # Return the DataFrame with the 'twt' column
        return em.drop(columns = 'flag')
    
    def _em_ext(self, stock, df= None):
        """ Returns the expected move for the foremost expiration date. """
        if df is None:
            df = self._get_em_data(stock)
            
        if len(df)<2:
            return None
            
        price_point = 'ask'
        itm = df.copy(); odf = df.copy()
        
        call_strike = itm[(itm['type'] == 'Call') & (itm['strike'] < itm.stk_price)]['strike'].max()
        put_strike = itm[(itm['type'] == 'Put') & (itm['strike'] > itm.stk_price)]['strike'].min()
        
        cols = ['expiry','stk_price','type','strike',price_point]
        call_em = itm[(itm.strike == call_strike) & (itm.type == 'Call')][cols]
        put_em = itm[(itm.strike == put_strike) & (itm.type == 'Put')][cols]
        
        em = pd.concat([call_em, put_em]).groupby(['expiry']).agg({'stk_price':'first', price_point:'sum'}).reset_index()
        em.rename(columns = {price_point:'em'}, inplace = True)
        em['empct'] = (0.95 * em['em']) / em['stk_price']
        if 'stock' in em.columns: 
            return em 
        else:
            em.insert(0, 'stock', stock)
            return em 
    
    def _em(self, stock, new_chain = None):
        if new_chain is not None: 
            em = self._em_ext(stock, new_chain)
        else:
            em = self._em_ext(stock)
        em = self.get_twt(em)
        # xxp = (pd.Timestamp.today() + pd.DateOffset(days = 7))
        # return em[(em.expiry == em.expiry.min())]
        return em 
    
    def __edit_expected_moves_table(self, df):
        """ Only include dates less than next friday """
        next_friday = self.__next_friday()
        return df[df.expiry <= next_friday]
               
    def _init_em_tables(self):
        pbar = tqdm(self.stocks['all_stocks'], desc = "Initializing Expected Move Table... ")
        out = []
        out_ext = []
        for stock in pbar:
            pbar.set_description(f"Initializing Expected Move Table ${stock.upper()}")
            d = self._em(stock)
            j = self._em_ext(stock)
            if d is not None:
                # d.insert(0, 'stock', stock)
                out.append(d)
                out_ext.append(j)
            else:
                continue
        out_df = pd.concat(out).reset_index(drop = True)
        out_df = self.__edit_expected_moves_table(out_df)
        out_df.to_sql('expected_moves', self.stats_db, if_exists = 'replace', index = False)
        out_ext_df = pd.concat(out_ext).reset_index(drop = True)
        out_ext_df.to_sql('exp_ext', self.stats_db, if_exists = 'replace', index = False)
        
        
           
    def gg(self):
        """ return expected_moves table """
        print('working')
        print('table1')
        g = pd.read_sql('select * from exp_ext', self.stats_db,parse_dates=['expiry'])
        print(g)
        print('table2')
        gg = pd.read_sql('select * from expected_moves', self.stats_db,parse_dates=['expiry'])
        print(gg)
        
        
if __name__ == "__main__":
    print("Control what you can Control.")
    connections = {
            'backup_db': 'bin/pipe/log/backup.db',
            'tracking_values_db': 'bin/pipe/test_data/option_criteria.db',
            'tracking_db': 'bin/pipe/test_data/option_criteria.db',
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
    oc = Exp(connections)
    oc._init_em_tables()
    
    oc.gg()