''' 
    This file will find Option Plays for a given stock. It was orginally written to find plays based on the implied move, but we will modify it 
        so that we can also give the class a range or percentage, which we will determine exteranlly from the trend module,or volatility estimation model. 
        
'''

import numpy as np 
import pandas as pd 
import datetime as dt 
from tqdm import tqdm 
import yfinance as yf
import sqlite3 as sql 
import json
import re


import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[3]))
from bin.models.


class dxp:
    def __init__(self, connections):
        self.stocks = json.load(open(connections['ticker_path'], 'r'))['all_stocks']
        self.option_db = sql.connect(connections['option_db'])
        self.change_db = sql.connect(connections['change_db'])
        self.verbose = True
        self.LOCAL = False
        
    def __breakeven(self, option_type, strike, option_price, stock_price):
        if option_type == 'Call':
            out = ((strike + option_price) - stock_price) / stock_price
        if option_type == 'Put':
            out = ((strike - option_price) - stock_price) / stock_price
        return out
    
    def __gyf(self, ticker):
        '''
        Get the option chain via api call to yfinance. 
        '''
        stock = ticker
        tk = yf.Ticker(stock)
        price = tk.history().iloc[-1]['Close']
        exps = tk.options[:5]
        options = []
        for e in exps:
            opt = tk.option_chain(e)
            calls = opt.calls
            calls['type'] = 'Call'
            puts = opt.puts
            puts['type'] = 'Put'
            option_df = pd.concat([calls, puts])
            option_df['expiry'] = e
            option_df['stk_price'] = price  
            option_df['cash'] = abs((price - option_df['strike'])* option_df['openInterest'])
            option_df.columns = [c.lower() for c in option_df.columns]
            options.append(option_df)    
        
        odf = pd.concat(options)
        odf['lastprice'] = (odf['lastprice'] + odf['bid']) / 2
        self.odf = odf
        return odf
    
    
    def _check_ticker(self, ticker):
        ''' Check if the ticker exist in the database '''
        all_stocks = self.stocks
        if ticker in all_stocks:
            return True
        else:
            print(f'{ticker} yfinance call...\n')
            return False
        
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
        
            # return bs_df(df)
            return df
        else: 
            return df 
        
        
    def _local_option_chain(self, stock):
        ''' Return the option chain from the CHANGE DB '''
        self.LOCAL = True
        out = self._parse_change_db(stock, today = True, bsdf = False)
        out['moneyness'] = out['strike'] / out['stk_price']
        self.odf = out.copy()
        return out
    
        
    def double_exp(self, df):
        '''
            At the very least the data frame must include: 
                1. strike
                2. expiry
                3. stk_price
                4. type
                5. lastprice
        '''
        itm = df.copy(); odf = df.copy()
        call_strike = itm[(itm['type'] == 'Call') & (itm['strike'] < itm.stk_price)]['strike'].max()
        put_strike = itm[(itm['type'] == 'Put') & (itm['strike'] > itm.stk_price)]['strike'].min()

        cols = ['expiry','stk_price','type','strike','lastprice', 'volume', 'openinterest']
        call_em = itm[(itm.strike == call_strike) & (itm.type == 'Call')][cols]
        put_em = itm[(itm.strike == put_strike) & (itm.type == 'Put')][cols]
        em = pd.concat([call_em, put_em]).groupby(['expiry']).agg({'stk_price':'first', 'lastprice':'sum'})
        self.dollar_amnt = em['lastprice'].iloc[0]
        em['empct'] = (0.95 * em['lastprice']) / em['stk_price']
        em.rename(columns = {'lastprice':'em'}, inplace = True)
        return em 
        
    
    def nearest_strike(self, x, strikes):
        return min(strikes, key=lambda y:abs(y-x))
    
    def dxp_analysis(self, odf, em):    
        mmm = em.em.iloc[0]
        price = em.stk_price.iloc[0]
        exps = list(odf.expiry.unique())

        call_targets = [price + (i * mmm) for i in range(1, 3)]
        put_targets = [price - (i * mmm) for i in range(1, 3)]
        avail_strikes = odf.strike.unique()
        
        # For each expiration, get the unique strikes, then find the target 
        targets = {}; flags = {}
        for e in exps: 
            strikes = odf[odf.expiry == e].strike.unique()
            targets[e] = {'Call': [self.nearest_strike(x, strikes) for x in call_targets], 
                        'Put': [self.nearest_strike(x, strikes) for x in put_targets]}
            flags[e] = {'Call': [1, 2], 'Put': [1, 2]}
            
        out = []
        for i in targets:
            if self.LOCAL != True:
                cols = ['type','expiry','stk_price','strike','lastprice','impliedvolatility','volume','openinterest','cash']
                o = odf[odf.expiry == i][cols]
            else:
                o = odf[odf.expiry == i].copy()
            for j in targets[i]:
                for k in targets[i][j]:
                    tmp = o[(o.type == j) & (o.strike == k)].copy()
                    tmp_o = flags[i][j][targets[i][j].index(k)]
                    tmp.insert(3, 'sd', tmp_o)
                    out.append(tmp)

        out_df = pd.concat(out).reset_index(drop = True)
        out_df['voi'] = out_df['volume'] / out_df['openinterest']
        out_df['be'] = out_df.apply(lambda x: self.__breakeven(x['type'], x['strike'], x['lastprice'], x['stk_price']), axis = 1)
        out_df['cash'] = abs((out_df['stk_price'] - out_df['strike'])* out_df['openinterest'])
        out_df = out_df.set_index(['expiry']).join(em[['em','empct']])
        return out_df
    
    def dxp(self, df):
        em = self.double_exp(df)
        out = self.dxp_analysis(df, em)
        return out
        
        
    def run(self,stock):
        '''
        Run the analysis for a given stock.
        '''
        if self._check_ticker(stock):
            df = self._local_option_chain(stock)
        else:
            df = self.__gyf(stock)
        
        
        df = self.dxp(df)
     
        if self.verbose: 
            pass
            
        return df.sort_values('strike')
    
    def summary_text(self, df):
        '''
        Return a summary text of the analysis. 
        '''
        if self.verbose == True: 
            print_text = True
        else:
            print_text = False
        
        df['cash'] = abs((df['stk_price'] - df['strike'])* df['openInterest'])
        gbdf = df.copy().groupby(['expiry','type']).agg({'volume':'sum', 'openinterest':'sum', 'cash':'sum'})
        expirations = sorted(list(df.expiry.unique()))
        exp = expirations[0]
        
    
    
    def scan(self, group = None):
        if group == None:
            group = self.stocks['all_stocks']
            
        out = []
        for stock in tqdm(group):
            out.append(self.run(stock))
        
        return pd.concat(out)
        
if __name__ == "__main__":
    from tqdm import tqdm 
    p = Play()
    print(p.run('iei'))