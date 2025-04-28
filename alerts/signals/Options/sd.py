''' 
    This file contains a class used to implement the "Double Expectation" strategy.
    This strategy is based on the idea that the stock price will move at least the expected move
    This file is designed to provide quick analysis of the strategy.
    
    The strategy is based on the following steps:
    1. Find the Expected move for a given expiraiton date
    2. Find strikes at Single, and Double Expeced Move
    3. Calculate Option Statistics for the strikes
    4. Determine which contracts to buy and sell
    
'''

import numpy as np 
import pandas as pd 
import datetime as dt 
from tqdm import tqdm 
import yfinance as yf
import sqlite3 as sql 
import json


class dxp(object):
    '''
    Initialize the class such that the use of YahooFinance will be optional to complete the analysis. 
    '''
    def __init__(self,connections):
        self.verbose = True
        
        self.stock_path = connections['ticker_path']
        self.option_path = connections['option_db']
        
        self.stocks = json.load(open(self.stock_path))
        
        # self.stock_path = stock_path
        # self.option_path = option_path
        
        
    def run(self,stock):
        '''
        Run the analysis for a given stock.
        '''
        if self._check_ticker(stock):
            df = self.local_option_chain(stock, self.option_path)
        else:
            df = self.gyf(stock)
        
        
        df = self.dxp(df)
     
        if self.verbose: 
            try:
                df.index = pd.to_datetime(df.index)
                current_em = df['empct'].iloc[0]
                current_stock_price = df.stk_price.iloc[0]
                current_expiration = df.index.min()
                
                ### Volume and Open interest Stats
                total_call_volume = df[(df['type'] == 'Call') & (df.index == current_expiration)]['volume'].sum()
                total_put_volume = df[(df['type'] == 'Put') & (df.index == current_expiration)]['volume'].sum()
                total_call_oi = df[(df['type'] == 'Call') & (df.index == current_expiration)]['openinterest'].sum()
                total_put_oi = df[(df['type'] == 'Put') & (df.index == current_expiration)]['openinterest'].sum()
                
                ### Premium Stats 
                total_call_prem = df[(df['type'] == 'Call') & (df.index == current_expiration)]['lastprice'].sum()
                total_put_prem = df[(df['type'] == 'Put') & (df.index == current_expiration)]['lastprice'].sum()
                
                ## Get the Strikes for the Single and Double Expected Move
                one_sd_call = df[(df.sd == 1) & (df['type'] == 'Call') & (df.index == current_expiration)].iloc[0]
                one_sd_put = df[(df.sd == 1) & (df['type'] == 'Put') & (df.index == current_expiration)].iloc[0]
                two_sd_call = df[(df.sd == 2) & (df['type'] == 'Call') & (df.index == current_expiration)].iloc[0]
                two_sd_put = df[(df.sd == 2) & (df['type'] == 'Put') & (df.index == current_expiration)].iloc[0]
                
                volume_skew = 'Calls 🟢' if total_call_volume > total_put_volume else 'Puts 🔴'
                oi_skew = 'Calls 🟢' if total_call_oi > total_put_oi else 'Puts 🔴'
                prem_skew = 'Calls 🟢' if total_call_prem > total_put_prem else 'Puts 🔴'


                txt = f'''
                ${stock.upper()} Pricing In about ±{current_em:.2%} move by {current_expiration:%m/%d}
                    - Volume Skew: {volume_skew}, OI Skew: {oi_skew}, Premium Skew: {prem_skew}
                    - Call Volume: {total_call_volume:,.0f}, Put Volume: {total_put_volume:,.0f}
                    - Call OI: {total_call_oi:,.0f}, Put OI: {total_put_oi:,.0f}
                
                For calls:
                    1. ${one_sd_call.strike} strk is priced @ ${one_sd_call.lastprice:.2f} w/ a {one_sd_call.be:.2%} move to BE 
                    2. ${two_sd_call.strike} strk is priced @ ${two_sd_call.lastprice:.2f} w/ a {two_sd_call.be:.2%} move to BE
                For Puts:
                    1. ${one_sd_put.strike} strk is priced @ ${one_sd_put.lastprice:.2f} w/ a {one_sd_put.be:.2%} move to BE
                    2. ${two_sd_put.strike} strk priced @ ${two_sd_put.lastprice:.2f} w/ a {two_sd_put.be:.2%} move to BE

                ${stock.upper()} Pricing ${self.dollar_amnt:.2f} (±{current_em:.2%})
                Calls: 
                    ${one_sd_call.strike} strk @ ${one_sd_call.lastprice:.2f} 
                    ${two_sd_call.strike} strk @ ${two_sd_call.lastprice:.2f}
                Puts:
                    ${one_sd_put.strike} strk @ ${one_sd_put.lastprice:.2f}
                    ${two_sd_put.strike} strk @ ${two_sd_put.lastprice:.2f}
                '''
                print(txt)     
            except Exception as e:
                print(e) 
                return df
        return df.sort_values('strike')
        
    def _check_ticker(self, ticker):
        ''' Check if the ticker exist in the database '''
        all_stocks = json.load(open(self.stock_path))['all_stocks']
        if ticker in all_stocks:
            return True
        else:
            print('yfinance call...\n')
            return False

    def gyf(self, ticker):
        '''
        Get the option chain via api call to yfinance. 
        '''
        stock = ticker
        tk = yf.Ticker(stock)
        price = tk.history().iloc[-1]['Close']
        exps = tk.options[:3]
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
            option_df.columns = [c.lower() for c in option_df.columns]
            options.append(option_df)    
        
        odf = pd.concat(options)
        odf['lastprice'] = (odf['lastprice'] + odf['bid']) / 2
        self.odf = odf
        return odf
    
    def local_option_chain(self, stock, path):
        conn = sql.connect(path)
        q = f''' 
            select 
                type,
                date(expiry) as expiry,
                stk_price,
                cast(strike as float) as strike,
                impliedvolatility,
                lastprice,
                volume,
                openinterest
            from {stock} 
            where datetime(gatherdate) = (select max(datetime(gatherdate)) from {stock})
        '''
        odf = pd.read_sql(q, conn, parse_dates = ['expiry'])
        exps = sorted(list(odf.expiry.unique()))[:3]
        odf = odf[odf.expiry.isin(exps)]
        self.odf = odf
        conn.close()
        return odf
    
    
    def dxp(self, df):
        em = self.double_exp(df)
        out = self.dxp_analysis(df, em)
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
    
    # Calculate Breakeven 
    def break_even(self, option_type, strike, option_price, stock_price):
        if option_type == 'Call':
            out = ((strike + option_price) - stock_price) / stock_price
        if option_type == 'Put':
            out = ((strike - option_price) - stock_price) / stock_price
        return out
    
    def dxp_analysis(self, odf, em):    
        mmm = em.em.iloc[0]
        price = em.stk_price.iloc[0]
        exps = list(odf.expiry.unique())
        cols = ['type','expiry','stk_price','strike','lastprice','impliedvolatility','volume','openinterest']

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
            o = odf[odf.expiry == i][cols]
            for j in targets[i]:
                for k in targets[i][j]:
                    tmp = o[(o.type == j) & (o.strike == k)].copy()
                    tmp_o = flags[i][j][targets[i][j].index(k)]
                    tmp.insert(3, 'sd', tmp_o)
                    out.append(tmp)

        out_df = pd.concat(out).reset_index(drop = True)
        out_df['voi'] = out_df['volume'] / out_df['openinterest']
        out_df['be'] = out_df.apply(lambda x: self.break_even(x['type'], x['strike'], x['lastprice'], x['stk_price']), axis = 1)
        out_df = out_df.set_index(['expiry']).join(em[['em','empct']])
        return out_df
        
if __name__ == '__main__':
    option_db = 'data/options/options.db'
    stock_names = 'data/stocks/tickers.json'
    
    connections = {
        'option_db': option_db, 
        'ticker_path': stock_names
    }
    
    dxp = dxp(connections)
    dxp.verbose = False
    out = dxp.run('amd')
    out = out.round(2)
    flag = np.where(out['volume'] / out['openinterest'] > 1, 1, 0)
    # out.insert(7, 'voi', flag)
    
    from tqdm import tqdm 
    begin = '\033[92m'
    endoc = '\033[0m'
    
    lodf = []

    pbar = tqdm(dxp.stocks['all_stocks'], desc = "Scan: ")
    for i in pbar:
        pbar.set_description(f"{begin}${i.upper()}{endoc}")
        try:
            out = dxp.run(i)
            out = out.round(2)
            flag = np.where(out['volume'] / out['openinterest'] > 1, 1, 0)
            out.insert(7, 'voi-flag', flag)
            out.insert(0, 'stock', i.upper())
            lodf.append(out[out.voi >= 1.5])
        except Exception as e:
            print(f'{begin}i{endoc} {e}')
            pass
        
    
    hot = pd.concat(lodf) 
    hot.to_csv('signals/Options/hot.csv', index = True)
    print(hot)