''' 
    2 Standard Deviation Theory:
           - This module will be used to find the best strike prices for options, given an expected range based on: 
                1. The expected move priced into the option chain 
                2. Volatility Indicators We obtain from observing the stock price movment over time. 
            - Compare the Expected Move with the % needed to break even on an option contract. 
                - this helps gauge weather or not an option is worthy of the risk. 
            
        Further Development:
            1. Implement Scenario Analysiss using Geometric Brownian Motion in a Mean Reverting System to evaluate the performance of an option contract. 
            
        All chosen options will be chosen from the next availible expiration date (+1 exp), with an option to choose further out. 
            - The strikes chosen will based off of the current expiration date (+0 exp) with the idea of choosing an option with time. 
'''

import numpy as np 
import pandas as pd 
import sqlite3 as sql 
import datetime as dt 
import json
from tqdm import tqdm

class expected_moves():
    def __init__(self, option_db, price_db, verbose = False):   
        try: 
            self.option_conn = sql.connect(option_db)
            self.price_conn = sql.connect(price_db)
            self.verbose = verbose 
        except Exception as e:
            print('Could not connect to the database: ', e)
                    
    def expected_moves(self, stock, lim = 3):
        ''' 
        Obtain the Expected Move from the Option Chain for the next availible expiration date
            : Use 85% Of the ATM Straddle Price, this is the sum of: 
                1. The Maximum Strike of the ITM Call option 
                2. The Minimum Strike of the ITM Put option     
        '''
        
        call_q = f'''
           select
                expiry, 
                stk_price,
                max(strike) as call_strike,
                lastprice as call_ask
            from
                {stock}
            where
                date(gatherdate) = (select max(date(gatherdate)) from {stock})
            and
                type = 'Call'
            and 
                stk_price > strike
            group by date(expiry)
        '''
        
        put_q = f'''
            select 
                expiry, 
                min(strike) as put_strike,
                lastprice as put_ask
            from
                {stock}
            where
                datetime(gatherdate) = (select max(datetime(gatherdate)) from {stock})
            and
                type = 'Put'
            and
                stk_price < strike
            group by date(expiry)
        '''
        
        exp_move = f'''
        with calls as ({call_q}), puts as ({put_q})
            select 
                date(calls.expiry) as expiry, 
                calls.stk_price, 
                (calls.call_ask + puts.put_ask) * 0.85 as em,
                --concat(concat("±", round(100 * (((calls.call_ask + puts.put_ask) * 0.85) / calls.stk_price), 2)), "%") as "%"
                round(( (calls.call_ask + puts.put_ask) * 0.85) / calls.stk_price, 4) as "%"
            from
                calls join puts on calls.expiry = puts.expiry
            where date(calls.expiry) > date('now')
            order by 
                date(calls.expiry) asc
            limit {lim + 1}
        '''

        d = pd.read_sql(exp_move, self.option_conn, parse_dates=['expiry'])
        d.insert(1, 'stock', stock.upper())
        d['em'] = d.em.iloc[0]
        d['1𝝈'] = list(zip((d.stk_price + d.em).round(2), (d.stk_price - d.em).round(2)))
        d['2𝝈'] = list(zip((d.stk_price + 2 * d.em).round(2), (d.stk_price - 2 * d.em).round(2)))
        if self.verbose: 
            print(d, '\n')
        return d 

    def find_strikes(self, d):
        ''' 
        Given the dataframe from the expected_moves method: 
            Find the Following:  
                1. The Strike Prices for Calls at +1𝝈, and +2𝝈
                2. The Strike Prices for Puts at -1𝝈, and -2𝝈
        
        Return a list of Call and Put Strikes for the next expiration date.
            1. Call Strikes: [1𝝈, 2𝝈]
            2. Put Strikes: [-1𝝈, -2𝝈]
            3. Expirations: [Next Expiration Date]
        '''
        # Printing Purposes: 
        stock = d.stock.to_list()[0]    
        curr_exp = d.expiry.to_list()[0]
        exps = d.expiry.to_list()[1:]
        curr_stock_price = d.stk_price.round(2).to_list()[0]
        curr_sd = d.em.round(2).to_list()[0]
        curr_sd_pct = d['%'].to_list()[0]
        
        # Gather the rough strikes for each expiration 
        call_strikes = [(x[0], y[0]) for x,y  in zip(list(d['1𝝈' ]), list(d['2𝝈']))]
        put_strikes = [(x[1], y[1]) for x,y  in zip(list(d['1𝝈' ]), list(d['2𝝈']))]
            
        # Get distinct Strikes For a given expiration date 
        q = lambda x: f''' 
                select 
                    distinct cast(strike as float) as strike   
                from {stock}
                where 
                    datetime(gatherdate) = (select max(datetime(gatherdate)) from {stock})
                and
                    date(expiry) = date("{x}")
                order by
                    cast(strike as float) asc
            '''
        
        # List of All Expirations, Dictionary with each expiration and availible strikes.  
        exps = d.expiry.to_list()
        exp_strike = {exp: pd.read_sql(q(exp), self.option_conn).strike.to_list() for exp in exps}
        
        # Create a dictionary to store strikes for each expiration 
        call_dict = dict(zip(exps, call_strikes)); put_dict = dict(zip(exps, put_strikes))
        strike_dict = {exp: {'Call': call_dict[exp], 'Put': put_dict[exp]} for exp in exps}
        
        # Now using exp_strike, find the closest strike prices in strike_dict for each expiration date and option type. 
        availible_strikes = {}
        
        # Iterate over each expiration to find the corresponding strikes for the expected move.
        # Closest Strikes = abs(strike - expected strike using 1 and 2 standard deviations)
        for exp in exps:
            targets = strike_dict[exp] # {Call: [x1, x2] , Put: [x1, x2]}
            strikes = exp_strike[exp] # [k1, k2, k3, k4, k5, ....]
            # minimize the the difference betweeen the actual strikes and the expected 
            cs1sd = min(strikes, key = lambda x: abs(x - targets['Call'][0]))
            cs2sd = min(strikes, key = lambda x: abs(x - targets['Call'][1]))
            ps1sd = min(strikes, key = lambda x: abs(x - targets['Put'][0]))
            ps2sd = min(strikes, key = lambda x: abs(x - targets['Put'][1]))
            
            availible_strikes[exp] = {'Call': (cs1sd, cs2sd), 'Put': (ps1sd, ps2sd)}
            if (cs2sd > cs1sd) and (ps2sd < ps1sd): 
                pass
            else:
                if self.verbose: 
                    print(f"Warning: Unable to find availible strikes for {exp} ")
                pass 
        
        if self.verbose:
            print(f'''
                Stock: {stock}
                Current Expiration: {curr_exp}
                Current Stock Price: {curr_stock_price}
                Current Expected Move: {curr_sd}
                Current Expected Move %: {curr_sd_pct}
                \n
            ''')
        
        return availible_strikes
        

    def get_option_prices(self, stock, strike_dict = None):
        ''' 
        Get the corresponding option prices and percentage needed to break even 
                for the strikes found in the find_strikes method. 
        Inputs:
            1. Stock: the stock to find the option prices for 
            2. Strike Dict: A dictionary containing the expiration dates and corresponding strikes for Calls and Puts. 
                    
        Process: 
            1. For each Expiration 
                - Find the Option Prices, and their break even percentage 
        
        '''
        if strike_dict is None: 
            strike_dict = self.find_strikes(self.expected_moves(stock, lim = 3))
        # Lambda Function to get the call and put contracts for a given expiration. 
        option_prices = lambda x, call_strikes, put_strikes: f'''
            select
                expiry,
                type, 
                cast(strike as float) as strike, 
                cast(lastprice as float) as ask, 
                case when type = 'Call' then (((strike + lastprice) - stk_price) / stk_price ) 
                    else (((strike - lastprice) - stk_price) / stk_price) end as "BE"
            from 
                {stock}
            where 
                datetime(gatherdate) = (select max(datetime(gatherdate)) from {stock}) and date(expiry) = date("{x}")
            and
                case when type = 'Call' then strike in {call_strikes} else strike in {put_strikes} end
            order by 
                cast(strike as float), type
        '''
        lodf = []
        for exp, strikes in strike_dict.items():
            c = pd.read_sql(option_prices(exp, strikes['Call'], strikes['Put']), self.option_conn, parse_dates=['expiry'])
            lodf.append(c)

        out = pd.concat(lodf).reset_index(drop=True)
        
        o = out[out.expiry > out.expiry.min()]
        o.index = out[out.expiry < out.expiry.max()].index
        o.columns = [f'{i}1' for i in o.columns]
        
        out = pd.concat([out, o], axis = 1).dropna()
        # Indicate weather the option is 1 or 2 standard deviations from the expected move. 
            # Using Strike Dict: {exp: {'Call': (x1, x2), 'Put': (x1, x2)}}, x1 = 1𝝈, x2 = 2𝝈
        sds_map = strike_dict.copy()
        for exp in list(sds_map.keys()):
            sds_map[exp]['Call'] = dict(zip(sds_map[exp]['Call'], ('1𝝈', '2𝝈')))
            sds_map[exp]['Put'] = dict(zip(sds_map[exp]['Put'], ('1𝝈', '2𝝈')))
        
        # Add the 1 and 2 standard deviation indicators to the data frame.
        out['sd'] = [sds_map[exp][i][j] for exp, i, j in zip(out.expiry, out.type, out.strike)]
        
        
        if self.verbose: 
            print('\n', out, '\n')

        return out
    
    
    def run(self, stock, lim = 3):
        ''' 
        Run the above methods to get all the data needed to construct the data frame which we will use. 
            -(For later use): Keep the data in a format so that a model is able to access it.  
        
        expected_moves -> data frame with columns: (expiry , stock, stk_price, em, %, 1𝝈, 2𝝈)
        find_strikes -> dictionary with expirations and strikes: {exp: {'Call': (x1, x2), 'Put': (x1, x2)}}
        get_option_prices -> data frame with columns: (expiry, stock, strike, type, ask, BE%)
        
        Note that we will be looking forward for all expirations dates:
            - However we will still use the expected move for the current expiration date
            - We Want to give the trade time to develop, so using the next expiration for purchase is ideal. 
            
        '''
        d = self.expected_moves(stock, lim = lim)
        sd = self.find_strikes(d)
        c = self.get_option_prices(stock, sd)
        d = d.drop(columns = ['1𝝈', '2𝝈'])
        cols = list(d.columns)[:] + ['type', 'strike']
        #c = c.drop(columns =['type1', 'strike1']).copy()
        out = d.merge(c, on = 'expiry', how = 'left')
        if self.verbose: 
            print('\n',out.set_index(cols).dropna(), '\n')    
        return out
    
if __name__ == '__main__':
    print('\nStop Doing Loser shit, if you dont want to be a Loser in Life.\n')
    options_file_path = 'data/options/options.db'
    price_file_path = 'data/prices/stocks.db'
    occ = expected_moves(option_db= options_file_path, price_db=price_file_path, verbose = True)
    occ.run('arm', lim = 5)