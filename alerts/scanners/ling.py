################################################################################################
"""
            Scanning Module for Finding Profitable Contracts
            ---------------------------------------------------
    1. This module is designed to scan through a list of stocks and find profitable contracts.
    2. We want to return a dictionary with flexible keys such as: 
        1. Top Volume Gainers 
        2. Top Open Interest Gainers
        3. Highest IV compared to 30 day averageq
        4. Highest Percentage Gain in Price. 
        5. Highest Volume to Open interest ratio 
        ....
        -- Etc. 
        We want to keep this flexibile so that later we can add more screening methods, like outputs from a machine learning model 

"""
################################################################################################
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import datetime as dt 
import json 



import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[3]))

from bin.alerts.iterator import Iterator 
from bin.alerts.scanners.dxp import dxp

class Scanner(Iterator):
    def __init__(self, connections = None):
        super().__init__(connections)
        self.stock_dict = json.load(open(connections['ticker_path'], 'r'))
        self.todays_date = dt.datetime.today().date()
        self.connections = connections
    
        
    def pct_chg(self, stock):
        """ Highest Percentage Gains in Contract price. 
                Looks for 1,000% gains in an option contract. 
        """
        q = f''' 
        select 
            contractsymbol, 
            datetime(gatherdate) as gatherdate,
            cast(stk_price as float) as stk_price,
            cast(stk_price_chg as float) as stk_price_chg,
            cast(ask as float) as ask,
            cast(bid as float) as bid,
            cast(volume as float) as volume,
            cast(vol_chg as float) as vol_chg,
            cast(openinterest as float) as openinterest,
            cast(oi_chg as float) as oi_chg,
            cast(impliedvolatility as float) as impliedvolatility,
            cast(iv_avg_30d as float) as iv_avg_30d,
            cast(percentchange as float) as percentchange,
            cast(pct_chg as float) as pct_chg
        from {stock} 
        where date(gatherdate) = (select max(date(gatherdate)) from {stock})
        and cast(percentchange as float) > 100
        and cast(pct_chg as float) > 80
        and cast(ask as float) > 0.05
        and cast(bid as float) > 0.01
        and abs(ask - bid) < 0.25
        and volume > 496
        order by percentchange desc
        '''
        return q
    
    def volume(self, stock):
        """ Highest Volume Contracts """
        q = f''' 
        select 
            --contractsymbol, 
            --datetime(gatherdate) as gatherdate
            *
        from {stock} 
        where date(gatherdate) = (select max(date(gatherdate)) from {stock})
        and volume > 1000
        and cast(ask as float) > 0.05
        and cast(bid as float) > 0.05
        and (ask - bid) < 0.10
        and cast(ask as float) < 2
        and cast(bid as float) < 2
        order by volume desc
        limit 5
        '''
        return q
    
    
    def voi(self, stock):
        """ Highest Volume to Open Interest Ratio """
        q = f''' 
        select 
            --contractsymbol, 
            --datetime(gatherdate) as gatherdate
            *,
            volume / openinterest as voi
        from {stock} 
        where date(gatherdate) = (select max(date(gatherdate)) from {stock})
        and volume > 1000
        and volume / openinterest > 1
        and cast(ask as float) > 0.1
        and cast(bid as float) > 0.1
        and (ask - bid) < 0.10
        order by volume / openinterest desc
        limit 5
        '''
        return q
    
    def iv_diff(self, stock):
        """ Highest IV Ranking, comparing current IV to the 30 day average. """
        q = f''' 
        select 
            --contractsymbol, 
            --datetime(gatherdate) as gatherdate
            *, 
            iv_avg_30d - impliedvolatility as iv_diff
        from {stock} 
        where date(gatherdate) = (select max(date(gatherdate)) from {stock})
        and cast(impliedvolatility as float) < cast(iv_avg_30d as float)
        and cast(ask as float) > 0.1
        and cast(bid as float) > 0.1
        and (ask - bid) < 0.10
        and cast(ask as float) < 2
        and cast(bid as float) < 2
        order by (iv_avg_30d - impliedvolatility) desc
        limit 5
        '''
        return q
    
    def oi_chg(self, stock):
        """ Highest Open Interest Change """
        q = f''' 
        select 
            --contractsymbol, 
            --datetime(gatherdate) as gatherdate
            *
        from {stock} 
        where date(gatherdate) = (select max(date(gatherdate)) from {stock})
        and oi_chg > 100
        and cast(ask as float) > 0.1
        and cast(bid as float) > 0.1
        and (ask - bid) < 0.10
        and cast(ask as float) < 2
        and cast(bid as float) < 2
        order by oi_chg desc    
        limit 5
        '''
        return q
    
    def amnt(self, stock):
        """ 
        Highest 'AMNT' Change, this is when the change of open interest is greater than the total volume from the previous day.
                - This implies that there were contracts traded after the market closed, OR that the real volume was not reported. 
        """
        q = f'''
        select 
            --contractsymbol, 
            --datetime(gatherdate) as gatherdate
            *
        from {stock}
        where date(gatherdate) = (select max(date(gatherdate)) from {stock})
        and amnt > 0 
        and cast(ask as float) > 0.1
        and cast(bid as float) > 0.1
        and (ask - bid) < 0.10
        and cast(ask as float) < 2
        and cast(bid as float) < 2
        order by amnt desc
        limit 5
        '''
        return q
    
    def one_cent_wonders(self, stock):
        """
        Contracts that are trading under 5 cents, but have an extremely high volume or open interest.
        """
        q = f'''
        select 
            --contractsymbol, 
            --datetime(gatherdate) as gatherdate
            *
        from {stock}
        where date(gatherdate) = (select max(date(gatherdate)) from {stock})
        and cast(ask as float) < 0.10
        and cast(bid as float) < 0.10
        and cast(ask as float) > 0.01
        and cast(bid as float) > 0.01
        and volume > 200
        and vol_chg > 200
        and openinterest > 500
        and oi_chg > 500
        order by volume desc
        limit 5
        '''
        return q
    
    
    def run(self):
        # return self.group_query_iterator(self.high_percent_changes, self.connection, group = 'all_stocks')
        # return self.query_iteroator(self.high_percent_changes, self.connection, group = 'all_stocks')
        list_of_functions = [self.pct_chg, self.volume, self.voi, self.iv_diff, self.oi_chg, self.amnt, self.one_cent_wonders]

        return self.list_iterator(list_of_functions, conn = 'change_db', group = 'all_stocks')

    def scan_contracts(self):
        out = self.run()
        table_names = list(out.keys())
        write_connection = self.get_connection('stats_db')
        for table in table_names:
            df = out[table].copy()
            df.to_sql(table, write_connection, if_exists = 'replace', index = False)
            print(df)

    def dxp(self, stock):
        """
        Run the dxp scanner on a specific stock.
        """
        dxp_scanner = dxp(self.connections)
        return dxp_scanner.run(stock)



if __name__ == "__main__":
    from bin.main import get_path
    connections = get_path()
    sc = Scanner(connections)
    # sc.scan_contracts()
    print(sc.dxp('amzn'))