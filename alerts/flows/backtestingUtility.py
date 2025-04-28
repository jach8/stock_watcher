import sys
from tqdm import tqdm
import numpy as np
import pandas as pd
import sqlite3 as sql
import datetime as dt
from itertools import chain 
import json 
import statsmodels.api as sm

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[3]))
from bin.models.option_stats_model_setup import data as daily_option_stats_data


from typing import List, Dict, Tuple, Union


class cp_backtesting_utility:
    def __init__(self, connections: Dict[str, Union[str, sql.Connection]]):
        self.verbose = False
        # Load in the Ticker Dictionary 
        self.stock_dict = json.load(open(connections['ticker_path']))

        # Save Connection Paths 
        self.daily_db = connections['daily_db']
        self.vol_db = connections['vol_db']
        self.stats_db = connections['stats_db']

        # Import the Daily Option Stats Class
        self.option_stats_data = daily_option_stats_data(connections)
        self.option_stats_data.verbose = self.verbose

        # Set-up a cache for the daily option stats data and for each stock \
        self.cached_data = {}
        self.cached_dost = {}

    def __query(self, query: str, db: Union[str, sql.Connection]) -> pd.DataFrame:
        if isinstance(db, str):
            conn = sql.connect(db)
        else:
            conn = db
        
        c = conn.cursor()
        c.execute(query)
        data = c.fetchall()
        columns = [column[0] for column in c.description]
        df = pd.DataFrame(data, columns=columns)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        if 'gatherdate' in df.columns:
            df['gatherdate'] = pd.to_datetime(df['gatherdate'])
        conn.close()
        return df
    
    def __reverse_dict(self):
        """
        Reverse the stock dictionary to map stock symbols to their respective groups.
        """
        reversed_dict = {}
        for group, stocks in self.stock_dict.items():
            if group == 'all_stocks':
                continue
            else:
                for stock in stocks:
                    reversed_dict[stock] = group
        return reversed_dict
    

    def calculate_ivr(self, df: pd.DataFrame, iv_col: str) -> pd.Series:
        """
        Calculate the Implied Volatility Rank (IVR) for a given column in the DataFrame.

        Args:
            df (pd.DataFrame): DataFrame containing historical data.
            iv_col (str): Column name for IV data.

        Returns:
            pd.Series: IVR values.
        """
        iv = df[iv_col]
        iv_52w_high = df[iv_col].rolling(window=252, min_periods=1).max()
        iv_52w_low = df[iv_col].rolling(window=252, min_periods=1).min()
        ivr = (iv - iv_52w_low) / (iv_52w_high - iv_52w_low) * 100
        return ivr
    
    def __dost_cleanup(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean up the Daily Option Stats DataFrame.
        """
        # Calculate Implied Volatility Rank (IVR) for calls, puts, ATM, and OTM
        df['call_ivr'] = self.calculate_ivr(df, 'call_iv')
        df['put_ivr'] = self.calculate_ivr(df, 'put_iv')
        df['atm_ivr'] = self.calculate_ivr(df, 'atm_iv')
        df['otm_ivr'] = self.calculate_ivr(df, 'otm_iv')

        df['blowoff_vol'] = df['total_vol'] >= df['total_vol'].quantile(0.99)
        df['blowoff_oi'] = df['total_oi'] >= df['total_oi'].quantile(0.99)
        
        # Round the IVR values to 4 decimal places
        df = df.round(4)

        # Replace zeros in change columns with NaN and forward fill
        oi_chng_cols = list(df.filter(regex='_chng').columns)
        for col in oi_chng_cols:
            df[col] = df[col].replace(0, np.nan)
            df[col] = df[col].fillna(method='ffill')
        
        # # Convert volume and open interest columns to integers
        # vol_oi_cols = list(df.filter(regex='vol|oi').columns)
        # df[vol_oi_cols] = df[vol_oi_cols].astype(int)

        # Drop percentage change columns
        # drop_cols = list(df.filter(regex='pct').columns)
        # df = df.drop(drop_cols, axis=1)
        return df
    
    def build_cp_table(self, date: str):
        """
        Build the Daily Option Stats table for a given date.
            
        """
        sg = self.__reverse_dict()
        out = []
        skip_stocks = ['lcid', 'itb']
        stocks = [x for x in self.stock_dict['all_stocks'] if x not in skip_stocks]
        for stock in tqdm(stocks, desc=f"Daily Option Stats for {date}"):
            try:
                # Get the stats data for the stock up to the given date
                query = f'''SELECT * FROM {stock} WHERE date(gatherdate) <= '{date}' ORDER BY datetime(gatherdate) ASC'''
                df = self.__query(query, self.vol_db)
                df['gatherdate'] = pd.to_datetime(df['gatherdate'])
                df.insert(0, 'stock', stock)
                df.insert(1, 'group', df['stock'].map(sg))
                df = self.__dost_cleanup(df)
                # self.cached_data['stock'] = df
                out.append(df.tail(1))
            except Exception as e:
                print(f"Error with {stock}: {e}")    
        combined = pd.concat([x for x in out if not x.empty])
        self.cached_dost[date] = combined
        return combined



if __name__ == "__main__":
    from bin.main import get_path 

    connections = get_path()


    cpdat = cp_backtesting_utility(connections)
    df = cpdat.build_cp_table("2025-03-26")
    print(df)
