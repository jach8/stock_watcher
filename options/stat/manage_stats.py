import sys
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
import yfinance as yf
import datetime as dt
from tqdm import tqdm
import time 
import logging
sys.path.append('/Users/jerald/Documents/Dir/Python/Stocks')
from bin.options.stat.em import Exp
from bin.options.stat.change_vars import ChangeVars
from bin.options.stat.cp import CP

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Stats(Exp, ChangeVars, CP):
    def __init__(self, connections: Dict[str, str]):
        super().__init__(connections)

    def update_stats(self, stock: str, new_chain: pd.DataFrame) -> None:
        """
        Update the Stats Database with the new option data.

        Args:
            stock (str): The ticker symbol of the stock.
            new_chain (pd.DataFrame): New option chain data.

        Raises:
            ValueError: If the input DataFrame is empty.
        """
        if new_chain.empty:
            logger.error(f"Empty DataFrame for stock {stock}")
            raise ValueError("New option chain data cannot be empty")
        self.update_change_vars(stock)
        self.update_cp(stock, new_chain)
        self._em(stock, new_chain)
        self._all_cp()

    def _init_change_db(self) -> None:
        """
        Initialize the change db if needed.

        Raises:
            Exception: If there's an issue initializing the database for any stock.
        """
        for stock in tqdm(self.stocks['all_stocks'], desc="Initializing Change DB"):
            try:
                self._initialize_change_db(stock)
            except Exception as e:
                logger.error(f"ChangeDB: Error initializing change db for {stock}: {e}")

    def _init_vol_db(self) -> None:
        """
        Initialize the vol db if needed.

        Raises:
            Exception: If there's an issue initializing the database for any stock.
        """
        for stock in tqdm(self.stocks['all_stocks'], desc="Initializing Vol DB"):
            try:
                self._initialize_vol_db(stock)
            except Exception as e:
                logger.error(f"VolDB: Error initializing vol db for {stock}: {e}")

    def clear_tables(self) -> None:
        """
        Save a log and clear all tables in the stats db.

        Raises:
            Exception: If there's an error during the backup or table operations.
        """
        try:
            self.stats_db.backup(self.backup)
            cursor = self.stats_db.cursor()
            tables = [i[0] for i in cursor.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()]
            for table in tables:
                cursor.execute(f'DROP TABLE {table}')
            self.stats_db.commit()
            # logger.info("__SYSTEM__: All tables cleared successfully.")
        except Exception as e:
            logger.error(f"__SYSTEM__: Error occurred while clearing tables: {e}")
            raise

    def cp_query(self, stock: str, n: int = 30) -> pd.DataFrame:
        """
        Fetch and combine old and current CP data.

        Args:
            stock (str): The ticker symbol of the stock.
            n (int): Number of days to look back.

        Returns:
            pd.DataFrame: Combined CP data.

        Raises:
            Exception: If there's an error in reading from the database.
        """
        try:
            old_df = self.get_cp_from_purged_db(stock, n=n)
        except Exception as e:
            logger.warning(f"No old CP data for {stock}: {e}")
            old_df = pd.DataFrame()
        
        current_df = self._calculation(self._cp(stock, n=n))
        new_df = pd.concat([old_df, current_df], axis=0).reset_index(drop=True)
        return new_df

    def _init_em_tables(self) -> None:
        """Initialize the expected moves tables."""
        self._initialize_em_tables()

    def reverse_dict(self) -> Dict[str, List[str]]:
        """
        Reverses the keys and values of a dictionary containing string keys and list of string values.

        Returns:
            Dict[str, List[str]]: A new dictionary with reversed keys and values.
        """
        d = self.stocks.copy()
        if 'all_stocks' in d:
            del d['all_stocks']
        reversed_dict = {}
        for key, values in d.items():
            for value in values:
                if value not in reversed_dict:
                    reversed_dict[value] = []
                reversed_dict[value].append(key)
        
        return {stockname: groups[0] if len(groups) == 1 else groups[1] for stockname, groups in reversed_dict.items()}
    
    def calculate_ivr(self, df: pd.DataFrame, col:str) -> pd.Series:
        """
        Calculate Implied Volatility Rank (IVR).
        Args:
            df (pd.DataFrame): DataFrame containing historical data.
            col (str): Column name for IV data.	
        Returns:
            pd.Series: IVR values.
        """
        iv = df[col]
        iv_52w_high = df[col].rolling(window=252, min_periods=1).max()
        iv_52w_low = df[col].rolling(window=252,  min_periods=1).min()
        ivr = (iv - iv_52w_low) / (iv_52w_high - iv_52w_low) * 100
        return ivr

    def _mdf(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Modify DataFrame by filling in missing data and calculating moving averages.

        Args:
            df (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: Modified DataFrame with computed statistics.
        """
        df['call_ivr'] = self.calculate_ivr(df, 'call_iv')
        df['put_ivr'] = self.calculate_ivr(df, 'put_iv')
        df['atm_ivr'] = self.calculate_ivr(df, 'atm_iv')
        df['otm_ivr'] = self.calculate_ivr(df, 'otm_iv')
        df['blowoff_vol'] = df['total_vol'] >= df['total_vol'].quantile(0.95)
        df['blowoff_oi'] = df['total_oi'] >= df['total_oi'].quantile(0.95)
        df['avg_call_vol'] = df['call_vol'].rolling(window=30, min_periods=1).mean()
        df['avg_put_vol'] = df['put_vol'].rolling(window=30, min_periods=1).mean()
        df['avg_call_oi'] = df['call_oi'].rolling(window=30, min_periods=1).mean()
        df['avg_put_oi'] = df['put_oi'].rolling(window=30, min_periods=1).mean()
        df = df.round(4)

        # Clean up DataFrame
        # vol_oi_cols = list(df.filter(regex = 'vol|oi').columns)
        # df[vol_oi_cols] = df[vol_oi_cols].astype(int)
        oi_chng_cols = list(df.filter(regex='_chng').columns)
        for col in oi_chng_cols:
            df[col] = df[col].replace(0, np.nan)
            df[col] = df[col].fillna(method='ffill')
        
        # drop_cols = list(df.filter(regex='pct').columns)
        # df = df.drop(drop_cols, axis=1)

        return df

    def _all_cp(self) -> pd.DataFrame:
        """
        Returns the Daily Option Stats for all stocks.

        Returns:
            pd.DataFrame: Concatenated DataFrame of daily option stats for all stocks.

        Raises:
            Exception: If data retrieval or processing fails for any stock.
        """
        # logger.info("Getting the latest Option stats")
        sg = self.reverse_dict()
        out = []
        skip_stocks = ['lcid', 'itb']
        pbar = tqdm(self.stocks['all_stocks'], desc="Daily Option Stats")
        for stock in pbar:
            pbar.set_description(f"Processing {stock.upper()}")
            if stock in skip_stocks:
                continue
            else:
                try:
                    # df = pd.read_sql(f'''SELECT * FROM {stock} where dte_flag = 'total' ORDER BY datetime(gatherdate) ASC''', self.vol_db, parse_dates=['gatherdate'])
                    df = pd.read_sql(f'''SELECT * FROM {stock} ORDER BY datetime(gatherdate) ASC''', self.vol_db, parse_dates=['gatherdate'])
                    df['gatherdate'] = pd.to_datetime(df['gatherdate'])
                    # df = df.drop(columns = ['dte_flag'])
                    prem_cols = list(df.filter(regex = 'spread').columns)
                    if len(prem_cols) > 0:
                        print(f"Prem cols: {prem_cols}: {stock}")
                        print(df)
                    df.insert(0, 'stock', stock)
                    df.insert(1, 'group', df['stock'].map(sg)) 
                    df = self._mdf(df)
                    if df.empty:
                        logger.warning(f'No data for {stock}')
                        continue
                    out.append(df.tail(1))
                except Exception as e:
                    logger.error(f"Error processing stock {stock}: {e}")
        
        combined = pd.concat([x for x in out if not x.empty])
        combined.to_sql('daily_option_stats', self.stats_db, if_exists='replace', index=False)
        return combined

if __name__ == "__main__":
    print("Control what you can Control.")

    sys.path.append('/Users/jerald/Documents/Dir/Python/Stocks')
    from bin.main import get_path
    connections = get_path()
    oc = Stats(connections)
    df = oc._all_cp()
    print(df[['stock', 'group', 'call_iv', 'put_iv', 'call_ivr', 'put_ivr']])
