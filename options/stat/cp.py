# improved_cp.py

import sys
from typing import Any, Dict, List, Tuple, Optional
import pandas as pd 
import numpy as np 
import yfinance as yf 
import datetime as dt 
from tqdm import tqdm
import sqlite3 as sql
import logging
from IPython.display import display
sys.path.append('/Users/jerald/Documents/Dir/Python/Stocks')
from bin.options.optgd.db_connect import Connector
from models.bsm.bsModel import bs_df
import time
# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class CP(Connector):
    """
    Engineering Features for each stock in the database. Using the option chain, the following script aims to extract critical information about the stock for that date. 
    Each entry in the table has a Datetime index named 'gatherdate', and is in the format %Y-%m-%d %H:%M:%S sometimes it is %Y-%m-%dT%H:%M:%S.
    
    The following features are derived from the option chain: 
        1. 'call_vol': The total volume of call options traded for that day.
        2. 'put_vol': The total volume of put options traded for that day.
        3. 'total_vol': The total volume of options traded for that day.
        4. 'call_oi': The total open interest of call options for that day.
        5. 'put_oi': The total open interest of put options for that day.
        6. 'total_oi': The total open interest of options for that day.
        7. 'call_prem': The total premium of call options for that day.
        8. 'put_prem': The total premium of put options for that day.
        9. 'total_prem': The total premium of options for that day.
        10. 'call_iv': The average implied volatility of call options for that day.
        11. 'put_iv': The average implied volatility of put options for that day.
        12. 'atm_iv': The average implied volatility of options that are at the money for that day.
        13. 'otm_iv': The average implied volatility of options that are out of the money for that day.
        14. 'put_spread': The average spread (ask - bid) of put options for that day.
        15. 'call_spread': The average spread (ask - bid) of call options for that day.
    """

    def __init__(self, connections: Dict[str, Any]):
        """ Import Connections """
        super().__init__(connections)
        try:
            self.dates_db = sql.connect(connections['dates_db'])
        except sql.Error as e:
            logging.error(f"DAILY OPTION STATS: Failed to connect to dates_db: {e}", exc_info=True)
            raise

    def __custom_query_option_db(self, q: str, connection: sql.Connection) -> pd.DataFrame:
        """ 
        Helper function to run custom queries on the option database 
            args: 
                q: str: query 
                connection: sql.Connection: connection to the database
            returns:
                pd.DataFrame: DataFrame of the query results
        """
        try:
            c = connection.cursor()
            c.execute(q)
            d = pd.DataFrame(c.fetchall(), columns=[desc[0] for desc in c.description])
            d['gatherdate'] = pd.to_datetime(d['gatherdate'])
            return d
        except sql.Error as e:
            logging.error(f"DAILY OPTION STATS: Error executing custom query '{q[:10]}...' Connection: {connection}... {e}", exc_info=False)
            raise
    
    def _check_for_stock_in_vol_db(self, stock: str) -> bool:
        ''' Check if the stock is in the vol_db
        
        args:
            stock: str: stock symbol
        returns:
            bool: True if the stock is in the vol_db, False otherwise
        
        '''
        cursor = self.vol_db.cursor()
        q = f"""
        select exists(select 1 from sqlite_master where type='table' and name='{stock}')
        """
        valid = cursor.execute(q).fetchone()[0]
        return bool(valid)

    def _get_query_str(self, stock: str) -> str:
        q = f'''
            SELECT 
            datetime(gatherdate) AS gatherdate,
            CAST(SUM(volume) AS INT) AS total_vol,
            CAST(SUM(cash) AS FLOAT) AS total_prem, 
            CAST(SUM(openinterest) AS INT) AS total_oi,
            CAST(SUM(CASE WHEN type = 'Call' THEN volume ELSE 0 END) AS INT) AS call_vol,
            CAST(SUM(CASE WHEN type = 'Put' THEN volume ELSE 0 END) AS INT) AS put_vol,
            CAST(SUM(CASE WHEN type = 'Call' THEN openinterest ELSE 0 END) AS INT) AS call_oi, 
            CAST(SUM(CASE WHEN type = 'Put' THEN openinterest ELSE 0 END) AS INT) AS put_oi,
            CAST(AVG(CASE WHEN type = 'Call' THEN impliedvolatility ELSE 0 END) AS FLOAT) AS call_iv,
            CAST(AVG(CASE WHEN type = 'Put' THEN impliedvolatility ELSE 0 END) AS FLOAT) AS put_iv,
            CAST(AVG(CASE WHEN stk_price / strike BETWEEN 0.95 AND 1.05 THEN impliedvolatility ELSE 0 END) AS FLOAT) AS atm_iv, 
            CAST(AVG(CASE WHEN stk_price / strike NOT BETWEEN 0.95 AND 1.05 THEN impliedvolatility ELSE 0 END) AS FLOAT) AS otm_iv
            FROM {stock}
            GROUP BY datetime(gatherdate)
            ORDER BY gatherdate ASC
            '''
        return q

    def _cp(self, stock: str) -> pd.DataFrame:
        """
        Calculate the daily option stats for each stock: 
            args:
                stock: str: stock symbol
            returns:
                pd.DataFrame: DataFrame of the option chain
        """
        q = self._get_query_str(stock)
        try:
            # logging.info(f"DAILY OPTION STATS: Running _cp for {stock.upper()}")
            return self.__custom_query_option_db(q, self.option_db)
        except Exception as e:
            logging.error(f"DAILY OPTION STATS: Error in _cp for {stock}: {e}", exc_info=True)
            raise

    def _calculation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the change in the features observed above. 
            args:
                df: pd.DataFrame: DataFrame of the option chain
            returns:
                pd.DataFrame: DataFrame with the change variables appended to the original dataframe 
        """
        
        try:
            if 'gatherdate' in df.columns: 
                df['gatherdate'] = pd.to_datetime(df['gatherdate'])
                df['date'] = df['gatherdate'].dt.date

            df['call_vol_pct'] = df['call_vol'] / df['total_vol']
            df['put_vol_pct'] = df['put_vol'] / df['total_vol']
            df['call_oi_pct'] = df['call_oi'] / df['total_oi']
            df['put_oi_pct'] = df['put_oi'] / df['total_oi']
            # lagges
            # lag_df = df.groupby(df.tmp_date).last().diff()
            # lag_df = lag_df.replace(0, np.nan).ffill()
            # lag_df.columns = [f'{x}_chng' for x in lag_df.columns]

            # Join on the index 
            # df = df.join(lag_df, on='tmp_date')
            # df.drop(columns=['tmp_date'], inplace=True)
            # return df
            # Get the last data entry for each day 
            
            df = df.set_index('date')
            gbdf = df.groupby(df.index).last().reset_index(drop=True).set_index('gatherdate').diff()
            gbdf.columns = [ f'{col}_chng' for col in gbdf.columns]

            final_change = df.merge(gbdf, left_on = 'gatherdate', right_index=True).reset_index(drop=True)
            
            return final_change
        except sql.Error as e:
            logging.error(f"DAILY OPTION STATS: Error in _calculation: {e}", exc_info=True)
            raise
        except pd.errors.EmptyDataError as e:
            logging.error(f"DAILY OPTION STATS: Empty data error in _calculation: {e}", exc_info=True)
            raise
        except Exception as e:
            logging.error(f"DAILY OPTION STATS: Error in _calculation: {e}", exc_info=True)

        except Exception as e:
            logging.error(f"DAILY OPTION STATS: Error in _calculation: {e}", exc_info=True)
            raise

    def _recent(self, stock: str) -> pd.DataFrame:
        ''' Returns the last n rows of the stock table '''
        q = f'''
        SELECT 
            datetime(gatherdate) AS gatherdate, 
            call_vol, 
            put_vol, 
            total_vol, 
            call_oi, 
            put_oi, 
            total_oi, 
            total_prem,
            call_iv, 
            put_iv,
            atm_iv,
            otm_iv
        FROM {stock}
        ORDER BY datetime(gatherdate) ASC
        '''
        try:
            df = pd.read_sql_query(q, self.vol_db, parse_dates=['gatherdate']).sort_values('gatherdate', ascending=True)
            logging.info(f"DAILY OPTION STATS: Fetched recent data for {stock}")
            return df 
        except sql.Error as e:
            logging.error(f"DAILY OPTION STATS: Error fetching recent data for {stock}: {e}", exc_info=True)
            raise

    def _last_dates(self, stock: str, N: int = 5) -> np.ndarray:
        ''' Return the last dates for each day for a stock in the db '''
        q = f'''
        SELECT DISTINCT
            MAX(datetime(gatherdate)) OVER (PARTITION BY date(gatherdate)) AS gatherdate
        FROM {stock}
        ORDER BY gatherdate DESC LIMIT ?
        '''
        try:
            cursor = self.vol_db.cursor()
            cursor.execute(q, (N,))
            df = pd.DataFrame(cursor.fetchall(), columns=['gatherdate'])
            return df['gatherdate'].unique()
        except sql.Error as e:
            logging.error(f"DAILY OPTION STATS: Error fetching last dates for {stock}: {e}", exc_info=True)
            raise

    def update_cp(self, stock: str, new_chain: pd.DataFrame) -> Optional[pd.DataFrame]:
        ''' Updates the table for stock with data from the new option chain '''
        try:
            new_chain['moneyness'] = new_chain['stk_price'] / new_chain['strike']
            
            chk = len(self._last_dates(stock)) > 3
            if not chk:
                logging.warning(f"Not enough historical data for {stock}. Skipping update.")
                return None
            else:
                gdate = new_chain.gatherdate.iloc[0]
                old_chain = self._recent(stock)
                calls = new_chain[new_chain['type'] == 'Call']
                puts = new_chain[new_chain['type'] == 'Put']
                newest_cp = pd.DataFrame({
                'gatherdate': [gdate],
                'call_vol': [calls['volume'].sum()],
                'put_vol': [puts['volume'].sum()],
                'total_vol': [calls['volume'].sum() + puts['volume'].sum()],
                'call_oi': [calls['openinterest'].sum()],
                'put_oi': [puts['openinterest'].sum()],
                'total_oi': [calls['openinterest'].sum() + puts['openinterest'].sum()],
                'total_prem': [calls['cash'].sum() + puts['cash'].sum()], 
                'atm_iv': [new_chain[(new_chain['moneyness'] >= 0.99) & (new_chain['moneyness'] <= 1.01)]['impliedvolatility'].mean()],
                'otm_iv': [new_chain[(new_chain['moneyness'] < 0.99) | (new_chain['moneyness'] > 1.01)]['impliedvolatility'].mean()],
                'call_iv': [calls[calls['moneyness'].between(0.95, 1.05)]['impliedvolatility'].mean()],
                'put_iv': [puts[puts['moneyness'].between(0.95, 1.05)]['impliedvolatility'].mean()],
                })
                
                ready = pd.concat([old_chain, newest_cp], axis=0, ignore_index=True)
                add_on = self._calculation(ready).tail(1).reset_index(drop = True)

                add_on.to_sql(f'{stock}', self.vol_db, if_exists='append', index=False)
                self.vol_db.commit()
                # logging.info(f"DAILY OPTION STATS: Updated {stock} in vol_db")
                return pd.read_sql(f'select * from {stock}', self.vol_db)
        except Exception as e:
            logging.error(f"DAILY OPTION STATS: Error updating {stock}: {e}", exc_info=True)
            self.vol_db.rollback()
            raise

    def get_cp_from_purged_db(self, stock: str,  inactive_db: str = None) -> pd.DataFrame:
        try:
            # Check if the stock is in the inactive_db
            if inactive_db is None:
                inactive_db = self.inactive_db
            else:
                try:
                    inactive_db = sql.connect(inactive_db)
                    logging.info(f"Connected to inactive_db: {inactive_db}")
                    q = self._get_query_str(stock)
                    df = self.__custom_query_option_db(q, inactive_db)
                    logging.info(f"Daily OptionStats: {stock.upper()} Successful retrieval from inactive_db")
                    return self._calculation(df)
                except sql.Error as e:
                    logging.error(f"Daily OptionStats: Failed to connect to inactive_db: {e}", exc_info=True)
                    raise
        except Exception as e:
            logging.error(f"Daily OptionStats: Error getting CP from purged db for {stock}", exc_info=True)
            pass


    def cp_query(self, stock: str, inactive_db:str = None ) -> pd.DataFrame:
        if inactive_db != None:
            try:
                old_df = self.get_cp_from_purged_db(stock, inactive_db=inactive_db)
            except:
                old_df = pd.DataFrame()
        else:
            old_df = pd.DataFrame()
        try:
            current_df = self._calculation(self._cp(stock,))
            new_df = pd.concat([old_df, current_df], axis=0).reset_index()
            return new_df
        except Exception as e:
            logging.error(f"Daily OptionStats: Error in CP query for {stock}: {e}", exc_info=True)
            raise

    def merge_cp(self, stock: str, inactive_db: str = None) -> pd.DataFrame:
        try:
            old_df = self.get_cp_from_purged_db(stock, inactive_db=inactive_db)
            current_df = self._calculation(self._cp(stock))
            new_df = pd.concat([old_df, current_df], axis=0)
            return new_df
        except Exception as e:
            logging.error(f"Daily OptionStats: Error merging CP for {stock}: {e}", exc_info=True)
            raise


    def _intialized_cp(self, stock: str,  inactive_db:str = None) -> None:
        ''' Initializes the cp table '''
        try:
            old_df = self.get_cp_from_purged_db(stock,inactive_db = inactive_db)
        except:
            old_df = pd.DataFrame()
        try:
            current_df = self._calculation(self._cp(stock))
            new_df = pd.concat([old_df, current_df], axis=0).drop_duplicates()

            new_df.to_sql(f'{stock}', self.vol_db, if_exists='replace', index=False)
            self.vol_db.commit()
            # logging.info(f"Daily OptionStats: Initialized CP for {stock}")
        except Exception as e:
            logging.error(f"Daily OptionStats: Error initializing CP for {stock}: {e}", exc_info=True)
            self.vol_db.rollback()
            raise

    def _initialize_vol_db(self, stock: str) -> pd.DataFrame:
        ''' Builds the table for the stock 
        
        args:
            stock: str: stock symbol
        returns:
            pd.DataFrame: DataFrame of the stock table
        
        '''
        try:
            df = self._cp(stock)
            df = self._calculation(df)
            df.index = pd.to_datetime(df.index)
            # check to see if the stock is already in the database
            ########################################################
            if self._check_for_stock_in_vol_db(stock):
                logging.warning(f"{stock} already in vol_db. Appendng only new entries.")
                existing_df = pd.read_sql(f'select * from {stock}', self.vol_db, parse_dates=['gatherdate']).sort_values('gatherdate', ascending=True)
                # Drop any values for gatherdate that are NA
                existing_df = existing_df.dropna(subset=['gatherdate'])
                # If the existing_df is greater than the new df, then we need to append the new df to the existing df
                if existing_df.shape[0] > df.shape[0]:
                    df.to_sql(f'{stock}', self.vol_db, if_exists='append', index=False)
                else:
                    df.to_sql(f'{stock}', self.vol_db, if_exists='replace', index=False)
            self.vol_db.commit()
            logging.info(f"DAILY OPTION STATS: Initialized vol_db for {stock}")
            return df
        except Exception as e:
            logging.error(f"DAILY OPTION STATS: Error initializing vol_db for {stock}: {e}", exc_info=True)
            self.vol_db.rollback()
            raise
    
    def initialize_stocks(self, stocks: list, inactive_db: str = None) -> pd.DataFrame:
        ''' Initialize the stocks in the vol_db '''
        try:
            pbar = tqdm(stocks, desc="Initializing CP")
            for stock in pbar:
                pbar.set_description(f"Initializing CP for {stock}")
                self._intialized_cp(stock, inactive_db=inactive_db)
            logging.info("Daily OptionStats: Initialized all stocks in vol_db")
        except Exception as e:
            logging.error(f"Daily OptionStats: Error initializing stocks: {e}", exc_info=True)
            raise
        # finally:
        #     self.vol_db.commit()
        #     self.vol_db.close()
        #     self.dates_db.close()




if __name__ == "__main__":
    print("(10.4) Spiritual Intelligence, Knowledge, freedom from false perception, compassion, trufhfullness, control of the senses, control of the mind, happiness, unhappiness, birth, death, fear and fearlessness, nonviolence, equanimity,  contentment, austerity, charity, fame, infamy; all these variegated diverse qualities of all living entities originate from Me alone.")
    import sys 
    sys.path.append('/Users/jerald/Documents/Dir/Python/Stocks')
    from bin.main import get_path
    connections = get_path()
    print()
    cp = CP(connections)
    ib = '/Volumes/Backup Plus/options-backup/inactive.db'

    stocks = cp.all_stocks

    cp.initialize_stocks(stocks, inactive_db=ib)

    print(pd.read_sql('select * from amzn', cp.vol_db)[['gatherdate','call_iv', 'put_iv']])
    print(cp._cp('amzn')[['gatherdate','call_iv', 'put_iv']])
