import sys
import pandas as pd
import numpy as np
import yfinance as yf
import datetime as dt
from tqdm import tqdm
import logging

import sys 
sys.path.append('/Users/jerald/Documents/Dir/Python/Stocks')
    
# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from bin.options.optgd.db_connect import Connector

class Exp(Connector):
    def __init__(self, connections: dict):
        super().__init__(connections)

    def __next_friday(self) -> dt.datetime:
        """
        Calculate the date of the next Friday from the current date.

        Returns:
            dt.datetime: Date of next Friday.
        """
        today = dt.datetime.today()
        if today.weekday() == 4:  # 4 is Friday
            return today + dt.timedelta(7)
        else:
            return today + dt.timedelta((4 - today.weekday()) % 7)

    def _get_em_data(self, stock: str) -> pd.DataFrame:
        """
        Fetch option chain data from the database for the ATM options for a stock 

        Args:
            stock (str): The ticker symbol of the stock.

        Returns:
            pd.DataFrame: DataFrame containing option chain data.

        Raises:
            ValueError: If no data is returned for the stock.
        """
        query = f'''
        SELECT 
            type, 
            DATETIME(gatherdate) AS gatherdate,
            DATE(expiry) AS expiry, 
            strike, 
            stk_price, 
            ask, 
            bid, 
            lastprice
        FROM {stock}
        WHERE DATETIME(gatherdate) = (SELECT MAX(DATETIME(gatherdate)) FROM {stock})
        AND DATE(expiry) >= DATE('now')
        and (strike/stk_price) BETWEEN 0.95 AND 1.05
        ORDER BY 
            DATE(expiry) ASC
        '''
        df = pd.read_sql_query(query, self.option_db, parse_dates=['expiry', 'gatherdate'])
        if df.empty:
            logger.error(f"Expected Moves: No data found for stock: {stock}")
            raise ValueError(f"No data available for stock {stock}")
        return df

    def get_twt(self, em: pd.DataFrame) -> pd.DataFrame:
        """
        Generate a 'twt' column with formatted expected move information.

        Args:
            em (pd.DataFrame): DataFrame containing expected move data.

        Returns:
            pd.DataFrame: Updated DataFrame with 'twt' column.
        """
        em['flag'] = em['empct'] > 0.05
        em['twt'] = em.apply(lambda x: f'${x["stock"].upper()} ±{x["em"]:.2f} ({x["empct"]:.2%})', axis=1)
        em.loc[em.flag == True, 'twt'] += ' 🔥'
        return em.drop(columns='flag')

    def _em_ext(self, stock: str, df: pd.DataFrame = None) -> pd.DataFrame | None:
        """
        Calculate the expected move for the nearest expiration date.

        Args:
            stock (str): The ticker symbol of the stock.
            df (pd.DataFrame, optional): Option chain data. If None, it will be fetched.

        Returns:
            pd.DataFrame: DataFrame with expected move metrics or None if data insufficient.

        Raises:
            ValueError: If the DataFrame has fewer than 2 rows.
        """
        if df is None:
            df = self._get_em_data(stock)
        
        if len(df) < 2:
            logger.warning(f"Expected Moves: Insufficient data for {stock} to calculate expected move.")
            return None
        
        price_point = 'lastprice'
        itm = df.copy()
        odf = df.copy()
        
        call_strike = itm[(itm['type'] == 'Call') & (itm['strike'] < itm.stk_price)]['strike'].max()
        put_strike = itm[(itm['type'] == 'Put') & (itm['strike'] > itm.stk_price)]['strike'].min()
        
        cols = ['expiry', 'stk_price', 'type', 'strike', price_point]
        call_em = itm[(itm.strike == call_strike) & (itm.type == 'Call')][cols]
        put_em = itm[(itm.strike == put_strike) & (itm.type == 'Put')][cols]
        
        em = pd.concat([call_em, put_em]).groupby(['expiry']).agg({'stk_price': 'first', price_point: 'sum'}).reset_index()
        em.rename(columns={price_point: 'em'}, inplace=True)
        em['empct'] = (0.95 * em['em']) / em['stk_price']
        
        if 'stock' not in em.columns:
            em.insert(0, 'stock', stock)
        return em

    def _em(self, stock: str, new_chain: pd.DataFrame = None) -> pd.DataFrame:
        """
        Compute and format the expected move for a stock.

        Args:
            stock (str): The ticker symbol of the stock.
            new_chain (pd.DataFrame, optional): New option chain data if available.

        Returns:
            pd.DataFrame: DataFrame with formatted expected move data.
        """
        em = self._em_ext(stock, new_chain)
        return self.get_twt(em)

    def __edit_expected_moves_table(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter the expected moves table to include only dates up to the next Friday.

        Args:
            df (pd.DataFrame): DataFrame with expected move data.

        Returns:
            pd.DataFrame: Filtered DataFrame.
        """
        next_friday = self.__next_friday()
        return df[df.expiry <= next_friday]

    def _initialize_em_tables(self) -> None:
        """
        Initialize and populate the expected moves tables in the database.
        """
        out = []
        out_ext = []
        
        for stock in tqdm(self.stocks['all_stocks'], desc="Initializing Expected Move Table..."):
            try:
                d = self._em(stock)
                j = self._em_ext(stock)
                if d is not None:
                    if not ((d.empct == 0).any() or (d.em == 0).any()):
                        out.append(d)
                        out_ext.append(j)
            except Exception as e:
                logger.error(f"Expected moves: Error processing stock {stock}: {e}")

        out_df = pd.concat(out).reset_index(drop=True)
        out_df = self.__edit_expected_moves_table(out_df)
        # Commit the smaller Expected Moves table to the database
        out_df.to_sql('expected_moves', self.stats_db, if_exists='replace', index=False)
        # logger.info(f"Expected moves: Front Month Expected Moves Recorded in 'expected_moves' table.")
        # Commit the larger Expected Moves table to the database
        out_ext_df = pd.concat(out_ext).reset_index(drop=True)
        out_ext_df.to_sql('exp_ext', self.stats_db, if_exists='replace', index=False)
        # logger.info(f"Expected moves: Extended Expected moves recorded in 'exp_ext' table.")

    def gg(self) -> None:
        """
        Print the contents of 'exp_ext' and 'expected_moves' tables.
        exp_ext: Expected move data for all stocks, across multiple future expiration dates
        expected_moves: Expected move data for all stocks, up to the next Friday
        
        """
        # logger.info('Expected Moves: Querying exp_ext table')
        g = pd.read_sql('SELECT * FROM exp_ext', self.stats_db, parse_dates=['expiry'])
        print(g)
        # logger.info('Expected Moves: Querying expected_moves table')
        gg = pd.read_sql('SELECT * FROM expected_moves', self.stats_db, parse_dates=['expiry'])
        print(gg)

if __name__ == "__main__":
    print("Control what you can Control.")
    import sys 
    sys.path.append('/Users/jerald/Documents/Dir/Python/Stocks')
    from bin.main import get_path
    connections = get_path()
    oc = Exp(connections)
    oc._initialize_em_tables()
    # oc.gg()
    # oc.gg()