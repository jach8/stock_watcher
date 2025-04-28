import sys
from typing import Optional, Dict
import pandas as pd
import numpy as np
import sqlite3 as sql
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class OptionsDB:
    """
    Manages options data for stocks, computing features from option chains.
    
    Features:
        - call_vol, put_vol, total_vol: Call, put, and total option volumes.
        - call_oi, put_oi, total_oi: Call, put, and total open interest.
        - call_prem, put_prem, total_prem: Call, put, and total premiums (sum of lastprice).
        - call_iv, put_iv, atm_iv, otm_iv: Implied volatilities for calls, puts, ATM, and OTM.
        - call_spread, put_spread: Average bid-ask spreads for calls and puts.
        - iv_rank, call_iv_rank, put_iv_rank: 52-week IV ranks for total group.
        - atm_straddle: Sum of ATM call and put lastprices.
        - stk_price: Stock price.
        - dte_flag: DTE group (0DTE, STE, MTE, LTE, total).
        - Derived: call_vol_pct, put_vol_pct, call_oi_pct, put_oi_pct, and change metrics.
    
    Args:
        vol_db_path (str): Path to SQLite volatility database.
        option_db_path (str): Path to SQLite options database.
    """
    
    def __init__(self, connections: Optional[Dict[str, str]]):
        """Initialize database connections."""
        try:
            vol_db_path = connections['vol2_db']
            option_db_path = connections['option_db']
            if not vol_db_path or not option_db_path:
                raise ValueError("Database paths are required.")
            
            self.vol_db = sql.connect(vol_db_path)
            self.option_db = sql.connect(option_db_path)
        except sql.Error as e:
            logging.error(f"Failed to connect to databases: {e}")
            raise
        
        # Ensure indexes for performance
        # self._ensure_indexes()
    
    def _ensure_indexes(self):
        """Create indexes on stock tables for faster queries."""
        cursor = self.vol_db.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        for table in tables:
            try:
                cursor.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_{table}_gatherdate_dte
                    ON {table}(gatherdate, dte_flag)
                """)
            except sql.Error as e:
                logging.warning(f"Failed to create index for {table.upper()}: {e}")
        self.vol_db.commit()
    
    def _execute_query(self, query: str, connection: sql.Connection) -> pd.DataFrame:
        """Execute SQL query and return DataFrame."""
        try:
            df = pd.read_sql_query(query, connection, parse_dates=['gatherdate'])
            return df
        except sql.Error as e:
            logging.error(f"Query failed: {query[:50]}... {e}")
            raise
    
    def _get_query_str(self, stock: str) -> str:
        """Return SQL query for option chain metrics."""
        return f'''

            WITH step1 AS (
                SELECT 
                    datetime(gatherdate) AS gatherdate,
                    date(expiry) AS expiry,
                    strike,
                    stk_price,
                    strike / stk_price AS moneyness,
                    type,
                    lastprice, 
                    volume, 
                    openinterest, 
                    impliedvolatility,
                    ask,
                    bid,
                    julianday(date(expiry)) - julianday(date(gatherdate)) AS dte,
                    cash
                FROM {stock}
            ),
            t2 AS (
                SELECT 
                    *,
                    CASE
                        WHEN dte <= 1 THEN '0DTE'
                        WHEN dte between 1 and 8 THEN 'STE'
                        WHEN dte BETWEEN 8 AND 35 THEN 'MTE'
                        ELSE 'LTE'
                    END AS dte_flag
                FROM step1
                UNION ALL
                SELECT 
                    *,
                    'total' AS dte_flag
                FROM step1
            ),
            first_expiry AS (
                SELECT 
                    gatherdate,
                    MIN(expiry) AS first_expiry
                FROM step1
                GROUP BY gatherdate
            ),
            atm_strike AS (
                SELECT 
                    t2.gatherdate,
                    t2.expiry,
                    t2.dte_flag,
                    t2.stk_price,
                    t2.strike AS atm_strike,
                    ROW_NUMBER() OVER (
                        PARTITION BY t2.gatherdate, t2.expiry, t2.dte_flag 
                        ORDER BY ABS(t2.strike - t2.stk_price)
                    ) AS rn
                FROM t2
            ),
            total_atm_straddle AS (
                SELECT 
                    t2.gatherdate,
                    'total' AS dte_flag,
                    SUM(CASE 
                        WHEN t2.type = 'Call' AND t2.strike = atm.atm_strike THEN t2.lastprice 
                        ELSE 0 
                    END) + SUM(CASE 
                        WHEN t2.type = 'Put' AND t2.strike = atm.atm_strike THEN t2.lastprice 
                        ELSE 0 
                    END) AS atm_straddle
                FROM t2
                JOIN atm_strike atm
                    ON t2.gatherdate = atm.gatherdate
                    AND t2.expiry = atm.expiry
                    AND t2.dte_flag = atm.dte_flag
                    AND atm.rn = 1
                JOIN first_expiry fe
                    ON t2.gatherdate = fe.gatherdate
                    AND t2.expiry = fe.first_expiry
                WHERE t2.dte_flag = 'total'
                GROUP BY t2.gatherdate
            )
            SELECT 
                t2.gatherdate,
                t2.dte_flag,
                
                SUM(t2.volume) AS total_vol,
                SUM(t2.openinterest) AS total_oi,
                
                SUM(CASE WHEN t2.strike = atm.atm_strike THEN t2.volume ELSE 0 END) AS atm_vol,
                SUM(CASE WHEN t2.strike = atm.atm_strike THEN t2.openinterest ELSE 0 END) AS atm_oi,
                
                SUM(CASE WHEN t2.strike != atm.atm_strike THEN t2.volume ELSE 0 END) AS otm_vol,
                SUM(CASE WHEN t2.strike != atm.atm_strike THEN t2.openinterest ELSE 0 END) AS otm_oi,
                
                SUM(CASE WHEN t2.type = 'Call' THEN t2.volume ELSE 0 END) AS call_vol,
                SUM(CASE WHEN t2.type = 'Put' THEN t2.volume ELSE 0 END) AS put_vol,
                
                SUM(CASE WHEN t2.type = 'Call' AND t2.strike != atm.atm_strike AND t2.stk_price < t2.strike THEN t2.volume ELSE 0 END) AS otm_call_vol,
                SUM(CASE WHEN t2.type = 'Put' AND t2.strike != atm.atm_strike AND t2.stk_price > t2.strike THEN t2.volume ELSE 0 END) AS otm_put_vol,
                
                SUM(CASE WHEN t2.type = 'Call' THEN t2.openinterest ELSE 0 END) AS call_oi, 
                SUM(CASE WHEN t2.type = 'Put' THEN t2.openinterest ELSE 0 END) AS put_oi,
                
                SUM(CASE WHEN t2.type = 'Call' AND t2.strike != atm.atm_strike AND t2.stk_price < t2.strike THEN t2.openinterest ELSE 0 END) AS otm_call_oi,
                SUM(CASE WHEN t2.type = 'Put' AND t2.strike != atm.atm_strike AND t2.stk_price > t2.strike THEN t2.openinterest ELSE 0 END) AS otm_put_oi,
                
                AVG(CASE WHEN t2.type = 'Call' AND t2.strike = atm.atm_strike THEN t2.impliedvolatility ELSE NULL END) AS call_iv,
                AVG(CASE WHEN t2.type = 'Put' AND t2.strike = atm.atm_strike THEN t2.impliedvolatility ELSE NULL END) AS put_iv,
                
                AVG(CASE WHEN t2.strike = atm.atm_strike THEN t2.impliedvolatility ELSE NULL END) AS atm_iv, 
                AVG(CASE WHEN t2.strike != atm.atm_strike THEN t2.impliedvolatility ELSE NULL END) AS otm_iv,
                CASE 
                    WHEN t2.dte_flag = 'total' THEN tas.atm_straddle
                    ELSE SUM(CASE 
                        WHEN t2.type = 'Call' AND t2.strike = atm.atm_strike THEN t2.lastprice 
                        ELSE 0 
                    END) + SUM(CASE 
                        WHEN t2.type = 'Put' AND t2.strike = atm.atm_strike THEN t2.lastprice 
                        ELSE 0 
                    END)
                END AS atm_straddle,
                MAX(t2.stk_price) AS stk_price,
                SUM(CASE WHEN t2.type = 'Call' THEN t2.cash ELSE 0 END) AS call_prem,
                SUM(CASE WHEN t2.type = 'Put' THEN t2.cash ELSE 0 END) AS put_prem,
                
                SUM(t2.cash) AS total_prem,
                
                AVG(CASE WHEN t2.type = 'Call' THEN t2.ask - t2.bid ELSE NULL END) AS call_spread,
                AVG(CASE WHEN t2.type = 'Put' THEN t2.ask - t2.bid ELSE NULL END) AS put_spread,

                AVG(CASE WHEN t2.type = 'Put' AND t2.moneyness < 0.95 THEN t2.impliedvolatility ELSE NULL END) -
                AVG(CASE WHEN t2.type = 'Call' AND t2.moneyness > 1.05 THEN t2.impliedvolatility ELSE NULL END) AS vol_skew

            FROM t2 
            JOIN atm_strike atm
                ON t2.gatherdate = atm.gatherdate 
                AND t2.expiry = atm.expiry 
                AND t2.dte_flag = atm.dte_flag
                AND atm.rn = 1

            LEFT JOIN total_atm_straddle tas
                ON t2.gatherdate = tas.gatherdate AND t2.dte_flag = tas.dte_flag
            GROUP BY t2.gatherdate, t2.dte_flag
            ORDER BY t2.gatherdate ASC, 
                    CASE 
                        WHEN t2.dte_flag = 'total' THEN 1 
                        ELSE 0 
                    END ASC, 
                    t2.dte_flag ASC;
        '''
    
    def calculate_change_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        """Example:
        
            tdf = df[non_change_columns].sort_index().copy().reset_index()
            tdf['date'] = tdf.gatherdate.dt.date

            #### GET THE CHANGES PER DATE AND DTE_FLAG
            ## Get max datetime per day 
            tdf = tdf.groupby(['date', 'dte_flag']).last().reset_index().drop(columns = ['gatherdate'])

            groups = tdf.dte_flag.unique()
            lodf = []

            for group in groups:
                # calcualte .diff per group
                out = tdf[tdf.dte_flag == group].set_index(['date', 'dte_flag']).diff().iloc[:]
                out.columns = [ f'{col}_chng' for col in out.columns ]
                lodf.append(out) # drop the first row since it is NaN


            change_df = pd.concat(lodf).sort_index()
        """

        non_change_columns = df.columns[~df.columns.str.contains('_chng')]
        # Keep only numeric non-change columns
        non_change_columns = list(df[non_change_columns].select_dtypes(include="number").columns)
        # Get the max datetime per day
        df = df[non_change_columns].sort_index().copy().reset_index()
        df['date'] = df.gatherdate.dt.date
        tdf = df.groupby(['date', 'dte_flag']).last().reset_index().drop(columns=['date'])
        lodf = []
        groups = tdf.dte_flag.unique()
        for group in groups:
            # Calculate .diff per group
            out = tdf[tdf.dte_flag == group].set_index(['gatherdate', 'dte_flag']).sort_index().diff().iloc[:]
            out.columns = [f'{col}_chng' for col in out.columns]
            lodf.append(out)
        change_df = pd.concat(lodf).sort_index()
        return change_df
    
    def _calculate_derived_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add percentage and change metrics to DataFrame."""
        df = df.copy()
        df['gatherdate'] = pd.to_datetime(df['gatherdate'])
        df = df.set_index(['gatherdate', 'dte_flag'])
        
        # Percentage metrics 
        df['call_vol_pct'] = df['call_vol'] / df['total_vol'].replace(0, np.nan)
        df['put_vol_pct'] = df['put_vol'] / df['total_vol'].replace(0, np.nan)
        df['call_oi_pct'] = df['call_oi'] / df['total_oi'].replace(0, np.nan)
        df['put_oi_pct'] = df['put_oi'] / df['total_oi'].replace(0, np.nan)
        
        # Change metrics
        # Check if change columns already exist
        if list(df.filter(regex='_chng')):
            df = df.drop(columns=list(df.filter(regex='_chng')))

        change_cols = [f'{col}_chng' for col in df.columns if '_chng' not in col]

        lag_df = self.calculate_change_cols(df)
        
    
        df = df.join(lag_df, how= 'inner')

        return df.reset_index()
    
    def _calculate_iv_ranks(self, df: pd.DataFrame, latest_date: pd.Timestamp) -> pd.DataFrame:
        """Add 52-week IV ranks for total group."""
        df = df.copy().set_index(['gatherdate', 'dte_flag'])
        
        # total_df = df[df['dte_flag'] == 'total'].copy()
        total_df = df.copy()
        
        if total_df.empty:
            df[['iv_rank', 'call_iv_rank', 'put_iv_rank']] = np.nan
            return df
        
        one_year_ago = latest_date - pd.Timedelta(days=365)
        hist_data = total_df[['call_iv', 'put_iv', 'atm_iv']]
        
        def compute_rank(series):
            if len(series) > 1 and not pd.isna(series.iloc[-1]):
                return series.rank(pct=True).iloc[-1] * 100  # Scale to 0-100
            return np.nan
        
        if not hist_data.empty:
            ranks = {
                'call_iv_rank': [],
                'put_iv_rank': [],
                'iv_rank': []
            }
            # sequentially calculate the ranks
            for i in range(len(hist_data)):
                call_iv_rank = compute_rank(hist_data['call_iv'].iloc[:i+1])
                put_iv_rank = compute_rank(hist_data['put_iv'].iloc[:i+1])
                iv_rank = compute_rank(hist_data['atm_iv'].iloc[:i+1])
                
                ranks['call_iv_rank'].append(call_iv_rank)
                ranks['put_iv_rank'].append(put_iv_rank)
                ranks['iv_rank'].append(iv_rank)

            for col in ranks:
                ranks[col] = pd.Series(ranks[col], index=hist_data.index)
        else:
            ranks = {'call_iv_rank': np.nan, 'put_iv_rank': np.nan, 'iv_rank': np.nan}

        df['call_iv_rank'] = ranks['call_iv_rank']
        df['put_iv_rank'] = ranks['put_iv_rank']
        df['iv_rank'] = ranks['iv_rank']
        return df.reset_index()
    
    def get_stock_metrics(self, stock: str, inactive_db_path: Optional[str] = None) -> pd.DataFrame:
        """
        Retrieve and compute metrics for a stock from option_db and optional inactive_db.
        
        Args:
            stock (str): Stock symbol.
            inactive_db_path (Optional[str]): Path to inactive database for historical data.
        
        Returns:
            pd.DataFrame: Metrics with derived columns and IV ranks.
        """
        try:
            # Query option_db
            query = self._get_query_str(stock)
            df = self._execute_query(query, self.option_db)
            
            # Merge with inactive_db if provided
            if inactive_db_path:
                try:
                    inactive_db = sql.connect(inactive_db_path)
                    inactive_df = self._execute_query(query, inactive_db)
                    df = pd.concat([inactive_df, df]).drop_duplicates(subset=['gatherdate', 'dte_flag'])
                    inactive_db.close()
                except sql.Error as e:
                    logging.warning(f"Could not access inactive_db {inactive_db_path}: {e}")
            
            if df.empty:
                logging.warning(f"No data found for {stock}")
                return pd.DataFrame()
            
            # Add derived metrics and IV ranks
            df = self._calculate_iv_ranks(df, df['gatherdate'].max())
            df = self._calculate_derived_metrics(df)
            return df.sort_values(['gatherdate', 'dte_flag'])
        
        except Exception as e:
            logging.error(f"Failed to get metrics for {stock}: {e}")
            raise
    
    def update_stock_metrics(self, stock: str, new_chain: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Update vol_db with new option chain data for a stock.
        
        Args:
            stock (str): Stock symbol.
            new_chain (pd.DataFrame): New option chain data with columns:
                gatherdate, expiry, strike, stk_price, type, lastprice, volume,
                openinterest, impliedvolatility, bid, ask.
        
        Returns:
            Optional[pd.DataFrame]: Updated metrics table or None if skipped.
        """
        try:
            # Validate input
            required_cols = {'gatherdate', 'expiry', 'strike', 'stk_price', 'type',
                            'lastprice', 'volume', 'openinterest', 'impliedvolatility',
                            'bid', 'ask'}
            if not required_cols.issubset(new_chain.columns):
                missing = required_cols - set(new_chain.columns)
                logging.error(f"Missing columns for {stock}: {missing}")
                return None
            
            new_chain = new_chain.copy()
            new_chain['gatherdate'] = pd.to_datetime(new_chain['gatherdate'])
            
            if new_chain.empty:
                logging.warning(f"Empty new_chain for {stock}")
                return self.get_stock_metrics(stock)
            
            # Check historical data
            cursor = self.vol_db.cursor()
            cursor.execute(f"SELECT COUNT(DISTINCT date(gatherdate)) FROM {stock}")
            date_count = cursor.fetchone()[0] or 0
            if date_count <= 3:
                logging.warning(f"Not enough historical data for {stock}: {date_count} days")
                return None
            
            # Create temporary table
            temp_table = f'temp_{stock}_{int(pd.Timestamp.now().timestamp())}'
            new_chain.to_sql(temp_table, self.vol_db, if_exists='replace', index=False)
            
            # Compute metrics
            query = self._get_query_str(temp_table)
            new_data = self._execute_query(query, self.vol_db)
            
            # Drop temporary table
            self.vol_db.execute(f"DROP TABLE IF EXISTS {temp_table}")
            
            if new_data.empty:
                logging.warning(f"No metrics computed for {stock}")
                return self.get_stock_metrics(stock)
            
            # Add IV ranks
            new_data = self._calculate_iv_ranks(new_data, new_data['gatherdate'].max())
            
            
            # Check for duplicates
            existing_data = pd.read_sql(
                f"SELECT gatherdate, dte_flag FROM {stock}",
                self.vol_db, parse_dates=['gatherdate']
            )
            new_data = new_data[
                ~new_data[['gatherdate', 'dte_flag']].apply(tuple, axis=1).isin(
                    existing_data[['gatherdate', 'dte_flag']].apply(tuple, axis=1)
                )
            ]
            
            if new_data.empty:
                logging.warning(f"No new data to append for {stock}")
                return self.get_stock_metrics(stock)
            
            # Append new data
            new_data.to_sql(stock, self.vol_db, if_exists='append', index=False)
            
            # Fetch and recalculate derived metrics
            updated_df = self.get_stock_metrics(stock)
            self.vol_db.commit()
            
            logging.info(f"Updated metrics for {stock}")
            return updated_df
        
        except Exception as e:
            logging.error(f"Failed to update {stock}: {e}")
            self.vol_db.rollback()
            self.vol_db.execute(f"DROP TABLE IF EXISTS {temp_table}")
            return None
    
    def initialize_stock_table(self, stock: str, inactive_db_path: Optional[str] = None) -> pd.DataFrame:
        """
        Initialize or update vol_db table for a stock, ensuring schema includes new columns.
        
        Args:
            stock (str): Stock symbol.
            inactive_db_path (Optional[str]): Path to inactive database.
        
        Returns:
            pd.DataFrame: Initialized table.
        """
        df = self.get_stock_metrics(stock, inactive_db_path)
        if df.empty:
            logging.warning(f"No data to initialize {stock}")
            return pd.DataFrame()
        
        # Ensure table schema
        cursor = self.vol_db.cursor()
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {stock} (
                gatherdate TEXT,
                dte_flag TEXT,
                total_vol REAL,
                total_oi REAL,
                atm_vol REAL,
                atm_oi REAL,
                otm_vol REAL,
                otm_oi REAL,
                call_vol REAL,
                put_vol REAL,
                call_oi REAL,
                put_oi REAL,
                call_iv REAL,
                put_iv REAL,
                atm_iv REAL,
                otm_iv REAL,
                atm_straddle REAL,
                stk_price REAL,
                call_prem REAL,
                put_prem REAL,
                total_prem REAL,
                call_spread REAL,
                put_spread REAL,
                vol_skew REAL,
                call_iv_rank REAL,
                put_iv_rank REAL,
                iv_rank REAL
            )
        """)
        try:
            cursor.execute(f"ALTER TABLE {stock} ADD COLUMN vol_skew REAL")
            cursor.execute(f"ALTER TABLE {stock} ADD COLUMN call_iv_rank REAL")
            cursor.execute(f"ALTER TABLE {stock} ADD COLUMN put_iv_rank REAL")
            cursor.execute(f"ALTER TABLE {stock} ADD COLUMN iv_rank REAL")
        except sql.Error:
            pass  # Columns may already exist
        
        df.to_sql(stock, self.vol_db, if_exists='replace', index=False)
        # self._ensure_indexes()
        self.vol_db.commit()
        
        logging.info(f"Initialized table for {stock}")
        return df
            
    def close(self):
        """Close database connections."""
        try:
            self.vol_db.close()
            self.option_db.close()
        except sql.Error as e:
            logging.error(f"Failed to close databases: {e}")

    def test_import(self, stock):
        return pd.read_sql(f'select * from {stock}', self.vol_db, parse_dates=['gatherdate'])
    
    def recalculate_metrics(self, stocks):
        """
        Recalculate metrics for a list of stocks.
        
        Args:
            stocks (list): List of stock symbols.
        
        Returns:
            pd.DataFrame: DataFrame containing metrics for all stocks.
        """
        if isinstance(stocks, str):
            stocks = [stocks]
        
        lodf = []
        pbar = tqdm(stocks, desc="Recalculating Metrics")
        for stock in pbar:
            pbar.set_description(f"Recalculating Metrics ${stock.upper()}")
            try:
                df = self.test_import(stock)
                # df = self._calculate_iv_ranks(df, df['gatherdate'].max())
                df = self._calculate_derived_metrics(df)
                if 'stock' not in df.columns: 
                    df.insert(0, 'stock', stock.upper())
                else:
                    df['stock'] = stock.upper()
                df.to_sql(stock, self.vol_db, if_exists='replace', index=False)
                lodf.append(df)
            except Exception as e:
                logging.error(f"Failed to recalculate metrics for {stock}: {e}")
                continue
        
        return pd.concat(lodf) if lodf else pd.DataFrame()



if __name__ == "__main__":
    from tqdm import tqdm 
    import sys 
    sys.path.append('/Users/jerald/Documents/Dir/Python/Stocks')
    from bin.main import get_path
    import json 
    connections = get_path()

    stocks = json.load(open(connections['ticker_path']))['all_stocks']
    # vol_db_path = 'data/options/vol.db'
    db = OptionsDB(connections)
    
    # # Example: Initialize and update
    # inactive_db = '/Volumes/Backup Plus/options-backup/inactive.db'
    
    # for stock in tqdm(stocks, desc="Initializing Stocks"): 
    #     try:
    #         db.initialize_stock_table(stock, inactive_db_path = inactive_db)
    #     except Exception as e:
    #         logging.error(f"Failed to initialize {stock}: {e}")

    df = db.recalculate_metrics(stocks)
    df = db.test_import('amd')
    print(df)
    db.close()

