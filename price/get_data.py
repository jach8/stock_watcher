from typing import Dict, List, Optional
import pandas as pd 
import numpy as np
import sqlite3 as sql
import yfinance as yf
import pandas_datareader.data as web
from datetime import datetime
import json
import logging
from pathlib import Path
from contextlib import contextmanager

# Configure logging
try:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/stock_updates.log')
        ]
    )
except FileNotFoundError:
    pass
logger = logging.getLogger(__name__)

class DatabaseConnectionError(Exception):
    """Custom exception for database connection errors"""
    pass

class StockDataError(Exception):
    """Custom exception for stock data retrieval/processing errors"""
    pass

@contextmanager
def database_connection(db_path: str):
    """Context manager for database connections"""
    conn = None
    try:
        conn = sql.connect(db_path)
        logger.debug(f"Connected to database: {db_path}")
        yield conn
    except sql.Error as e:
        logger.error(f"Database connection error: {e}")
        raise DatabaseConnectionError(f"Failed to connect to database {db_path}: {e}")
    finally:
        if conn:
            conn.close()
            logger.debug(f"Closed connection to database: {db_path}")

class UpdateStocks:
    def __init__(self, connections: Dict[str, str]) -> None:
        """
        Initialize the UpdateStocks class.
        
        Args:
            connections: Dictionary containing database and file paths
                Required keys: 'daily_db', 'intraday_db', 'ticker_path'
        """
        self.validate_connections(connections)
        self.stocks_db: str = connections['daily_db']
        self.stocks_intraday_db: str = connections['intraday_db']
        self.ticker_path: str = connections['ticker_path']

    @staticmethod
    def validate_connections(connections: Dict[str, str]) -> None:
        """Validate the connections dictionary"""
        required_keys = {'daily_db', 'intraday_db', 'ticker_path'}
        if missing_keys := required_keys - set(connections.keys()):
            raise ValueError(f"Missing required connection keys: {missing_keys}")
        
        # Validate paths exist
        for key, path in connections.items():
            if not Path(path).parent.exists():
                raise ValueError(f"Directory for {key} does not exist: {Path(path).parent}")

    def stock_names(self) -> List[str]:
        """
        Get list of stock symbols from ticker file.
        
        Returns:
            List of stock symbols
        
        Raises:
            FileNotFoundError: If ticker file is not found
            json.JSONDecodeError: If ticker file is invalid JSON
        """
        try:
            with open(self.ticker_path, 'r') as f:
                stocks = json.load(f)
            return stocks['all_stocks']
        except FileNotFoundError:
            logger.error(f"Ticker file not found: {self.ticker_path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in ticker file: {e}")
            raise

    def update_stocks(self) -> None:
        """
        Update daily stock data in database.
        
        Raises:
            DatabaseConnectionError: If database connection fails
            StockDataError: If stock data retrieval/processing fails
        """
        try:
            with database_connection(self.stocks_db) as conn:
                logger.info('Connected to daily database')
                stocks = self.stock_names()
                stock_symbols = ' '.join(stocks)
                
                # Get latest date from database
                query = 'SELECT date(max(Date)) FROM spy'
                latest_date = pd.read_sql_query(query, conn).iloc[0][0]
                
                logger.debug(f"Fetching data for {len(stocks)} stocks from {latest_date}")
                data = yf.download(stock_symbols, start="1990-01-01")
                if data.empty:
                    raise StockDataError("No data retrieved from Yahoo Finance")
                
                # Process data
                data = data.swaplevel(0, 1, axis=1).sort_index(axis=1)
                stocks_upper = [s.upper() for s in stocks]
                
                # Create dictionary of dataframes per stock
                stock_data = {
                    s: data[s].drop_duplicates() 
                    for s in stocks_upper
                }
                
                # Update database for each stock
                for symbol, stock_df in stock_data.items():
                    try:
                        clean_df = stock_df[~stock_df.index.duplicated(keep='last')].dropna()
                        clean_df.to_sql(symbol, con=conn, if_exists='replace')
                        logger.debug(f"Updated daily data for {symbol}")
                    except Exception as e:
                        logger.error(f"Error updating {symbol}: {e}")
                        continue
                        
                logger.info('Daily data update completed successfully')
                
        except Exception as e:
            logger.error(f"Failed to update daily stock data: {e}")
            raise StockDataError(f"Daily stock update failed: {e}")

    def update_stocks_intraday(self) -> None:
        """
        Update intraday stock data in database.
        
        Raises:
            DatabaseConnectionError: If database connection fails
            StockDataError: If stock data retrieval/processing fails
        """
        try:
            stocks = self.stock_names()
            stock_symbols = ' '.join(stocks)
            
            with database_connection(self.stocks_intraday_db) as conn:
                logger.info('Connected to Intraday database')
                
                data = yf.download(stock_symbols, period='5d', interval='1m')
                if data.empty:
                    raise StockDataError("No intraday data retrieved from Yahoo Finance")
                
                # Process data
                data = data.swaplevel(0, 1, axis=1).sort_index(axis=1)
                data.index = [str(x).split('-04:00')[0] for x in data.index]
                stocks_upper = [s.upper() for s in stocks]
                
                # Create dictionary of dataframes per stock
                stock_data = {
                    s: data[s].drop_duplicates() 
                    for s in stocks_upper
                }
                
                # Update database for each stock
                for idx, symbol in enumerate(stocks):
                    try:
                        db_add = stock_data[stocks_upper[idx]].copy()
                        db_add = db_add[~db_add.index.duplicated(keep='last')].dropna()
                        db_add = db_add.reset_index()
                        db_add.rename(columns={'index': 'Date'}, inplace=True)
                        
                        # Clean datetime format
                        db_add['Date'] = (db_add['Date'].str[:19].str.replace('T', ' ').pipe(pd.to_datetime))
                        
                        db_add = db_add.drop_duplicates()
                        db_add.to_sql(stocks_upper[idx], con=conn, 
                                    if_exists='append', index=False)
                        logger.debug(f"Updated intraday data for {symbol}")
                    except Exception as e:
                        logger.error(f"Error updating intraday data for {symbol}: {e}")
                        continue
                
                logger.info('Intraday data update completed successfully')
                
        except Exception as e:
            logger.error(f"Failed to update intraday stock data: {e}")
            raise StockDataError(f"Intraday stock update failed: {e}")

    def update(self) -> None:
        """Update both daily and intraday stock data"""
        try:
            self.update_stocks()
            self.update_stocks_intraday()
            logger.info("Completed full stock data update")
        except Exception as e:
            logger.error(f"Failed to complete full update: {e}")
            raise

if __name__ == '__main__':
    try:
        logger.info("Starting Stock Price Database update...")
        
        connections = {
            'daily_db': 'data/prices/stocks.db',
            'intraday_db': 'data/prices/stocks_intraday.db',
            'ticker_path': 'data/stocks/tickers.json'
        }
        
        price_update = UpdateStocks(connections)
        price_update.update()
        logger.info("Stock Price Database update completed successfully")
        
    except Exception as e:
        logger.error(f"Stock Price Database update failed: {e}")
        raise