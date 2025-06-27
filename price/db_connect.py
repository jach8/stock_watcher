from typing import Dict, List, Optional, Union, Any, Tuple
import numpy as np 
import pandas as pd 
import sqlite3 as sql 
import datetime as dt 
from tqdm import tqdm
import time
import json 
import logging
from logging.handlers import RotatingFileHandler
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))
from bin.price.indicators import Indicators
from bin.price.get_data import UpdateStocks
from bin.price.technicals.utils import combine_timeframes
from bin.price.models.simulations.prices import optimize_and_simulate

# Custom exceptions
class DatabaseConnectionError(Exception):
    """Raised when database connection fails"""
    pass

class QueryExecutionError(Exception):
    """Raised when SQL query execution fails"""
    pass

class InvalidParameterError(Exception):
    """Raised when invalid parameters are provided"""
    pass

# Configure logging
def setup_logger(name: str) -> logging.Logger:
    """Configure logger with console handler only"""
    logger = logging.getLogger(name)
    
    # Only add handler if none exist
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        try:
            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)
        except Exception as e:
            print(f"Failed to set up logger: {str(e)}")
            # Return a basic logger if setup fails
            basic_logger = logging.getLogger(f"{name}_basic")
            basic_logger.addHandler(logging.StreamHandler())
            return basic_logger
            
    return logger

logger = setup_logger(__name__)

class Prices(UpdateStocks):
    """Class for managing stock price database connections and queries"""
    
    def __enter__(self):
        """Enable context management support"""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Ensure connections are closed when exiting context"""
        self.close_connections()
    
    def _cleanup_connections(self) -> None:
        """Clean up any open database connections"""
        for conn in [getattr(self, 'names_db', None),
                    getattr(self, 'daily_db', None),
                    getattr(self, 'intraday_db', None)]:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass

    def __init__(self, connections: Dict[str, str], timeout: int = 30) -> None:
        """
        Initialize price database connections
        
        Args:
            connections: Dictionary containing database connection paths
            timeout: Database connection timeout in seconds
        
        Raises:
            DatabaseConnectionError: If database connection fails
            FileNotFoundError: If required files are not found

        Methods:
            - update_stock_prices: Update stock prices in the database
            - custom_q: Execute a custom SQL query on the daily database
            - _get1minCl: Retrieve 1-minute close prices for a specified stock
            - get_intraday_close: Retrieve intraday closing prices for multiple stocks
            - close_connections: Close all database connections
            


        """
        super().__init__(connections)
        self.execution_start_time = time.time()
        self.timeout = timeout
        
        try:
            # Validate connection parameters
            required_keys = ['stock_names', 'daily_db', 'intraday_db', 'ticker_path']
            if not all(key in connections for key in required_keys):
                raise InvalidParameterError(f"Missing required connection parameters: {required_keys}")
            
            # Validate file existence
            for key, path in connections.items():
                if not Path(path).exists():
                    raise FileNotFoundError(f"File not found: {path} for {key}")
            
            # Establish database connections with timeout
            self.names_db = self._connect_db(connections['stock_names'])
            self.daily_db = self._connect_db(connections['daily_db'])
            self.intraday_db = self._connect_db(connections['intraday_db'])
            
            # Load ticker data
            try:
                with open(connections['ticker_path'], 'r') as f:
                    self.stocks = json.load(f)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse ticker JSON file: {e}")
                self._cleanup_connections()  # Clean up before re-raising
                raise
                
            logger.info(f"PriceDB Initialized successfully at {dt.datetime.now()}")
            logger.info("Established 3 database connections")
            
        except (sql.Error, FileNotFoundError) as e:
            self._cleanup_connections()  # Clean up any open connections
            error_msg = f"Initialization failed: {str(e)}"
            logger.error(error_msg)
            raise DatabaseConnectionError(error_msg) from e
    
    def _connect_db(self, db_path: str) -> sql.Connection:
        """
        Establish database connection with timeout and error handling
        
        Args:
            db_path: Path to SQLite database
            
        Returns:
            SQLite connection object
            
        Raises:
            DatabaseConnectionError: If connection fails
        """
        try:
            connection = sql.connect(db_path, timeout=self.timeout)
            connection.execute("SELECT 1")  # Test connection
            return connection
        except sql.Error as e:
            error_msg = f"Failed to connect to database {db_path}: {str(e)}"
            logger.error(error_msg)
            raise DatabaseConnectionError(error_msg) from e
            
    def update_stock_prices(self) -> None:
        """
        Update stock prices in the database.

        This method triggers the stock price update process inherited from UpdateStocks class.
        It refreshes all stock price data in both daily and intraday databases.

        Raises:
            Exception: If any error occurs during the update process
        """
        logger.info('Starting stock price update')
        try:
            self.update()
            logger.info('Successfully updated stock prices')
        except Exception as e:
            logger.error(f'Failed to update stock prices: {str(e)}')
            raise
            
    def custom_q(self, q: str) -> pd.DataFrame:
        """
        Execute a custom query on the daily_db
        
        Args:
            q: SQL query string
            
        Returns:
            DataFrame containing query results
            
        Raises:
            QueryExecutionError: If query execution fails
        """
        try:
            cursor = self.daily_db.cursor()
            cursor.execute(q)
            results = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            return pd.DataFrame(results, columns=columns)
        except sql.Error as e:
            error_msg = f"Query execution failed: {str(e)}\nQuery: {q}"
            logger.error(error_msg)
            raise QueryExecutionError(error_msg) from e
    
    def _get1minCl(self, stock: str, agg: str = '1min') -> pd.DataFrame:
        """
        Retrieve 1-minute close prices for a specified stock.

        This internal method fetches 1-minute interval closing prices from the intraday
        database and optionally aggregates them to a different time interval.

        Args:
            stock (str): The stock symbol to retrieve prices for
            agg (str, optional): Aggregation interval for resampling. Defaults to '1min'.
                               Common values: '1min', '5min', '15min', '1H'

        Returns:
            pd.DataFrame: DataFrame with datetime index and closing prices.
                        Column name is the stock symbol.

        Raises:
            sqlite3.Error: If there's an error executing the database query
            pd.errors.EmptyDataError: If no data is found for the stock
        """
        try:
            q = f'''select datetime(date) as date, close from {stock} order by datetime(date) asc'''
            cursor = self.intraday_db.cursor()
            cursor.execute(q)
            df = pd.DataFrame(cursor.fetchall(), columns=['date', stock])
            df.date = pd.to_datetime(df.date)
            df = df.set_index('date')
            if agg != '1min':
                df = df.resample(agg).last()
            return df
        except (sql.Error, pd.errors.EmptyDataError) as e:
            logger.error(f"Failed to get 1-minute close for {stock}: {str(e)}")
            raise
    
    def get_intraday_close(self, stocks: List[str], agg: str = '1min') -> pd.DataFrame:
        """
        Retrieve intraday closing prices for multiple stocks with optional time aggregation.

        This method fetches intraday closing prices for a list of stocks and allows
        for resampling the data to different time intervals. The resulting DataFrame
        contains columns for each stock with their respective closing prices.

        Args:
            stocks (List[str]): List of stock symbols to retrieve prices for
            agg (str, optional): Time aggregation interval. Defaults to '1min'.
                               Common values: '1min', '5min', '15min', '1H'

        Returns:
            pd.DataFrame: DataFrame with datetime index and columns for each stock's
                        closing prices. Each column is named after its stock symbol.

        Raises:
            InvalidParameterError: If stocks parameter is not a list
            Exception: If there's an error fetching or processing the data

        Example:
            >>> prices = db.get_intraday_close(['AAPL', 'MSFT'], agg='5min')
            >>> print(prices.head())
                               AAPL    MSFT
            2024-02-11 09:30  188.25  401.50
            2024-02-11 09:35  188.30  401.75
            ...
        """
        if not isinstance(stocks, list):
            raise InvalidParameterError("Input must be a list of stocks")
            
        try:
            out = [self._get1minCl(stock) for stock in stocks]
            out = [i.resample(agg).last() for i in out]
            return pd.concat(out, axis=1)
        except Exception as e:
            logger.error(f"Failed to get intraday close prices: {str(e)}")
            raise

    def _getClose(self, stock: str) -> pd.DataFrame:
        """
        Retrieve daily closing prices for a specified stock.

        This internal method fetches daily closing prices from the daily database
        and returns them as a DataFrame with a datetime index.

        Args:
            stock (str): The stock symbol to retrieve prices for

        Returns:
            pd.DataFrame: DataFrame with datetime index and closing prices.
                        The column is named after the stock symbol.

        Raises:
            sqlite3.Error: If there's an error executing the database query
            pd.errors.DatabaseError: If there's an error creating the DataFrame
        """
        try:
            q = f'''select date(date) as date, close as "Close" from {stock} order by date(date) asc'''
            df = pd.read_sql_query(q, self.daily_db, parse_dates=['date'], index_col='date')
            return df.rename(columns={'Close': stock})
        except (sql.Error, pd.errors.DatabaseError) as e:
            logger.error(f"Failed to get close prices for {stock}: {str(e)}")
            raise

    def get_close(self, stocks: List[str], start: Optional[str] = None, end: Optional[str] = None) -> pd.DataFrame:
        """
        Retrieve daily closing prices for multiple stocks with optional date filtering.

        This method fetches daily closing prices for a list of stocks and allows filtering
        the data by start and end dates. The resulting DataFrame contains columns for
        each stock's closing prices.

        Args:
            stocks (List[str]): List of stock symbols to retrieve prices for
            start (Optional[str], optional): Start date for filtering in 'YYYY-MM-DD' format
            end (Optional[str], optional): End date for filtering in 'YYYY-MM-DD' format

        Returns:
            pd.DataFrame: DataFrame with datetime index and columns for each stock's
                        closing prices. Each column is named after its stock symbol.

        Raises:
            InvalidParameterError: If stocks parameter is not a list
            Exception: If there's an error fetching or processing the data

        Example:
            >>> prices = db.get_close(['AAPL', 'MSFT'], start='2024-01-01', end='2024-02-01')
            >>> print(prices.head())
                          AAPL    MSFT
            2024-01-01  185.75  376.04
            2024-01-02  185.85  377.50
            ...
        """
        if not isinstance(stocks, list):
            raise InvalidParameterError("Input must be a list of stocks")
            
        try:
            out = [self._getClose(stock) for stock in stocks]
            df = pd.concat(out, axis=1)
            
            if start is not None:
                df = df[df.index >= start]
            if end is not None:
                df = df[df.index <= end]
                
            return df
        except Exception as e:
            logger.error(f"Failed to get close prices: {str(e)}")
            raise

    def ohlc(self, stock: str, daily: bool = True, start: Optional[str] = None, end: Optional[str] = None) -> pd.DataFrame:
        """
        Retrieve OHLCV (Open, High, Low, Close, Volume) data for a stock.

        This method fetches either daily or intraday OHLCV data for a specified stock,
        with optional date filtering. It can retrieve data from either the daily or
        intraday database based on the 'daily' parameter.

        Args:
            stock (str): The stock symbol to retrieve data for
            daily (bool, optional): If True, fetches daily data; if False, fetches intraday data. Defaults to True.
            start (Optional[str], optional): Start date for filtering in 'YYYY-MM-DD' format
            end (Optional[str], optional): End date for filtering in 'YYYY-MM-DD' format

        Returns:
            pd.DataFrame: DataFrame with datetime index and columns for OHLCV data:
                        - open: Opening price
                        - high: Highest price
                        - low: Lowest price
                        - close: Closing price
                        - volume: Trading volume

        Raises:
            Exception: If there's an error fetching or processing the data

        Example:
            >>> # Get daily OHLCV data for AAPL
            >>> daily_data = db.ohlc('AAPL', daily=True, start='2024-01-01')
            >>> print(daily_data.head())
                          open    high     low   close     volume
            2024-01-01  185.0   186.5   184.5   185.75   5000000
            2024-01-02  186.0   187.2   185.8   186.50   6200000
            ...
        """
        try:
            if daily:
                q = f'''select date(date) as "Date", open, high, low, close, volume from {stock} order by date(date) asc'''
                cursor = self.daily_db.cursor()
            else:
                if start is None:
                    q = f'''select datetime(date) as "Date", open, high, low, close, volume from {stock} order by datetime(date) asc'''
                else:
                    q = f'''
                    select 
                        datetime(date) as "Date", 
                        open, 
                        high, 
                        low, 
                        close, 
                        volume 
                    from {stock} 
                    where 
                        date(date) >= date("{start}") 
                    order by datetime(date) asc'''
                cursor = self.intraday_db.cursor()
                
            cursor.execute(q)
            df = pd.DataFrame(cursor.fetchall(), columns=[i[0] for i in cursor.description])
            df.Date = pd.to_datetime(df.Date)
            df = df.drop_duplicates(subset='Date')
            df = df.set_index('Date')
            df = df.sort_index()
            
            if start is not None:
                df = df[df.index >= start]
            if end is not None:
                df = df[df.index <= end]
                
            return df
            
        except Exception as e:
            logger.error(f"Failed to get OHLC data for {stock}: {str(e)}")
            raise

    def _exclude_duplicate_ticks(self, group: str) -> List[str]:
        """
        Exclude duplicate tickers from a group
        
        Args:
            group: Ticker group name
            
        Returns:
            List of unique tickers
        """
        try:
            if group == 'etf':
                g = list(set(self.stocks['etf']) - set(self.stocks['market']) - set(self.stocks['bonds']))
            else:
                g = self.stocks[group]
            return g
        except KeyError as e:
            logger.error(f"Invalid group name: {group}")
            raise InvalidParameterError(f"Invalid group name: {group}") from e

    def get_aggregates(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Get daily, weekly, monthly aggregates
        
        Args:
            df: DataFrame with datetime index
            
        Returns:
            Dictionary of aggregated DataFrames
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            raise InvalidParameterError("DataFrame index must be DatetimeIndex")
            
        try:
            daily = df.resample('B').last().dropna()
            weekly = df.resample('W').last().dropna()
            monthly = df.resample('M').last().dropna()
            return {'B': daily, 'W': weekly, 'M': monthly}
        except Exception as e:
            logger.error(f"Failed to compute aggregates: {str(e)}")
            raise

    def intra_day_aggs(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Generate multiple intraday time aggregations from a DataFrame.

        This method creates various time-based aggregations of intraday data,
        including 3-minute, 6-minute, 18-minute, 1-hour, and 4-hour intervals.
        Each aggregation uses the last value within its time window.

        Args:
            df (pd.DataFrame): DataFrame with datetime index containing price data

        Returns:
            Dict[str, pd.DataFrame]: Dictionary of aggregated DataFrames with keys:
                                   - '3min': 3-minute aggregation
                                   - '6min': 6-minute aggregation
                                   - '18min': 18-minute aggregation
                                   - '1H': 1-hour aggregation
                                   - '4H': 4-hour aggregation

        Raises:
            InvalidParameterError: If DataFrame index is not DatetimeIndex
            Exception: If there's an error computing the aggregations

        Example:
            >>> df = db.get_intraday_close(['AAPL'])
            >>> aggs = db.intra_day_aggs(df)
            >>> print(aggs['1H'].head())  # View hourly aggregation
        """
        df.index = pd.to_datetime(df.index)
        if not isinstance(df.index, pd.DatetimeIndex):
            raise InvalidParameterError("DataFrame index must be DatetimeIndex")
            
        try:
            return {
                '3min': df.resample('3T').last().dropna(),
                '6min': df.resample('6T').last().dropna(),
                '18min': df.resample('18T').last().dropna(),
                '1H': df.resample('H').last().dropna(),
                '4H': df.resample('4H').last().dropna()
            }
        except Exception as e:
            logger.error(f"Failed to compute intraday aggregates: {str(e)}")
            raise

    def daily_aggregates(self, stock: str) -> Dict[str, pd.DataFrame]:
        """
        Calculate daily, weekly, and monthly aggregates for a stock.

        This method retrieves daily closing prices for a stock and computes
        various time-based aggregations. It returns business daily (B),
        weekly (W), and monthly (M) aggregated data.

        Args:
            stock (str): The stock symbol to calculate aggregates for

        Returns:
            Dict[str, pd.DataFrame]: Dictionary containing aggregated DataFrames:
                                   - 'B': Business daily aggregation
                                   - 'W': Weekly aggregation
                                   - 'M': Monthly aggregation

        Raises:
            InvalidParameterError: If input data cannot be properly aggregated
            Exception: If there's an error retrieving data or computing aggregates

        Example:
            >>> aggs = db.daily_aggregates('AAPL')
            >>> print(aggs['W'].head())  # View weekly aggregation
        """
        try:
            df = self._getClose(stock)
            return self.get_aggregates(df)
        except Exception as e:
            logger.error(f"Failed to get daily aggregates for {stock}: {str(e)}")
            raise


    def model_preparation(self, stock: str, daily: bool = True, ma: str = 'ema',
                         start_date: Optional[str] = None, end_date: Optional[str] = None
                         ) -> Dict[str, Union[str, pd.DataFrame, List[str], pd.Series]]:
        """
        Prepare data for model training by computing technical indicators and preparing features
        
        Args:
            stock: Stock symbol to prepare data for
            daily: Whether to use daily or intraday data
            ma: Type of moving average to use ('ema' or 'sma')
            start_date: Optional start date for filtering data (YYYY-MM-DD format)
            end_date: Optional end date for filtering data (YYYY-MM-DD format)
            
        Returns:
            Dictionary containing:
                - 'stock': Stock symbol
                - 'df': Full DataFrame with all features
                - 'X': Feature DataFrame (without close price and target)
                - 'y': Target Series (next day's close price)
                - 'features': List of feature column names
                - 'target': List containing target column name
                
        Raises:
            InvalidParameterError: If invalid parameters are provided
            Exception: If data preparation fails
        """
        if not isinstance(stock, str):
            raise InvalidParameterError("Stock symbol must be a string")
        if ma not in ['ema', 'sma', 'kama', 'wma']:
            raise InvalidParameterError("Moving average type must be 'ema', 'sma', 'kama', or 'wma'")
            
        try:
            i = Indicators()
            df = self.ohlc(stock, daily=daily, start=start_date, end=end_date)
            if df.empty:
                raise ValueError(f"No data found for stock {stock}")
                
            mdf = i.all_indicators(df, ma).dropna().drop(columns=['open', 'high', 'low'])
            
            # Create target (next day's close price)
            close_series = mdf["close"].iloc[:, 0] if isinstance(mdf["close"], pd.DataFrame) else mdf["close"]
            mdf["target"] = close_series.pct_change().shift(-1)
            test_data = mdf.drop(columns = ['close', 'target']).iloc[-1]
            mdf = mdf.dropna()
            
            if mdf.empty:
                raise ValueError(f"No valid data after computing indicators for stock {stock}")
                
            return {
                'stock': stock,
                'df': mdf,
                'X': mdf.drop(columns=['close', 'target']),
                'y': mdf['target'],
                'features': list(mdf.drop(columns=['close', 'target']).columns),
                'target': ['target'],
                'test_data': test_data
            }
        except Exception as e:
            logger.error(f"Failed to prepare data for model: {str(e)}")
            raise
    
    def get_multi_frame(self, stock:str, **kwargs) -> pd.DataFrame:
        """
        Get multi-timeframe data for a stock by combining daily and intraday data.
        
        Args:
            stock: Stock symbol to retrieve data for
            daily: Whether to use daily or intraday data
            ma: Type of moving average to use ('ema', 'sma', 'kama', 'wma')
            KWARGS:
                - start_date: Optional start date for filtering data (YYYY-MM-DD format)
                - end_date: Optional end date for filtering data (YYYY-MM-DD format)
                - min_df: DataFrame with intraday data
                
            
        Returns:
            DataFrame containing combined daily and intraday data with indicators
        """
        ma = kwargs.get('ma', 'ema')
        start_date = kwargs.get('start_date', (dt.datetime.now() - dt.timedelta(days=30)).strftime('%Y-%m-%d'))
        
        if not isinstance(stock, str):
            raise InvalidParameterError("Stock symbol must be a string")
        if ma not in ['ema', 'sma', 'kama', 'wma']:
            raise InvalidParameterError("Moving average type must be 'ema', 'sma', 'kama', or 'wma'")
        
        try:
            i = Indicators()
            min_df = self.ohlc(stock, daily=False, start = start_date)
            daily_df = self.ohlc(stock, daily=True)
            min_df = i.moving_average_ribbon(min_df, ma=ma).dropna()
            daily_df = i.moving_average_ribbon(daily_df, ma=ma).dropna()
            return combine_timeframes(min_df, daily_df)
        except Exception as e:
            logger.error(f"Failed to get multi-timeframe data for {stock}: {str(e)}")
            raise

    def get_daily_technicals(self, stock: str, **kwargs) -> pd.DataFrame:
        """
        Get daily technical indicators for a stock.

        Args:
            stock (str): The stock symbol to retrieve data for
            kwargs: Additional parameters for technical indicator calculation

        Returns:
            DataFrame containing daily technical indicators for the specified stock
        """
        start_date = kwargs.get('start_date', None)

        try:
            i = Indicators()
            df = self.ohlc(stock, daily=True)
            out =  i.all_indicators(df).dropna()
            out.index = pd.to_datetime(out.index)
            out.index.name = 'date'
            if start_date is not None:
                out = out[out.index >= start_date]
            return out
        except Exception as e:
            logger.error(f"Failed to get daily technical indicators for {stock}: {str(e)}")
            raise

    def simulate_stock(self, stock: str, **kwargs) -> np.array:
        """
        Simulate stock prices using the specified model and parameters.

        Args:
            stock (str): The stock symbol to simulate
            kwargs:
                - method = [poisson, gbm, merton_jump, heston_path]
        
        returns: 
            np.array: Simulated stock prices
        """
        df = self.ohlc(stock, daily=True)
        verbose = kwargs.get('verbose', True)
        return optimize_and_simulate(stock = stock, df = df, verbose = verbose, **kwargs)
    
    def get_moving_averages(self, df:pd.DataFrame, ma: str = 'ema', **kwargs) -> pd.DataFrame:
        """
        Get moving averages for a stock.

        Args:
            stock: Stock symbol to retrieve data for
            ma: Type of moving average to use ('ema', 'sma', 'kama', 'wma')
            kwargs: Additional parameters for moving average calculation
            
        Returns:
            DataFrame containing moving averages for the specified stock
        """
        if not isinstance(df, pd.DataFrame):
            raise InvalidParameterError("Input must be a DataFrame")
        if ma not in ['ema', 'sma', 'kama', 'wma']:
            raise InvalidParameterError("Moving average type must be 'ema', 'sma', 'kama', or 'wma'")
            
        try:
            i = Indicators()
            return i.moving_average_ribbon(df, ma=ma, **kwargs)
        except Exception as e:
            logger.error(f"Failed to get moving averages: {str(e)}")
            raise

        
    def close_connections(self) -> None:
        """Close all database connections"""
        try:
            for conn in [self.names_db, self.daily_db, self.intraday_db]:
                conn.close()
            
            end_time = time.time()
            runtime_min = (end_time - self.execution_start_time) / 60
            logger.info(f"Connections closed successfully. Total runtime: {runtime_min:.2f} min")
            
        except Exception as e:
            logger.error(f"Error closing connections: {str(e)}")
            raise

    


if __name__ == "__main__":
    print("\n(26) To whatever and wherever the restless and unsteady mind wanders this mind should be restrained then and there and brought under the control of the self alone. (And nothing else) \n")
    
    connections = {
        'daily_db': 'data/prices/stocks.db',
        'intraday_db': 'data/prices/stocks_intraday.db',
        'ticker_path': 'data/stocks/tickers.json',
        'stock_names': 'data/stocks/stock_names.db'
    }
     

    p = Prices(connections)
    print(p.get_daily_technicals('amzn', start_date = '2024-08-01' ))

    # d = p.model_preparation('spy', daily = True, ma = 'kama', start_date = "2023-01-01")
    # for i in d: 
    #     try:
    #         print(i, d[i].shape)
    #     except:
    #         print(i, d[i])



    # try:
    #     with Prices(connections) as p:
    #         print('\n\n\n')
    #         result = p.get_multi_frame('spy', ma = 'sma')
    #         print(result, '\n', result.columns, '\n')
    #         print(p.simulate_stock('spy', method='heston_path'))
    #         # Connection cleanup handled by context manager
    # except Exception as e:
    #     logger.error(f"Main execution failed: {str(e)}")
    #     raise