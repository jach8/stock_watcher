# from bin.models.trends.trend_detector import TrendAnalyzer
# from bin.models.option_stats_model_setup import data as OptionsData
# from bin.models.indicator_model_setup import data as IndicatorData

from datetime import datetime, date
import pandas as pd
import re
import shutil
from pathlib import Path

from dataclasses import dataclass, field
from typing import Optional, Any, Dict

@dataclass(slots=True)
class StockData:
    stock: str
    manager: Any
    cache_dir: str = "data_cache"
    _cache: Optional[dict] = field(init=False, default=None, repr=False, compare=False)
    _daily_option_stats_cache: Dict[str, pd.DataFrame] = field(init=False, default_factory=dict, repr=False, compare=False)
    X: Optional[pd.DataFrame] = field(init=False, default=None, repr=False, compare=False)
    y: Optional[pd.DataFrame] = field(init=False, default=None, repr=False, compare=False)

    def __post_init__(self):
        Path(self.cache_dir).mkdir(exist_ok=True)
        self._cache = None

    def _get_cache_file(self):
        return Path(self.cache_dir) / f"{self.stock}_cache.pkl"

    def _load_cache(self):
        cache_file = self._get_cache_file()
        if cache_file.exists():
            try:
                self._cache = pd.read_pickle(cache_file)
            except Exception:
                self._cache = {}
        else:
            self._cache = {}

    def _save_cache(self):
        cache_file = self._get_cache_file()
        pd.to_pickle(self._cache, cache_file)

    def _load_or_cache(self, data_type: str, fetch_func, cache_key: str = None) -> pd.DataFrame:
        cache_key = cache_key or data_type
        if self._cache is None:
            self._load_cache()
        if cache_key in self._cache:
            data = self._cache[cache_key]
            if data_type == "daily_option_stats":
                if not isinstance(data.index, pd.DatetimeIndex):
                    print(f"Invalid cache for {cache_key}: Expected datetime index. Reloading.")
                    del self._cache[cache_key]
                else:
                    today = date.today()
                    latest_date = data.index.max().date()
                    is_trading_day = today.weekday() < 5
                    if is_trading_day and latest_date < today:
                        print(f"Cache for {cache_key} outdated: Latest date {latest_date}. Reloading.")
                        del self._cache[cache_key]
                    else:
                        return data
            else:
                return data
        data = fetch_func()
        if data_type == "daily_option_stats":
            if not isinstance(data.index, pd.DatetimeIndex):
                raise ValueError(f"Fetched {data_type} for {self.stock} does not have a datetime index")
            latest_date = data.index.max().date()
            today = date.today()
            if today.weekday() < 5 and latest_date < today:
                print(f"Warning: Fetched {data_type} for {self.stock} may be outdated (latest: {latest_date})")
        self._cache[cache_key] = data
        self._save_cache()
        return data

    @property
    def price_data(self) -> pd.DataFrame:
        if self._cache is None or 'price_data' not in self._cache:
            self._cache = self._cache or {}
            self._cache['price_data'] = self._load_or_cache(
                "price_data", lambda: self.manager.Pricedb.ohlc(self.stock)
            )
            self._cache['price_data'] = self._cache['price_data'].reset_index()
            self._cache['price_data'].columns = self._cache['price_data'].columns.str.lower()
            self._save_cache()
        return self._cache['price_data'].set_index("date")

    @property
    def indicators(self) -> pd.DataFrame:
        if self._cache is None or 'indicators' not in self._cache:
            self._cache = self._cache or {}
            self._cache['indicators'] = self._load_or_cache(
                "indicators", lambda: self.manager.Pricedb.get_daily_technicals(self.stock)
            )
            self._save_cache()
        return self._cache['indicators']

    @property
    def daily_option_stats(self) -> pd.DataFrame:
        if self._cache is None or 'daily_option_stats' not in self._cache:
            self._cache = self._cache or {}
            self._cache['daily_option_stats'] = self._load_or_cache(
                "daily_option_stats", lambda: self.manager.Optionsdb.get_daily_option_stats(self.stock)
            )
            self._save_cache()
        return self._cache['daily_option_stats']
    
    @property
    def option_chain(self) -> pd.DataFrame:
        if self._cache is None or 'option_chain' not in self._cache:
            self._cache = self._cache or {}
            self._cache['option_chain'] = self._load_or_cache(
                "option_chain", lambda: self.manager.Optionsdb._parse_change_db(self.stock)
            )
            self._save_cache()
        return self._cache['option_chain']

    @property
    def intraday_data_price_data(self) -> pd.DataFrame:
        """
        Get intraday price data for the stock.
        
        Returns:
            pd.DataFrame: DataFrame with intraday price data.
        """
        if self._cache is None or 'intraday_data_price_data' not in self._cache:
            self._cache = self._cache or {}
            self._cache['intraday_data_price_data'] = self._load_or_cache(
                "intraday_data_price_data", lambda: self.manager.Pricedb.get_intraday_price_data(self.stock, daily = False, )
            )
            self._save_cache()
        return self._cache['intraday_data_price_data']

    def get_daily_option_stats(self, dropCols: str = None) -> pd.DataFrame:
        """
        Get daily_option_stats with optional column dropping based on regex.
        
        Args:
            dropCols (str, optional): Regex pattern to match columns to drop (e.g., 'vol|oi|iv').
        
        Returns:
            pd.DataFrame: Filtered or full daily_option_stats DataFrame.
        
        Notes:
            Filtered DataFrames are re-cached if the base daily_option_stats is updated.
        """
        cache_key = f"daily_option_stats_{dropCols}" if dropCols else "daily_option_stats"
        base_data = self.daily_option_stats  # Ensure base data is fresh
        
        # Check if cached filtered data is still valid
        if cache_key in self._daily_option_stats_cache:
            cached_data = self._daily_option_stats_cache[cache_key]
            if cached_data.index.equals(base_data.index):  # Check if index matches base data
                return cached_data
            else:
                print(f"Invalid cache for {cache_key}: Index mismatch. Recreating.")
        
        # Create filtered DataFrame
        data = base_data.copy()
        if dropCols:
            try:
                columns_to_drop = [col for col in data.columns if re.search(dropCols, col)]
                if not columns_to_drop:
                    print(f"Warning: No columns matched regex '{dropCols}'")
                data = data.drop(columns=columns_to_drop, errors="ignore")
            except re.error:
                raise ValueError(f"Invalid regex pattern: {dropCols}")

        self._daily_option_stats_cache[cache_key] = data
        return data



    def clear_cache(self, disk: bool = False, stock_specific: bool = True) -> None:
        """
        Clear in-memory and optionally disk caches.
        
        Args:
            disk (bool): If True, delete Parquet files from cache_dir.
            stock_specific (bool): If True, only delete files for this stock. If False, delete all files.
        
        Raises:
            FileNotFoundError: If cache_dir doesn't exist and disk=True.
        """
        self._cache = None
        self._daily_option_stats_cache.clear()
        if disk:
            cache_file = self._get_cache_file()
            if cache_file.exists():
                cache_file.unlink()

    def get_features(self, drop_columns: str = None) -> pd.DataFrame:
        """
        Base feature extraction for ML.
        
        Args:
            drop_columns (str, optional): Regex pattern to drop columns from daily_option_stats.
        
        Returns:
            pd.DataFrame: Combined DataFrame of daily_option_stats and price_data.
        """
        option_stat = self.get_daily_option_stats(dropCols=drop_columns)
        price_data = self.indicators
        option_stat = option_stat.resample('1D').last()
        price_df = price_data.loc[option_stat.index[0]:]
        option_df = option_stat.loc[price_df.index[0]:]
        df = pd.merge(option_df, price_df, left_index=True, right_index=True)
        return df.dropna()



if __name__ == "__main__":
    from main import Manager 
    import numpy as np 
    # Example usage
    manager = Manager()  # Your Manager class
    sd = StockData(stock="spy", manager=manager, cache_dir="data_cache")
    sd.clear_cache(disk=True, stock_specific=False)
    df = sd.get_features()
    x = df.drop(columns=["close", "open", "high", "low"])
    y = df["close"].pct_change().shift(-1)
    y = pd.Series(np.sign(y), index=y.index)
    y = y.dropna()
    x = x.loc[y.index]
    print(df)

    option_chain = sd.option_chain
    print(f"Option Chain for {sd.stock}:\n{option_chain.head()}")

    # # Load Cache Files
    # cache_dir = Path("data_cache")
    # if cache_dir.exists():
    #     # Print the keys in the cache
    #     cache_files = list(cache_dir.glob("*.parquet"))
    #     if cache_files:
    #         print("Cache files found:")
    #         for file in cache_files:
    #             print(f"- {file.name}")
    #             cdf = pd.read_parquet(file, engine="pyarrow")
    #             print(f"  - Shape: {cdf.shape}, Columns: {cdf.columns.tolist()}")
    #     else:
    #         print("No cache files found.")
