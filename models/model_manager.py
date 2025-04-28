from trends.trend_detector import TrendAnalyzer
from option_stats_model_setup import data as OptionsData
from indicator_model_setup import data as IndicatorData

import pandas as pd 
import numpy as np 
from statsmodels.tsa.seasonal import seasonal_decompose 
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class StockData:
    stock: str
    price_data: pd.DataFrame
    indicators: pd.DataFrame
    daily_option_stats: pd.DataFrame
    option_chain: pd.DataFrame




if __name__ == "__main__":

    import sys 
    import logging
    sys.path.append('/Users/jerald/Documents/Dir/Python/Stocks')
    from bin.main import get_path 
    from bin.main import Manager
    from bin.plots.utils import pretty_print
    
    manager = Manager()

    # Example of things i want to store for quick access

    stock = 'cmre'
    price_data = manager.Pricedb.ohlc(stock)
    indicators = manager.Pricedb.get_multi_frame(stock)
    daily_option_stats = manager.Optionsdb.get_daily_option_stats(stock)
    option_chain = manager.Optionsdb._parse_change_db(stock)


    # Create a StockData object
    stock_data = StockData(stock, price_data, indicators, daily_option_stats, option_chain)

    print(stock_data)

