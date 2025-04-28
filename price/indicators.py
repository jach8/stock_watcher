import pandas as pd 
import numpy as np 
import sqlite3 as sql
import logging
from typing import Union, Tuple, Dict, List, Optional
from pandas import Series, DataFrame


# from technicals.vol import volatility 
# from technicals.others import descriptive_indicators
# from technicals.ma import moving_avg
# from technicals.mom import momentum


from .technicals.vol import volatility 
from .technicals.others import descriptive_indicators
from .technicals.ma import moving_avg
from .technicals.mom import momentum

# Configure logging
logging.basicConfig(
    # filename='logs/indicators.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Indicators:
    def __init__(self, *args, **kwargs) -> None:
        self.__moving_average = moving_avg()
        self.__volatility = volatility()
        self.__momentum = momentum()
        self.__descriptive = descriptive_indicators()
        self.__moving_average.windows = np.array([6, 10, 20, 28, 96, 108])
        self.__volatility.windows = np.array([6, 10, 20, 28])
        self.__descriptive.windows = np.array([10, 20])
    
    def moving_average_ribbon(self, df: pd.DataFrame, ma:str='sma') -> pd.DataFrame:
        """ Generate a moving average ribbon """
        assert ma in ['sma', 'ema', 'wma', 'kama'], 'Invalid moving average type'
        return self.__moving_average.ribbon(df, ma=ma)
    
    def volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """ Generate volatility indicators """
        df = self.__volatility._validate_dataframe(df)
        return self.__volatility.vol_indicators(df)

    def descriptive_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """ Generate descriptive indicators """
        df = self.__descriptive._validate_dataframe(df)
        return self.__descriptive.descriptive_indicators(df)
    
    def momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """ Generate momentum indicators """
        df = self.__momentum._validate_dataframe(df)
        return self.__momentum.mom_indicators(df)
    
    def all_indicators(self, df: pd.DataFrame, ma: str = 'sma') -> pd.DataFrame:
        """ Generate all indicators """
        out = pd.concat([
            self.moving_average_ribbon(df, ma),
            self.momentum_indicators(df),
            self.volatility_indicators(df),
            self.descriptive_indicators(df)
        ], axis=1)
        return out

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from main import Manager, get_path 

    connections = get_path()
    m = Manager(connections)
    # df = m.Pricedb.ohlc('spy', daily=False).resample('3T').last()
    df = m.Pricedb.ohlc('spy', daily=True)
    i = Indicators()

    indicator_df = i.all_indicators(df)

    # We need to find out why some columns are being added twice
    cols = list(indicator_df.columns)
    column_counts = {col: cols.count(col) for col in cols}
    print("Columns: ",len(cols))
    print("Counts: ")
    for col, count in column_counts.items():
        print(f"{col}: {count}")
    
    print('\n\n')

    for col in cols:
        if column_counts[col] > 1:
            print(f"!!!Duplicate column: {col}")


