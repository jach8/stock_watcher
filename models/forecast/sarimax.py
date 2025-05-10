""" Wrapper for Sarimax Model from Statsmodels API """

import pandas as pd 
import numpy as np 
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from typing import Tuple, Dict, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class SarimaxModel:
    """Wrapper for SARIMAX model from Statsmodels API."""
    
    order: Tuple[int, int, int]
    seasonal_order: Tuple[int, int, int, int]
    trend: str = 'c'
    exogenous: pd.DataFrame = None
    model: Any = None
    fitted_model: Any = None
    
    def __post_init__(self):
        self.model = SARIMAX(
            endog=None,
            exog=self.exogenous,
            order=self.order,
            seasonal_order=self.seasonal_order,
            trend=self.trend
        )



if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parents[3]))
    from bin.main import get_path
    from main import Manager 
    connections = get_path()
    manager = Manager(connections)
    close_prices = manager.Pricedb.ohlc('spy')['Close'].to_frame()
    
    model = SARIMAX(
    endog=close_prices,
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 12),
    trend='c'
    ).fit()
