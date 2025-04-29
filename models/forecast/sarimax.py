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