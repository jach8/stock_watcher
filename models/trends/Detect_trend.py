import pandas as pd 
import numpy as np 
from statsmodels.tsa.seasonal import seasonal_decompose 
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import lru_cache

@dataclass
class TimeSeriesData:
    """Data container for time series analysis."""
    original_data: np.ndarray| pd.Series
    trend: np.ndarray| pd.Series
    seasonal: np.ndarray| pd.Series
    residual: np.ndarray| pd.Series
    observed: np.ndarray| pd.Series

    def __post_init__(self):
        self.df = pd.DataFrame({
            'trend': self.trend,
            'seasonal': self.seasonal,
            'residual': self.residual,
            'observed' : self.observed
        })

class TrendAnalyzer(ABC):
    """Base class for time series trend decomposition."""
    
    def __init__(self, period: int = 7, model: str = 'additive'):
        """
        Initialize the trend analyzer.
        
        Args:
            period (int): Seasonality period (e.g., 7 for weekly patterns)
            model (str): Type of decomposition model ('additive' or 'multiplicative')
        """
        self.period = period
        self.model = model
        
    
    def decompose(self, data: np.ndarray) -> TimeSeriesData:
        """
        Decompose time series into trend, seasonal, and residual components.
        
        Args:
            data (np.ndarray): Time series data to decompose
            
        Returns:
            TimeSeriesData: Decomposed components
            
        Raises:
            ValueError: If data length is insufficient for decomposition
        """
        if len(data) < self.period * 2:
            raise ValueError(f"Data length must be at least {self.period * 2} points")
            
        result = seasonal_decompose(
            data,
            period=self.period,
            model=self.model,
            extrapolate_trend='freq'
        )
        
        return TimeSeriesData(
            original_data=data,
            trend=result.trend,
            seasonal=result.seasonal,
            residual=result.resid,
            observed=result.observed
        )
    
    def __trend_analysis(self, data: TimeSeriesData, resample_periods) -> str:
        """
        Analyze the trend of the time series data.
        
        Args:
            data (TimeSeriesData): Decomposed time series data
            resample_periods (list): List of periods to check trend
            
        Returns:
            str: 'up' or 'down' based on trend analysis
        """
        checks = resample_periods
        name = data.observed.name
        # Report the trend for each check, get the slope of the trend line 
        bull = 0; bear = 0
        for check in checks:
            # last_trend = data.trend.resample(check).last().diff().tail(1).values[0]
            trend_slope = self.__lr_slope(data)
            if trend_slope > 0:
                bull += 1
            elif trend_slope < 0:
                bear += 1
            else:
                if bull > bear:
                    bull += 1
                else:
                    bear += 1
        if bull > bear:
            return 'up'
        else:
            return 'down'
        
    
    def __seasonality(self, data: TimeSeriesData) -> str:
        """
        Analyze the seasonality of the time series data.
        
        Args:
            data (TimeSeriesData): Decomposed time series data
            
        Returns:
            str: 'up' or 'down' based on seasonality analysis
        """
        # Report the seasonality 
        s = data.seasonal
        upper = s.quantile(0.9)
        lower = s.quantile(0.1)
        if s.tail(1).values[0] >= upper:
            return 'high'
        elif s.tail(1).values[0] <= lower:
            return 'low'
        else:
            return 'normal'

    # Linear Regression
    def __lr_slope(self, data: TimeSeriesData) -> float:
        """
        Calculate the slope of the linear regression line for the trend.
        
        Args:
            data (TimeSeriesData): Decomposed time series data
            
        Returns:
            float: Slope of the linear regression line
        """
        x = np.arange(len(data.trend))
        y = data.trend.values
        A = np.vstack([x, np.ones(len(x))]).T
        m, c = np.linalg.lstsq(A, y, rcond=None)[0]
        return m

    def analyze(self, data: np.ndarray) -> TimeSeriesData:
        """
        Analyze the trend of the time series data.
        
        Args:
            data (np.ndarray): Time series data to analyze
            
        Returns:
            TimeSeriesData: Decomposed components
        """
        # Normaliz
        data = data.bfill().ffill()
        data = (data / data[0]) - 1
        tmp = self.decompose(data)
        self.data = tmp
        checks = ['6D', '10D','28D']
        name = data.name
        trend = self.__trend_analysis(tmp, checks)
        season = self.__seasonality(tmp)
        slope = self.__lr_slope(tmp)
        return trend, season, slope
    

if __name__ == "__main__":
    # Example usage
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parents[3]))
    # from bin.alerts.flows.backtestingUtility import cp_backtesting_utility
    from bin.models.option_stats_model_setup import data
    from bin.main import get_path
    from main import Manager 
    
    # Connections Dictionary
    connections = get_path()

    data = Manager(connections)

    stocks = data.Pricedb.stocks['all_stocks']
    trend = TrendAnalyzer()

    out = pd.DataFrame(columns = ['stock', 'trend'])

    for stock in stocks:
        ohlcv = data.Pricedb.ohlc(stock).tail(90)
        try:
            x = ohlcv['Close']
            trend_direction, seasonality, slope = trend.analyze(x)
            tmp = pd.DataFrame({
                'stock': [stock], 
                'trend': [trend_direction], 
                'seasonality': [seasonality], 
                'slope': [slope]
                
            })
            out = pd.concat([out, tmp])
        except ValueError as e:
            pass

    print(out.sort_values('trend', ascending=False).set_index('stock'))

