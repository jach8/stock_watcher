import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.signal import find_peaks, argrelmax, argrelmin
from typing import Dict, Optional, List, Set
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import lru_cache
import logging

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

@dataclass
class PeakData:
    peak_values: int|float
    peak_dates: str

    valley_values: int|float
    valley_dates: str

    changepoint_values: int|float
    changepoint_dates: str



class PeakDetector(ABC):
    """Base class for peak detection in time series data."""
    
    def __init__(self, prominence: float = 1.0, distance: int = 1):
        """
        Initialize the peak detector.
        
        Args:
            prominence (float): Required prominence of peaks
            distance (int): Minimum distance between peaks
            
        Raises:
            ValueError: If prominence is not positive or distance is less than 1
        """
        if prominence <= 0:
            raise ValueError("Prominence must be positive")
        if distance < 1:
            raise ValueError("Distance must be at least 1")
        
        self.prominence = prominence
        self.distance = distance
        logger.debug(f"Initialized PeakDetector with prominence={prominence}, distance={distance}")
    
    def find_peaks(self, data: np.ndarray) -> List[int]:
        """
        Find peaks in the time series data.
        
        Args:
            data (np.ndarray): Time series data to analyze
            
        Returns:
            List[int]: Indices of detected peaks
            
        Raises:
            ValueError: If data is not a numpy array or is empty
        """
        if not isinstance(data, np.ndarray):
            raise ValueError("Input data must be a numpy array")
        if data.size == 0:
            raise ValueError("Input data array is empty")
        
        try:
            peaks, _ = find_peaks(
                data,
                prominence=self.prominence,
                distance=self.distance
            )
            logger.debug(f"Found {len(peaks)} peaks in data of length {len(data)}")
            return peaks.tolist()
        except Exception as e:
            logger.error(f"Error finding peaks: {str(e)}")
            raise
    
    def find_valleys(self, data: np.ndarray) -> List[int]:
        """
        Find valleys in the time series data.
        
        Args:
            data (np.ndarray): Time series data to analyze
            
        Returns:
            List[int]: Indices of detected valleys
            
        Raises:
            ValueError: If data is not a numpy array or is empty
        """
        if not isinstance(data, np.ndarray):
            raise ValueError("Input data must be a numpy array")
        if data.size == 0:
            raise ValueError("Input data array is empty")
        
        try:
            valleys = argrelmin(data, order=self.distance)[0]
            logger.debug(f"Found {len(valleys)} valleys in data of length {len(data)}")
            return valleys.tolist()
        except Exception as e:
            logger.error(f"Error finding valleys: {str(e)}")
            raise
    
    def find_local_extrema(self, data: np.ndarray) -> List[int]:
        """
        Find local extrema (peaks and valleys) in the time series data.
        
        Args:
            data (np.ndarray): Time series data to analyze
            
        Returns:
            List[int]: Indices of detected local extrema
            
        Raises:
            ValueError: If data is not a numpy array or is empty
        """
        if not isinstance(data, np.ndarray):
            raise ValueError("Input data must be a numpy array")
        if data.size == 0:
            raise ValueError("Input data array is empty")
        
        try:
            peaks = self.find_peaks(data)
            valleys = self.find_valleys(data)
            # Ensure unique indices
            extrema = sorted(set(peaks + valleys))
            logger.debug(f"Found {len(extrema)} local extrema in data of length {len(data)}")
            return extrema
        except Exception as e:
            logger.error(f"Error finding local extrema: {str(e)}")
            raise

    def find_trend_change_points(self, data: np.ndarray) -> List[int]:
        """
        Find trend change points in the time series data.
        
        Args:
            data (np.ndarray): Time series data to analyze
            
        Returns:
            List[int]: Indices of detected trend change points
            
        Raises:
            ValueError: If data is not a numpy array or is empty
        """
        if not isinstance(data, np.ndarray):
            raise ValueError("Input data must be a numpy array")
        if data.size == 0:
            raise ValueError("Input data array is empty")
        
        try:
            local_extrema = self.find_local_extrema(data)
            logger.debug(f"Found {len(local_extrema)} trend change points")
            return local_extrema
        except Exception as e:
            logger.error(f"Error finding trend change points: {str(e)}")
            raise

    def get_all_peak_data(self, data: pd.Series) -> Dict[str, List]:
        """
        Get all peak data including peaks, valleys, and change points.
        
        Args:
            data (pd.Series): Time series data to analyze

        Returns:
            Dict[str, List]: Dictionary containing peaks, valleys, and change points
        """
        peaks = self.find_peaks(data.to_numpy())
        valleys = self.find_valleys(data.to_numpy())
        changepoints = self.find_trend_change_points(data.to_numpy())
        
        dates = data.index
        peak_dates = [dates[i].strftime("%Y-%m-%d") for i in peaks]
        valley_dates = [dates[i].strftime("%Y-%m-%d") for i in valleys]
        changepoint_dates = [dates[i].strftime("%Y-%m-%d") for i in changepoints]

        return {
            "peaks": peaks,
            "peak_dates": peak_dates,
            "valleys": valleys,
            "valley_dates": valley_dates,
            "changepoints": changepoints,
            "changepoint_dates": changepoint_dates
        }

    def get_peak_data(self, data: pd.Series) -> PeakData:
        """
        Get the last peak data including peaks, valleys, and change points.
        
        Args:
            data (pd.Series): Time series data to analyze

        Returns:
            PeakData: Object containing peaks, valleys, and change points
        """
        data = data.bfill().ffill().dropna()
        peaks = self.find_peaks(data.to_numpy())
        valleys = self.find_valleys(data.to_numpy())
        changepoints = self.find_trend_change_points(data.to_numpy())
        
        dates = data.index
        peak_dates = [dates[i].strftime("%Y-%m-%d") for i in peaks]
        valley_dates = [dates[i].strftime("%Y-%m-%d") for i in valleys]
        changepoint_dates = [dates[i].strftime("%Y-%m-%d") for i in changepoints]

        peak_values = data.iloc[peaks].tolist()
        valley_values = data.iloc[valleys].tolist()
        changepoint_values = data.iloc[changepoints].tolist()

        return PeakData(
            peak_values=peak_values[-1],
            peak_dates=peak_dates[-1],
            valley_values=valley_values[-1],
            valley_dates=valley_dates[-1],
            changepoint_values=changepoint_values[-1],
            changepoint_dates=changepoint_dates[-1]
        )

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parents[3]))
    from bin.models.trends.Detect_trend import TrendAnalyzer, TimeSeriesData
    # Configure logging for the example
    logging.basicConfig(level=logging.INFO)

    # Example usage
    try:
        data = np.random.randn(100)  # Replace with actual time series data
        data = pd.Series(data, index=pd.date_range(start='2023-01-01', periods=100, freq='D'))

        peak_detector = PeakDetector(prominence=0.5, distance=2)
        peak_data = peak_detector.get_peak_data(data)
        print("Peaks:", peak_data.peak_values)
        print("Lows:", peak_data.valley_values)
        print("Highs:", peak_data.peak_dates)
        print("Change Points:", peak_data.changepoint_values)

        print("Peaks Dates:", peak_data.peak_dates)
        print("Lows Dates:", peak_data.valley_dates)
        print("Trend Change Points:", peak_data.changepoint_dates)
    except Exception as e:
        logger.error(f"Error in example: {str(e)}")