import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.signal import find_peaks, argrelmax, argrelmin
from typing import Dict, Optional, List, Set
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import lru_cache
from .trend_detector import TrendAnalyzer, TimeSeriesData
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
    peaks: List[int]
    peak_dates: List[pd.Timestamp]

    valleys: List[int]
    valley_dates: List[pd.Timestamp]

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

if __name__ == "__main__":
    # Configure logging for the example
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    try:
        data = np.random.randn(100)  # Replace with actual time series data
        trend_analyzer = TrendAnalyzer(period=7, model='additive')
        decomposed_data = trend_analyzer.decompose(data)
        print("Trend:", decomposed_data.trend)
        print("Seasonal:", decomposed_data.seasonal)
        print("Residual:", decomposed_data.residual)
        print("Observed:", decomposed_data.observed)

        peak_detector = PeakDetector(prominence=0.5, distance=2)
        peaks = peak_detector.find_peaks(data)
        valleys = peak_detector.find_valleys(data)
        print("Peaks:", peaks)
        print("Lows:", valleys)
        print("Highs:", peak_detector.find_local_extrema(data))
        print("Trend Change Points:", peak_detector.find_trend_change_points(data))
    except Exception as e:
        logger.error(f"Error in example: {str(e)}")