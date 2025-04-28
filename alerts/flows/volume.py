import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.signal import find_peaks, argrelmax, argrelmin
from typing import Dict, Optional, List, Set
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import lru_cache
from trending import TrendAnalyzer, TimeSeriesData

@dataclass
class PeakData:
    peaks: List[int]
    valleys: List[int]

class PeakDetector(ABC):
    """Base class for peak detection in time series data."""
    
    def __init__(self, prominence: float = 1.0, distance: int = 1):
        """
        Initialize the peak detector.
        
        Args:
            prominence (float): Required prominence of peaks
            distance (int): Minimum distance between peaks
        """
        self.prominence = prominence
        self.distance = distance
    
    
    def find_peaks(self, data: np.ndarray) -> List[int]:
        """
        Find peaks in the time series data.
        
        Args:
            data (np.ndarray): Time series data to analyze
            
        Returns:
            List[int]: Indices of detected peaks
        """
        peaks, _ = find_peaks(
            data,
            prominence=self.prominence,
            distance=self.distance
        )
        return peaks.tolist()
    
    def find_valleys(self, data: np.ndarray) -> List[int]:
        """
        Find valleys in the time series data.
        
        Args:
            data (np.ndarray): Time series data to analyze
            
        Returns:
            List[int]: Indices of detected valleys
        """
        valleys = argrelmin(data, order=self.distance)[0]
        return valleys.tolist()
    
    def find_local_extrema(self, data: np.ndarray) -> List[int]:
        """
        Find local extrema (peaks and valleys) in the time series data.
        
        Args:
            data (np.ndarray): Time series data to analyze
            
        Returns:
            List[int]: Indices of detected local extrema
        """
        peaks = self.find_peaks(data)
        valleys = self.find_valleys(data)
        # Ensure unique indices
        return sorted(set(peaks + valleys))

    def find_trend_change_points(self, data: np.ndarray) -> List[int]:
        """
        Find trend change points in the time series data.
        
        Args:
            data (np.ndarray): Time series data to analyze
            
        Returns:
            List[int]: Indices of detected trend change points
        """
        local_extrema = self.find_local_extrema(data)
        return local_extrema
    



if __name__ == "__main__":

    # Example usage
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