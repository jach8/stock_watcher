import pandas as pd
import numpy as np
from typing import Union, Tuple, Dict, Optional
from itertools import product
from pandas import Series, DataFrame
from statsmodels.tsa.seasonal import seasonal_decompose
from dataclasses import dataclass
from trend_detector import TrendAnalyzer

class ChangePointDetector:
    """A class for detecting change points in time series data with parameter optimization."""
    
    def __init__(self, 
            data: Union[Series, DataFrame], 
            scale: bool = True, 
            period: int = 21, 
            window_size: Optional[int] = None
        ) -> None:
        """
        Initialize the detector with data.
        
        Parameters:
        - data: pandas Series or DataFrame (1D), preferably with datetime index
        - scale: bool, whether to normalize the data
        - period: int, seasonality period for decomposition (default: 21 for monthly in trading days)
        - window_size: int, size of rolling window for local mean (None for global mean, default: 30)
        
        Raises:
        - ValueError: If data is empty, not 1D, contains NaNs, or window_size is invalid
        """
        if data.empty:
            raise ValueError("Input data cannot be empty")
        if isinstance(data, DataFrame) and data.shape[1] > 1:
            raise ValueError("Input data must be 1-dimensional")
        if data.isna().any().any():
            raise ValueError("Input data cannot contain NaN values")
        if window_size is not None and (not isinstance(window_size, int) or window_size < 1):
            raise ValueError("window_size must be a positive integer")
            
        self.data: Series = data if isinstance(data, Series) else data.iloc[:, 0]
        self.scale: bool = scale
        self.period: int = period
        self.window_size: Optional[int] = window_size if window_size is not None else 30  # Default to 30 days
        
        # Decompose the time series
        self.trend_analyzer = TrendAnalyzer(period=self.period, model='additive')
        self.decomposed_data = self.trend_analyzer.decompose(self.data)
        
        # Use residual for change point detection
        self.processed_data: np.ndarray = self._preprocess_data()
        
    def _preprocess_data(self) -> np.ndarray:
        """Normalize the residual component if scale is True.
        
        Returns:
        - numpy array of processed data (residuals)
        
        Raises:
        - ValueError: If data range is zero and normalization is requested
        """
        data = np.array(self.decomposed_data.residual, dtype=np.float64)
        if self.scale:
            data_min, data_max = np.min(data), np.max(data)
            if data_max == data_min:
                raise ValueError("Cannot normalize data with zero range (all values identical)")
            return (data - data_min) / (data_max - data_min)
        return data
    
    def _compute_change_points(self, sensitivity: float, threshold: float, normalize_s_t: bool = False) -> Tuple[DataFrame, float]:
        """Core change point detection algorithm on residuals.
        
        Parameters:
        - sensitivity: float, critical level (C)
        - threshold: float, threshold level (T)
        - normalize_s_t: bool, whether to normalize S_T for optimization
        
        Returns:
        - DataFrame with detection results
        - float, maximum S_T value (for normalization)
        
        Raises:
        - ValueError: If sensitivity or threshold is negative
        """
        if sensitivity < 0 or threshold < 0:
            raise ValueError("Sensitivity and threshold must be non-negative")
        if len(self.processed_data) < 2:
            return pd.DataFrame({
                'X_MU_C': [0.0],
                'S_T': [0.0],
                'Signal': [0.0],
                'Sensitivity': [sensitivity]
            }, index=self.data.index[:1]).round(4), 0.0
            
        # Compute rolling mean on residuals
        rolling_mean = pd.Series(self.processed_data).rolling(
            window=self.window_size, min_periods=1, center=False
        ).mean().to_numpy() if self.window_size is not None else np.full_like(self.processed_data, np.mean(self.processed_data))
        
        # Use absolute deviations to detect changes in either direction
        x_mu_c = abs(self.processed_data - rolling_mean) - sensitivity
        x_mu_c[0] = 0
            
        s_t = np.zeros_like(x_mu_c)
        for i in range(1, len(x_mu_c)):
            s_t[i] = max(0, s_t[i-1] + x_mu_c[i])
        
        # Store maximum S_T for normalization
        s_t_max = np.max(s_t) if np.max(s_t) > 0 else 1.0  # Avoid division by zero
        if normalize_s_t:
            s_t_normalized = s_t / s_t_max
            signals = (s_t_normalized > threshold).astype(float)
        else:
            s_t_normalized = s_t
            signals = (s_t > threshold).astype(float)
        
        return pd.DataFrame({
            'X_MU_C': x_mu_c,
            'S_T': s_t_normalized,
            'Signal': signals,
            'Sensitivity': sensitivity
        }, index=self.data.index).round(4), s_t_max
    
    def assess_sensitivity(
        self,
        sensitivity_range: Tuple[float, float, float] = (0.05, 1.0, 0.1),
        threshold: float = 0.1
    ) -> Dict[float, DataFrame]:
        """
        Assess the number of triggers for a range of sensitivity values.
        
        Parameters:
        - sensitivity_range: tuple (start, stop, step) for sensitivity (C)
        - threshold: float, threshold level (T) to use for assessment
        
        Returns:
        - Dict mapping sensitivity values to result DataFrames
        """
        if any(x <= 0 for x in sensitivity_range):
            raise ValueError("Sensitivity range values must be positive")
        if sensitivity_range[0] >= sensitivity_range[1]:
            raise ValueError("Range start must be less than stop")
            
        sensitivities = np.arange(*sensitivity_range)
        results = {}
        for sensitivity in sensitivities:
            result_df, _ = self._compute_change_points(sensitivity, threshold)
            results[round(sensitivity, 2)] = result_df
        return results
    
    def optimize_parameters(
        self,
        sensitivity_range: Tuple[float, float, float] = (0.01, 0.9, 0.01),
        threshold_range: Tuple[float, float, float] = (0.1, 2.0, 0.1),
        min_triggers: int = 5,
        max_triggers: int = 10
    ) -> Tuple[Dict[str, Union[float, str]], DataFrame]:
        """
        Find optimal sensitivity and threshold within trigger range.
        
        Parameters:
        - sensitivity_range: tuple (start, stop, step) for sensitivity (C)
        - threshold_range: tuple (start, stop, step) for threshold (T)
        - min_triggers: minimum acceptable number of triggers (N)
        - max_triggers: maximum acceptable number of triggers (M)
        
        Returns:
        - best_params: dict with optimal sensitivity and threshold
        - result_df: DataFrame with detection results
        
        Raises:
        - ValueError: If ranges are invalid or min_triggers > max_triggers
        """
        if any(x <= 0 for x in sensitivity_range + threshold_range):
            raise ValueError("Range values must be positive")
        if sensitivity_range[0] >= sensitivity_range[1] or threshold_range[0] >= threshold_range[1]:
            raise ValueError("Range start must be less than stop")
        if min_triggers > max_triggers:
            raise ValueError("min_triggers must not exceed max_triggers")
            
        sensitivities = np.arange(*sensitivity_range)
        thresholds = np.arange(*threshold_range)
        
        # First pass: compute S_T for all combinations to find normalization factor
        s_t_max_global = 0.0
        for sensitivity, threshold in product(sensitivities, thresholds):
            _, s_t_max = self._compute_change_points(sensitivity, threshold)
            s_t_max_global = max(s_t_max_global, s_t_max)
        
        # Second pass: optimize with normalized S_T
        best_params: Optional[Dict[str, Union[float, str]]] = None
        best_result: Optional[DataFrame] = None
        best_distance_to_target: float = float('inf')
        target_triggers = (min_triggers + max_triggers) / 2  # Aim for midpoint of range
        
        for sensitivity, threshold in product(sensitivities, thresholds):
            result_df, _ = self._compute_change_points(sensitivity, threshold / s_t_max_global, normalize_s_t=True)
            triggers = int(result_df['Signal'].sum())
            
            # Select parameters with triggers in [min_triggers, max_triggers], minimizing distance to target
            if min_triggers <= triggers <= max_triggers:
                distance = abs(triggers - target_triggers)
                if distance < best_distance_to_target:
                    best_distance_to_target = distance
                    best_params = {
                        'sensitivity': sensitivity,
                        'threshold': threshold
                    }
                    best_result = result_df
                
        if best_params is None:
            # Fallback: highest sensitivity (lowest C) that yields <= max_triggers
            for sensitivity in sorted(sensitivities, reverse=True):
                for threshold in sorted(thresholds, reverse=True):  # Start with higher thresholds
                    result_df, _ = self._compute_change_points(sensitivity, threshold / s_t_max_global, normalize_s_t=True)
                    triggers = int(result_df['Signal'].sum())
                    if triggers <= max_triggers:
                        best_params = {
                            'sensitivity': sensitivity,
                            'threshold': threshold
                        }
                        best_result = result_df
                        return best_params, best_result
            # If still no valid parameters, use median values
            best_params = {
                'sensitivity': float(np.median(sensitivities)),
                'threshold': float(np.median(thresholds))
            }
            best_result, _ = self._compute_change_points(
                best_params['sensitivity'], 
                best_params['threshold'] / s_t_max_global,
                normalize_s_t=True
            )
            
        return best_params, best_result
    
    def detect(self, sensitivity: float = 0.45, threshold: float = 0.1) -> DataFrame:
        """
        Detect change points with specified parameters.
        
        Parameters:
        - sensitivity: float, critical level (C)
        - threshold: float, threshold level (T)
        
        Returns:
        - DataFrame with detection results
        """
        result_df, _ = self._compute_change_points(sensitivity, threshold)
        return result_df
    
    def get_last_change_point(
        self,
        sensitivity_range: Tuple[float, float, float] = (0.01, 0.9, 0.1),
        threshold_range: Tuple[float, float, float] = (0.1, 2.0, 0.1),
        min_triggers: int = 5,
        max_triggers: int = 10
        ) -> DataFrame:
        """Get the last detected change point."""
        _, result_df = self.optimize_parameters(
            sensitivity_range=sensitivity_range,
            threshold_range=threshold_range,
            min_triggers=min_triggers,
            max_triggers=max_triggers
        )
        return result_df.iloc[-1]

if __name__ == "__main__":
    
    dates = pd.date_range(start='2024-11-15', end='2025-05-13', freq='6H')  # 4 points/day
    n = len(dates)  # 720 points
    trend = np.linspace(240, 170, n)
    noise = np.random.normal(0, 5, n)
    simulated_data = trend + noise
    data = pd.Series(simulated_data, index=dates)


    # Initialize detector
    detector = ChangePointDetector(data, scale=True, period=28, window_size=20)  # period=28 (weekly for 4 points/day), window_size=20 (~5 days)

    # Sensitivity assessment
    test_sens = np.arange(0.05, 1.0, 0.1)
    results = {}
    print("Sensitivity Assessment:")
    for i in test_sens:
        i = round(i, 2)
        res_df = detector.detect(sensitivity=i, threshold=0.1)
        triggers = res_df['Signal'].sum()
        print(f'Sensitivity Level {i}: {triggers}')
        results[i] = res_df

    # Optimized parameters
    best_params, best_result_df = detector.optimize_parameters(
        sensitivity_range=(0.05, 1.0, 0.1),
        threshold_range=(0.1, 2.0, 0.1),
        min_triggers=5,
        max_triggers=20
    )
    print("\nOptimized Parameters:")
    print(f"Best Parameters: {best_params}")
    print(f"Triggers: {best_result_df['Signal'].sum()}")