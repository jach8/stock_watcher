import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.signal import find_peaks
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

class MarketAnalyzer:
    def __init__(self, data):
        """
        Initialize the MarketAnalyzer with historical data.
        
        Args:
            data (dict): Dictionary with stock symbols as keys and DataFrames as values.
                         Each DataFrame should have columns: 'date', 'price', 'volume', 'openinterest'.
        
        Raises:
            ValueError: If data dictionary is empty or has invalid structure
        """
        if not isinstance(data, dict) or not data:
            raise ValueError("Data must be a non-empty dictionary")
            
        # Validate data structure
        required_columns = {'date', 'price', 'volume', 'openinterest'}
        for symbol, df in data.items():
            if not isinstance(df, pd.DataFrame):
                raise ValueError(f"Value for symbol {symbol} must be a DataFrame")
            if not required_columns.issubset(df.columns):
                raise ValueError(f"DataFrame for {symbol} missing required columns: {required_columns - set(df.columns)}")
        
        self.data = data
        self.trend_model = 'additive'  # Can be 'additive' or 'multiplicative'
        self.period = 7  # Seasonality period (e.g., 7 for weekly)
        self.verbose = False  # Debug mode flag
        self._cache = {}  # Cache for expensive calculations

    def _calculate_trend_decomposition(self, df, value_col, date_col):
        """
        Extract the trend component using seasonal decomposition.
        
        Args:
            df (pd.DataFrame): DataFrame with time series data.
            value_col (str): Column name of the value to decompose.
            date_col (str): Column name of the date.
            
        Returns:
            pd.Series: Trend component, or None if an error occurs.
        """
        try:
            df = df.set_index(date_col).sort_index()
            decomposition = seasonal_decompose(df[value_col], model=self.trend_model, period=self.period)
            return decomposition.trend
        except Exception as e:
            print(f"Error in decomposition for {value_col}: {e}")
            return None

    def _detect_anomalies(self, df, value_col):
        """
        Detect anomalies in the residuals after decomposition.
        
        Args:
            df (pd.DataFrame): DataFrame with residuals.
            value_col (str): Column name of the residuals.
            
        Returns:
            list: Indices of anomalies.
        """
        residuals = df[value_col].dropna()
        if residuals.empty:
            return []
        scaler = StandardScaler()
        z_scores = scaler.fit_transform(residuals.values.reshape(-1, 1))
        anomalies = residuals.index[np.abs(z_scores) > 3].tolist()
        return anomalies

    def _find_peaks(self, df, value_col, prominence=1, distance=1):
        """
        Detect peaks in the time series.
        
        Args:
            df (pd.DataFrame): DataFrame with time series data.
            value_col (str): Column name of the value to analyze.
            prominence (float): Minimum prominence for peaks.
            distance (int): Minimum distance between peaks.
            
        Returns:
            list: Indices of detected peaks.
        """
        series = df[value_col].dropna()
        if series.empty:
            return []
        peaks, _ = find_peaks(series, prominence=prominence, distance=distance)
        return series.index[peaks].tolist()

    def _overall_sentiment(self, stock, lookback_days):
        """
        Determine overall market sentiment based on price, volume, and OI trends.
        
        Args:
            stock (str): Stock symbol.
            lookback_days (int): Number of days to look back.
            
        Returns:
            str: Sentiment label ('strong', 'weak', 'neutral').
        """
        df = self.data.get(stock)
        if df is None or len(df) < lookback_days:
            return 'neutral'
        
        df = df.tail(lookback_days)
        price_trend = self._calculate_trend_decomposition(df, 'price', 'date')
        volume_trend = self._calculate_trend_decomposition(df, 'volume', 'date')
        oi_trend = self._calculate_trend_decomposition(df, 'openinterest', 'date')
        
        if price_trend is None or volume_trend is None or oi_trend is None:
            return 'neutral'
        
        price_rising = price_trend.iloc[-1] > price_trend.iloc[0]
        volume_rising = volume_trend.iloc[-1] > volume_trend.iloc[0]
        oi_rising = oi_trend.iloc[-1] > oi_trend.iloc[0]
        
        if price_rising and volume_rising and oi_rising:
            return 'strong'
        elif not price_rising and not volume_rising and not oi_rising:
            return 'weak'
        else:
            return 'neutral'

    def _intensity_behind_move(self, stock, lookback_days):
        """
        Calculate the intensity behind a price move.
        
        Args:
            stock (str): Stock symbol.
            lookback_days (int): Number of days to look back.
            
        Returns:
            float: Intensity score.
        """
        df = self.data.get(stock)
        if df is None or len(df) < lookback_days:
            return 0.0
        
        df = df.tail(lookback_days)
        price_change = (df['price'].iloc[-1] - df['price'].iloc[0]) / df['price'].iloc[0]
        avg_volume = df['volume'].mean()
        hist_avg_volume = self.data[stock]['volume'].mean()
        volume_factor = avg_volume / hist_avg_volume if hist_avg_volume != 0 else 1
        intensity = price_change * volume_factor
        return intensity

    def _detect_divergence(self, stock, lookback_days):
        """
        Detect divergence between price and volume trends.
        
        Args:
            stock (str): Stock symbol.
            lookback_days (int): Number of days to look back.
            
        Returns:
            bool: True if divergence is detected.
        """
        df = self.data.get(stock)
        if df is None or len(df) < lookback_days:
            return False
        
        df = df.tail(lookback_days)
        price_trend = self._calculate_trend_decomposition(df, 'price', 'date')
        volume_trend = self._calculate_trend_decomposition(df, 'volume', 'date')
        
        if price_trend is None or volume_trend is None:
            return False
        
        price_rising = price_trend.iloc[-1] > price_trend.iloc[0]
        volume_rising = volume_trend.iloc[-1] > volume_trend.iloc[0]
        
        return price_rising != volume_rising

    def _detect_reversal(self, stock, lookback_days):
        """
        Detect potential market reversals.
        
        Args:
            stock (str): Stock symbol.
            lookback_days (int): Number of days to look back.
            
        Returns:
            bool: True if reversal is detected.
        """
        df = self.data.get(stock)
        if df is None or len(df) < lookback_days:
            return False
        
        df = df.tail(lookback_days)
        peaks = self._find_peaks(df, 'price', prominence=0.05 * df['price'].std(), distance=5)
        if not peaks:
            return False
        
        last_peak = peaks[-1]
        volume_at_peak = df.loc[last_peak, 'volume']
        avg_volume = df['volume'].mean()
        
        # Check for blowoff top or selling climax
        if df['price'].iloc[-1] < df.loc[last_peak, 'price'] and volume_at_peak > 1.5 * avg_volume:
            return True
        return False

    def _calculate_pressure(self, stock, lookback_days):
        """
        Calculate the pressure of a move (buying or selling).
        
        Args:
            stock (str): Stock symbol.
            lookback_days (int): Number of days to look back.
            
        Returns:
            float: Pressure score (positive for buying, negative for selling).
        """
        df = self.data.get(stock)
        if df is None or len(df) < lookback_days:
            return 0.0
        
        df = df.tail(lookback_days)
        price_change = (df['price'].iloc[-1] - df['price'].iloc[0]) / df['price'].iloc[0]
        volume_factor = df['volume'].mean() / self.data[stock]['volume'].mean()
        pressure = price_change * volume_factor
        return pressure


    ####### This method does not work well, whats the issue here? #######
    def _detect_new_money_flow(self, stock, lookback_days):
        """
        Detecting New Money Flowing Into A Stock. This is based on Analyzing the trend of Open Interest
        In conjunction with the trend of price. 
        
        Args:
            stock (str): Stock symbol.
            lookback_days (int): Number of days to look back.
            
        Returns:
            bool: True if new money is flowing in.
        """
        df = self.data.get(stock)
        if df is None or len(df) < lookback_days:
            return False
        
        df = df.tail(lookback_days)
        price_trend = self._calculate_trend_decomposition(df, 'price', 'date')
        oi_trend = self._calculate_trend_decomposition(df, 'openinterest', 'date')
        
        if price_trend is None or oi_trend is None:
            print(f'Error: {stock}: Failed to calculate trend decomposition')
            return False
        
        return price_trend.iloc[-1] > price_trend.iloc[0] or oi_trend.iloc[-1] > oi_trend.iloc[0]

    def _detect_short_covering(self, stock, lookback_days):
        """
        Detect short covering rallies.
        
        Args:
            stock (str): Stock symbol.
            lookback_days (int): Number of days to look back.
            
        Returns:
            bool: True if short covering is detected.
        """
        df = self.data.get(stock)
        if df is None or len(df) < lookback_days:
            return False
        
        df = df.tail(lookback_days)
        price_trend = self._calculate_trend_decomposition(df, 'price', 'date')
        oi_trend = self._calculate_trend_decomposition(df, 'openinterest', 'date')
        
        if price_trend is None or oi_trend is None:
            return False
        
        return price_trend.iloc[-1] > price_trend.iloc[0] and oi_trend.iloc[-1] < oi_trend.iloc[0]

    def _detect_liquidation(self, stock, lookback_days):
        """
        Detect liquidation events.
        
        Args:
            stock (str): Stock symbol.
            lookback_days (int): Number of days to look back.
            
        Returns:
            bool: True if liquidation is detected.
        """
        df = self.data.get(stock)
        if df is None or len(df) < lookback_days:
            return False
        
        df = df.tail(lookback_days)
        price_trend = self._calculate_trend_decomposition(df, 'price', 'date')
        oi_trend = self._calculate_trend_decomposition(df, 'openinterest', 'date')
        
        if price_trend is None or oi_trend is None:
            return False
        
        return price_trend.iloc[-1] < price_trend.iloc[0] and oi_trend.iloc[-1] < oi_trend.iloc[0]

    def _analyze_oi_activity(self, stock, lookback_days):
        """
        Analyze open interest activity as bullish or bearish.
        
        Args:
            stock (str): Stock symbol.
            lookback_days (int): Number of days to look back.
            
        Returns:
            str: Activity label ('bullish', 'bearish', 'neutral').
        """
        df = self.data.get(stock)
        if df is None or len(df) < lookback_days:
            return 'neutral'
        
        df = df.tail(lookback_days)
        price_trend = self._calculate_trend_decomposition(df, 'price', 'date')
        oi_trend = self._calculate_trend_decomposition(df, 'openinterest', 'date')
        
        if price_trend is None or oi_trend is None:
            return 'neutral'
        
        price_rising = price_trend.iloc[-1] > price_trend.iloc[0]
        oi_rising = oi_trend.iloc[-1] > oi_trend.iloc[0]
        
        if price_rising and oi_rising:
            return 'bullish'
        elif not price_rising and oi_rising:
            return 'bearish'
        else:
            return 'neutral'

    def _detect_high_oi(self, stock, lookback_days, percentile=95):
        """
        Detect unusually high open interest.
        
        Args:
            stock (str): Stock symbol.
            lookback_days (int): Number of days to look back.
            percentile (int): Percentile threshold for high OI.
            
        Returns:
            bool: True if OI is unusually high.
        """
        df = self.data.get(stock)
        if df is None or len(df) < lookback_days:
            return False
        
        oi_series = df['openinterest'].dropna()
        if oi_series.empty:
            return False
        
        threshold = np.percentile(oi_series, percentile)
        current_oi = oi_series.iloc[-1]
        return current_oi > threshold

    def _detect_oi_increase(self, stock, lookback_days, increase_threshold=0.2):
        """
        Detect unusual increase in open interest.
        
        Args:
            stock (str): Stock symbol.
            lookback_days (int): Number of days to look back.
            increase_threshold (float): Minimum percentage increase.
            
        Returns:
            bool: True if unusual increase is detected.
        """
        df = self.data.get(stock)
        if df is None or len(df) < lookback_days:
            return False
        
        df = df.tail(lookback_days)
        oi_start = df['openinterest'].iloc[0]
        oi_end = df['openinterest'].iloc[-1]
        if oi_start == 0:
            return False
        increase = (oi_end - oi_start) / oi_start
        return increase > increase_threshold

    def analyze_market_sentiment(self, group, lookback_days=30, top_n=None):
        """
        Analyze market sentiment for a group of stocks.
        
        Args:
            group (list): List of stock symbols.
            lookback_days (int): Number of days to look back.
            top_n (int): Number of stocks to report (optional).
            
        Returns:
            dict: Analysis results for each stock.
            
        Raises:
            ValueError: If group is empty or lookback_days is invalid.
        """
        if not isinstance(group, (list, tuple)) or not group:
            raise ValueError("Group must be a non-empty list or tuple of stock symbols")
        if lookback_days < self.period:
            raise ValueError(f"lookback_days must be >= period ({self.period})")
            
        results = {}
        for stock in group:
            try:
                df = self.data.get(stock)
                if df is None:
                    if self.verbose:
                        print(f"No data found for stock {stock}")
                    continue
                
                if len(df) < lookback_days:
                    if self.verbose:
                        print(f"Insufficient data points for stock {stock}")
                    continue
                    
                df_subset = df.tail(lookback_days)
                
                try:
                    sentiment = self._overall_sentiment(stock, lookback_days)
                    intensity = self._intensity_behind_move(stock, lookback_days)
                    divergence = self._detect_divergence(stock, lookback_days)
                    reversal = self._detect_reversal(stock, lookback_days)
                    pressure = self._calculate_pressure(stock, lookback_days)
                    new_money_flow = self._detect_new_money_flow(stock, lookback_days)
                    short_covering = self._detect_short_covering(stock, lookback_days)
                    liquidation = self._detect_liquidation(stock, lookback_days)
                    oi_activity = self._analyze_oi_activity(stock, lookback_days)
                    high_oi = self._detect_high_oi(stock, lookback_days)
                    oi_increase = self._detect_oi_increase(stock, lookback_days)
                
                    # Detect anomalies in price residuals
                    try:
                        decomposition = seasonal_decompose(df_subset['price'], model=self.trend_model, period=self.period)
                        residuals = decomposition.resid.dropna()
                        anomalies = self._detect_anomalies(pd.DataFrame({'residuals': residuals}), 'residuals')
                    except Exception as e:
                        if self.verbose:
                            print(f"Error detecting anomalies for {stock}: {e}")
                        anomalies = []
                    
                    # Find peaks in price
                    try:
                        peaks = self._find_peaks(df_subset, 'price', prominence=0.05 * df_subset['price'].std(), distance=5)
                    except Exception as e:
                        if self.verbose:
                            print(f"Error finding peaks for {stock}: {e}")
                        peaks = []
                    
                    results[stock] = {
                        'sentiment': sentiment,
                        'intensity': intensity,
                        'divergence': divergence,
                        'reversal': reversal,
                        'pressure': pressure,
                        'new_money_flow': new_money_flow,
                        'short_covering': short_covering,
                        'liquidation': liquidation,
                        'oi_activity': oi_activity,
                        'high_oi': high_oi,
                        'oi_increase': oi_increase,
                        'anomalies': anomalies,
                        'peaks': peaks
                    }
                except Exception as e:
                    if self.verbose:
                        print(f"Error calculating metrics for {stock}: {e}")
                    continue
                    
            except Exception as e:
                if self.verbose:
                    print(f"Error processing stock {stock}: {e}")
                continue
        
        if top_n:
            # Sort by intensity and take top_n
            sorted_results = sorted(results.items(), key=lambda x: x[1]['intensity'], reverse=True)[:top_n]
            results = dict(sorted_results)
        
        return results

    def ams(self, group, lookback_days=30, top_n=None):
        """
        Analyze market sentiment for a group of stocks.
        
        Args:
            group (list): List of stock symbols.
            lookback_days (int): Number of days to look back.
            top_n (int): Number of stocks to report (optional).
            
        Returns: Th
            dict: Analysis results for each stock.
            
        Raises:
            ValueError: If group is empty or lookback_days is invalid.
        """
        if not isinstance(group, (list, tuple)) or not group:
            raise ValueError("Group must be a non-empty list or tuple of stock symbols")
        if lookback_days < self.period:
            raise ValueError(f"lookback_days must be >= period ({self.period})")
            
        results = {}
        for stock in group:
            try:
                df = self.data.get(stock)
                if df is None:
                    if self.verbose:
                        print(f"No data found for stock {stock}")
                    continue
                    
                if len(df) < lookback_days:
                    if self.verbose:
                        print(f"Insufficient data points for stock {stock}")
                    continue
                        
                # Since 'date' is the index, no need to set it again
                df_subset = df.tail(lookback_days)
                
                # Sentiment analysis
                sentiment = self._overall_sentiment(stock, lookback_days)
                intensity = self._intensity_behind_move(stock, lookback_days)
                divergence = self._detect_divergence(stock, lookback_days)
                reversal = self._detect_reversal(stock, lookback_days)
                pressure = self._calculate_pressure(stock, lookback_days)
                new_money_flow = self._detect_new_money_flow(stock, lookback_days)
                short_covering = self._detect_short_covering(stock, lookback_days)
                liquidation = self._detect_liquidation(stock, lookback_days)
                oi_activity = self._analyze_oi_activity(stock, lookback_days)
                high_oi = self._detect_high_oi(stock, lookback_days)
                oi_increase = self._detect_oi_increase(stock, lookback_days)

                # Detect anomalies in price residuals
                try:
                    # 'date' is already the index
                    decomposition = seasonal_decompose(df_subset['price'], model=self.trend_model, period=self.period)
                    residuals = decomposition.resid.dropna()
                    anomalies = self._detect_anomalies(pd.DataFrame({'residuals': residuals}, index=residuals.index), 'residuals')
                except Exception as e:
                    if self.verbose:
                        print(f"Error detecting anomalies for {stock}: {e}")
                    anomalies = []
                    
                # Find peaks
                peaks = self._find_peaks(df_subset, 'price', prominence=0.05 * df_subset['price'].std(), distance=5)
                
                results[stock] = {
                    'sentiment': sentiment,
                    'intensity': intensity,
                    'divergence': divergence,
                    'reversal': reversal,
                    'pressure': pressure,
                    'new_money_flow': new_money_flow,
                    'short_covering': short_covering,
                    'liquidation': liquidation,
                    'oi_activity': oi_activity,
                    'high_oi': high_oi,
                    'oi_increase': oi_increase,
                    'anomalies': anomalies,
                    'peaks': peaks
                }
                if top_n:
                    # Sort by intensity and take top_n
                    sorted_results = sorted(results.items(), key=lambda x: x[1]['intensity'], reverse=True)[:top_n]
                    results = dict(sorted_results)
                    return results
            except Exception as e:
                if self.verbose:
                    print(f"Error processing stock {stock}: {e}")
                continue
        
        if top_n:
            sorted_results = sorted(results.items(), key=lambda x: x[1]['intensity'], reverse=True)[:top_n]
            results = dict(sorted_results)
            
        return results
    


# Example usage
if __name__ == "__main__":
    # Sample data (replace with actual data)
    sample_data = {
        'AAPL': pd.DataFrame({
            'date': pd.date_range(start='2023-01-01', periods=100),
            'price': np.random.randn(100).cumsum() + 100,
            'volume': np.random.randint(1000, 10000, 100),
            'openinterest': np.random.randint(1000, 10000, 100)
        })
    }
    analyzer = MarketAnalyzer(sample_data)
    results = analyzer.analyze_market_sentiment(group=['AAPL'], lookback_days=30)
    print(results)