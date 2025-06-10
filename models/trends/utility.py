from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from pandas_market_calendars import get_calendar
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[3]))

from bin.models.trends.data_utility import StockData, DataUtility
from bin.models.trends.Detect_trend import TrendAnalyzer
from bin.models.trends.Detect_class import Classifier
from bin.models.trends.Detect_peak import PeakDetector
from bin.main import get_path

logging.basicConfig(
    level=logging.DEBUG,  # Increased to DEBUG for detailed logging
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class TrendUtility:
    """Utility for analyzing trends in stock data across multiple periods."""
    periods_config: Dict[str, Dict] = None
    classification_logs: list = field(default_factory=list)

    def __post_init__(self):
        """Initialize trend analyzers for each configured period."""
        if self.periods_config is None:
            self.periods_config = {
                'short_term': {
                    'lookback': 30,
                    'window_size': 5,
                    'analyzer_period': 3,
                    'min_data_points': 15,
                    'min_unique_values': 2  # Stricter requirement for short_term
                },
                'long_term': {
                    'lookback': 90,
                    'window_size': 30,
                    'analyzer_period': 21,
                    'min_data_points': 90,
                    'min_unique_values': 10
                },
                'ytd': {
                    'lookback': None,  # Computed dynamically
                    'window_size': 150,
                    'analyzer_period': 10,
                    'min_data_points': 20,
                    'min_unique_values': 5
                }
            }

        self.calendar = get_calendar('NYSE')
        self.analyzers = {
            period: TrendAnalyzer(period=config['analyzer_period'])
            for period, config in self.periods_config.items()
        }

    def get_metric_maps(self, price_data: pd.DataFrame, options_data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Get a mapping of metric names to their corresponding data series.

        Args:
            price_data (pd.DataFrame): OHLCV data with 'Close', 'Volume', 'returns'.
            options_data (pd.DataFrame): Option data with 'total_vol', 'total_oi', etc.

        Returns:
            Dict[str, pd.Series]: Mapping of metric names to data series.
        """
        metrics = {
            'price': price_data['Close'],
            'returns': price_data['Returns'],
            'stock_volume': price_data['Volume'],
            'options_volume': options_data['total_vol'],
            'oi': options_data['total_oi'],
            'pcr_volume': options_data['pcr_volume'],
            'pcr_oi': options_data['pcr_oi'],
            'total_vol_oi': options_data['total_vol_oi'], 
            'atm_iv': options_data['atm_iv'],
            'call_volume': options_data['call_vol'],
            'put_volume': options_data['put_vol']
        }
        return {k: v for k, v in metrics.items()}
    

    def get_metric_change_maps(self, price_data: pd.DataFrame, options_data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Get a mapping of metric names to their corresponding change series.

        Args:
            price_data (pd.DataFrame): OHLCV data with 'Close', 'Volume', 'returns'.
            options_data (pd.DataFrame): Option data with 'total_vol', 'total_oi', etc.

        Returns:
            Dict[str, pd.Series]: Mapping of metric names to change series.
        """
        metrics = {
            'price': price_data['Close'].diff(),
            'returns': price_data['Returns'].diff(),
            'stock_volume': price_data['Volume'],
            'options_volume': options_data['total_vol_chng'],
            'oi': options_data['total_oi_chng'],
            'pcr_volume': options_data['pcr_volume'].diff(),
            'pcr_oi': options_data['pcr_oi'].diff(),
            'total_vol_oi': options_data['total_vol_oi'].diff(), 
            'atm_iv': options_data['atm_iv_chng'],
            'call_volume': options_data['call_vol_chng'],
            'put_volume': options_data['put_vol_chng']
        }
        return {k: v for k, v in metrics.items() }


    def compute_ytd_lookback(self, backtest_date: Optional[datetime] = None) -> int:
        """
        Compute the number of trading days from the start of the year to the backtest date.

        Args:
            backtest_date (datetime, optional): Backtest date. Defaults to current date.

        Returns:
            int: Number of trading days for YTD period.
        """
        if backtest_date is None:
            backtest_date = pd.to_datetime(datetime.now().date())
        else:
            backtest_date = pd.to_datetime(backtest_date)

        year_start = pd.to_datetime(f"{backtest_date.year}-01-01")
        trading_days = self.calendar.valid_days(
            start_date=year_start,
            end_date=backtest_date
        ).tz_localize(None)
        return len(trading_days)

    def analyze_trends(self, stock_data: StockData) -> Dict:
        """
        Analyze trends for a stock across configured periods using price, volume, and option data.

        Args:
            stock_data (StockData): Stock data containing ohlcv, option_df, and returns.

        Returns:
            Dict: Trend metrics for each period, including trend direction, seasonality, slope,
                  category, and blowoff status.
        """
        try:
            ohlcv = stock_data.ohlcv
            option_df = stock_data.option_df
            stock = stock_data.stock
            backtest_date = stock_data.backtest_date
            peak_detection = PeakDetector(prominence=0.01, distance=2)
            current_date = ohlcv.index[-1].strftime("%Y-%m-%d") 

            if ohlcv.empty or option_df.empty:
                logger.debug(f"Empty OHLCV or option data for {stock}")
                return {}

            results = {}
            for period, config in self.periods_config.items():
                try:
                    lookback = config['lookback']
                    min_data_points = config['min_data_points']
                    min_unique_values = config['min_unique_values']
                    if period == 'ytd':
                        lookback = self.compute_ytd_lookback(current_date)
                        if lookback < min_data_points:
                            logger.debug(f"Insufficient YTD data for {stock}: {lookback} trading days, need {min_data_points}")
                            results[period] = {}
                            continue

                    if len(ohlcv) < min_data_points or len(option_df) < min_data_points:
                        logger.debug(f"Insufficient data for {period} analysis of {stock}: need at least {min_data_points} days")
                        results[period] = {}
                        continue

                    period_ohlcv = ohlcv.tail(lookback)
                    period_option_df = option_df.tail(lookback)

                    if period_ohlcv.empty or period_option_df.empty:
                        logger.debug(f"Empty data after truncation for {period} analysis of {stock}")
                        results[period] = {}
                        continue

                    metrics = self.get_metric_maps(period_ohlcv, period_option_df)
                    period_results = {}

                    for metric_name, data in metrics.items():
                        try:
                            logger.debug(f"Analyzing {metric_name} for {stock} in {period}: len={len(data)}, non-NaN={data.count()}, unique={data.nunique()}, min={data.min()}, max={data.max()}")
                            if data.empty or data.count() < min_data_points:
                                logger.debug(f"Insufficient non-NaN data for {metric_name} in {period} for {stock}: {data.count()} valid points")
                                period_results[metric_name] = {}
                                continue

                            # Skip if too few unique values
                            if data.nunique() < min_unique_values:
                                logger.debug(f"Skipping {metric_name} in {period} for {stock}: only {data.nunique()} unique values, need {min_unique_values}")
                                period_results[metric_name] = {}
                                continue

                            # Fill NaN values for analysis
                            data_filled = data.copy()
                            if len(data_filled) < min_data_points or data_filled.count() < min_data_points:
                                logger.debug(f"Insufficient data after filling NaNs for {metric_name} in {period} for {stock}: {data_filled.count()} valid points")
                                period_results[metric_name] = {}
                                continue

                            # Check for identical values after filling
                            if data_filled.max() == data_filled.min():
                                logger.debug(f"Skipping {metric_name} in {period} for {stock}: all values identical after filling ({data_filled.max()})")
                                period_results[metric_name] = {}
                                continue

                            # Additional length check for analyzer
                            if len(data_filled) < config['analyzer_period'] + 1:
                                logger.debug(f"Skipping {metric_name} in {period} for {stock}: data length {len(data_filled)} too short for analyzer period {config['analyzer_period']}")
                                period_results[metric_name] = {}
                                continue

                            trend_direction, seasonality, slope = self.analyzers[period].analyze(data_filled)
                            classifier = Classifier(
                                data=data_filled,
                                lookback=lookback,
                                period=config['analyzer_period'],
                                window_size=config['window_size']
                            )
                            category, blowoff, classification_log = classifier.classify(stock, metric_name)
                            peak_data = peak_detection.get_peak_data(data)
                            peaks = peak_data.peak_values
                            valleys = peak_data.valley_values
                            change_points = peak_data.changepoint_values
                            data_value = data_filled.iloc[-1] if not data_filled.empty else None
                            self.classification_logs.append(classification_log)

                            period_results[metric_name] = {
                                'value': data_value,
                                'trend': trend_direction,
                                'seasonality': seasonality,
                                'slope': slope,
                                'status': category,
                                'blowoff': blowoff,
                                'peaks': peaks,
                                'valleys': valleys,
                                'change_point': change_points
                            }
                        except Exception as e:
                            # Find the exact line the error occurred
                            line = sys.exc_info()[-1].tb_lineno
                            logger.error(f"Error analyzing {metric_name} for {stock} in {period} at line {line}: {str(e)}")
                            logger.debug(f"Error analyzing {metric_name} for {stock} in {period}: {str(e)}")
                            period_results[metric_name] = {}

                    results[period] = period_results

                except Exception as e:
                    line = sys.exc_info()[-1].tb_lineno
                    logger.debug(f"Error in {period} analysis for {stock}: {str(e)}, line {line}")
                    results[period] = {}

            if not any(results.values()):
                logger.debug(f"No trend analysis results generated for {stock}")
                return {}

            return {current_date: results}

        except Exception as e:
            line = sys.exc_info()[-1].tb_lineno
            logger.error(f"Error in trend analysis for {stock}: {str(e)}, line {line}")
            return {}


if __name__ == "__main__":
    import json 
    connections = get_path()
    data_util = DataUtility(connections)
    trend_util = TrendUtility()
    stock = 'AAPL'
    backtest_date = '2025-03-01'

    try:
        stock_data = data_util.get_aligned_data(stock, backtest_date=backtest_date)
        trends = trend_util.analyze_trends(stock_data)
        trends = {k: {k2: str(v2) for k2, v2 in v.items()} for k, v in trends.items()}
        print(f"Trend Analysis for {stock} ({backtest_date}):\n{json.dumps(trends, indent=2)}\n")

        stock_data_full = data_util.get_aligned_data(stock)
        trends_full = trend_util.analyze_trends(stock_data_full)
        trends_full = {k: {k2: str(v2) for k2, v2 in v.items()} for k, v in trends_full.items()}
        print(f"Full Trend Analysis for {stock}:\n{json.dumps(trends_full, indent=2)}\n")

    except ValueError as e:
        logger.error(f"Error in main execution: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error in main execution: {str(e)}")
        raise
    finally:
        data_util.data_manager.close_connection()