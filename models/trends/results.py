from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
import pandas as pd
import numpy as np
from tqdm import tqdm
import logging
from pandas_market_calendars import get_calendar
from datetime import datetime

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[3]))

from bin.models.trends.utility import TrendUtility, DataUtility
from bin.models.trends.Detect_class import Classifier, ClassificationLog
from bin.models.trends.result_signals import VolumeOpenInterestWorksheet
from bin.main import get_path
from main import Manager

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

@dataclass
class TrendResult:
    stock: str
    name: str
    trend_direction: str
    seasonality: str
    slope: float
    change_point: Optional[float] = None
    valley: Any = None
    peaks: Any = None
    metric_status: Optional[str] = None
    blowoff_flag: bool = False

    def to_dict(self) -> Dict:
        return {
            "stock": self.stock,
            "name": self.name,
            "trend_direction": self.trend_direction,
            "seasonality": self.seasonality,
            "slope": self.slope,
            "change_point": self.change_point,
            "valley": self.valley,
            "peaks": self.peaks,
            "metric_status": self.metric_status,
            "blowoff_flag": self.blowoff_flag
        }

@dataclass
class ResultsWorksheet:
    worksheets: List[VolumeOpenInterestWorksheet]
    timestamp: datetime
    total_bullish: int = None
    total_bearish: int = None

    def __post_init__(self):
        self.total_bullish = sum(ws.bullish_signals for ws in self.worksheets)
        self.total_bearish = sum(ws.bearish_signals for ws in self.worksheets)

    def to_df(self) -> pd.DataFrame:
        if not self.worksheets:
            return pd.DataFrame()
        return pd.concat([ws.to_df() for ws in self.worksheets], ignore_index=True)

    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "total_bullish": self.total_bullish,
            "total_bearish": self.total_bearish,
            "worksheets": [ws.to_df().to_dict(orient='records') for ws in self.worksheets]
        }

class TResults:
    def __init__(self, connections: Dict|str, lookback_days: int = 90, window_size: int = 30, period: int = 3):
        self.lookback_days = lookback_days
        self.window_size = window_size
        self.period = period
        self.data_utility = DataUtility(connections)
        self.trend_utility = TrendUtility()
        self.stocks = self.data_utility.stocks
        self.all_worksheets = []
        self.returns_df = []
        self.results_worksheet = ResultsWorksheet(worksheets=[], timestamp=datetime.now())

    def get_aligned_data(self, stock: str, backtest_date: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        ohlcv, option_df = self.data_utility.get_aligned_data(stock, backtest_date=backtest_date)
        if backtest_date is not None: 
            self.returns_df.append(self.data_utility.returns_df.copy())
        return ohlcv, option_df

    def analyze_single_stock(self, stock: str) -> Optional[List[TrendResult]]:
        try:
            ohlcv, option_df = self.get_aligned_data(stock)
            if ohlcv is None or option_df is None:
                logger.warning(f"Skipping analysis for {stock} due to invalid or missing data")
                return None

            min_data_points = 15
            if ohlcv['Price'].count() < min_data_points or option_df['total_vol'].count() < min_data_points:
                logger.warning(f"Insufficient non-NaN data for {stock}: OHLCV {ohlcv['Price'].count()}, Option {option_df['total_vol'].count()}")
                return None


            stock_data = self.data_utility.get_aligned_data(stock, backtest_date='2025-03-01' if self.returns_df else None)
            trend_results = self.trend_utility.analyze_trends(stock_data)
            if not trend_results:
                logger.warning(f"No trend analysis results for {stock}")
                return None

            # Align metrics with utility.py's analyzed metrics
            metrics = {
                'returns': ohlcv['returns'],
                'close_prices': ohlcv['Price'],
                'stock_volume': ohlcv['Volume'],
                'options_volume': option_df['total_vol'],
                'oi': option_df['total_oi'],
                'total_vol_oi': option_df['total_vol_oi'],
                'pcr_volume': option_df['pcr_volume'],
                'pcr_oi': option_df['pcr_oi']
            }
            # Additional metrics for VolumeOpenInterestWorksheet
            supplementary_metrics = {
                'atm_iv': option_df['atm_iv'],
                'call_oi': option_df['call_oi'],
                'put_oi': option_df['put_oi'],
                'call_volume': option_df['call_vol'],
                'put_volume': option_df['put_vol']
            }
            default_chng = pd.Series(0.0, index=ohlcv.index)
            chng_map = {
                'close_prices': default_chng,
                'stock_volume': default_chng,
                'options_volume': default_chng,
                'oi': default_chng,
                'total_vol_oi': default_chng,
                'pcr_volume': default_chng,
                'pcr_oi': default_chng,
                'atm_iv': default_chng,
                'call_oi': default_chng,
                'put_oi': default_chng,
                'call_volume': default_chng,
                'put_volume': default_chng
            }

            results = []
            price_data = ohlcv['Price']
            price_chng_data = ohlcv['close_chng']
            returns_data = ohlcv['returns']
            returns_chng_data = ohlcv['returns'].diff().bfill()

            pcr_vol = option_df['pcr_volume']
            pcr_oi = option_df['pcr_oi']

            trend_directions = {}
            classification_logs = {}

            # Process analyzed metrics
            for metric_name, data in metrics.items():
                if data is None or data.empty or data.count() < min_data_points:
                    logger.warning(f"Skipping {metric_name} for {stock} due to missing or insufficient non-NaN data")
                    continue

                # Try long_term, then short_term, then ytd
                long_term_results = trend_results.get('long_term', {}).get(metric_name, {})
                short_term_results = trend_results.get('short_term', {}).get(metric_name, {})
                ytd_results = trend_results.get('ytd', {}).get(metric_name, {})
                
                results_data = long_term_results or short_term_results or ytd_results or {}
                if not results_data:
                    logger.warning(f"Skipping {metric_name} for {stock} due to empty trend results in all periods")
                    continue

                classifier = Classifier(
                    data=data,
                    open_interest=chng_map.get(metric_name),
                    lookback=self.lookback_days,
                    period=self.period,
                    window_size=self.window_size
                )
                category, blowoff, log_entry = classifier.classify(stock, metric_name)
                classification_logs[metric_name] = log_entry

                trend_direction = results_data.get('trend', 'unknown')
                seasonality = results_data.get('seasonality', 'unknown')
                slope = results_data.get('slope', 0.0)
                change_point = results_data.get('change_point')
                peaks = results_data.get('peaks')
                valleys = results_data.get('valleys')

                trend_directions[metric_name] = trend_direction

                trend_result = TrendResult(
                    stock=stock,
                    name=metric_name,
                    trend_direction=trend_direction,
                    seasonality=seasonality,
                    slope=slope,
                    change_point=change_point,
                    valley=valleys,
                    peaks=peaks,
                    metric_status={'Low': 'below', 'Average': 'at', 'High': 'above', 'Blowoff': 'above'}.get(category, 'unknown'),
                    blowoff_flag=blowoff
                )
                results.append(trend_result)

            # Process supplementary metrics for classification only
            for metric_name, data in supplementary_metrics.items():
                if data is None or data.empty or data.count() < min_data_points:
                    logger.warning(f"Skipping {metric_name} for {stock} due to missing or insufficient non-NaN data")
                    continue

                classifier = Classifier(
                    data=data,
                    open_interest=chng_map.get(metric_name),
                    lookback=self.lookback_days,
                    period=self.period,
                    window_size=self.window_size
                )
                category, blowoff, log_entry = classifier.classify(stock, metric_name)
                classification_logs[metric_name] = log_entry

                # No trend analysis for supplementary metrics
                trend_result = TrendResult(
                    stock=stock,
                    name=metric_name,
                    trend_direction='unknown',  # Explicitly set as not analyzed
                    seasonality='unknown',
                    slope=0.0,
                    change_point=None,
                    valley=None,
                    peaks=None,
                    metric_status={'Low': 'below', 'Average': 'at', 'High': 'above', 'Blowoff': 'above'}.get(category, 'unknown'),
                    blowoff_flag=blowoff
                )
                results.append(trend_result)

            if not results:
                logger.warning(f"No valid trend results generated for {stock}")
                return None

            if pcr_vol.count() >= min_data_points and pcr_vol.nunique() >= 5:
                classifier_pcr_vol = Classifier(
                    data=pcr_vol,
                    lookback=self.lookback_days,
                    period=self.period,
                    window_size=self.window_size
                )
                _, _, pcr_vol_log = classifier_pcr_vol.classify(stock, 'pcr_vol')
                classification_logs['pcr_vol'] = pcr_vol_log
            else:
                logger.warning(f"Skipping pcr_volume classification for {stock}: {pcr_vol.count()} valid points, {pcr_vol.nunique()} unique values")
                classification_logs['pcr_vol'] = None

            if pcr_oi.count() >= min_data_points and pcr_oi.nunique() >= 5:
                classifier_pcr_oi = Classifier(
                    data=pcr_oi,
                    lookback=self.lookback_days,
                    period=self.period,
                    window_size=self.window_size
                )
                _, _, pcr_oi_log = classifier_pcr_oi.classify(stock, 'pcr_oi')
                classification_logs['pcr_oi'] = pcr_oi_log
            else:
                logger.warning(f"Skipping pcr_oi classification for {stock}: {pcr_oi.count()} valid points, {pcr_oi.nunique()} unique values")
                classification_logs['pcr_oi'] = None

            worksheet = VolumeOpenInterestWorksheet(
                stock=stock,
                classification_logs=classification_logs,
                price_data=price_data,
                price_chng_data=price_chng_data,
                returns_data=returns_data,
                volume_data=metrics['stock_volume'],
                volume_chng_data=chng_map['stock_volume'],
                oi_data=metrics['oi'],
                oi_chng_data=chng_map['oi'],
                option_volume_data=metrics['options_volume'],
                option_volume_chng_data=chng_map['options_volume'],
                put_volume=option_df['put_vol'],
                call_volume=option_df['call_vol'],
                put_oi=option_df['put_oi'],
                call_oi=option_df['call_oi'],
                trend_directions=trend_directions
            )
            self.all_worksheets.append(worksheet)
            self.results_worksheet = ResultsWorksheet(worksheets=self.all_worksheets, timestamp=datetime.now())

            return results
        except Exception as e:
            logger.error(f"Error analyzing {stock}: {str(e)}")
            return None

    def __convert_to_dataframe(self, results: List[TrendResult]) -> pd.DataFrame:
        data = []
        for i in results:
            if i is None:
                continue
            for result in i:
                data.append({
                    'stock': result.stock,
                    'name': result.name,
                    'trend_direction': result.trend_direction,
                    'seasonality': result.seasonality,
                    'slope': result.slope,
                    'change_point': result.change_point,
                    'last_valley': result.valley,
                    'last_peak': result.peaks,
                    'metric_status': result.metric_status,
                    'blowoff_flag': result.blowoff_flag,
                    'slope_discrepancy': (
                        (result.trend_direction == 'up' and result.slope < 0) or
                        (result.trend_direction == 'down' and result.slope > 0)
                    )
                })
        df = pd.DataFrame(data)
        return df

    def analyze_stocks(self, stocks: Optional[List[str]] = None) -> Tuple[pd.DataFrame, ResultsWorksheet]:
        stocks_to_analyze = stocks if stocks is not None else self.stocks
        results = []
        self.all_worksheets = []
        self.results_worksheet = ResultsWorksheet(worksheets=[], timestamp=datetime.now())

        pbar = tqdm(stocks_to_analyze, desc='Analyzing Stocks')
        for stock in pbar:
            pbar.set_description(f'Processing {stock}')
            result = self.analyze_single_stock(stock)
            if result:
                results.append(result)
                pbar.set_postfix({'Success': True})
            else:
                pbar.set_postfix({'Success': False})

        self.result_df = self.__convert_to_dataframe(results)
        return self.result_df, self.results_worksheet

def main():
    """Example usage of the TResults class."""
    import json
    connections = get_path()
    stocks = json.load(open(connections['ticker_path'], 'r'))['all_stocks']
    stocks = ['aapl']
    detector = TResults(connections, lookback_days=150)
    results_df, worksheet = detector.analyze_stocks(stocks=stocks)
    wdf = worksheet.to_df()
    print("\nTrend Analysis Results:")
    print(results_df)
    print("\nWorksheet Summary:")
    print(f"Total Bullish Signals: {worksheet.total_bullish}")
    if not wdf.empty and 'Bullish' in wdf.columns:
        print(wdf[wdf['Bullish'] == True])
    else:
        print("No bullish signals (empty or missing Bullish column)")
    print(f"Total Bearish Signals: {worksheet.total_bearish}")
    if not wdf.empty and 'Bearish' in wdf.columns:
        print(wdf[wdf['Bearish'] == True])
    else:
        print("No bearish signals (empty or missing Bearish column)")
    print("\nWorksheet Details (Cross-Stock Analysis):")
    print(wdf)
    cols = ['Stock', 'PCR_OI', 'Price_Delta', 'Volume_Category', 'OI_Category']
    if not wdf.empty and all(col in wdf.columns for col in cols):
        high_pcr_oi_bearish = wdf[(wdf['PCR_OI'] > 1) & (wdf['Bearish'])][cols]
        print("\nStocks with Elevated PCR OI (>1.0) and Bearish Signals:")
        print(high_pcr_oi_bearish if not high_pcr_oi_bearish.empty else "No stocks meet the criteria.")
    else:
        print("\nNo stocks with elevated PCR OI (>1.0) and Bearish signals (empty or missing columns)")
    results_df.to_csv("trend_analysis_results.csv", index=False)
    if not wdf.empty:
        wdf.to_csv("worksheet_results.csv", index=False)
    rdf = pd.concat(detector.returns_df) if detector.returns_df else pd.DataFrame()
    print(rdf)
    if not rdf.empty:
        rdf.to_csv("returns_data.csv", index=True)

if __name__ == "__main__":
    main()