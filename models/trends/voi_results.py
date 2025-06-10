





from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
import pandas as pd
import numpy as np
from tqdm import tqdm
import logging
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[3]))

from bin.models.trends.Detect_trend import TrendAnalyzer
from bin.models.trends.Detect_peak import PeakDetector
from bin.models.trends.voi import VolumeOpenInterestWorksheet
from bin.models.trends.Detect_class import Classifier, ClassificationLog
from datetime import datetime
from bin.main import get_path
from main import Manager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

@dataclass
class TrendResult:
    stock: str
    name: str
    trend_direction: str
    seasonality: str
    slope: float
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
        self.data_manager = Manager(connections)
        self.stocks = self.data_manager.Pricedb.stocks['equities']
        self.trend_analyzer = TrendAnalyzer(period=self.period)
        self.peak_detector = PeakDetector(prominence=0.5, distance=2)
        self.all_worksheets = []  # Persistent list to store all worksheets

    def get_aligned_data(self, stock: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        ohlcv = self.data_manager.Pricedb.ohlc(stock).dropna().sort_index()
        ohlcv['close_chng'] = ohlcv['Close'].diff()
        ohlcv['returns'] = ohlcv['Close'].pct_change()
        ohlcv = ohlcv.tail(self.lookback_days)
        option_db = self.data_manager.Optionsdb.get_daily_option_stats(stock).dropna().sort_index()
        if ohlcv.empty or option_db.empty:
            raise ValueError(f"No data available for {stock}")
        return ohlcv, option_db
    
    def analyze_single_stock(self, stock: str) -> Optional[List[TrendResult]]:
        try:
            ohlcv, option_db = self.get_aligned_data(stock)
            metrics = {
                'returns': ohlcv['returns'],
                'close_prices': ohlcv['Close'],
                'stock_volume': ohlcv['Volume'],
                'options_volume': option_db['total_vol'],
                'oi': option_db['total_oi'],
                'atm_iv': option_db['atm_iv'], 
                'call_oi': option_db['call_oi'],
                'put_oi': option_db['put_oi'],
                'call_volume': option_db['call_vol'],
                'put_volume': option_db['put_vol'],
            }
            chng_map = {
                'close_prices': None,
                'stock_volume': option_db['total_vol_chng'],
                'options_volume': option_db['total_vol_chng'],
                'oi': option_db['total_oi_chng'],
                'atm_iv': option_db['atm_iv_chng'],
                'call_oi': option_db['call_oi_chng'],
                'put_oi': option_db['put_oi_chng'],
                'call_volume': option_db['call_vol_chng'],
                'put_volume': option_db['put_vol_chng'],
            }

            results = []
            price_data = ohlcv['Close']
            price_chng_data = ohlcv['close_chng']
            returns_data = ohlcv['returns']
            returns_chng_data = ohlcv['returns'].diff().bfill()

            # Compute PCR volume and OI ratios
            pcr_vol = option_db['put_vol'] / option_db['call_vol'].replace(0, pd.NA)
            pcr_vol = pcr_vol.fillna(0.0)
            pcr_oi = option_db['put_oi'] / option_db['call_oi'].replace(0, pd.NA)
            pcr_oi = pcr_oi.fillna(0.0)

            # Store trend directions for relevant metrics
            trend_directions = {}

            # Classify all metrics, including PCRs
            classification_logs = {}
            for metric_name, data in metrics.items():
                classifier = Classifier(
                    data=data,
                    open_interest=chng_map.get(metric_name),
                    lookback=self.lookback_days,
                    period=self.period,
                    window_size=self.window_size
                )
                category, blowoff, log_entry = classifier.classify(stock, metric_name)
                classification_logs[metric_name] = log_entry

                trend_direction, seasonality, slope = self.trend_analyzer.analyze(data)
                # Store trend direction for relevant metrics
                if metric_name in ['close_prices', 'stock_volume', 'oi', 'options_volume']:
                    trend_directions[metric_name] = trend_direction

                peaks = self.peak_detector.find_peaks(data.values)
                peak_dates = data.index[peaks]
                valleys = self.peak_detector.find_valleys(data.values)
                valley_dates = data.index[valleys]

                trend_results = TrendResult(
                    stock=stock,
                    name=metric_name,
                    trend_direction=trend_direction,
                    seasonality=seasonality,
                    slope=slope,
                    valley=valley_dates.max() if len(valley_dates) > 0 else None,
                    peaks=peak_dates.max() if len(peak_dates) > 0 else None,
                    metric_status={'Low': 'below', 'Average': 'at', 'High': 'above', 'Blowoff': 'above'}[category],
                    blowoff_flag=blowoff
                )
                ### End Loop appending TrendResults. 
                results.append(trend_results)

            # Classify PCRs
            classifier_pcr_vol = Classifier(
                data=pcr_vol,
                lookback=self.lookback_days,
                period=self.period,
                window_size=self.window_size
            )
            _, _, pcr_vol_log = classifier_pcr_vol.classify(stock, 'pcr_vol')
            classification_logs['pcr_vol'] = pcr_vol_log

            classifier_pcr_oi = Classifier(
                data=pcr_oi,
                lookback=self.lookback_days,
                period=self.period,
                window_size=self.window_size
            )
            _, _, pcr_oi_log = classifier_pcr_oi.classify(stock, 'pcr_oi')
            classification_logs['pcr_oi'] = pcr_oi_log

            # Create a single worksheet per stock with all metrics
            worksheet = VolumeOpenInterestWorksheet(
                stock=stock,
                classification_logs=classification_logs,
                price_data=price_data,
                price_chng_data=price_chng_data,
                volume_data=metrics['stock_volume'],
                volume_chng_data=chng_map['stock_volume'],
                oi_data=metrics['oi'],
                oi_chng_data=chng_map['oi'],
                option_volume_data=metrics['options_volume'],
                option_volume_chng_data=chng_map['options_volume'],
                put_volume=option_db['put_vol'],
                call_volume=option_db['call_vol'],
                put_oi=option_db['put_oi'],
                call_oi=option_db['call_oi'],
                trend_directions=trend_directions  # Pass the trend directions
            )
            self.all_worksheets.append(worksheet)  # Append to persistent list

            self.results_worksheet = ResultsWorksheet(worksheets=self.all_worksheets, timestamp=datetime.now())
            return results
        except Exception as e:
            logging.error(f"Error analyzing {stock}: {str(e)}")
            return None
        
    def __convert_to_dataframe(self, results: List[TrendResult]) -> pd.DataFrame:
        data = []
        for i in results:
            for result in i:
                data.append({
                    'stock': result.stock,
                    'name': result.name,  # Changed from 'metric' to 'name' to match TrendResult and trend_summary.py
                    'trend_direction': result.trend_direction,
                    'seasonality': result.seasonality,
                    'slope': result.slope,
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
        self.all_worksheets = []  # Reset the list for a new analysis
        
        pbar = tqdm(stocks_to_analyze, desc='Analyzing Stocks')
        for stock in pbar:
            pbar.set_description(f'Processing {stock}')
            result = self.analyze_single_stock(stock)
            if result:
                results.append(result)
                pbar.set_postfix({'Success': True})

        self.result_df = self.__convert_to_dataframe(results)
        return self.result_df, self.results_worksheet
    
def main():
    """Example usage of the TResults class."""
    import json 
    connections = get_path()
    stocks = json.load(open(connections['ticker_path'], 'r'))['mag8']
    detector = TResults(connections, lookback_days=500)
    results_df, worksheet = detector.analyze_stocks(stocks=stocks)
    print("\nTrend Analysis Results:")
    print(results_df)
    print("\nWorksheet Summary:")
    print(f"Total Bullish Signals: {worksheet.total_bullish}")
    print(f"Total Bearish Signals: {worksheet.total_bearish}")
    print("\nWorksheet Details (Cross-Stock Analysis):")
    wdf = worksheet.to_df()
    print("Bullish Stocks:")
    print(wdf[wdf.Bullish == True] if not wdf[wdf.Bullish == True].empty else "No Bullish Stocks")
    print("Bearish Stocks:")
    print(wdf[wdf.Bearish == True])
    print("\nWorksheet DataFrame:")
    print(wdf)
    cols = [
    'Stock',
    'Price_Trend_Direction',
    'Volume_Category',
    'Volume_Trend_Direction',
    'OI_Category',
    'OI_Trend_Direction',
    'Option_Volume_Category',
    'Option_Volume_Trend_Direction',
    "Bullish",
    "Bearish",
    ]
    print(wdf[cols])


if __name__ == "__main__":
    main()