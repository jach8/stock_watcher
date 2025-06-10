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
    seasonality: float
    change_point: float 
    valley: Any 
    peaks: Any
    metric_status: str
    blowoff_flag: bool

    def to_dict(self) -> Dict:
        return {
            "stock": self.stock,
            "name": self.name,
            "trend_direction": self.trend_direction,
            "seasonality": self.seasonality,
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
        stock_data = self.data_utility.get_aligned_data(stock, backtest_date=backtest_date)
        if backtest_date is not None: 
            self.returns_df.append(self.data_utility.returns_df.copy())
        return stock_data
    
    def analyze_stock_trend(self, stock: str, backtest_date: Optional[str] = None): 
        stock_data = self.get_aligned_data(stock, backtest_date=backtest_date)
        trend_results = self.trend_utility.analyze_trends(stock_data)
        trend_results['stock'] = stock
        return trend_results
    
    def trend_result_to_df(self, trend_result: TrendResult) -> pd.DataFrame:
        d = trend_result.copy()
        date = list(d.keys())[0]
        short_term = d[date]['short_term']
        long_term = d[date]['long_term']
        ytd = d[date]['ytd']
        st = pd.DataFrame(short_term).T
        lt = pd.DataFrame(long_term).T
        yt = pd.DataFrame(ytd).T
        st['timeframe'] = 'short_term'
        lt['timeframe'] = 'long_term'
        yt['timeframe'] = 'ytd'
        out = pd.concat([st, lt, yt]).sort_index().reset_index().rename(columns = {'index': 'name'})
        out.insert(0, 'date', date)
        out.insert(1, 'stock', trend_result['stock'])
        return out
    
    def worksheet_entry(self, stock: str, trend_results: Dict) -> VolumeOpenInterestWorksheet:
        try:
            df = self.trend_result_to_df(trend_results)
            if df.empty:
                logger.warning(f"No trend data for {stock}, skipping worksheet entry")
                return None
            
            date = df['date'].iloc[0]
            stock_data = self.get_aligned_data(stock, backtest_date=date)
            if stock_data is None:
                logger.warning(f"No stock data for {stock} at {date}, skipping worksheet entry")
                return None

            # Get metric maps
            metric_map = self.trend_utility.get_metric_maps(stock_data.ohlcv, stock_data.option_df)
            metric_change_map = self.trend_utility.get_metric_change_maps(stock_data.ohlcv, stock_data.option_df)

            # Get classification logs
            classification_logs = {
                log.metric: log for log in self.trend_utility.classification_logs
                if log.stock.lower() == stock.lower() and log.date == pd.to_datetime(date)
            }

            # Extract long_term trends, seasonality, peaks, valleys, and trend comparisons
            trend_directions = {}
            trend_comparisons = {}
            seasonality = 'normal'
            peaks = None
            valleys = None

            required_metrics = ['price', 'stock_volume', 'options_volume', 'oi']
            for metric in required_metrics:
                long_term_row = df[(df['name'] == metric) & (df['timeframe'] == 'long_term')]
                short_term_row = df[(df['name'] == metric) & (df['timeframe'] == 'short_term')]
                
                # Trend directions (long_term)
                trend_directions[metric] = long_term_row['trend'].iloc[0] if not long_term_row.empty else 'unknown'
                
                # Trend comparisons
                if not long_term_row.empty and not short_term_row.empty:
                    long_trend = long_term_row['trend'].iloc[0]
                    short_trend = short_term_row['trend'].iloc[0]
                    trend_comparisons[metric] = 'Aligned' if long_trend == short_trend else 'Divergent'
                else:
                    trend_comparisons[metric] = 'Unknown'

                # Seasonality, peaks, valleys for price
                if metric == 'price' and not long_term_row.empty:
                    seasonality = long_term_row['seasonality'].iloc[0]
                    peaks = long_term_row['peaks'].iloc[0]
                    valleys = long_term_row['valleys'].iloc[0]

            # Validate inputs
            required_metrics = ['price', 'stock_volume', 'options_volume', 'oi', 'pcr_volume', 'pcr_oi']
            missing_metrics = [m for m in required_metrics if m not in metric_map or metric_map[m].empty]
            missing_logs = [m for m in required_metrics if m not in classification_logs]
            if missing_metrics or missing_logs:
                logger.warning(f"Missing metrics {missing_metrics} or logs {missing_logs} for {stock}")
                return None

            # Create worksheet
            worksheet = VolumeOpenInterestWorksheet(
                stock=stock,
                date=pd.to_datetime(date),
                metric_map=metric_map,
                metric_change_map=metric_change_map,
                classification_logs=classification_logs,
                trend_directions=trend_directions,
                trend_comparisons=trend_comparisons,
                seasonality=seasonality,
                peaks=peaks,
                valleys=valleys
            )
            return worksheet
        except Exception as e:
            logger.error(f"Error creating worksheet for {stock}: {str(e)}")
            return None

def main():
    """Example usage of the TResults class."""
    import json
    connections = get_path()
    stocks = json.load(open(connections['ticker_path'], 'r'))['all_stocks']
    detector = TResults(connections, lookback_days=150)
    lodf = []; lowdf = []
    for stock in tqdm(stocks, desc="Analyzing Stocks"):
        try:
            trend_results = detector.analyze_stock_trend(stock)
            if not trend_results:
                logger.warning(f"No trend results for {stock}")
                continue
            
            worksheet = detector.worksheet_entry(stock, trend_results)
            if worksheet is None:
                logger.warning(f"Failed to create worksheet for {stock}")
                continue
            
            detector.all_worksheets.append(worksheet)
            lodf.append(detector.trend_result_to_df(trend_results))
            lowdf.append(worksheet.to_df())
        except Exception as e:
            logger.error(f"Error processing {stock}: {str(e)}")

    # Print the results
    if lodf:
        lodf = pd.concat(lodf, ignore_index=True)
        print("Trend Results DataFrame:")
        print(lodf)
        lodf.to_csv('trend_results.csv', index=False)
    else:
        print("No trend results found.")
    if lowdf:
        lowdf = pd.concat(lowdf, ignore_index=True)
        print("Worksheets DataFrame:")
        print(lowdf)
        lowdf.to_csv('worksheets.csv', index=False)


if __name__ == "__main__":
    main()