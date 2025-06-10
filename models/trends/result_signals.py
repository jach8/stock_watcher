"""
Signal Classification for Price, Volume, and Open Interest Trends
- Categorical signals only.
"""

from dataclasses import dataclass
from datetime import datetime
import pandas as pd
import logging
from typing import List, Optional, Dict

from bin.models.trends.Detect_class import ClassificationLog

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

@dataclass
class WorksheetEntry:
    date: datetime
    volume_category: str
    oi_category: str
    option_volume_category: str
    pcr_volume_category: str
    pcr_oi_category: str
    price_trend_direction: str
    volume_trend_direction: str
    oi_trend_direction: str
    option_volume_trend_direction: str
    seasonality: str
    returns_category: Optional[str]
    is_peak: bool
    is_valley: bool
    price_trend_alignment: str
    volume_trend_alignment: str
    options_volume_trend_alignment: str
    oi_trend_alignment: str
    today_price_category: str
    today_volume_category: str
    today_options_volume_category: str
    today_oi_category: str

@dataclass
class VolumeOpenInterestWorksheet:
    stock: str
    date: datetime
    entries: List[WorksheetEntry]
    bullish_signals: int
    bearish_signals: int

    def validate_metric_maps(
        self,
        metric_map: Dict[str, pd.Series],
        metric_change_map: Dict[str, pd.Series]
    ) -> None:
        """Validate metric_map and metric_change_map for required metrics."""
        required_metrics = ['price', 'stock_volume', 'options_volume', 'oi', 'pcr_volume', 'pcr_oi']
        missing_metrics = [m for m in required_metrics if m not in metric_map or metric_map[m].empty]
        missing_change_metrics = [m for m in required_metrics if m not in metric_change_map]
        
        if missing_metrics:
            logger.warning(f"Missing or empty metrics for {self.stock}: {missing_metrics}")
            for m in missing_metrics:
                metric_map[m] = pd.Series()
        
        if missing_change_metrics:
            logger.warning(f"Missing change metrics for {self.stock}: {missing_change_metrics}")
            for m in missing_change_metrics:
                metric_change_map[m] = pd.Series()

    def validate_classification_logs(
        self,
        classification_logs: Dict[str, ClassificationLog]
    ) -> None:
        """Validate classification_logs for required metrics."""
        required_metrics = ['price', 'stock_volume', 'options_volume', 'oi', 'pcr_volume', 'pcr_oi']
        missing_logs = [m for m in required_metrics if m not in classification_logs]
        if missing_logs:
            logger.warning(f"Missing classification logs for {self.stock}: {missing_logs}")
            for m in missing_logs:
                classification_logs[m] = ClassificationLog(category='Average', seasonality='normal')

    def validate_trend_directions(
        self,
        trend_directions: Dict[str, str]
    ) -> None:
        """Validate trend_directions for price key."""
        required_metrics = ['price', 'stock_volume', 'options_volume', 'oi']
        missing_trends = [m for m in required_metrics if m not in trend_directions]
        if missing_trends:
            logger.warning(f"Missing trend directions for {self.stock}: {missing_trends}")
            for m in missing_trends:
                trend_directions[m] = 'unknown'

    def validate_trend_comparisons(
        self,
        trend_comparisons: Dict[str, str]
    ) -> None:
        """Validate trend_comparisons for required metrics."""
        required_metrics = ['price', 'stock_volume', 'options_volume', 'oi']
        missing_comparisons = [m for m in required_metrics if m not in trend_comparisons]
        if missing_comparisons:
            logger.warning(f"Missing trend comparisons for {self.stock}: {missing_comparisons}")
            for m in missing_comparisons:
                trend_comparisons[m] = 'Unknown'

    def compute_bullish_signals(self, entry: WorksheetEntry) -> int:
        """Compute bullish signals based on categorical conditions."""
        return 1 if (
            entry.price_trend_direction == 'up' and
            (entry.volume_category in ['High', 'Blowoff'] or entry.oi_category in ['High', 'Blowoff']) and
            entry.pcr_oi_category in ['Low', 'Average'] and
            entry.seasonality == 'high'
        ) else 0

    def compute_bearish_signals(self, entry: WorksheetEntry) -> int:
        """Compute bearish signals based on categorical conditions."""
        return 1 if (
            entry.price_trend_direction == 'down' and
            (entry.volume_category in ['Low', 'Average'] or entry.oi_category in ['Low', 'Average']) and
            entry.pcr_oi_category in ['High', 'Blowoff'] and
            entry.seasonality == 'high'
        ) else 0

    def __init__(
        self,
        stock: str,
        date: datetime,
        metric_map: Dict[str, pd.Series],
        metric_change_map: Dict[str, pd.Series],
        classification_logs: Dict[str, ClassificationLog],
        trend_directions: Dict[str, str],
        trend_comparisons: Dict[str, str],
        seasonality: str,
        peaks: float,
        valleys: float
    ):
        self.stock = stock
        self.date = date
        self.entries = []

        # Validate inputs
        self.validate_metric_maps(metric_map, metric_change_map)
        self.validate_classification_logs(classification_logs)
        self.validate_trend_directions(trend_directions)
        self.validate_trend_comparisons(trend_comparisons)

        # Extract returns_category
        returns = metric_map.get('returns', pd.Series()).tail(1).iloc[0] if 'returns' in metric_map and not metric_map['returns'].empty else 0.0
        returns_category = 'Up' if returns > 0 else 'Down' if returns < 0 else None

        # Compute peak and valley signals
        current_price = metric_map.get('price', pd.Series()).tail(1).iloc[0] if 'price' in metric_map and not metric_map['price'].empty else 0.0
        is_peak = current_price >= peaks if peaks is not None else False
        is_valley = current_price <= valleys if valleys is not None else False

        # Extract today's categories
        today_price_category = classification_logs['price'].category
        today_volume_category = classification_logs['stock_volume'].category
        today_options_volume_category = classification_logs['options_volume'].category
        today_oi_category = classification_logs['oi'].category

        # Create entry
        entry = WorksheetEntry(
            date=self.date,
            volume_category=classification_logs['stock_volume'].category,
            oi_category=classification_logs['oi'].category,
            option_volume_category=classification_logs['options_volume'].category,
            pcr_volume_category=classification_logs['pcr_volume'].category,
            pcr_oi_category=classification_logs['pcr_oi'].category,
            price_trend_direction=trend_directions.get('price', 'unknown'),
            volume_trend_direction=trend_directions.get('stock_volume', 'unknown'),
            oi_trend_direction=trend_directions.get('oi', 'unknown'),
            option_volume_trend_direction=trend_directions.get('options_volume', 'unknown'),
            seasonality=seasonality,
            returns_category=returns_category,
            is_peak=is_peak,
            is_valley=is_valley,
            price_trend_alignment=trend_comparisons.get('price', 'Unknown'),
            volume_trend_alignment=trend_comparisons.get('stock_volume', 'Unknown'),
            options_volume_trend_alignment=trend_comparisons.get('options_volume', 'Unknown'),
            oi_trend_alignment=trend_comparisons.get('oi', 'Unknown'),
            today_price_category=today_price_category,
            today_volume_category=today_volume_category,
            today_options_volume_category=today_options_volume_category,
            today_oi_category=today_oi_category
        )
        self.entries.append(entry)

        # Compute signals
        self.bullish_signals = self.compute_bullish_signals(entry)
        self.bearish_signals = self.compute_bearish_signals(entry)

    def to_df(self) -> pd.DataFrame:
        """Return a DataFrame with categorical data only."""
        try:
            data = []
            for entry in self.entries:
                data.append({
                    'Date': entry.date,
                    'Stock': self.stock,
                    'Volume_Category': entry.volume_category,
                    'OI_Category': entry.oi_category,
                    'Options_Volume_Category': entry.option_volume_category,
                    'PCR_Volume_Category': entry.pcr_volume_category,
                    'PCR_OI_Category': entry.pcr_oi_category,
                    'Price_Trend_Direction': entry.price_trend_direction,
                    'Volume_Trend_Direction': entry.volume_trend_direction,
                    'OI_Trend_Direction': entry.oi_trend_direction,
                    'Options_Volume_Trend_Direction': entry.option_volume_trend_direction,
                    'Seasonality': entry.seasonality,
                    'Returns_Category': entry.returns_category,
                    'Is_Peak': entry.is_peak,
                    'Is_Valley': entry.is_valley,
                    'Price_Trend_Alignment': entry.price_trend_alignment,
                    'Volume_Trend_Alignment': entry.volume_trend_alignment,
                    'Options_Volume_Trend_Alignment': entry.options_volume_trend_alignment,
                    'OI_Trend_Alignment': entry.oi_trend_alignment,
                    'Today_Price_Category': entry.today_price_category,
                    'Today_Volume_Category': entry.today_volume_category,
                    'Today_Options_Volume_Category': entry.today_options_volume_category,
                    'Today_OI_Category': entry.today_oi_category,
                    'Bullish': self.bullish_signals > 0,
                    'Bearish': self.bearish_signals > 0
                })
            return pd.DataFrame(data)
        except Exception as e:
            logger.error(f"Error converting {self.stock} to DataFrame: {str(e)}")
            return pd.DataFrame()