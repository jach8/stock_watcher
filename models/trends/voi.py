# voi.py

from dataclasses import dataclass
from datetime import datetime
import pandas as pd
from typing import List, Optional
from bin.models.trends.Detect_class import ClassificationLog

@dataclass
class WorksheetEntry:
    date: datetime
    price: float
    price_delta: float
    volume: float
    volume_delta: float
    oi: float
    oi_delta: float
    option_volume: float
    option_volume_delta: float
    pcr_vol: float
    pcr_oi: float
    volume_category: str
    oi_category: str
    option_volume_category: str
    pcr_vol_category: str
    pcr_oi_category: str
    price_trend_direction: str  # New field
    volume_trend_direction: str  # New field
    oi_trend_direction: str      # New field
    option_volume_trend_direction: str  # New field

@dataclass
class VolumeOpenInterestWorksheet:
    stock: str
    date: datetime
    entries: List[WorksheetEntry]
    bullish_signals: int
    bearish_signals: int

    def __init__(
        self,
        stock: str,
        classification_logs: dict,
        price_data: pd.Series,
        price_chng_data: pd.Series,
        volume_data: pd.Series,
        volume_chng_data: pd.Series,
        oi_data: pd.Series,
        oi_chng_data: pd.Series,
        option_volume_data: pd.Series,
        option_volume_chng_data: pd.Series,
        put_volume: pd.Series,
        call_volume: pd.Series,
        put_oi: pd.Series,
        call_oi: pd.Series,
        trend_directions: dict  # New parameter to pass trend directions
    ):
        self.stock = stock
        self.date = datetime.now()
        self.entries = []

        # Use only the last row
        price_data = price_data.tail(1)
        price_deltas = price_chng_data.tail(1)
        volume_data = volume_data.tail(1)
        volume_deltas = volume_chng_data.tail(1)
        oi_data = oi_data.tail(1)
        oi_deltas = oi_chng_data.tail(1)
        option_volume_data = option_volume_data.tail(1)
        option_volume_deltas = option_volume_chng_data.tail(1)
        put_volume = put_volume.tail(1)
        call_volume = call_volume.tail(1)
        put_oi = put_oi.tail(1)
        call_oi = call_oi.tail(1)

        # Calculate put-to-call ratios
        pcr_vol = (put_volume.iloc[-1] / call_volume.iloc[-1]) if call_volume.iloc[-1] != 0 else 0.0
        pcr_oi = (put_oi.iloc[-1] / call_oi.iloc[-1]) if call_oi.iloc[-1] != 0 else 0.0

        # Get classifications from classification_logs
        volume_category = "Average"
        oi_category = "Average"
        option_volume_category = "Average"
        pcr_vol_category = "Average"
        pcr_oi_category = "Average"

        for metric, log in classification_logs.items():
            if metric == 'stock_volume':
                volume_category = log.category
            elif metric == 'oi':
                oi_category = log.category
            elif metric == 'options_volume':
                option_volume_category = log.category
            elif metric == 'pcr_vol':
                pcr_vol_category = log.category
            elif metric == 'pcr_oi':
                pcr_oi_category = log.category

        # Create the single entry
        for i, (date, price) in enumerate(price_data.items()):
            self.entries.append(WorksheetEntry(
                date=date,
                price=price,
                price_delta=price_deltas.iloc[i] if not pd.isna(price_deltas.iloc[i]) else 0.0,
                volume=volume_data.iloc[i],
                volume_delta=volume_deltas.iloc[i] if not pd.isna(volume_deltas.iloc[i]) else 0.0,
                oi=oi_data.iloc[i],
                oi_delta=oi_deltas.iloc[i] if not pd.isna(oi_deltas.iloc[i]) else 0.0,
                option_volume=option_volume_data.iloc[i],
                option_volume_delta=option_volume_deltas.iloc[i] if not pd.isna(option_volume_deltas.iloc[i]) else 0.0,
                pcr_vol=pcr_vol,
                pcr_oi=pcr_oi,
                volume_category=volume_category,
                oi_category=oi_category,
                option_volume_category=option_volume_category,
                pcr_vol_category=pcr_vol_category,
                pcr_oi_category=pcr_oi_category,
                price_trend_direction=trend_directions.get('close_prices', 'unknown'),
                volume_trend_direction=trend_directions.get('stock_volume', 'unknown'),
                oi_trend_direction=trend_directions.get('oi', 'unknown'),
                option_volume_trend_direction=trend_directions.get('options_volume', 'unknown')
            ))

        # Compute bullish/bearish signals for the last row
        entry = self.entries[0]
        self.bullish_signals = 1 if (
            entry.price_delta > 0 and 
            entry.volume_category in ['High', 'Blowoff'] and 
            entry.oi_delta > 0
        ) else 0
        self.bearish_signals = 1 if (
            entry.price_delta < 0 and 
            entry.volume_category in ['High', 'Blowoff'] and 
            entry.oi_delta > 0
        ) else 0

    def to_df(self) -> pd.DataFrame:
        data = []
        for entry in self.entries:
            data.append({
                'Date': entry.date,
                'Stock': self.stock,
                'Price': entry.price,
                'Price_Delta': entry.price_delta,
                'Volume': entry.volume,
                'Volume_Delta': entry.volume_delta,
                'OI': entry.oi,
                'OI_Delta': entry.oi_delta,
                'Option_Volume': entry.option_volume,
                'Option_Volume_Delta': entry.option_volume_delta,
                'PCR_Vol': entry.pcr_vol,
                'PCR_OI': entry.pcr_oi,
                'Volume_Category': entry.volume_category,
                'OI_Category': entry.oi_category,
                'Option_Volume_Category': entry.option_volume_category,
                'PCR_Vol_Category': entry.pcr_vol_category,
                'PCR_OI_Category': entry.pcr_oi_category,
                'Price_Trend_Direction': entry.price_trend_direction,  # New field
                'Volume_Trend_Direction': entry.volume_trend_direction,  # New field
                'OI_Trend_Direction': entry.oi_trend_direction,  # New field
                'Option_Volume_Trend_Direction': entry.option_volume_trend_direction,  # New field
                'Bullish': self.bullish_signals > 0,
                'Bearish': self.bearish_signals > 0
            })
        return pd.DataFrame(data)