from dataclasses import dataclass
from typing import Union, Optional, Tuple, List
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from datetime import datetime

@dataclass
class ClassificationLog:
    date: datetime
    stock: str
    metric: str
    lookback: int
    low_threshold: float
    high_threshold: float
    blowoff_threshold: float
    category_counts: dict  # {'Low': float, 'Average': float, 'High': float}
    oi_adjustment: float
    category: str  # Low/Average/High/Blowoff
    blowoff: bool

class Classifier:
    def __init__(
        self,
        data: Union[pd.Series, pd.DataFrame],
        open_interest: Optional[pd.Series] = None,
        lookback: int = 30,
        event_dates: Optional[List[datetime]] = None,
        blowoff_percentile: float = 90.0,
        category_balance: Tuple[float, float] = (0.2, 0.4),
        period: int = 21,
        window_size: Optional[int] = None
    ):
        # Validate inputs (as in your original)
        self.data = data if isinstance(data, pd.Series) else data.iloc[:, 0]
        self.open_interest = open_interest
        self.lookback = lookback
        self.blowoff_percentile = blowoff_percentile
        self.category_balance = category_balance
        self.period = period
        self.window_size = window_size or 30
        self.worksheet: List[ClassificationLog] = []
        self.last_evaluation = None

    def classify(self, stock: str, metric: str) -> Tuple[str, bool, ClassificationLog]:
        valid_data = self.data.tail(self.lookback)
        oi_factor = 1.0
        # if self.open_interest is not None:
        #     oi_trend = np.polyfit(np.arange(len(self.open_interest)), self.open_interest, 1)[0]
        #     oi_factor = 1 + (oi_trend / self.open_interest.mean()) if oi_trend != 0 else 1.0

        # Compute thresholds
        low_threshold = valid_data.quantile(0.25) * oi_factor
        high_threshold = valid_data.quantile(0.75) * oi_factor
        blowoff_threshold = valid_data.quantile(self.blowoff_percentile / 100) * oi_factor

        # Optimize for 20–40% balance
        counts = self._compute_category_counts(valid_data, low_threshold, high_threshold)
        if not all(self.category_balance[0] <= v <= self.category_balance[1] for v in counts.values()):
            low_threshold, high_threshold = self._optimize_thresholds(valid_data, oi_factor)

        # Classify
        current_value = self.data.iloc[-1]
        blowoff = current_value >= blowoff_threshold
        category = "Blowoff" if blowoff else ("Low" if current_value < low_threshold else ("High" if current_value > high_threshold else "Average"))

        # Log
        log_entry = ClassificationLog(
            date=datetime.now(),
            stock=stock,
            metric=metric,
            lookback=self.lookback,
            low_threshold=low_threshold,
            high_threshold=high_threshold,
            blowoff_threshold=blowoff_threshold,
            category_counts=counts,
            oi_adjustment=oi_factor,
            category=category,
            blowoff=blowoff
        )
        self.worksheet.append(log_entry)
        return category, blowoff, log_entry
    

    def _compute_category_counts(self, data: pd.Series, low: float, high: float) -> dict:
        # Compute % in each category
        return {
            'Low': (data < low).mean(),
            'Average': ((data >= low) & (data <= high)).mean(),
            'High': (data > high).mean()
        }

    def _optimize_thresholds(self, data: pd.Series, oi_factor: float) -> Tuple[float, float]:
        # Iteratively adjust thresholds to achieve 20–40% balance
        # (Simplified; use optimization algorithm in practice)
        low = data.quantile(0.2) * oi_factor
        high = data.quantile(0.8) * oi_factor
        return low, high

    def reevaluate(self):
        # Recalculate thresholds if >7 days since last evaluation
        if self.last_evaluation is None or (datetime.now() - self.last_evaluation).days >= 7:
            self.worksheet = []  # Reset or append based on preference
            self.last_evaluation = datetime.now()


if __name__ == "__main__":

    # Example usage
    data = pd.Series(np.random.randn(100), index=pd.date_range(start='2023-01-01', periods=100))
    open_interest = pd.Series(np.random.randn(100), index=pd.date_range(start='2023-01-01', periods=100))
    classifier = Classifier(data, open_interest)
    category, blowoff, log_entry = classifier.classify(stock="AAPL", metric="call_volume")
    print(f"Category: {category}, Blowoff: {blowoff}")
    print(f"Log Entry: {log_entry}")
    
