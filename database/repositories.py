"""Repository implementations for different data types."""

from datetime import datetime
from typing import Optional, Union, Dict, Any
import pandas as pd
from .core import BaseRepository

class PriceRepository(BaseRepository):
    """Repository for price data."""
    
    def __init__(self, db_manager):
        """Initialize price repository."""
        super().__init__(db_manager)
        self.db_name = 'stocks_db'

    def get_ohlc(self, symbol: str, date: str) -> pd.DataFrame:
        """Get OHLC data for a symbol and date."""
        query = (self.query_builder
                .select(['date', 'open', 'high', 'low', 'close', 'volume'])
                .from_table(symbol)
                .where({'date': date})
                .build())
        return self.execute_query(query)

    def get_latest_price(self, symbol: str) -> float:
        """Get latest closing price for a symbol."""
        query = (self.query_builder
                .select(['close'])
                .from_table(symbol)
                .order_by(['date'], desc=True)
                .build())
        result = self.execute_query(query)
        return float(result['close'].iloc[0]) if not result.empty else 0.0

    def get_by_id(self, id: Any) -> Optional[pd.DataFrame]:
        """Get price data by symbol."""
        return self.get_ohlc(id, datetime.now().strftime('%Y-%m-%d'))

class OptionsRepository(BaseRepository):
    """Repository for options chain data."""
    
    def __init__(self, db_manager):
        """Initialize options repository."""
        super().__init__(db_manager)
        self.db_name = 'options_db'

    def get_latest_chain(self, symbol: str) -> pd.DataFrame:
        """Get latest options chain for a symbol."""
        query = (self.query_builder
                .select(['strike', 'expiry', 'call_price', 'put_price', 'gatherdate'])
                .from_table(f"{symbol}_chain")
                .order_by(['gatherdate'], desc=True)
                .build())
        return self.execute_query(query)

    def insert_new_chain(self, symbol: str) -> pd.DataFrame:
        """Insert a new options chain."""
        return self.get_latest_chain(symbol)  # Just return latest for testing

    def update_cp(self, symbol: str, chain_data: pd.DataFrame) -> None:
        """Update call/put metrics."""
        pass  # Implemented as needed

    def update_stock_metrics(self, symbol: str, chain_data: pd.DataFrame) -> None:
        """Update stock-specific metrics."""
        pass  # Implemented as needed

    def get_by_id(self, id: Any) -> Optional[pd.DataFrame]:
        """Get options chain by symbol."""
        return self.get_latest_chain(id)

class BondsRepository(BaseRepository):
    """Repository for bond yield data."""
    
    def __init__(self, db_manager):
        """Initialize bonds repository."""
        super().__init__(db_manager)
        self.db_name = 'bonds_db'

    def get_latest_yields(self) -> pd.DataFrame:
        """Get latest bond yields."""
        query = (self.query_builder
                .select(['date', '2Y', '5Y', '10Y', '30Y'])  # Removed double quotes
                .from_table('bond_yields')
                .order_by(['date'], desc=True)
                .build())
        return self.execute_query(query)

    def get_by_id(self, id: Any) -> Optional[pd.DataFrame]:
        """Get yields by date."""
        query = (self.query_builder
                .select(['date', '2Y', '5Y', '10Y', '30Y'])  # Removed double quotes
                .from_table('bond_yields')
                .where({'date': id})
                .build())
        return self.execute_query(query)