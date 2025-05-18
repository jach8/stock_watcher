import logging
from typing import List, Optional, Dict
import pandas as pd
from tqdm import tqdm

from .core import DBManager, get_path
from .repositories import PriceRepository, OptionsRepository, BondsRepository

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Manager:
    """Base manager class implementing the repository pattern for database operations.
    
    This class serves as the integration layer between the repository pattern
    and the existing system, providing connection management and repository access.
    """
    def __init__(self, connections: Optional[Dict[str, str]] = None):
        """Initialize manager with database connections and repositories.
        
        Args:
            connections: Optional dictionary of database connection strings.
                       If None, uses default paths from get_path().
        """
        self.db_manager = DBManager(connections or get_path(), pool_size=5)
        
        # Initialize repositories
        self.price_repository = PriceRepository(self.db_manager)
        self.options_repository = OptionsRepository(self.db_manager)
        self.bonds_repository = BondsRepository(self.db_manager)
        
        # Legacy compatibility attributes
        self.Pricedb = self.price_repository
        self.Optionsdb = self.options_repository
        self.Bonds = self.bonds_repository

    def close_connection(self):
        """Close all database connections."""
        self.db_manager.close_all()

class Pipeline(Manager):
    """Pipeline class for orchestrating database operations and updates.
    
    Inherits from Manager to utilize the repository pattern while maintaining
    backwards compatibility with the existing system.
    """
    def quick_update(self, stocks: Optional[List[str]] = None) -> None:
        """Quickly update option chain data for specified or default stocks.
        
        Args:
            stocks: Optional list of stock symbols to update.
                   Defaults to ['spy', 'qqq', 'iwm', 'dia', 'vxx'].
                   
        Raises:
            ValueError: If an invalid stock symbol is provided.
        """
        stocks_to_update = stocks if stocks else ['spy', 'qqq', 'iwm', 'dia', 'vxx']
        
        for stock in tqdm(stocks_to_update, desc="Quick Update"):
            try:
                # Using repository pattern while maintaining legacy compatibility
                new_chain = self.options_repository.insert_new_chain(stock)
                if new_chain is not None:
                    self.options_repository.update_change_vars(stock)
                    self.options_repository.update_cp(stock, new_chain)
                    self.options_repository.update_stock_metrics(stock, new_chain)
            except Exception as e:
                logger.error(f"Error updating {stock}: {e}")

    def update_options(self) -> None:
        """Update all options data and run associated workflows."""
        try:
            # Using repository pattern for options updates
            self.options_repository.update_all_options()
        except Exception as e:
            logger.error(f"Failed to update options: {e}")

    def update_stock_prices(self) -> None:
        """Update stock price data."""
        try:
            # Using repository pattern for price updates
            self.price_repository.update_prices()
        except Exception as e:
            logger.error(f"Failed to update stock prices: {e}")

    def update_bonds(self) -> None:
        """Update bond market data."""
        try:
            # Using repository pattern for bond updates
            self.bonds_repository.update_bonds()
        except Exception as e:
            logger.error(f"Failed to update bonds: {e}")

    def master_run(self) -> None:
        """Execute complete update cycle for all data types."""
        try:
            self.update_stock_prices()
            self.update_options()
            self.update_bonds()
            self.close_connection()
        except Exception as e:
            logger.error(f"Master run failed: {e}")
            raise