"""
Database Integration Examples

This module demonstrates how to use the database layer components for stock data management.
It covers basic database operations and pipeline integration patterns.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from bin.database.core import DBManager
from bin.database.repositories import PriceRepository, OptionsRepository
from bin.database.manager import Manager

# Basic Usage Example
def demonstrate_basic_usage():
    """
    Demonstrates basic database operations using the Manager class.
    Shows connection management, repository usage, and proper cleanup.
    """

    # Initialize manager with connection pooling
    manager = Manager()

    try:
        # Price repository operations
        spy_data = manager.price_repository.get_ohlc('SPY', '2025-01-01')
        print(f"SPY OHLC data: {spy_data}")

        latest_price = manager.price_repository.get_latest_price('QQQ')
        print(f"Latest QQQ price: {latest_price}")

        # Options repository operations
        chain = manager.options_repository.get_latest_chain('AAPL')
        print(f"AAPL latest options chain: {chain}")

        # Update options change variables
        manager.options_repository.update_change_vars('AAPL')
        print("Updated AAPL options change variables")

    except Exception as e:
        print(f"Error during database operations: {e}")
    finally:
        # Always ensure proper cleanup of database connections
        manager.close_connection()


# Pipeline Integration Example
def demonstrate_pipeline_usage():
    """
    Demonstrates how to use the Pipeline class for automated data updates.
    Shows batch operations and proper resource management.
    """
    from bin.database.manager import Pipeline

    pipeline = Pipeline()

    try:
        # Quick update for specific symbols
        pipeline.quick_update(['SPY', 'QQQ'])
        print("Completed quick update for SPY and QQQ")

        # Full stock price update
        pipeline.update_stock_prices()
        print("Completed full stock price update")

    except Exception as e:
        print(f"Error during pipeline operations: {e}")
    finally:
        # Ensure connection cleanup
        pipeline.close_connection()


if __name__ == '__main__':
    print("\n=== Basic Usage Example ===")
    demonstrate_basic_usage()
    
    # print("\n=== Pipeline Integration Example ===")
    # demonstrate_pipeline_usage()