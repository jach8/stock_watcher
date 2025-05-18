import unittest
import os
import sqlite3
import pandas as pd
from bin.database.core import DBManager, QueryBuilder
from bin.database.repositories import (
    PriceRepository,
    OptionsRepository,
    BondsRepository
)
from bin.database.tests.test_base import DatabaseTestCase

class TestDatabaseIntegration(DatabaseTestCase):
    def setUp(self):
        """Set up test environment"""
        super().setUp()
        
        # Initialize database schemas
        self._initialize_test_data()
        
        # Initialize repositories
        self.price_repo = PriceRepository(self.db_manager)
        self.options_repo = OptionsRepository(self.db_manager)
        self.bonds_repo = BondsRepository(self.db_manager)

    def _initialize_test_data(self):
        """Initialize test databases with sample data"""
        # Set up stocks database
        conn = self.create_test_database(self.db_paths['stocks_db'])
        conn.execute("""
            CREATE TABLE IF NOT EXISTS AAPL (
                date TEXT PRIMARY KEY,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER
            )
        """)
        conn.execute("""
            INSERT INTO AAPL VALUES
            ('2025-05-17', 100.0, 105.0, 98.0, 103.0, 1000000),
            ('2025-05-16', 98.0, 101.0, 97.0, 100.0, 900000)
        """)
        conn.commit()
        conn.close()
        
        # Set up options database
        conn = self.create_test_database(self.db_paths['options_db'])
        conn.execute("""
            CREATE TABLE IF NOT EXISTS AAPL_chain (
                strike REAL,
                expiry TEXT,
                call_price REAL,
                put_price REAL,
                gatherdate TEXT,
                PRIMARY KEY (strike, expiry)
            )
        """)
        conn.execute("""
            INSERT INTO AAPL_chain VALUES
            (100.0, '2025-06-21', 5.0, 2.0, '2025-05-17 10:00:00'),
            (110.0, '2025-06-21', 2.0, 7.0, '2025-05-17 10:00:00')
        """)
        conn.commit()
        conn.close()
        
        # Set up bonds database
        conn = self.create_test_database(self.db_paths['bonds_db'])
        conn.execute("""
            CREATE TABLE IF NOT EXISTS bond_yields (
                date TEXT PRIMARY KEY,
                "2Y" REAL,
                "5Y" REAL,
                "10Y" REAL,
                "30Y" REAL
            )
        """)
        conn.execute("""
            INSERT INTO bond_yields VALUES
            ('2025-05-17', 4.5, 4.7, 4.9, 5.1)
        """)
        conn.commit()
        conn.close()

    def test_price_retrieval_flow(self):
        """Test complete price data retrieval flow"""
        # Test OHLC data retrieval
        ohlc_data = self.price_repo.get_ohlc('AAPL', '2025-05-16')
        self.assertIsInstance(ohlc_data, pd.DataFrame)
        self.assertEqual(len(ohlc_data), 1)
        self.assertEqual(ohlc_data['close'].iloc[0], 100.0)
        
        # Test latest price
        latest_price = self.price_repo.get_latest_price('AAPL')
        self.assertEqual(latest_price, 103.0)

    def test_options_chain_flow(self):
        """Test options chain retrieval and processing flow"""
        # Get latest chain
        chain = self.options_repo.get_latest_chain('AAPL')
        self.assertIsInstance(chain, pd.DataFrame)
        self.assertEqual(len(chain), 2)
        
        # Test chain insertion
        result = self.options_repo.insert_new_chain('AAPL')
        self.assertIsNotNone(result)
        
        # Verify chain metrics update
        self.options_repo.update_cp('AAPL', chain)
        self.options_repo.update_stock_metrics('AAPL', chain)

    def test_bonds_yield_flow(self):
        """Test bonds yield retrieval flow"""
        yields = self.bonds_repo.get_latest_yields()
        self.assertIsInstance(yields, pd.DataFrame)
        self.assertEqual(yields['10Y'].iloc[0], 4.9)

    def test_connection_pooling_under_load(self):
        """Test connection pooling with concurrent operations"""
        def run_queries():
            for _ in range(5):
                self.price_repo.get_latest_price('AAPL')
                self.options_repo.get_latest_chain('AAPL')
                self.bonds_repo.get_latest_yields()
        
        # Run multiple operations that should exercise the connection pool
        run_queries()
        
        # Verify pool still has expected size
        for db_name in self.db_paths:
            self.assertEqual(self.db_manager._pools[db_name].qsize(), 3)

if __name__ == '__main__':
    unittest.main()