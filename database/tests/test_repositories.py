"""Test repository implementations."""

import unittest
import pandas as pd
import os
import sqlite3
from unittest.mock import patch, MagicMock
from datetime import datetime
from bin.database.core import (
    BaseRepository, 
    DBManager, 
    DatabaseError, 
    QueryError, 
    RepositoryError,
    ConnectionError
)
from bin.database.repositories import (
    PriceRepository,
    OptionsRepository,
    BondsRepository
)
from bin.database.tests.test_base import DatabaseTestCase

class TestBondsRepository(DatabaseTestCase):
    def setUp(self):
        """Set up test environment"""
        super().setUp()
        self.repository = BondsRepository(self.db_manager)
        self.repository.db_name = 'bonds_db'

    def test_get_latest_yields(self):
        """Test latest bond yields retrieval"""
        # Create test table and data
        with self.db_manager.connection('bonds_db') as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS bond_yields (
                    date TEXT PRIMARY KEY,
                    [2Y] REAL,
                    [5Y] REAL,
                    [10Y] REAL,
                    [30Y] REAL
                )
            """)
            conn.execute("""
                INSERT INTO bond_yields (date, [2Y], [5Y], [10Y], [30Y])
                VALUES ('2025-05-17', 4.5, 4.7, 4.9, 5.1)
            """)
            conn.commit()
            
        result = self.repository.get_latest_yields()
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 1)
        self.assertIn('10Y', result.columns)
        self.assertEqual(float(result['10Y'].iloc[0]), 4.9)

    def test_error_handling(self):
        """Test repository error handling"""
        try:
            self.repository.get_latest_yields()
            self.fail("Expected RepositoryError to be raised")
        except ConnectionError as e:
            # Check that the error contains the correct message about missing table
            self.assertIn("no such table: bond_yields", str(e))

class TestPriceRepository(DatabaseTestCase):
    def setUp(self):
        """Set up test environment"""
        super().setUp()
        
        # Create test database schema
        with self.db_manager.connection('stocks_db') as conn:
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
                INSERT INTO AAPL (date, open, high, low, close, volume)
                VALUES ('2025-05-17', 100.0, 105.0, 98.0, 103.0, 1000000)
            """)
            conn.commit()
        
        self.repository = PriceRepository(self.db_manager)
        self.repository.db_name = 'stocks_db'

    def test_get_ohlc(self):
        """Test OHLC data retrieval"""
        result = self.repository.get_ohlc('AAPL', '2025-05-17')
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 1)
        self.assertEqual(float(result['close'].iloc[0]), 103.0)

    def test_get_latest_price(self):
        """Test latest price retrieval"""
        price = self.repository.get_latest_price('AAPL')
        self.assertEqual(price, 103.0)

class TestOptionsRepository(DatabaseTestCase):
    def setUp(self):
        """Set up test environment"""
        super().setUp()
        
        # Create test database schema
        with self.db_manager.connection('options_db') as conn:
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
                INSERT INTO AAPL_chain (strike, expiry, call_price, put_price, gatherdate)
                VALUES 
                    (100.0, '2025-06-21', 5.0, 2.0, '2025-05-17 10:00:00'),
                    (110.0, '2025-06-21', 2.0, 7.0, '2025-05-17 10:00:00')
            """)
            conn.commit()
        
        self.repository = OptionsRepository(self.db_manager)
        self.repository.db_name = 'options_db'

    def test_get_latest_chain(self):
        """Test latest options chain retrieval"""
        result = self.repository.get_latest_chain('AAPL')
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 2)
        self.assertIn('strike', result.columns)
        self.assertIn('call_price', result.columns)

    def test_insert_new_chain(self):
        """Test options chain insertion"""
        result = self.repository.insert_new_chain('AAPL')
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 2)

    def test_update_cp_metrics(self):
        """Test call/put metrics update"""
        chain = self.repository.get_latest_chain('AAPL')
        self.repository.update_cp('AAPL', chain)
        # Test passes if no exception is raised

if __name__ == '__main__':
    unittest.main()