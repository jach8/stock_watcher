"""Base test class for database tests"""

import unittest
import os
import sqlite3
import logging
from bin.database.core import DBManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseTestCase(unittest.TestCase):
    """Base test case that handles database setup and cleanup"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        # Get absolute path to test directory
        cls.test_dir = os.path.dirname(os.path.abspath(__file__))
        cls.data_dir = os.path.join(cls.test_dir, 'data')
        
        # Ensure test data directory exists
        if not os.path.exists(cls.data_dir):
            os.makedirs(cls.data_dir)
    
    def get_db_path(self, name: str) -> str:
        """Get absolute path for a test database file"""
        return os.path.abspath(os.path.join(self.data_dir, f'{name}.db'))
            
    def setUp(self):
        """Set up test databases before each test"""
        # Create database file paths - use exact names that repositories expect
        self.db_paths = {
            'stocks_db': self.get_db_path('stocks'),
            'options_db': self.get_db_path('options'),
            'bonds_db': self.get_db_path('bonds'),
            'test_db': self.get_db_path('test')
        }
        
        # Clean up any existing test databases
        self.cleanup_databases()
        
        # Initialize all test databases
        for db_name, db_path in self.db_paths.items():
            logger.info(f"Initializing test database {db_name}: {db_path}")
            try:
                conn = sqlite3.connect(db_path)
                conn.row_factory = sqlite3.Row
                # Create a test table to ensure database is writable
                conn.execute('CREATE TABLE IF NOT EXISTS test_table (id INTEGER PRIMARY KEY)')
                conn.commit()
                conn.close()
                logger.info(f"Successfully created test database {db_name}")
            except Exception as e:
                logger.error(f"Error creating test database {db_name}: {e}")
                raise
        
        # Create manager with test databases
        self.db_manager = DBManager(self.db_paths, pool_size=3)
        
    def create_test_database(self, db_path: str) -> sqlite3.Connection:
        """Create a test database file
        
        Args:
            db_path: Path where the database should be created
            
        Returns:
            SQLite connection to the database
        """
        try:
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            
            # Create a test table to ensure database is writable
            conn.execute('CREATE TABLE IF NOT EXISTS test_table (id INTEGER PRIMARY KEY)')
            conn.commit()
            
            logger.info(f"Successfully created test database: {db_path}")
            return conn
            
        except Exception as e:
            logger.error(f"Error creating test database {db_path}: {e}")
            raise
        
    def cleanup_databases(self):
        """Remove test database files"""
        # First cleanup any existing connection pools
        if hasattr(self, 'db_manager'):
            try:
                self.db_manager.cleanup()
            except Exception as e:
                logger.error(f"Error cleaning up connection pools: {e}")
            
        # Then remove database files
        for db_name, db_path in self.db_paths.items():
            if os.path.exists(db_path):
                try:
                    os.remove(db_path)
                    logger.info(f"Removed test database {db_name}: {db_path}")
                except OSError as e:
                    logger.error(f"Error removing database {db_name}: {e}")
                    
    def tearDown(self):
        """Clean up after each test"""
        self.cleanup_databases()
        
        # Remove test data directory if empty
        try:
            if os.path.exists(self.data_dir) and not os.listdir(self.data_dir):
                os.rmdir(self.data_dir)
        except OSError:
            pass

    @classmethod
    def tearDownClass(cls):
        """Clean up test environment"""
        # Remove test data directory if it exists and is empty
        try:
            if os.path.exists(cls.data_dir) and not os.listdir(cls.data_dir):
                os.rmdir(cls.data_dir)
        except OSError:
            pass