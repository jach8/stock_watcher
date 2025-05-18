import unittest
import sqlite3
import os
from unittest.mock import patch, MagicMock
from bin.database.core import DBManager, ConnectionError
from bin.database.tests.test_base import DatabaseTestCase

class TestDBManager(DatabaseTestCase):
    def test_connection_pool_initialization(self):
        """Test that connection pool is properly initialized"""
        for db_name in self.db_paths:
            self.assertIn(db_name, self.db_manager._pools)
            self.assertEqual(self.db_manager._pools[db_name].qsize(), 3)
            self.assertIn(db_name, self.db_manager._locks)

    def test_connection_context_manager(self):
        """Test connection acquisition and release"""
        test_db_name = 'test_db'
        initial_pool_size = self.db_manager._pools[test_db_name].qsize()
        
        with self.db_manager.connection(test_db_name) as conn:
            # Check connection is valid
            self.assertIsInstance(conn, sqlite3.Connection)
            # Check pool size decreased
            self.assertEqual(self.db_manager._pools[test_db_name].qsize(), initial_pool_size - 1)
            
            # Verify connection works
            cur = conn.cursor()
            cur.execute('INSERT INTO test_table VALUES (1)')
            cur.execute('SELECT * FROM test_table')
            self.assertEqual(cur.fetchone()[0], 1)
            
        # Check connection returned to pool
        self.assertEqual(self.db_manager._pools[test_db_name].qsize(), initial_pool_size)

    def test_connection_error_handling(self):
        """Test error handling for bad connections"""
        # Create a nonexistent subdirectory path
        bad_dir = os.path.join(self.test_dir, "nonexistent_dir")
        bad_db_path = os.path.join(bad_dir, "test.db")
        
        # Test creating manager with nonexistent directory
        with self.assertRaisesRegex(ConnectionError, "Database directory does not exist"):
            DBManager({'bad_db': bad_db_path})
            
        # Test attempting to use invalid database name
        with self.assertRaisesRegex(ConnectionError, "Unknown database"):
            with self.db_manager.connection('nonexistent_db'):
                pass

    def test_cleanup(self):
        """Test proper cleanup of connection pools"""
        # Create multiple test connections
        test_db_name = 'test_db'
        conn_list = []
        
        # Get connections from pool
        for _ in range(3):
            with self.db_manager.connection(test_db_name) as conn:
                conn_list.append(conn)
                cur = conn.cursor()
                cur.execute('SELECT 1')
                self.assertEqual(cur.fetchone()[0], 1)
        
        # Store a reference to one connection
        test_conn = None
        with self.db_manager.connection(test_db_name) as conn:
            test_conn = conn
            
        # Clean up pools
        self.db_manager.cleanup()
        
        # Verify pools are empty
        self.assertEqual(len(self.db_manager._pools), 0)
        
        # Verify connection is closed
        with self.assertRaises(sqlite3.ProgrammingError):
            test_conn.execute('SELECT 1')

if __name__ == '__main__':
    unittest.main()