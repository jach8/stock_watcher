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
            self.assertEqual(cur.fetchone()['id'], 1)
            
        # Check connection returned to pool
        self.assertEqual(self.db_manager._pools[test_db_name].qsize(), initial_pool_size)

    def test_connection_error_handling(self):
        """Test error handling for bad connections"""
        nonexistent_path = os.path.join(self.test_dir, 'data', 'nonexistent.db')
        bad_manager = DBManager({'bad_db': nonexistent_path}, 1)
        
        with self.assertRaises(ConnectionError):
            with bad_manager.connection('bad_db'):
                pass

    def test_cleanup(self):
        """Test proper cleanup of connection pools"""
        test_db_name = 'test_db'
        
        # First verify connection works
        with self.db_manager.connection(test_db_name) as conn:
            cur = conn.cursor()
            cur.execute('INSERT INTO test_table VALUES (1)')
            cur.execute('SELECT * FROM test_table')
            self.assertEqual(cur.fetchone()['id'], 1)
        
        # Clean up pools
        self.db_manager.cleanup()
        
        # Verify pools are empty and connections are closed
        for pool in self.db_manager._pools.values():
            self.assertTrue(pool.empty())
        
        # Verify can't get connection after cleanup
        with self.assertRaises(Exception):
            with self.db_manager.connection(test_db_name):
                pass

if __name__ == '__main__':
    unittest.main()