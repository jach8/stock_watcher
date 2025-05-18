"""Core database components."""

import logging
import sqlite3
import os
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Generator, Union
import pandas as pd
from queue import Queue
from threading import Lock

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Base exceptions
class DatabaseError(Exception):
    """Base exception class for database-related errors."""
    pass

class ConnectionError(DatabaseError):
    """Exception raised for database connection errors."""
    pass

class QueryError(DatabaseError):
    """Exception raised for query execution errors."""
    pass

class RepositoryError(DatabaseError):
    """Exception raised for repository-related errors."""
    pass

@dataclass
class SQLQuery:
    """SQL query with parameters."""
    text: str
    params: Dict[str, Any]

class QueryBuilder:
    """Builds SQL queries safely with parameter binding."""
    
    def __init__(self) -> None:
        """Initialize query builder."""
        self.reset()
        
    def reset(self) -> None:
        """Reset builder state."""
        self._query_parts = []
        self._params = {}
        
    def select(self, columns: List[str]) -> 'QueryBuilder':
        """Add SELECT clause."""
        cols = ', '.join(f'[{self._sanitize_identifier(col)}]' for col in columns)
        self._query_parts.append(f"SELECT {cols}")
        return self
        
    def from_table(self, table: str) -> 'QueryBuilder':
        """Add FROM clause."""
        self._query_parts.append(f"FROM [{self._sanitize_identifier(table)}]")
        return self
        
    def where(self, conditions: Dict[str, Any]) -> 'QueryBuilder':
        """Add WHERE clause with conditions."""
        clauses = []
        for col, value in conditions.items():
            param_name = f"p_{len(self._params)}"
            clauses.append(f"[{self._sanitize_identifier(col)}] = :{param_name}")
            self._params[param_name] = value
        
        if clauses:
            self._query_parts.append("WHERE " + " AND ".join(clauses))
        return self
        
    def order_by(self, columns: List[str], desc: bool = False) -> 'QueryBuilder':
        """Add ORDER BY clause."""
        cols = ', '.join(f'[{self._sanitize_identifier(col)}]' for col in columns)
        direction = "DESC" if desc else "ASC"
        self._query_parts.append(f"ORDER BY {cols} {direction}")
        return self
        
    def build(self) -> SQLQuery:
        """Build final query with parameters."""
        query = SQLQuery(
            text=' '.join(self._query_parts),
            params=self._params.copy()  # Copy params to avoid state issues
        )
        self.reset()  # Reset state after building
        return query
        
    @staticmethod
    def _sanitize_identifier(identifier: str) -> str:
        """Prevent SQL injection in identifiers."""
        return ''.join(c for c in identifier if c.isalnum() or c in ['_', '.', 'Y'])

class DBManager:
    """Manages database connections with connection pooling."""
    
    def __init__(self, connections: Dict[str, str], pool_size: int = 5):
        """Initialize database manager.
        
        Args:
            connections: Map of database names to their file paths
            pool_size: Maximum connections per database
        """
        self.connections = connections
        self.pool_size = pool_size
        self._pools = {}
        self._locks = {}
        self._initialize_pools()
        
    def _initialize_pools(self) -> None:
        """Initialize connection pools for all databases."""
        for db_name, db_path in self.connections.items():
            # Verify database directory exists
            db_dir = os.path.dirname(db_path)
            if not os.path.exists(db_dir):
                raise ConnectionError(f"Database directory does not exist: {db_dir}")
                
            # Create connection pool
            self._pools[db_name] = Queue(maxsize=self.pool_size)
            self._locks[db_name] = Lock()
            
            # Create and test one connection first
            try:
                test_conn = self._create_connection(db_path)
                test_conn.close()
            except sqlite3.Error as e:
                raise ConnectionError(f"Failed to connect to {db_name} at {db_path}: {e}")
            except Exception as e:
                raise ConnectionError(f"Unexpected error connecting to {db_name}: {e}")
            
            # Pre-populate pool with connections
            try:
                for _ in range(self.pool_size):
                    conn = self._create_connection(db_path)
                    self._pools[db_name].put(conn)
            except Exception as e:
                # Clean up any connections that were created
                while not self._pools[db_name].empty():
                    try:
                        conn = self._pools[db_name].get()
                        conn.close()
                    except:
                        pass
                raise ConnectionError(f"Failed to initialize connection pool for {db_name}: {e}")
    
    def _create_connection(self, db_path: str) -> sqlite3.Connection:
        """Create a new database connection."""
        try:
            conn = sqlite3.connect(db_path, detect_types=sqlite3.PARSE_DECLTYPES)
            conn.row_factory = sqlite3.Row
            return conn
        except sqlite3.Error as e:
            logger.error(f"Failed to create connection to {db_path}: {e}")
            raise ConnectionError(f"Could not connect to {db_path}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error connecting to {db_path}: {e}")
            raise ConnectionError(f"Error connecting to {db_path}: {e}")
    
    @contextmanager
    def connection(self, db_name: str) -> Generator[sqlite3.Connection, None, None]:
        """Get a connection from the pool."""
        if db_name not in self._pools:
            raise ConnectionError(f"Unknown database: {db_name}")
            
        conn = None
        try:
            conn = self._pools[db_name].get()
            yield conn
            self._pools[db_name].put(conn)
        except sqlite3.Error as e:
            logger.error(f"Database error for {db_name}: {e}")
            if conn:
                try:
                    conn.close()
                except:
                    pass
            raise ConnectionError(f"Database error for {db_name}: {e}")
        except Exception as e:
            logger.error(f"Error managing connection for {db_name}: {e}")
            if conn:
                try:
                    conn.close()
                except:
                    pass
            raise ConnectionError(f"Connection error for {db_name}: {e}")
            
    def cleanup(self) -> None:
        """Clean up all database connections."""
        for db_name, pool in self._pools.items():
            while not pool.empty():
                try:
                    conn = pool.get_nowait()
                    conn.close()
                except:
                    continue
        self._pools.clear()
        logger.info("Database connections cleaned up")

class BaseRepository(ABC):
    """Base class for all repositories."""
    
    def __init__(self, db_manager: DBManager):
        """Initialize repository."""
        self.db_manager = db_manager
        self.query_builder = QueryBuilder()
        
    def execute_query(self, query: SQLQuery) -> pd.DataFrame:
        """Execute a query and return results as DataFrame."""
        with self.db_manager.connection(self.db_name) as conn:
            try:
                return pd.read_sql(query.text, conn, params=query.params)
            except sqlite3.OperationalError as e:
                logger.error(f"Query execution error in {self.__class__.__name__}: {e}")
                raise RepositoryError(f"Query failed: {e}")
            except Exception as e:
                logger.error(f"Unexpected error in {self.__class__.__name__}: {e}")
                raise RepositoryError(f"Repository operation failed: {e}")
    
    @abstractmethod
    def get_by_id(self, id: Any) -> Optional[pd.DataFrame]:
        """Get entity by ID."""
        pass