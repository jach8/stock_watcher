# Database Layer Design

## 1. Core Components

### Database Manager with Connection Pooling

```python
# Existing robust DBManager implementation provides:
- Connection pooling
- Resource cleanup
- Error handling
- Logging
- Context managers
```

### Query Builder Addition

```python
from typing import List, Dict, Optional, Any
from dataclasses import dataclass

@dataclass
class SQLQuery:
    """Represents a SQL query with parameters"""
    text: str
    params: Dict[str, Any]

class QueryBuilder:
    """Builds SQL queries safely with parameter binding"""
    
    def __init__(self) -> None:
        self._query_parts: List[str] = []
        self._params: Dict[str, Any] = {}
        
    def select(self, columns: List[str]) -> 'QueryBuilder':
        cols = ', '.join(self._sanitize_identifier(col) for col in columns)
        self._query_parts.append(f"SELECT {cols}")
        return self
        
    def from_table(self, table: str) -> 'QueryBuilder':
        self._query_parts.append(f"FROM {self._sanitize_identifier(table)}")
        return self
        
    def where(self, conditions: Dict[str, Any]) -> 'QueryBuilder':
        clauses = []
        for col, value in conditions.items():
            param_name = f"p_{len(self._params)}"
            clauses.append(f"{self._sanitize_identifier(col)} = :{param_name}")
            self._params[param_name] = value
        
        if clauses:
            self._query_parts.append("WHERE " + " AND ".join(clauses))
        return self
        
    def order_by(self, columns: List[str], desc: bool = False) -> 'QueryBuilder':
        cols = ', '.join(self._sanitize_identifier(col) for col in columns)
        direction = "DESC" if desc else "ASC"
        self._query_parts.append(f"ORDER BY {cols} {direction}")
        return self
        
    def build(self) -> SQLQuery:
        """Build the final query with parameters"""
        return SQLQuery(
            text=' '.join(self._query_parts),
            params=self._params
        )
        
    @staticmethod
    def _sanitize_identifier(identifier: str) -> str:
        """Prevent SQL injection in identifiers"""
        return ''.join(c for c in identifier if c.isalnum() or c in ['_', '.'])
```

### Base Repository

```python
from abc import ABC, abstractmethod
import pandas as pd
from typing import Optional, List, Dict, Any

class BaseRepository(ABC):
    """Base class for all repositories"""
    
    def __init__(self, db_manager: DBManager):
        self.db_manager = db_manager
        self.query_builder = QueryBuilder()
        
    def execute_query(self, query: SQLQuery) -> pd.DataFrame:
        """Execute a query and return results as DataFrame"""
        with self.db_manager.connection(self.db_name) as conn:
            try:
                return pd.read_sql(query.text, conn, params=query.params)
            except Exception as e:
                logger.error(f"Query execution error in {self.__class__.__name__}: {e}")
                raise RepositoryException(f"Query failed: {e}")

    @abstractmethod
    def get_by_id(self, id: Any) -> Optional[pd.DataFrame]:
        """Get entity by ID"""
        pass
```

## 2. Specific Repositories

### Price Repository

```python
class PriceRepository(BaseRepository):
    """Handles price data operations"""
    
    def __init__(self, db_manager: DBManager):
        super().__init__(db_manager)
        self.db_name = "daily_db"
    
    def get_ohlc(self, symbol: str, start_date: str) -> pd.DataFrame:
        query = self.query_builder\
            .select(['date', 'open', 'high', 'low', 'close', 'volume'])\
            .from_table(symbol)\
            .where({'date': start_date})\
            .order_by(['date'])\
            .build()
            
        return self.execute_query(query)
```

### Options Repository

```python
class OptionsRepository(BaseRepository):
    """Handles options data operations"""
    
    def __init__(self, db_manager: DBManager):
        super().__init__(db_manager)
        self.db_name = "options_db"
    
    def get_latest_chain(self, symbol: str) -> pd.DataFrame:
        subquery = f"""
        SELECT MAX(datetime(gatherdate)) 
        FROM {self._sanitize_identifier(symbol)}
        """
        
        query = self.query_builder\
            .select(['*'])\
            .from_table(symbol)\
            .where({'datetime(gatherdate)': subquery})\
            .build()
            
        return self.execute_query(query)
```

## 3. Usage Example

```python
# Configuration
connections = {
    'daily_db': 'path/to/daily.db',
    'options_db': 'path/to/options.db'
}

# Initialize manager
db_manager = DBManager(connections, pool_size=5)

# Use repositories
price_repo = PriceRepository(db_manager)
options_repo = OptionsRepository(db_manager)

# Get data
with db_manager:  # Ensures connections are properly closed
    spy_data = price_repo.get_ohlc('SPY', '2023-01-01')
    spy_options = options_repo.get_latest_chain('SPY')
```

## 4. Implementation Strategy

1. **Phase 1: Core Infrastructure** (Week 1)
   - Implement DBManager with connection pooling
   - Implement QueryBuilder with parameter binding
   - Create BaseRepository class

2. **Phase 2: Repository Migration** (Week 2)
   - Create PriceRepository
   - Create OptionsRepository
   - Add required queries to each repository

3. **Phase 3: Service Integration** (Week 3)
   - Integrate with existing services
   - Update service layer to use new repositories
   - Add error handling and logging

4. **Phase 4: Testing & Validation** (Week 4)
   - Unit tests for each component
   - Integration tests
   - Performance benchmarks
   - Migration validation

## 5. Questions for Discussion

1. Should we add caching layer for frequently accessed data?
2. Do we need additional connection pool settings?
3. Should we implement query result caching?
4. What additional logging metrics would be useful?