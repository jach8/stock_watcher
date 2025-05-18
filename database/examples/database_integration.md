# Database Layer Integration Design

## Current Pipeline Analysis

The Pipeline class has these key characteristics:
1. Inherits from Manager class
2. Handles multiple database operations:
   - Options data updates
   - Stock price updates
   - Bond data updates
3. Uses connection management through the Manager base class
4. Performs batch operations with progress tracking
5. Includes comprehensive error handling and logging

## Integration Strategy

### 1. Database Manager Integration

```python
class Manager:
    def __init__(self, connections: Optional[dict] = None):
        self.db_manager = DBManager(
            connections or get_path(),
            pool_size=5
        )
        # Initialize repositories with db_manager
        self.price_repository = PriceRepository(self.db_manager)
        self.options_repository = OptionsRepository(self.db_manager)
        self.bonds_repository = BondsRepository(self.db_manager)

    def close_connection(self):
        self.db_manager.close_all()
```

### 2. Repository Pattern Implementation

```python
class OptionsRepository(BaseRepository):
    def __init__(self, db_manager: DBManager):
        super().__init__(db_manager)
        self.db_name = "options_db"
        
    def insert_new_chain(self, stock: str) -> Optional[pd.DataFrame]:
        """Insert a new options chain into the database"""
        with self.db_manager.connection(self.db_name) as conn:
            try:
                # Existing logic moved to repository
                # Using QueryBuilder for safe query construction
                return self._insert_chain(conn, stock)
            except Exception as e:
                logger.error(f"Failed to insert new chain for {stock}: {e}")
                return None

    def update_change_vars(self, stock: str) -> None:
        """Update change variables for a stock"""
        with self.db_manager.connection(self.db_name) as conn:
            try:
                # Existing logic moved to repository
                # Using QueryBuilder for safe query construction
                self._update_changes(conn, stock)
            except Exception as e:
                logger.error(f"Failed to update changes for {stock}: {e}")
                raise
```

### 3. Modified Pipeline Class

```python
class Pipeline(Manager):
    def quick_update(self, stocks: Optional[List[str]] = None) -> None:
        stocks_to_update = stocks if stocks else ['spy', 'qqq', 'iwm', 'dia', 'vxx']
        
        for stock in tqdm(stocks_to_update, desc="Quick Update"):
            try:
                # Using repository pattern
                new_chain = self.options_repository.insert_new_chain(stock)
                if new_chain is not None:
                    self.options_repository.update_change_vars(stock)
                    self.options_repository.update_cp(stock, new_chain)
                    self.options_repository.update_stock_metrics(stock, new_chain)
            except Exception as e:
                logger.error(f"Error updating {stock}: {e}")

    def update_stock_prices(self) -> None:
        try:
            # Using repository pattern
            self.price_repository.update_prices()
        except Exception as e:
            logger.error(f"Failed to update stock prices: {e}")
```

## Key Benefits of Integration

1. **Connection Management**
   - Automated connection pooling
   - Proper resource cleanup
   - Connection reuse for better performance

2. **Query Safety**
   - Parameterized queries prevent SQL injection
   - Consistent error handling
   - Query building with type safety

3. **Resource Efficiency**
   - Connection pooling reduces overhead
   - Automatic connection cleanup
   - Optimal connection reuse during batch operations

4. **Maintainability**
   - Clear separation of concerns
   - Centralized database access logic
   - Consistent error handling and logging

## Migration Steps

1. **Phase 1: Infrastructure Setup**
   ```python
   # Create new DBManager instance
   db_manager = DBManager(connections, pool_size=5)
   
   # Initialize repositories
   options_repo = OptionsRepository(db_manager)
   price_repo = PriceRepository(db_manager)
   ```

2. **Phase 2: Repository Migration**
   ```python
   # Migrate existing database operations to repositories
   class OptionsRepository:
       def update_chain(self, stock: str) -> None:
           with self.db_manager.connection(self.db_name) as conn:
               # Existing logic using connection pooling
               pass
   ```

3. **Phase 3: Pipeline Updates**
   ```python
   class Pipeline(Manager):
       def update_options(self) -> None:
           for stock in tqdm(self.stocks):
               with self.db_manager:  # Ensures proper connection handling
                   self.options_repository.update_chain(stock)
   ```

## Error Handling Strategy

```python
class DatabaseError(Exception):
    """Base exception for database operations"""
    pass

class ConnectionError(DatabaseError):
    """Connection-related errors"""
    pass

class QueryError(DatabaseError):
    """Query execution errors"""
    pass

# Usage in repositories
def execute_query(self, query: SQLQuery) -> pd.DataFrame:
    try:
        with self.db_manager.connection(self.db_name) as conn:
            return pd.read_sql(query.text, conn, params=query.params)
    except sql.Error as e:
        raise QueryError(f"Query failed: {e}")
    except Exception as e:
        raise DatabaseError(f"Unexpected error: {e}")
```

## Conclusion

This integration strategy maintains the existing functionality while adding the benefits of:
- Connection pooling
- Safe query building
- Proper resource management
- Consistent error handling
- Clear separation of concerns

The migration can be done incrementally, ensuring minimal disruption to the existing system while improving its robustness and maintainability.