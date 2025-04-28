""" cvar.py - Module for calculating the change variables for the option contracts. """

import sys
import logging
# import pytest
import time
import psutil
import pandas as pd
import sqlite3 as sql
from typing import Optional, List, Dict, Any

sys.path.append('/Users/jerald/Documents/Dir/Python/Stocks')
from bin.options.optgd.db_connect import Connector

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class ChangeVars(Connector):
    """
    Module for calculating the change variables for the option contracts.
    Inherits from the Connector Class.
    """

    def __init__(self, connections: Dict[str, Any]):
        super().__init__(connections)
        try:
            self.date_db = sql.connect(connections['dates_db'])
        except sql.Error as e:
            logging.error(f"ChangeVars: Failed to connect to date_db: {e}")
            raise

    def _initialize_date_db(self, stock: str, date_col: str = 'gatherdate') -> None:
        """
        Initialize the date database with the unique dates for each stock.
        """
        try:
            query = f"SELECT DISTINCT datetime({date_col}) FROM {stock}"
            cursor = self.option_db_cursor
            cursor.execute(query)
            dates = [x[0] for x in cursor.fetchall()]
            df = pd.DataFrame({'stock': [stock] * len(dates), 'gatherdate': dates})
            df.to_sql(stock, self.date_db, if_exists='replace', index=False)
            self.date_db.commit()
        except sql.Error as e:
            logging.error(f"ChangeVars: Error initializing date_db for {stock}: {e}")
            raise

    def _last_dates(self, stock: str, N: int = 5) -> List[str]:
        """
        Get the last N dates for a stock in the vol.db.
        """
        try:
            query = f"SELECT datetime(gatherdate) FROM {stock} ORDER BY datetime(gatherdate) DESC LIMIT {N}"
            cursor = self.date_db.cursor()
            dates = [x[0] for x in cursor.execute(query).fetchall()]
            return sorted(dates)
        except sql.Error as e:
            logging.error(f"ChangeVars: Error fetching last dates for {stock}: {e}")
            raise

    def _calc_changes(self, stock: str, N: Optional[int] = None) -> pd.DataFrame:
        """
        Calculate the change values for each contract in the option chain.
        """
        try:
            if N is None:
                dte = 'date(gatherdate) > date("2022-11-17")'
            else:
                recent_dates = self._last_dates(stock, N=N)
                if not recent_dates:
                    raise ValueError(f"No dates found for stock {stock}")
                dte = f'date(gatherdate) BETWEEN date("{recent_dates[0]}") AND date("{recent_dates[-1]}")'

            # logging.info(f"Change Variables: Executing query for {stock} with date range: {dte}")

            query = f"""
                WITH t0 AS (
                    SELECT 
                        MAX(datetime(gatherdate)) AS gatherdate,
                        contractsymbol,  
                        stk_price,
                        lastprice,
                        ask, 
                        bid,
                        change, 
                        CAST(percentchange AS FLOAT) AS percentchange,
                        CAST(IFNULL(volume, 0) AS INT) AS vol,
                        CAST(IFNULL(openinterest, 0) AS INT) AS oi,
                        impliedvolatility
                    FROM {stock}
                    WHERE {dte}
                    GROUP BY contractsymbol, date(gatherdate)
                    ORDER BY datetime(gatherdate) ASC
                ),
                t1 AS (
                    SELECT 
                        *,
                        stk_price - LAG(stk_price, 1) OVER (PARTITION BY contractsymbol ORDER BY datetime(gatherdate)) AS stk_price_chg,
                        AVG(stk_price) OVER (PARTITION BY contractsymbol ORDER BY datetime(gatherdate) ROWS BETWEEN 29 PRECEDING AND CURRENT ROW) AS stk_price_avg_30d,
                        AVG(stk_price) OVER (PARTITION BY contractsymbol ORDER BY datetime(gatherdate) ROWS BETWEEN 5 PRECEDING AND CURRENT ROW) AS stk_price_avg_5d,
                        lastprice - LAG(lastprice, 1) OVER (PARTITION BY contractsymbol ORDER BY datetime(gatherdate)) AS lastprice_chg,
                        AVG(lastprice) OVER (PARTITION BY contractsymbol ORDER BY datetime(gatherdate) ROWS BETWEEN 29 PRECEDING AND CURRENT ROW) AS lastprice_avg_30d,
                        AVG(lastprice) OVER (PARTITION BY contractsymbol ORDER BY datetime(gatherdate) ROWS BETWEEN 5 PRECEDING AND CURRENT ROW) AS lastprice_avg_5d,
                        100 * ((lastprice - LAG(lastprice, 1) OVER (PARTITION BY contractsymbol ORDER BY datetime(gatherdate))) / LAG(lastprice, 1) OVER (PARTITION BY contractsymbol ORDER BY datetime(gatherdate))) AS pct_chg,
                        impliedvolatility - LAG(impliedvolatility, 1) OVER (PARTITION BY contractsymbol ORDER BY datetime(gatherdate)) AS iv_chg,
                        AVG(impliedvolatility) OVER (PARTITION BY contractsymbol ORDER BY datetime(gatherdate) ROWS BETWEEN 5 PRECEDING AND CURRENT ROW) AS iv_avg_5d,
                        AVG(impliedvolatility) OVER (PARTITION BY contractsymbol ORDER BY datetime(gatherdate) ROWS BETWEEN 29 PRECEDING AND CURRENT ROW) AS iv_avg_30d,
                        AVG(impliedvolatility) OVER (PARTITION BY contractsymbol ORDER BY datetime(gatherdate)) AS iv_avg_all,
                        vol - LAG(vol, 1) OVER (PARTITION BY contractsymbol ORDER BY datetime(gatherdate)) AS vol_chg,
                        oi - LAG(oi, 1) OVER (PARTITION BY contractsymbol ORDER BY datetime(gatherdate)) AS oi_chg,
                        CASE WHEN (oi - LAG(oi, 1) OVER (PARTITION BY contractsymbol ORDER BY datetime(gatherdate))) > LAG(vol, 1) OVER (PARTITION BY contractsymbol ORDER BY datetime(gatherdate)) THEN 1 ELSE 0 END AS flag,
                        CASE WHEN (oi - LAG(oi, 1) OVER (PARTITION BY contractsymbol ORDER BY datetime(gatherdate))) > LAG(vol, 1) OVER (PARTITION BY contractsymbol ORDER BY datetime(gatherdate)) THEN ((oi - LAG(oi, 1) OVER (PARTITION BY contractsymbol ORDER BY datetime(gatherdate))) - LAG(vol, 1) OVER (PARTITION BY contractsymbol ORDER BY datetime(gatherdate))) ELSE 0 END AS amnt
                    FROM t0
                    ORDER BY datetime(gatherdate) ASC
                )
                SELECT * FROM t1
            """
            cursor = self.option_db_cursor
            cursor.execute(query)
            data = cursor.fetchall()
            columns = [x[0] for x in cursor.description]
            df = pd.DataFrame(data, columns=columns).rename(columns={'oi': 'openinterest', 'vol': 'volume'})

            # Data integrity checks
            assert not df.empty, f"No data returned for stock {stock}"
            assert all(col in df.columns for col in ['gatherdate', 'contractsymbol', 'stk_price', 'lastprice']), "Missing required columns"
            assert df['gatherdate'].isnull().sum() == 0, "Null values found in gatherdate"
            return df

        except Exception as e:
            logging.error(f"ChangeVars: Error in _calc_changes for {stock}: {e}")
            raise

    def update_change_vars(self, stock: str) -> None:
        """
        Updates the change_db with new changes for each contractsymbol.
        Checks for duplicates (entire row) before and after updating and removes them.
        Handles potential errors during the update process.
        """
        try:
            # Validate stock symbol
            if not stock or not isinstance(stock, str):
                raise ValueError("Invalid stock symbol provided.")

            # Check if the stock exists in the change_db
            if not self.__check_for_stock_in_change_db(stock):
                # logging.info(f"Change Variables: Initializing change_db for stock: {stock}")
                self.__initialize_change_db(stock)
            else:
                # Calculate changes for the stock
                # logging.info(f"Change Variables: Calculating changes for stock: {stock}")
                df = self._calc_changes(stock, N=4)

                # Ensure data is not empty
                if df.empty:
                    logging.warning(f"Change Variables: No data found for stock {stock}. Skipping update.")
                    return

                # Filter for the latest data
                latest_data = df[df.gatherdate == df.gatherdate.max()]

                # Ensure latest data is not empty
                if latest_data.empty:
                    logging.warning(f"Change Variables: No new data found for stock {stock}. Skipping update.")
                    return

                # Check for duplicates in the new data before appending (whole row)
                if not latest_data.duplicated().empty:
                    logging.warning(f"Change Variables: Duplicates found in new data for {stock}. Dropping duplicates.")
                    latest_data = latest_data.drop_duplicates(keep='last')

                # Read existing data from change_db to check for duplicates
                existing_data = pd.read_sql_query(f"SELECT * FROM {stock}", self.change_db)
                
                # Combine and check for duplicates (whole row)
                combined = pd.concat([existing_data, latest_data], ignore_index=True)
                if not combined.duplicated().empty:
                    # logging.info(f"Change Variables: Removing duplicates after combining new data with existing for {stock}")
                    combined = combined.drop_duplicates(keep='last')

                # Replace the table with the updated data without duplicates
                try:
                    # logging.info(f"Change Variables: Updating change_db for stock: {stock} after checking for duplicates")
                    combined.to_sql(stock, self.change_db, if_exists='replace', index=False)
                    self.change_db.commit()
                except sql.Error as e:
                    logging.error(f"ChangeVars: Database error while updating {stock}: {e}")
                    self.change_db.rollback()  # Rollback in case of failure
                    raise
                except Exception as e:
                    logging.error(f"ChangeVars: Unexpected error while updating {stock}: {e}")
                    self.change_db.rollback()  # Rollback in case of failure
                    raise

        except ValueError as ve:
            logging.error(f"ChangeVars: Validation error in update_change_vars: {ve}")
            raise
        except sql.Error as se:
            logging.error(f"ChangeVars: SQL error in update_change_vars: {se}")
            raise
        except Exception as e:
            logging.error(f"ChangeVars: Unexpected error in update_change_vars: {e}")
            raise

    
    def __check_for_stock_in_change_db(self, stock: str) -> bool:
        """
        Check if the stock is in the change_db.
        """
        try:
            query = f"SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='{stock}')"
            return bool(self.change_db_cursor.execute(query).fetchone()[0])
        except sql.Error as e:
            logging.error(f"ChangeVars: Error checking for stock {stock} in change_db: {e}")
            raise

    def __initialize_change_db(self, stock: str) -> None:
        """
        Initialize the change_db for a stock.
        """
        try:
            df = self._calc_changes(stock, N=None)
            df.to_sql(stock, self.change_db, if_exists='replace', index=False)
            self.change_db.commit()
        except Exception as e:
            logging.error(f"ChangeVars: Error initializing change_db for {stock}: {e}")
            raise
        


# # Runtime Tests
# def test_calc_changes(connections: Dict[str, str] = None):
#     cv = ChangeVars(connections)

#     def test_valid_stock():
#         df = cv._calc_changes("oklo", N=5)
#         assert not df.empty
#         assert all(col in df.columns for col in ['gatherdate', 'contractsymbol'])

#     def test_invalid_stock():
#         with pytest.raises(sql.OperationalError):
#             cv._calc_changes("^GSPC", N=5)

#     def test_date_range():
#         df = cv._calc_changes("cvx", N=5)
#         dates = df['gatherdate'].unique()
#         assert len(dates) <= 5

#     def test_calculations():
#         df = cv._calc_changes("k", N=5)
#         assert not df.empty
#         assert all(col in df.columns for col in ['stk_price', 'lastprice', 'impliedvolatility'])
#         assert not df['stk_price'].isnull().all()
        

#     test_valid_stock()
#     test_invalid_stock()
#     test_date_range()
#     test_calculations()


if __name__ == "__main__":
    import sys
    sys.path.append('/Users/jerald/Documents/Dir/Python/Stocks')
    from bin.main import get_path
    connections = get_path()    
    
    # test_calc_changes(connections)
    
    oc = ChangeVars(connections)
    cur = oc.change_db_cursor
    q = "select * from spy"
    cur.execute(q)
    d = cur.fetchall()
    df = pd.DataFrame(d, columns = [x[0] for x in cur.description])
    print(df)
    print(df.drop_duplicates())
    # oc.update_change_vars('spy')
    
    oc.close_connections()
