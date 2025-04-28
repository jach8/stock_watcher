import pandas as pd  
import datetime as dt
import sqlite3 as sql 
import json

def check_if_stock_exists(conn, d = None):
    tables = pd.read_sql(''' select name from sqlite_master where type = 'table' ''', conn)
    if 'stocks' not in tables['name'].to_list():
        if d is None:
            raise ValueError('No stock data to enter in the database')
        else:
            d.to_sql('stocks', conn, index = False)
            return d
    else:
        stock_df = pd.read_sql(''' select * from stocks ''', conn)
        stock_list = stock_df['stock'].to_list()
        if d is None:
            return stock_df
        else:
            d = d[~d['stock'].isin(stock_list)]
            d.to_sql('stocks', conn, index = False, if_exists = 'append')
            return d
    

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def update_json(conn, path):
    stock_df = pd.read_sql(''' select * from stocks ''', conn)
    stock_list = stock_df['stock'].to_list()
    j = load_json(path)
    j['all_stocks'] = stock_list
    with open(path, 'w') as f:
        json.dump(j, f)
    
def add_stock(conn, path, stock): 
    """
    Add a stock to the database: 
    
    Args: 
        conn: Connection to the stock_names database 
        path: Path to the ticker json file 
        stock: Stock to add to the database 
    
    """
    stock_less_special_chars = stock.replace('^', '')
    d = pd.DataFrame({'date': [dt.datetime.today().strftime('%Y-%m-%d')], 'stock': [stock]})
    stock_df = check_if_stock_exists(conn, d)
    conn.commit()
    update_json(conn, path)
    print(f'{stock} added to database')
    
if __name__ == "__main__":
    import sys
    sys.path.append('/Users/jerald/Documents/Dir/Python/stocker')
    from bin.main import get_path 
    
    conn = sql.connect(get_path()['stock_names'])
    path = get_path()['ticker_path']
    add_stock(conn, path, '^IXIC')
    add_stock(conn, path, '^GSPC')
    
