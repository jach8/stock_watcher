import pandas as pd  
import datetime as dt
import sqlite3 as sql 
import json



def delete_stock(conn, stock = None):
    stock_df = pd.read_sql(''' select * from stocks ''', conn)
    out = stock_df[stock_df.stocks != stock]
    out.to_sql('stocks', conn, if_exists='replace', index=False)
    # with open('data/stocks/log/hist.txt', 'a') as t:
    #     t.write(f'{dt.datetime.today()}\t{stock} deleted from database\n')
    update_json()
    print(f'{stock} deleted from database')
