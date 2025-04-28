"""
Grabs the Earnings Data for equities in the data/stocks/tickers.json file
The data is stored in the value/earnings.json file

The format is as follows: 
stock: {
    balance_sheet: Pandas DataFrame,
    income_statement: Pandas DataFrame,
    cashflow: Pandas DataFrame,
    earnings: Pandas DataFrame
}
"""

from tqdm import tqdm 
import yfinance as yf 
import numpy as np 
import pandas as pd 
import json 
import time
import datetime as dt
import pickle




def get_earnings(stock):
    tick = yf.Ticker(stock)
    bs = tick.quarterly_balance_sheet.T
    income = tick.quarterly_income_stmt.T
    cf = tick.quarterly_cash_flow.T
    ed = tick.earnings_dates
    names = ['balance_sheet', "income_statement", "cashflow", "earnings"]
    lodfs = [bs.T, income.T, cf.T, ed]
    # lodfs = [df.reset_index().astype(str).rename(columns = {'index':"Date"}).to_dict('records') for df in lodfs]
    # return {stock: dict(zip(names, lodfs))}
    return dict(zip(names, lodfs))


def DownloadEarnings(stocks, path):
    pbar = tqdm(stocks, desc = "Earnings Data")
    earnings = {}
    begin = '\033[92m'
    endoc = '\033[0m'
    for stock in pbar:
        pbar.set_description(f"Earnings Data {begin}${stock.upper()}{endoc}")
        try:
            earnings[stock] = get_earnings(stock)
            time.sleep(5)
        except Exception as e:
            error_color = '\033[93m'
            print(f"{error_color}Error: {e}{endoc}")
            continue
    
    with open(path, 'wb') as f:
        pickle.dump(earnings, f)
    return earnings


def LoadEarnings(path):
    """ 
    Load the Earnings Dates from the Pickle File 
    """
    earnings = pickle.load(open(path, 'rb'))
    stocks = list(earnings.keys())
    [earnings.pop(x) for x in ['hsbc', 'djt']]
    stocks = list(earnings.keys())
    print(dt.datetime.now())
    out = []
    for x in stocks:
        df = earnings[x]['earnings'].copy().reset_index()
        earn_dates = [str(x).split(':00-')[0] for x in df['Earnings Date']]
        df['Earnings Date'] = pd.to_datetime(earn_dates)
        df.insert(0, 'stock', x.upper())
        start = dt.datetime.now()
        end = start + dt.timedelta(days=15)
        outdf = df[
            (df['Earnings Date'] >= start.date().strftime('%Y-%m-%d')) & 
            (df['Earnings Date'] <= end.date().strftime('%Y-%m-%d'))
        ]
        out.append(outdf)
        
    out = pd.concat(out)
    report_time = np.where(out['Earnings Date'].dt.hour < 11, 'AM', 'PM')
    out.insert(2, 'Hour', out['Earnings Date'].dt.hour)
    out.insert(3, 'Time', report_time)
    out['Earnings Date'] = out['Earnings Date'].dt.strftime('%Y-%m-%d')

    out_final = out.drop_duplicates(subset = ['stock']).rename(columns = {
        'Earnings Date': 'Date',
        'EPS Estimate': 'EPS',
    }).drop(columns = ['Reported EPS', 'Surprise(%)']).reset_index(drop = True)
    return out_final


if __name__ == '__main__':
    stocks = json.load(open('data/stocks/tickers.json'))['equities']
    earnings = DownloadEarnings(stocks, 'value/earnings/earnings.pkl')
    print(earn)
        