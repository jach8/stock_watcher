"""
Grabs the Earnings Data for equities in the data/stocks/tickers.json file
The data is stored in the value/earnings.json file

The format is as follows: 

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


def UpdateEarnings(stocks, path):
    """ Load in th Earnings Data, and update the files accordingly. """
    earnings = pickle.load(open(path, 'rb'))
    ekeys = list(earnings.keys())
    pbar = tqdm(stocks, desc = "Earnings Data")
    begin = '\033[92m'
    endoc = '\033[0m'
    for stock in pbar:
        pbar.set_description(f"Earnings Data {begin}${stock.upper()}{endoc}")
        try: 
            new_data = get_earnings(stock)
            # If stock is alreaqdy in the dictionary, we need to append the data 
            if stock in ekeys: 
                for key in new_data.keys(): 
                    # Load Old Data
                    old_df = earnings[stock][key]
                    # Get New Data
                    new_df = new_data[key]
                    # If key 
                    if key != 'earnings':
                        # Check if old df columns are a datetime object, OR datetime string
                        if isinstance(old_df.columns, pd.DatetimeIndex):
                            df = concatenate_statements(old_df, new_df).T
                        else:
                            df = pd.concat([old_df.T, new_df], axis = 1)
                        df = df.T.sort_index().T
                    else: 
                        df = pd.concat([old_df, new_df], axis = 0)
                    
                    earnings[stock][key] = df
            else: 
                # If stock not in the dictionary, just add the new data. 
                earnings[stock] = new_data
                
        except Exception as e:
            error_color = '\033[93m'
            print(f"{error_color}{stock.upper()} Error: {e}{endoc}")
            pass    
        
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
    
    out = []
    for x in stocks:
        df = earnings[x]['earnings'].copy().reset_index()
        earn_dates = [str(x).split(':00-')[0] for x in df['Earnings Date']]
        df['Earnings Date'] = pd.to_datetime(earn_dates)
        df.insert(0, 'stock', x.upper())
        start = dt.datetime.now()
        end = start + dt.timedelta(days=60)
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
        'EPS Estimate': 'estimated_eps',
    }).drop(columns = ['Reported EPS', 'Surprise(%)']).reset_index(drop = True)
    return out_final.sort_values('Date')

def LoadAllEarningsDates(path):
    """ 
    Load the Earnings Dates from the Pickle File 
    """
    earnings = pickle.load(open(path, 'rb'))
    stocks = list(earnings.keys())
    [earnings.pop(x) for x in ['hsbc', 'djt']]
    stocks = list(earnings.keys())
    
    out = []
    for x in stocks:
        df = earnings[x]['earnings'].copy().reset_index()
        earn_dates = [str(x).split(':00-')[0] for x in df['Earnings Date']]
        df['Earnings Date'] = pd.to_datetime(earn_dates)
        df.insert(0, 'stock', x.upper())
        out.append(df)
        
    out = pd.concat(out)
    report_time = np.where(out['Earnings Date'].dt.hour < 11, 'AM', 'PM')
    out.insert(2, 'Hour', out['Earnings Date'].dt.hour)
    out.insert(3, 'Time', report_time)
    out['Earnings Date'] = out['Earnings Date'].dt.strftime('%Y-%m-%d')

    out_final = out.rename(columns = {
        'Earnings Date': 'Date',
        'EPS Estimate': 'EPS',
    }).reset_index(drop = True)
    
    out_final = out.copy()
    
    # out_final.Date = pd.to_datetime(out_final.Date)
    return out_final.sort_values('Earnings Date')    


def concatenate_statements(old_df, new_df):
    """ Concatenates the old and new dataframes """
    # If Index is string of accounting entries, we need to transpose the dataframes
    if isinstance(old_df.columns, pd.DatetimeIndex):
        pass
    else:
        old_df = old_df.T
        
    if isinstance(new_df.columns, pd.DatetimeIndex):
        pass
    else:
        new_df = new_df.T
    
    
    df = pd.concat([old_df, new_df], axis = 1).T.sort_index()
    return df[~df.index.duplicated(keep='first')]


if __name__ == '__main__':
    stocks = json.load(open('data/stocks/tickers.json'))['equities']
    # earnings = DownloadEarnings(stocks, 'value/earnings/earnings.pkl')
    # print(earn)
    print(LoadEarnings('data/earnings/earnings.pkl'))
    print(LoadAllEarningsDates('data/earnings/earnings.pkl'))