""" Get the earnings calander using ALPHA ADVANTAGE API """

import sys 
sys.path.append('/Users/jerald/Documents/Dir/Python/Stocks')
from alphaadv.bin.pid.key import alpha_key
import requests
import pandas as pd
import datetime as dt
from tqdm import tqdm 
from time import sleep , time 
from pickle import load, dump
import os 


alpha_key = os.getenv("ALPHA_ADVANTAGE_API_KEY")


class EarningsAlpha: 
    def __init__(self, alpha_key = alpha_key): 
        self.api_key = f'&apikey={alpha_key}'
        self.base_url = 'https://www.alphavantage.co/query?'
        
    def income_statement(self, stock, write = True):
        ''' Get Intraday data from alpha advantage API '''
        url = f'{self.base_url}function=INCOME_STATEMENT&symbol={stock}{self.api_key}'
        r = requests.get(url)
        data = r.json()
        print(data.keys())
        income = pd.DataFrame(data['quarterlyReports'])

        return income
    
    def balance_sheet(self, stock, write = True):
        ''' Get Intraday data from alpha advantage API '''
        url = f'{self.base_url}function=BALANCE_SHEET&symbol={stock}{self.api_key}'
        r = requests.get(url)
        data = r.json()
        print(data.keys())
        balance = pd.DataFrame(data['quarterlyReports'])

        return balance
    
    def cash_flow(self, stock, write = True):
        ''' Get Intraday data from alpha advantage API '''
        url = f'{self.base_url}function=CASH_FLOW&symbol={stock}{self.api_key}'
        r = requests.get(url)
        data = r.json()
        print(data.keys())
        cash = pd.DataFrame(data['quarterlyReports'])

        return cash
    
    
    def earnings_info(self, stock, write = True):
        ''' Get Intraday data from alpha advantage API '''
        url = f'{self.base_url}function=EARNINGS&symbol={stock}{self.api_key}'
        r = requests.get(url)
        data = r.json()
        print(data.keys())
        earnings = pd.DataFrame(data['quarterlyEarnings'])

        return earnings
    
    def load_earnings(self):
        file = open(f'alphaadv/data/earnings.pkl', 'rb') 
        return load(file)
    
    def write_earnings(self, earnings_dict):
        with open(f'alphaadv/data/earnings.pkl', 'wb') as f:
            dump(earnings_dict, f)
    
    def _get_all(self, stock, write = True):
        income = self.income_statement(stock, False)
        balance = self.balance_sheet(stock, False)
        cash = self.cash_flow(stock, False)
        earnings = self.earnings_info(stock, False)
        if write:
            try:
                d = self.load_earnings()
                d[stock] = {'income': income, 'balance': balance, 'cash': cash, 'earnings': earnings}
            except (FileNotFoundError, EOFError):
                d = {stock: {'income': income, 'balance': balance, 'cash': cash, 'earnings': earnings}}
            
            self.write_earnings(d)
                
    
    def _get_list(self, stocks, write = True):
        for stock in tqdm(stocks):
            try:
                self._get_all(stock, write)
            except Exception as e:
                print(f'Error: {e}')
                continue
            sleep(3)
                
            
            
            
if __name__ == '__main__':
    import sys 
    sys.path.append('/Users/jerald/Documents/Dir/Python/Stocks')
    from bin.main import Manager 
    m = Manager()
    stocks = m.Optionsdb.stocks['equities']
    
    d = EarningsAlpha()
    
    # earnings_dict = d._get_list(stocks, True)
    # print(earnings_dict)
    
    # d._get_all('msft')
    stocks = ['smci']
    d._get_list(stocks, True)
    # print(d.load_earnings().keys())
