"""
Earnings Data Gathering.


"""


import pandas as pd 
import numpy as np 
from pickle import load, dump
import datetime as dt 
import json 
import sys
sys.path.append('/Users/jerald/Documents/Dir/Python/Stocks')
from bin.earnings.util import *

class Earnings:
    def __init__(self, connections):
        self.earnings_path = connections['earnings_dict']
        self.stocks = json.load(open(connections['ticker_path'], 'r'))['equities']
        self.earnings = self._earnings_dict()
        self.calendar = pd.read_csv(connections['earnings_calendar'])
        

    def _download_single_stock_earnings(self, stock):
        """ Downloads Earnings Data from yahoo finance. 

        Args:
            stock: str: Stock ticker
        
        Returns: 
            Dictionary Containting the earnings data. The keys of the dictionary are: 
            balance_sheet, income_statement, cashflow, earnings
        
        """
        earnings = get_earnings(stock)
        return earnings


    def _download_multi_stock_earnings(self):
        """ 
        Downloads Earnings Data from yahoo finance. 
        Once the files are downloaded, Read in the pickle file and concatenate the dataframes, dropping any duplicate rows. 
        Finally, save the Earnings Dates to a CSV file. 

        Args:
            stocks: list: List of stock tickers
        
        Returns: 
            Nested Dictionary with the Keys as Stocks, and the second set of keys "balance_sheet, income_statement, cashflow, earnings"
            
        """
        stocks = self.stocks    
        earnings = DownloadEarnings(stocks, self.earnings_path)
        out = LoadEarnings(self.earnings_path)
        return earnings
    
    def update_earnings(self, stocks = None):
        """ 
        Load in th Earnings Data, and update the files accordingly. 

        Args:
            stocks: list: List of stock tickers
        
        Returns: 
            Nested Dictionary with the Keys as Stocks, and the second set of keys "balance_sheet, income_statement, cashflow, earnings"
        """
        if stocks is None:
            stocks = self.stocks
        return UpdateEarnings(stocks, self.earnings_path)
    
    def get_earning_dates(self):
        """ 
        Load the Earnings Dates from the Pickle File 
        
        Returns: 
            Nested Dictionary with the Keys as Stocks, and the second set of keys "balance_sheet, income_statement, cashflow, earnings"
        """
        return LoadEarnings(self.earnings_path).sort_values('Date')

    def _earnings_dict(self):
        """ 
        Load the Earnings Dates from the Pickle File 
        
        Returns: 
            Nested Dictionary with the Keys as Stocks, and the second set of keys "balance_sheet, income_statement, cashflow, earnings"
        """
        return load(open(self.earnings_path, 'rb'))
    
    def load_balance_sheet(self, stock):
        """ 
        Load the Balance Sheet from the Earnings Dictionary 
        
        Args:
            stock: str: Stock Ticker
        
        Returns: 
            DataFrame: Balance Sheet Data
        """
        return self.earnings[stock]['balance_sheet']
    
    def load_cashflow(self, stock):
        """ 
        Load the Cashflow from the Earnings Dictionary 
        
        Args:
            stock: str: Stock Ticker
        
        Returns: 
            DataFrame: Cashflow Data
        """
        return self.earnings[stock]['cashflow']
    
    def load_income_statement(self, stock):
        """ 
        Load the Income Statement from the Earnings Dictionary 
        
        Args:
            stock: str: Stock Ticker
        
        Returns: 
            DataFrame: Income Statement Data
        """
        return self.earnings[stock]['income_statement']
    
    
    def load_earnings_dates(self, stock):
        """ 
        Load the Earnings Dates from the Earnings Dictionary 
        
        Args:
            stock: str: Stock Ticker
        
        Returns: 
            DataFrame: Earnings Dates
        """
        return self.earnings[stock]['earnings']
    
    def common_balance_sheet_change(self):
        """ 
        QoQ Percent Changes for common entries found in the Balance Sheet. 
        """
        earnings = self.earnings.copy()
        stocks = list(earnings.keys())
        
        bs_entries = {x:list(earnings[x]['balance_sheet'].index) for x in stocks}
        # Now get all of entries that are in common between all stocks

        ents = []
        for x in stocks:
            ents.append(set(bs_entries[x]))
            
        common_ents = list(set.intersection(*ents))

        balance_sheets= []

        for x in stocks:
            df = earnings[x]['balance_sheet'].T[common_ents].sort_index().ffill().pct_change().iloc[-1].to_frame().T*100
            # df = earnings[x]['balance_sheet'].T[common_ents].sort_index().iloc[-1].to_frame().T *100
            df = df.reset_index().rename(columns = {'index': 'lastEarningsDate'})
            df.insert(0, 'stock', x.upper())   
            balance_sheets.append(df)
            
            
        bs = pd.concat(balance_sheets).set_index('stock')
        return bs 
        
        
    def common_income_change(self):
        """
        QoQ Percent Changes for common entries found in the Income Statement. 
        """
        ####################
        #  Percent changes for each stock for common entries found in the income statement
        ####################
        earnings = self.earnings.copy()
        stocks = list(earnings.keys())
        income_statement_entries = {x:list(earnings[x]['income_statement'].index) for x in stocks}

        # Now get all of entries that are in common between all stocks

        ents = []
        for x in stocks:
            ents.append(set(income_statement_entries[x]))
            
        common_ents = list(set.intersection(*ents))

        income_statements= []

        for x in stocks:
            df = earnings[x]['income_statement'].T[common_ents].sort_index().ffill().pct_change().iloc[-1].to_frame().T*100
            df = df.reset_index().rename(columns = {'index': 'lastEarningsDate'})
            df.insert(0, 'stock', x.upper())   
            income_statements.append(df)

        income = pd.concat(income_statements).set_index('stock')
        return income
        
        
    def upcoming_earnings(self, n = 30):
        """ 
        Get the Earnings for the next n days, if n = 0 Returns the current day earnings. 
        
        Returns: 
            DataFrame: Earnings for Today
        
        """
        e = self.get_earning_dates()
        e.Date = pd.to_datetime(e.Date)
        start_date = dt.datetime.now().date().strftime('%Y-%m-%d')
        upcoming = e[e.Date >= start_date]
        
        if n == 0: 
            out = upcoming[upcoming.Date == dt.datetime.now().date()]
        else: 
            max_date = dt.datetime.now() + dt.timedelta(days = n)
            max_date = max_date.strftime('%Y-%m-%d')
            out =  upcoming[upcoming['Date'] <= max_date]
        
        if len(out) == 0:
            print("No Earnings Found")
        else: 
            return out 
    
    def next_earnings(self):
        """
        Return a list of stocks that are next up in earnings 
        """
        edf = self.get_earning_dates()
        edf.Date = pd.to_datetime(edf.Date)
        dtes = sorted(list(edf.Date.unique()))
        edf = edf[edf.Date <= dtes[1]]
        edf.stock =  [x.lower() for x in edf.stock]
        return edf.sort_values(['Date', 'Time'], ascending = [True, True])
    
    def earnings_calendar(self, n = 7, stocks = True):
        """ 
        Get the Earnings for the next n days, if n = 0 Returns the current day earnings. 
        
        Returns: 
            DataFrame: Earnings for Today
        
        """
        df = self.calendar.copy()
        df.reportDate = pd.to_datetime(df.reportDate)
        df.fiscalDateEnding = pd.to_datetime(df.fiscalDateEnding)
        # start_date = dt.datetime.now()
        start_date = dt.datetime.now().date()
        end_date = start_date + dt.timedelta(days = n) 
        df = df[(df.reportDate >= start_date.strftime('%Y-%m-%d')) & (df.reportDate <= end_date.strftime('%Y-%m-%d'))]
        df.symbol = [str(x).lower() for x in df.symbol]
        out = df[df.symbol.isin(self.stocks)].copy().sort_values('reportDate')
        print(f"{df.shape[0]:,.0f} Companies report Earnings in the next {n} days, {out.shape[0]:,.0f} are currently tracked.") 
        if stocks == True: 
            return out
        else: 
            return df 
        
        


    
    

if __name__ == '__main__':
    from bin.main import get_path
    connections = get_path()
    stocks = json.load(open('data/stocks/tickers.json'))['equities']    
    
    e = Earnings(connections)
    # stocks = ['amzn', 'aapl']
    e.earnings_calendar(n = 30)

    # print(e.upcoming_earnings(n = 10))
    # print(e.next_earnings())
