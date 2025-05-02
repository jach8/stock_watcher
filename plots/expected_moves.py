import json
import numpy as np 
import pandas as pd 
import sqlite3 as sql
import datetime as dt 
import mplfinance as mpf
from itertools import chain 
import matplotlib.pyplot as plt 
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
from matplotlib.offsetbox import (AnnotationBbox, DrawingArea, OffsetImage, TextArea)

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))
from bin.price.db_connect import Prices as PriceDB
from bin.price.models.simulations.prices import optimize_and_simulate, simulated_prices

class exp_plots:
    def __init__(self, connections):
        self.priceDB = PriceDB(connections)
        self.stats_db = sql.connect(connections['stats_db'])
        self.daily_db = sql.connect(connections['daily_db'])
        self.stock_dict = json.load(open(connections['ticker_path'], 'r'))
        self.groups = [x for x in list(self.stock_dict.keys()) if x != 'all_stocks']
        
    
    def get_exp_moves(self, stock, last_close = None):
        """ 
        Return a table of the expected moves for the stock and its expiration date. 
        Args:
            stock: str: Stock ticker
            last_close: float: Last closing price of the stock. default is None
        Returns:
            pd.DataFrame: Table of the expected moves for the stock and its upper and lower bounds. 
        
        """
        # Query the expected Moves for the stock 
        emdf = pd.read_sql(f'select * from exp_ext where stock = "{stock}"', self.stats_db, parse_dates = ['expiry'])
        
        # If the last close is not provided, use the last closing price from expected moves table. 
        if last_close is None: 
            last_close = emdf.stk_price.iloc[-1]
        
        # Calculate the upper and lower bounds 
        emdf['upper'] = last_close + emdf['em']
        emdf['lower'] = last_close - emdf['em']
        
        # Return dataframe, sorted by expiration date.
        return emdf.sort_values('expiry')
        
    def get_prices(self, stock):
        """ 
        Return the OHLCV data for the stock 
        args: 
            stock: str: The stock ticker
        returns: 
            pd.DataFrame: The OHLCV data for the stock.
        """
        q = f'''select date(Date) as Date, Open, High, Low, Close, Volume from {stock} order by Date asc'''
        return pd.read_sql(q, self.daily_db, index_col = 'Date', parse_dates = ['Date'])
    
    def exp_move_data(self, stock, n = 30):
        """ 
        Return the expected move data for the next n days for a given stock 
        Args: 
            stock: str: Stock ticker
            n: int: Number of days to look ahead. Default is 30
        Returns:
            tuple: pd.DataFrame, pd.DataFrame: OHLCV data and Expected Move data
        """
        if stock in self.stock_dict['market']:
            days_until_next_month = (pd.Timestamp.today() + pd.DateOffset(days = 14))
        else:
            days_until_next_month = (pd.Timestamp.today() + pd.DateOffset(months = 1))
        
        pdf_price = self.get_prices(stock).tail(n)
        last_close = pdf_price.Close.iloc[-1]

        emdf = self.get_exp_moves(stock, last_close)
        emdf = emdf[(emdf.expiry >= pdf_price.index.max()) & (emdf.expiry <= days_until_next_month)]
        return pdf_price, emdf
    
    def get_sims(self, stock, df, n = 30, method = 'heston_path'):
        """ 
        Get the simulated prices for the stock using the Heston model. 
        Args: 
            stock: str: Stock ticker
            df: pd.DataFrame: Dataframe containing the OHLCV data for the stock
            n: int: Number of days to simulate. Default is 30
            method: str: Method to use for simulation. ('gbm', 'poisson_jump', 'merton_jump', 'heston_path')
        Returns:
            pd.DataFrame: Simulated prices for the stock
        """
        # Get the simulated prices Default method is the Heston model
        paths = simulated_prices(stock, df, method = method, days = n)
        return paths
    

    def exp_move_plot(self, stock, fig, ax, n = 20, skip_exp = 1, method = 'heston_path'):
        """ 
        Plot the expecvted move for the stock 
        
        Args: 
            stock: str: Stock ticker
            fig: plt.figure: Figure object
            ax: plt.axis: Axis object
            n: int: Number of days to look ahead. Default is 20
            skip_exp: int: Skip the 1st expected move. Default is 1
        Returns:
            tuple: plt.figure, plt.axis: Figure and Axis object
        """
        
        pdf_price, emdf = self.exp_move_data(stock, n)

        # if len(emdf) <= 2:
        #     print(stock, emdf)
        #     return fig, ax 
        
        if pd.Timestamp.today() < emdf.expiry.iloc[0]:
            title = f'${stock.upper()} ± {emdf["empct"].iloc[0]:.2%} (\\${emdf["em"].iloc[0]:.2f}) By {emdf["expiry"].iloc[0].strftime("%-m/%-d")}'
        else:
            title = f'${stock.upper()} ± {emdf["empct"].iloc[1]:.2%} (\\${emdf["em"].iloc[1]:.2f}) By {emdf["expiry"].iloc[1].strftime("%-m/%-d")}'
            
        ax.set_title(title, fontsize = 9, fontweight = 'bold')

        # Expected Moves 
        ax.plot(emdf['expiry'], emdf['upper'], '-', color = 'green')
        ax.plot(emdf['expiry'], emdf['lower'], '-', color = 'red')
        ax.scatter(emdf['expiry'].iloc[:-1], emdf['upper'].iloc[:-1], color = 'green', label = 'Upper Bound', alpha = 1, s = 5)
        ax.scatter(emdf['expiry'].iloc[:-1], emdf['lower'].iloc[:-1], color = 'red', label = 'Lower Bound', alpha = 1, s = 5)
        
        # Scatter the expected moves
        ax.scatter(emdf['expiry'].iloc[-1], emdf['upper'].iloc[-1], color = 'green', label = 'Upper Bound', alpha = 1)
        ax.scatter(emdf['expiry'].iloc[-1], emdf['lower'].iloc[-1], color = 'red', label = 'Lower Bound', alpha = 1)
        
        # Fill the area between the upper and lower bounds
        ax.fill_between(emdf['expiry'], emdf['upper'], emdf['lower'], color = 'grey', alpha = 0.4, label = 'Expected Move')

        # put the yaxis on the left 
        ax.yaxis.tick_left()
        ax.yaxis.set_label_position("left")

        # Simulations
        # days = emdf.expiry.max() - pd.Timestamp.today()
        # paths = self.get_sims(stock, pdf_price, n = days.days, method = method)
        # print(f'\n\nLast Date: {pdf_price.index[-1]} - Days to Expiration: {days.days}, Max Date : {paths.index[-1]}, Expiry Max: {emdf.expiry.max()}')
        # ax.plot(paths.iloc[:emdf.shape[0]], color = 'grey', alpha = 0.1, linewidth = 0.5)
                    
        # Candle Chart 
        mpf.plot(
            pdf_price, 
            ax = ax, 
            type = 'candle', 
            volume = False, 
            style = 'charles', 
            show_nontrading = True, 
            # datetime_format = '%-m/%-d',
            )
        
        # Add Gridlines
        ax.grid(True)
    
        return fig, ax
    
    def em_plot(self,stocks = None, n = 30, group = "mag8", figsize = (5.25, 10), skip_exp = 1):
        """
        Plot the expected moves for the stocks in the group. 
            - We are allowed 4 pictures per tweet. So to make them more visable we will plot 4 stocks per figure. 
            - Unless there are less than 4 stocks. In this case we will stick to one per figure.
        
        Args:
            stocks: list: List of stock tickers to plot. Default is None
            n: int: Number of days to look ahead. Default is 30 
            group: str: The group of stocks to plot. Default is "mag8"
            figsize: tuple: The size of the figure. Default is (25, 9)
            skip_exp: int: Skip the 1st expected move. Default is 1
        Returns:
            list: List of figures containing the expected moves for the stocks in the group.
        """
            
        if stocks is None: 
            stocks = self.stock_dict[group]
        

        n_stock = len(stocks)
        if len(stocks) < 4:
            nrows = max(3, np.ceil(n_stock / 2).astype(int))
        else:
            nrows = 4
        ncols = 1
        
        gs = GridSpec(nrows, ncols)
        if len(stocks) <= 4:
            # One plot 
            figsize = figsize
            fig = plt.figure(figsize = figsize, dpi = 200)
            for i, stock in enumerate(stocks[:]):
                ax = fig.add_subplot(gs[i])
                fig, ax = self.exp_move_plot(stock, fig, ax, n, skip_exp=skip_exp)
            
            fig.autofmt_xdate(rotation=0)
            todays_date = dt.datetime.today().strftime('%-m/%-d/%y')
            fig.suptitle(f'Expected Moves as of {todays_date}', fontweight = 'bold')
            fig.tight_layout()
            return [fig]
        else:
            # Create two figures, one for each group of 4 stocks
            fig1 = plt.figure(figsize = (figsize[0], figsize[1]), dpi = 200)
            fig2 = plt.figure(figsize = (figsize[0], figsize[1]), dpi = 200)

            # First 4 stocks
            first_group = stocks[:4]
            gs = GridSpec(4, 1)
            for i, stock in enumerate(first_group):
                ax = fig1.add_subplot(gs[i])
                fig1, ax = self.exp_move_plot(stock, fig1, ax, n, skip_exp=skip_exp)
            fig1.autofmt_xdate(rotation=0)

            
            second_group = stocks[4:8] if n_stock > 4 else stocks[4:]
            gs = GridSpec(len(second_group), 1)
            for i, stock in enumerate(second_group):
                ax = fig2.add_subplot(gs[i])
                fig2, ax = self.exp_move_plot(stock, fig2, ax, n, skip_exp=skip_exp)
            fig2.autofmt_xdate(rotation=0)

            todays_date = dt.datetime.today().strftime('%-m/%-d/%y')
            fig1.suptitle(f'Expected Moves as of {todays_date}', fontweight = 'bold')
            fig2.suptitle(f'Expected Moves as of {todays_date}', fontweight = 'bold')
            fig1.tight_layout()
            fig2.tight_layout()
            return [fig1, fig2]

if __name__ == "__main__":
    import time
    import json 
    from bin.main import get_path
    connections = get_path()
    u = exp_plots(connections)
    figs = u.em_plot(group = 'mag8', figsize = (4.5, 7 ))
    figs[0].savefig('em_with_sim.png', dpi = 200, bbox_inches = 'tight')
