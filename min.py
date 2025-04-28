
import pandas as pd 
import numpy as np 

import os 

import sys
sys.path.append('/Users/jerald/Documents/Dir/Python/Stocks')

from bin.price.db_connect import Prices
from bin.options.manage_all import Manager as Optionsdb
from bin.earnings.get_earnings import Earnings 


def get_path(pre=''):
	""" Must be in the '/Users/jerald/Documents/Dir/Python/Stocks' directory to run this function. """
	connections = {
				##### Bonds Data ###########################
				'bonds_db': f'{pre}data/bonds/bonds.db', 
				##### Price Data ###########################
				'daily_db': f'{pre}data/prices/stocks.db', 
				'intraday_db': f'{pre}data/prices/stocks_intraday.db',
				'ticker_path': f'{pre}data/stocks/tickers.json',
				##### Options Data ###########################
				'inactive_db': f'{pre}data/options/log/inactive.db',
				'backup_db': f'{pre}data/options/log/backup.db',
				'tracking_values_db': f'{pre}data/options/tracking_values.db',
				'tracking_db': f'{pre}data/options/tracking.db',
				'stats_db': f'{pre}data/options/stats.db',
				'vol_db': f'{pre}data/options/vol.db',
				'vol2_db':f'{pre}data/options/vol2.db',
				'change_db': f'{pre}data/options/option_change.db', 
				'option_db': f'{pre}data/options/options.db', 
				'options_stat': f'{pre}data/options/options_stat.db',
				'dates_db': f'{pre}data/options/dates.db',
				##### Earnings + Company Info ###########################
				'earnings_dict': f'{pre}data/earnings/earnings.pkl',
				'stock_names' : f'{pre}data/stocks/stock_names.db',
                'stock_info_dict': f'{pre}data/stocks/stock_info.json',
                'earnings_calendar': f'{pre}data/earnings/earnings_dates_alpha.csv',
				#### Pictures Path ######
				'pics': f'{pre}alerts/pics/'

		}
	return connections



def check_path(connections):
	""" 
	Check if the files exist 
	Args: 
		connections: dict: Dictionary of file paths. 
	Returns:
		bool: True if all the files exist, False otherwise
	"""
	checks = []
	for key, value in connections.items():
		if not os.path.exists(value):
			raise ValueError(f'{value} does not exist')
		checks.append(True)
	return all(checks)

	 
class Manager:
	def __init__(self, connections=None):
		"""
        Initialize the Manager Class which is a wrapper for the Options database, Price Database, Earnings Database, Scanner, and dxp signal processing unit. 
        This class allows the user to access:
            
            Options Data (Optionsdb):
                - options_db: Options Database containing the raw option chain data
                - stats_db: Database which contains high volume contracts that have been gathered from all stocks 
                - vol_db: Database which contains daily features obtained from the raw option chain data 
                - change_db: Database which contains a simplified version of the raw option chain data, and includes change variables. 
                
            Price Data (Pricedb):
                - daily_db: Database containing the daily stock prices
                - intraday_db: Database containing the intraday stock prices
            
        Args:
            connections: dict: Dictionary
		"""
		
		#  If type is string, or is None
		if type(connections) == str:
			connections = get_path(connections)
		if connections == None:
			connections = get_path()
		if type(connections) != dict:
			raise ValueError('Connections must be a dictionary')
		# Check if the files exist
		if check_path(connections) == False:
			raise ValueError('Files do not exist')

		# Initialize the Connections: 
		self.Optionsdb = Optionsdb(connections) 
		self.Pricedb = Prices(connections)
		self.Earningsdb = Earnings(connections)
		
		# Save the Connection Dict.
		self.connection_paths = connections
		  
	def close_connection(self):
		self.Optionsdb.close_connections()
		self.Pricedb.close_connections()

	def run(self, stock):
		pass

	
if __name__ == "__main__":
	pre = ''
	connections = {
				##### Price Report ###########################
				'daily_db': f'{pre}data/prices/stocks.db', 
				'intraday_db': f'{pre}data/prices/stocks_intraday.db',
				'ticker_path': f'{pre}data/stocks/tickers.json',
				##### Price Report ###########################
				'inactive_db': f'{pre}data/options/log/inactive.db',
				'backup_db': f'{pre}data/options/log/backup.db',
				'tracking_values_db': f'{pre}data/options/tracking_values.db',
				'tracking_db': f'{pre}data/options/tracking.db',
				'stats_db': f'{pre}data/options/stats.db',
				'vol_db': f'{pre}data/options/vol.db',
				'change_db': f'{pre}data/options/option_change.db', 
				'option_db': f'{pre}data/options/options.db', 
				'options_stat': f'{pre}data/options/options_stat.db',
				'stock_names' : f'{pre}data/stocks/stocks.db'

		}
	connections = get_path()
	m = Manager(connections)
	print(m.Optionsdb._em_ext('gme'))
 
 