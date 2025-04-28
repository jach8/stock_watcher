""""
___ Option Statistics as Features: ______
	This module is used for gathering the data from the vol_db for each stock:
	VOL_DB: Contias daily Option Statitstics for each stock in the database.
	The Vol db will be used for setting up our features for the model.
	--->
	The features consist of:
		call_vol : --> int64 : Call Volume for the day.
		put_vol : --> int64 : Put Volume for the day.
		total_vol : --> int64 : Total Volume for the day.
		call_oi : --> int64 : Call Open Interest for the day.
		put_oi : --> int64 : Put Open Interest for the day.
		total_oi : --> int64 : Total Open Interest for the day.
		call_prem : --> float64 : Call Premium for the day.
		put_prem : --> float64 : Put Premium for the day.
		total_prem : --> float64 : Total Premium for the day.
		call_vol_pct : --> float64 : Call Volume Percentage for the day.
		put_vol_pct : --> float64 : Put Volume Percentage
		call_oi_pct : --> float64 : Call OI percentage
		put_oi_pct : --> float64 : Put OI percentage
		call_vol_chng : --> int64 :  Call Volume Change
		put_vol_chng : --> int64 :  Put Volume Change
		total_vol_chng : --> int64 :  Total Volume Change
		call_oi_chng : --> int64 :  Call OI Change
		put_oi_chng : --> int64 :  Put OI Change
		total_oi_chng : --> int64 :  Total OI Change
		call_prem_chng : --> int64 :  Call Premium Change ($Notional Value Open)
		put_prem_chng : --> int64 :  Put Premium Change ($Notional Value Open)
		total_prem_chng : --> int64 :  Total Premium Change ($Notional Value Open)
		call_vol_pct_chng : --> float64 :  Call Volume Percentage Change
		put_vol_pct_chng : --> float64 :  Put Volume Percentage Change
		call_oi_pct_chng : --> float64 :  Call OI Percentage Change
		put_oi_pct_chng : --> float64 :  Put OI Percentage Change
		net_call_oi_change5d : --> float64 :  Net Call OI Change 5 Day
		net_put_oi_change5d : --> float64 :      Net Put OI Change 5 Day
"""
import sys
sys.path.append('/Users/jerald/Documents/Dir/Python/Stocks')
import sqlite3 as sql
import pandas as pd
import numpy as np
import json 

class data:
	def __init__(self, connections):
		# Connect to vol_db: Contains the daily option statistics for each stock.
		self.vol_db = connections['vol_db']
		# Connect to the daily_db: Contains the daily price data for each stock.
		self.daily_db = connections['daily_db']
		self.stocks = json.load(open(connections['ticker_path'], 'r'))

	@staticmethod
	def connect_to_db(db, query):
		""" Connect to the database """
		with sql.connect(db) as conn:
			out = pd.read_sql(query, conn)
		return out
        
    
	def get_stats(self, stock):
		"""
		Get the daily option statistics for the stock.
		Args:
			stock: The stock to get the statistics for.
		Returns:
			A DataFrame with the daily option statistics for the stock.
		"""
		# query = f'select * from {stock} where dte_flag = "total"'
		# out = self.connect_to_db(self.vol_db, query)
		# out = out.drop(columns = ['dte_flag'])
		query = f'select * from {stock}'
		out = self.connect_to_db(self.vol_db, query)
		return out

	def numeric_df(self, df):
		"""
		Convert the DataFrame to numeric.
		Args:
			df: The DataFrame to convert.
		Returns:
			The DataFrame converted to numeric.
		"""
		return df.apply(pd.to_numeric, errors = 'coerce')

	def drop_columns_that_contain(self, df, string):
		"""
		Drop columns that contain a string.
		Args:
			df: The DataFrame to drop the columns from.
			string: The string to search for in the columns.
		Returns:
			The DataFrame with the columns dropped.
		"""
		return df[df.columns.drop(list(df.filter(regex=string)))]
	
	def calculate_ivr(self, df, col):
		"""
		Calculate Implied Volatility Rank (IVR).
		Args:
			df (pd.DataFrame): DataFrame containing historical data.
			col (str): Column name for IV data.	
		Returns:
			float: IVR between 0 and 100.
		"""
		iv = df[col]
		iv_52w_high = df[col].rolling(window=252, min_periods=1).max()
		iv_52w_low = df[col].rolling(window=252,  min_periods=1).min()
		ivr = (iv - iv_52w_low) / (iv_52w_high - iv_52w_low) * 100
		return ivr

	def daily_opt_stat(self, stock):
		"""
		Get the daily option statistics for the stock.
		Args:
			stock: The stock to get the statistics for.
		Returns:
			A DataFrame with the daily option statistics for the stock.
		"""
		# current_features = self.Optionsdb.option_custom_q(f'''select min(datetime(gatherdate)) as gd, * from {stock} group by date(gatherdate)''', 'vol_db')
		current_features = self.get_stats(stock)
		current_features['pcr'] = current_features['put_vol'] / current_features['call_vol']
		current_features['pcr_rank'] = self.calculate_ivr(current_features, 'pcr')
		current_features['call_ivr'] = self.calculate_ivr(current_features, 'call_iv')
		current_features['put_ivr'] = self.calculate_ivr(current_features, 'put_iv')
		current_features['atm_ivr'] = self.calculate_ivr(current_features, 'atm_iv')
		current_features['otm_ivr'] = self.calculate_ivr(current_features, 'otm_iv')
		f1 = current_features.copy()
		f1.gatherdate = pd.to_datetime(f1.gatherdate)
		f1.insert(0, 'date', f1.gatherdate.dt.date)
		# f1.drop(columns = ['gatherdate'], inplace = True)
		# f1 = self.drop_columns_that_contain(f1, 'pct|gd|total')
		f1.date = pd.to_datetime(f1.date)
		self.features = list(f1.columns[1:])
		# Now we need to change the 0s in columns with "oi_chng" to NaN
		oi_chng_cols = list(f1.filter(regex='_chng').columns)
		
		for col in oi_chng_cols:
			f1[col] = f1[col].replace(0, np.nan)
			f1[col] = f1[col].fillna(method='ffill')
			
		return f1

	def add_features_from_another_stock(self, stocks):
		"""
		Add the features from another stock to the stock.
		Args:
			stock2: The stock to add the features from.
		Returns:
			A DataFrame with the features from the other stock added.
		"""
		stock1 = stocks[0]
		stock2 = stocks[1]
		f1 = self.daily_opt_stat(stock1)
		f1['stock'] = stock1
		f2 = self.daily_opt_stat(stock2)
		f2['stock'] = stock2
		out = pd.concat([f1, f2], axis = 0).drop(columns = ['gatherdate']).dropna()
		out = out.melt(id_vars = ['date', 'stock'], var_name = 'feature', value_name = 'value')
		return out.sort_values(['date'])

	def daily_price_data(self, stock):
		"""
		Get the daily price data for the stock.
		Args:
			stock: The stock to get the price data for.
		Returns:
			A DataFrame with the daily price data for the stock.
		"""
		# target_data = self.Pricedb.custom_q(f'select date(date) as date, close from {stock}')
		# target_data = pd.read_sql(f'select date(date) as date, close, volume as "stock_volume" from {stock}', self.daily_db, parse_dates = 'date')
		query = 'select date(date) as date, close, volume as "stock_volume" from ' + stock
		target_data = self.connect_to_db(self.daily_db, query)
		target_data.date = pd.to_datetime(target_data.date)
		target_data['returns'] = target_data.Close.pct_change()
		target_data['target'] = target_data.Close.pct_change().shift(-1)
		return target_data

	def mdf_keep_close(self, stock):
		"""
		Get the merged data for the stock.
		Args:
			stock: The stock to get the merged data for.
		Returns:
			A DataFrame with the merged data for the stock.
		"""
		features = self.daily_opt_stat(stock)
		target = self.daily_price_data(stock)
		mdf = features.merge(target, on = 'date', how = 'inner')
		self.binary_target = np.where(mdf.returns > 0, 1, 0)
		self.multi_target = np.where(mdf.returns > 0.003, 1, np.where(mdf.returns < -0.003, 2, 0))
		mdf.date = pd.to_datetime(mdf.date)
		# mdf = mdf.drop_duplicates()
		return mdf.set_index('date').sort_index()

	def merge_data(self, feature_data, target_data):
		"""
		Merge the price data with the option data.
		Args:
			stock: The stock to merge the data for.
			price_data: The price data for the stock.
			opt_data: The option data for the stock.
		Returns:
			A DataFrame with the merged data.
		"""
		mdf = feature_data.merge(target_data, on = 'date')
		self.binary_target = np.where(mdf.returns > 0, 1, 0)
		self.multi_target = np.where(mdf.returns > 0.003, 1, np.where(mdf.returns < -0.003, 2, 0))
		mdf = mdf.drop(columns = ['Close', 'returns'])
		mdf.date = pd.to_datetime(mdf.date)
		mdf = mdf.set_index('date').sort_index()
		mdf = mdf[~mdf.index.duplicated(keep='last')]
		return mdf 

	def mdf(self, stock):
		"""
		Get the merged data for the stock.
		Args:
			stock: The stock to get the merged data for.
		Returns:
			A DataFrame with the merged data for the stock.
		"""
		features = self.daily_opt_stat(stock)
		target = self.daily_price_data(stock)
		return self.merge_data(features, target)
	
	

	def _returnxy(self, stock, start_date = None, end_date = None):
		"""_summary_: Return X Y in the form of a tupple, 
		X is the features obtained from daily Option statistics
		y is the target, which is the shifted returns for the stock. 
		Args:
			stock (_type_): Stock that we are looking at

		Returns:
			_type_: tuple -> x,y
		"""
		# mdf = self.mdf(stock).reset_index(drop = True).set_index('gatherdate')
		mdf = self.mdf(stock).drop(columns = 'gatherdate')
		mdf.index = pd.to_datetime(mdf.index)

		# The last rows, we dont have a target for. But we need it so that we can generate a prediction for the next day 

		held_out = mdf[mdf.target.isna()]
		if not held_out.empty:
			mdf = mdf[~mdf.index.isin(held_out.index)]
		
		# Now we need to drop the columns that contain 'pct' and 'chng' as they are not needed for the model
		mdf = self.drop_columns_that_contain(mdf, 'pct')

		if start_date is not None: 
			mdf = mdf[mdf.index >= start_date]
		if end_date is not None:
			mdf = mdf[mdf.index <= end_date]
		
		x = mdf.drop(columns = ['target'])
		y = mdf['target']
		self.feature_names = list(x.columns)
		return (x, y)
	
	def _get_test_data(self, stock):
		# mdf = self.mdf(stock).reset_index(drop = True).set_index('gatherdate')
		mdf = self.mdf(stock).drop(columns = 'gatherdate')
		mdf.index = pd.to_datetime(mdf.index)

		# The last rows, we dont have a target for. But we need it so that we can generate a prediction for the next day 

		held_out = mdf[mdf.target.isna()]
		mdf = mdf[mdf.index.isin(held_out.index)]
		
		# Now we need to drop the columns that contain 'pct' and 'chng' as they are not needed for the model
		# mdf = self.drop_columns_that_contain(mdf, 'pct')

		if 'target' in mdf.columns:
			mdf = mdf.drop(columns = ['target'])
		
		return mdf 
	
	def price_data(self, stock):
		"""
		Get the price data for the stock.
		Args:
			stock: The stock to get the price data for.
		Returns:
			A DataFrame with the price data for the stock.
		"""
		conn = sql.connect(self.daily_db)
		out = pd.read_sql(f'select date(date) as date, open, high, low, close, volume from {stock}', conn, parse_dates = 'date').set_index('date')
		out.columns = [x.lower() for x in out.columns]
		return out
	
	def _mdf(self, stock):
		features = self.daily_opt_stat(stock)
		target = self.price_data(stock)
		mdf = features.merge(target, on = 'date')
		return mdf



if __name__ == "__main__":
	from pathlib import Path 
	import sys
	sys.path.append(str(Path(__file__).resolve().parents[2]))
	from bin.main import get_path
	d = data(connections = get_path())
	print(d._mdf('spy'))