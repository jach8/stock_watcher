"""
Data Preparation for Models: 
This module is used for preprocessing the data for the models. 
"""

import sys
import numpy as np 
import pandas as pd 
import datetime as dt 
from tqdm import tqdm 
import scipy.stats as stats 
from logging import getLogger, DEBUG, INFO, Formatter, StreamHandler
from typing import Union, Optional, Dict, List
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, KBinsDiscretizer

# Configure module logger
logger = getLogger(__name__)
logger.setLevel(DEBUG)
if not logger.handlers:
    handler = StreamHandler()
    handler.setLevel(DEBUG)
    formatter = Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class data:
    def __init__(self, 
                 df: pd.DataFrame, 
                 feature_names: List = None, 
                 target_names: List = None,
                 stock: str = None) -> None:
        """
        Initialize the data class
        Args:
            df: pd.DataFrame: The DataFrame to use for the data
            feature_names: List: The names of the features
            target_names: List: The names of the targets
            stock: str: The stock symbol
        """
        self.stock = stock
        self.df = df
        self.feature_names = feature_names if feature_names is not None else []
        self.target_names = target_names if target_names is not None else []
        self.features = self.df[self.feature_names] if self.feature_names else pd.DataFrame()
        self.target = self.df[self.target_names] if self.target_names else pd.DataFrame()
        
    def binary_convert(self, x, thresh=0.003, buy=1, sell=0):
        """
        Convert the target to binary. 
        Args:
            x: The target to convert. 
        Returns:
            The converted target. 
        """
        return np.where(x > thresh, buy, sell)
    
    def multi_convert(self, x, thresh=0.003, buy=1, sell=2, hold=0):
        """
        Convert the target to multi-class. 
        Args:
            x: The target to convert. 
        Returns:
            The converted target. 
        """
        return np.where(x > thresh, buy, np.where(x < -thresh, sell, hold))

    def anomaly_convert(self, x, thresh=0.02, anom = -1, norm = 1):
            """
            Convert the target to anomaly detection. 
            Args:
                x: The target to convert. 
            Returns:
                The converted target. 
            """
            return np.where(np.abs(x) > thresh, anom, norm)
    
    def numeric_df(self, df):
        """
        Convert the DataFrame to numeric. 
        Args:
            df: The DataFrame to convert. 
        Returns:
            The DataFrame converted to numeric. 
        """
        return df.apply(pd.to_numeric, errors='coerce')
    
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
    
    def temporal_split(
            self, 
            x: pd.DataFrame,
            y: pd.DataFrame,
            t: int = 100,
            start_date: str = None,
            end_date: str = None
    ) -> Union[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Temporally Split the data into training and testing sets using the last t days for testing. 
        Args:
            x: pd.DataFrame: The features
            y: pd.DataFrame: The target
            t: int: The number of days to predict
            start_date: str: The start date
            end_date: str: The end date
        Returns:
            Numpy Array: xtrain, ytrain, xtest, ytest
        """
        self.training_dates = x.iloc[:-t].index
        self.testing_dates = x.iloc[-t:].index
        self.xtrain = x.iloc[:-t, :].to_numpy()
        self.xtest = x.iloc[-t:, :].to_numpy()
        self.ytrain = y.iloc[:-t].to_numpy()
        self.ytest = y.iloc[-t:].to_numpy()
        return self.xtrain, self.ytrain, self.xtest, self.ytest
    
    def time_split(
            self, 
            x: pd.DataFrame, 
            y: pd.DataFrame, 
            n_splits: int = 5
    ) -> TimeSeriesSplit:
        """
        Split the data using TimeSeriesSplit
        Args:
            x: pd.DataFrame: The features
            y: pd.DataFrame: The target
            n_splits: int: The number of splits
        Returns:
            The TimeSeriesSplit object
        """
        return TimeSeriesSplit(n_splits=n_splits).split(x, y)



class setup(data):
    def __init__(self, df, feature_names, target_names, stock):
        super().__init__(df, feature_names, target_names, stock)
        self.price_data = df  # Initialize price_data
        self.verbose = True  # Enable verbose for debugging
        
    def initialize(self, indicator_type=None, **kwargs):
        """
        Initialize the setup with data preprocessing steps
        Args:
            indicator_type: Optional regex pattern to filter technical indicators (e.g., 'EMA', 'BB', 'ADX')
            **kwargs: Additional keyword arguments
                scaler: sklearn scaler object (default: MinMaxScaler)
                discretize: bool (default: False)
                y_format: str (default: 'cont')
                test_size: float (default: 0.2)
                nbins: int (default: 5)
                strategy: str (default: 'kmeans')
                encode: str (default: 'ordinal')
        Returns:
            self: The setup instance
        """
        # logger.debug(f"Features before processing: {self.features.shape}")
        # logger.debug(f"Feature names: {self.feature_names}")
        # logger.debug(f"Target names: {self.target_names}")
        
        # Validate inputs
        if not isinstance(self.features, pd.DataFrame) or self.features.empty:
            raise ValueError("Features DataFrame is empty or not properly initialized")
            
        # if not self.target_names:raise ValueError("Target names must be specified")
            
        self.scaler = kwargs.get('scaler', StandardScaler())
        self.discretizer = kwargs.get('discritize', False)
        self.y_format = kwargs.get('y_format', 'anomaly_convert')  # Default to anomaly detection
        self.test_size = kwargs.get('test_size', 0.1)
        
        # Scale features first
        self.features_scaled = pd.DataFrame(
            self.scaler.fit_transform(self.features), 
            columns=self.features.columns,
            index=self.features.index
        )
        
        # Apply indicator type filtering if specified
        if indicator_type is not None:
            filtered_features = self.features_scaled.filter(regex=indicator_type)
            if filtered_features.empty:
                logger.warning(f"No features match the pattern: {indicator_type}")
                logger.warning("Available indicators: ema, sma, kama, and wma")
            else:
                self.features_scaled = filtered_features
            
        # logger.debug(f"Features after processing: {self.features_scaled.shape}")
        # logger.debug(f"Selected features: {list(self.features_scaled.columns)}")
            
        if self.discretizer:
            self.nbins = kwargs.get('nbins', 5)
            self.strategy = kwargs.get('strategy', 'kmeans')
            self.encode = kwargs.get('encode', 'ordinal')
            self.discretizer = KBinsDiscretizer(
                n_bins=self.nbins, 
                encode=self.encode,
                strategy=self.strategy
            )
            
        self.xtrain, self.xtest = train_test_split(
            self.features_scaled, 
            test_size=self.test_size,
            shuffle=False
        )
        
        # Handle different y_format cases
        target_data = self.df[self.target_names]
        func = getattr(self, self.y_format)
        anoms = target_data.apply(func)

        counts = np.unique(anoms, return_counts=True)
        logger.debug(f"Target Initialized with {len(counts[0])} classes. The counts are: {dict(zip(*counts))}")
        # logger.debug(f"Target shape: {anoms.shape}")
            
        self.ytrain = anoms.loc[self.xtrain.index]
        self.ytest = anoms.loc[self.xtest.index]
        self.xtrain = self.features_scaled.loc[self.xtrain.index]  # Use scaled features
        self.xtest = self.features_scaled.loc[self.xtest.index]    # Use scaled features
    
        return self




if __name__ == "__main__":
    ############################################################################################################
    from pathlib import Path 
    import sys
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from main import Manager, get_path
    
    ############################################################################################################
    # Get data
    get_path = get_path()
    m = Manager(get_path)
    
    # Initialize the Anomaly Model
    sys.path.append(str(Path(__file__).resolve().parent))
    from anom.model import StackedAnomalyModel
    from anom.view import EnhancedViewer as viewer

    # stock = np.random.choice(m.Optionsdb.all_stocks)
    stock = 'spy'
    d = m.Pricedb.model_preparation(stock, start_date = "2000-01-01", end_date = "2025-03-13")
    model = viewer(df=d['df'],feature_names=d['features'],target_names=d['target'],stock=d['stock'],verbose=True)
    model.initialize(y_format = 'anomaly_convert')  # Initialize with EMA features only
    model.fit()
    model.save_models()
    # model.plot_all()

    d = m.Pricedb.model_test_data(stock, start_date = dt.datetime(2025,3,13))
    model = viewer(df=d['df'].tail(1),feature_names=d['features'],target_names=None,stock=d['stock'],verbose=True)
    model.load_and_predict(new_data = d['df'].tail(1),stock = d['stock'])