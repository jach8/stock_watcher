import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[3]))
from bin.main import get_path
from bin.models.option_stats_model_setup import data as options_data
from bin.price.db_connect import Prices as price_data
from bin.models.anom.model import StackedAnomalyModel, PCAKMeansDetector, ICAKMeansDetector, BaseAnomalyDetector
from bin.models.anom.icr import IncrementalAnomalyModel
from statsmodels.stats.outliers_influence import variance_inflation_factor


from scipy.stats import pearsonr
import numpy as np 
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class AnomalyScanner:
    def __init__(self, connections):
        if connections is None:
            raise ValueError("Connections parameter cannot be None")
        self.data_connector = options_data(connections=connections)

    def _select_features(self, X, y, direction="both"):
        """Select features using mutual information and VIF for redundancy."""
        
        # Compute mutual information (y is categorical: 0 or 1)
        mi_scores = mutual_info_classif(X, y, random_state=999)
        feature_scores = pd.DataFrame({'mi': mi_scores}, index=X.columns)
        
        # Select top features based on MI (threshold > 0.02 or top 10)
        mi_threshold = 0.02
        selected = feature_scores[feature_scores['mi'] > mi_threshold]
        if len(selected) < 5:  # Ensure at least 5 features
            selected = feature_scores.nlargest(10, 'mi')  # Top 10 as starting point
        
        selected_features = selected.index.tolist()
        
        # Compute VIF and remove highly collinear features
        if selected_features:
            X_selected = X[selected_features]
            vif_data = pd.DataFrame()
            vif_data['feature'] = X_selected.columns
            vif_data['VIF'] = [variance_inflation_factor(X_selected.values, i) 
                            for i in range(X_selected.shape[1])]
            
            # Iteratively remove features with VIF > 10
            while vif_data['VIF'].max() > 10:
                max_vif_feature = vif_data.loc[vif_data['VIF'].idxmax(), 'feature']
                selected_features.remove(max_vif_feature)
                X_selected = X[selected_features]
                vif_data = pd.DataFrame()
                vif_data['feature'] = X_selected.columns
                vif_data['VIF'] = [variance_inflation_factor(X_selected.values, i) 
                                for i in range(X_selected.shape[1])]
            
            logging.debug(f"VIF values for {direction}:\n{vif_data}")
        
        if not selected_features:
            logging.warning(f"No features selected for {direction}. Using all features.")
            selected_features = X.columns.tolist()
        
        logging.debug(f"Feature scores for {direction}:\n{feature_scores}")
        logging.debug(f"Selected features for {direction}: {selected_features}")
        return X[selected_features]

    def _stock_data(self, stock, start_date=None, end_date=None, anomaly_threshold=0.01, direction="both"):
        if not isinstance(stock, str):
            raise TypeError("Stock must be a string")
        try:
            x, y_raw = self.data_connector._returnxy(stock, start_date=start_date, end_date=end_date)
            if x.empty or y_raw.empty:
                raise ValueError(f"No data returned for stock {stock} between {start_date} and {end_date}")
            
            if direction not in ["both", "bullish", "bearish"]:
                raise ValueError("Direction must be 'both', 'bullish', or 'bearish'")
            
            if direction == "both":
                y = pd.Series(np.where(np.abs(y_raw) > anomaly_threshold, 1, 0), index=y_raw.index, name='target')
            elif direction == "bullish":
                y = pd.Series(np.where(y_raw > anomaly_threshold, 1, 0), index=y_raw.index, name='target')
            elif direction == "bearish":
                y = pd.Series(np.where(y_raw < -anomaly_threshold, 1, 0), index=y_raw.index, name='target')
            
            x = x.loc[y.index]
            if x.empty or y.empty:
                raise ValueError(f"No data after filtering for anomalies/features for stock {stock}")
            if x.shape[0] != y.shape[0]:
                raise ValueError(f"Mismatch in rows between x and y for {stock}: {x.shape[0]} vs {y.shape[0]}")
            
            logging.debug(f"Loaded data for {stock} ({direction}): X {x.shape}, Y {y.shape}, Features: {x.columns.tolist()}")
            return x, y
        except Exception as e:
            logging.error(f"Failed to load stock data for {stock}: {str(e)}")
            raise

    def log_counts(self, preds, y, phase='Training'):
        predicted_count = preds.value_counts()
        actual_count = y.value_counts()
        logging.info(f"{phase} Anomaly Counts - Predicted {phase}: {predicted_count.get(1, 0)} - Actual {phase}: {actual_count.get(1, 0)}")

    def incremental_model(self, stock, contamination=0.1, direction="both", **kwargs):
        x, y = self._stock_data(stock, direction=direction, **kwargs)
        # Select features based on correlation and mutual information
        x = self._select_features(x, y,  direction=direction)
        model = IncrementalAnomalyModel(stock=stock, contamination=contamination, direction=direction)
        model.fit(x, y)
        preds = model.predict(x)
        self.log_counts(preds, y, phase='Training')
        self.save_model(model)
        return model
    
    def stacked_model(self, stock, contamination=0.1, direction="both", save_path = 'bin/models/anom/saved', **kwargs):
        x, y = self._stock_data(stock, direction=direction, **kwargs)
        model = StackedAnomalyModel()
        training_preds = model.train(x = x, y= y, stock = stock, contamination=contamination)
        self.log_counts(training_preds, y, phase='Stacked Model Training')
        model.save_models(direction = direction, directory = save_path)
        logging.debug(f"Stacked model saved for {stock} ({direction}) to {save_path}")
        return model

    def save_model(self, model, directory="bin/models/anom/saved"):
        try:
            model.save_model(directory)
            logging.debug(f"Model saved for {model.stock} ({model.direction}) to {directory}")
        except Exception as e:
            logging.error(f"Failed to save model for {model.stock}: {str(e)}")
            raise

    def load_model(self, stock, direction, directory="bin/models/anom/saved"):
        try:
            model = IncrementalAnomalyModel(stock=stock, direction=direction)
            model.load_model(directory)
            logging.debug(f"Model loaded for {stock} ({direction}) from {directory}")
            return model
        except FileNotFoundError as e:
            logging.error(f"Failed to load model for {stock} ({direction}): {str(e)}")
            raise

    def __track_n_day_returns(self, stock, preds, X, n_days=5):

        if preds is None or X is None or preds.size == 0 or X.empty:
            logging.warning(f"No predictions or data available for {stock}. Cannot track returns.")
            return []
        
        # If no anomalies are detected, return an empty list
        if not np.any(preds == 1):
            logging.info(f"No anomalies detected for {stock}. No returns to track.")
            return []

        try:
            # signal_dates = X.index[preds == 1]
            signal_dates = X.index[preds.squeeze() == 1]
            price_data = self.data_connector.price_data(stock)
            returns = []
            for date in signal_dates:
                if date in price_data.index:
                    idx = price_data.index.get_loc(date)
                    if idx + n_days < len(price_data):
                        start_price = price_data.iloc[idx]['close']
                        end_price = price_data.iloc[idx + n_days]['close']
                        ret = (end_price / start_price - 1) * 100
                        returns.append(ret)
            return returns
        except Exception as e:
            pass

    def run(self, stock, contamination=0.1, start_date=None, end_date=None, anomaly_threshold=0.01, direction="both", n_days=5):
        logging.info(f"${stock.upper()} contamination: {contamination:.2f}, checking for {direction} anomalies")

        # Fit and Save the model 
        self.incremental_model(
            stock, 
            contamination, 
            direction=direction, 
            start_date=start_date, 
            end_date=end_date, 
            anomaly_threshold=anomaly_threshold
        )

        # Stacked Model 
        stacked_anomaly_model = self.stacked_model(
            stock, 
            contamination,
            direction=direction, 
            start_date=start_date, 
            end_date=end_date, 
            anomaly_threshold=anomaly_threshold
        )


        # Load the Saved Model 
        model = self.load_model(stock, direction)
        xtest, ytest = self._stock_data(stock, start_date="2025-01-01", anomaly_threshold=anomaly_threshold, direction=direction)
        logging.debug(f'X_test: {xtest.shape}, Y_test: {ytest.shape}')
        
        incremental_predictions = model.predict(xtest)
        stacked_predictions = stacked_anomaly_model.load_and_predict(xtest, direction = direction, directory = 'bin/models/anom/saved')

        self.log_counts(incremental_predictions, ytest, phase='ICR MODEL Testing')
        self.log_counts(stacked_predictions, ytest, phase='STACKED MODEL Testing')
        
        returns = self.__track_n_day_returns(stock, incremental_predictions, xtest, n_days=n_days)
        if returns:
            avg_ret = np.mean(returns)
            max_ret = np.max(returns)
            min_ret = np.min(returns)
            std_ret = np.std(returns)
            logging.info(f"Average {n_days}D returns for detected anomalies: {avg_ret:.2f}%, max: {max_ret:.2f}%, min: {min_ret:.2f}%, std: {std_ret:.2f}%")
        
        stacked_returns = self.__track_n_day_returns(stock, stacked_predictions, xtest, n_days=n_days)
        if stacked_returns:
            avg_ret = np.mean(stacked_returns)
            max_ret = np.max(stacked_returns)
            min_ret = np.min(stacked_returns)
            std_ret = np.std(stacked_returns)
            logging.info(f"Average {n_days}D returns for stacked model anomalies: {avg_ret:.2f}%, max: {max_ret:.2f}%, min: {min_ret:.2f}%, std: {std_ret:.2f}%")

        return model
    
if __name__ == "__main__":
    connections = get_path()

    scanner = AnomalyScanner(connections)
    stocks = ['spy', 'mo', 'iwm']
    directions = ['bullish', 'bearish', 'both']
    for stock in stocks:
        for direction in directions:
            for contamination in [0.05, 0.10]:
                scanner.run(stock, contamination=contamination, end_date="2025-01-01", direction=direction, n_days=5)
            print('\n')
        print('\n\n')
