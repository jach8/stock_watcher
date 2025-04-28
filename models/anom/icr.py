import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.linear_model import SGDClassifier
from sklearn.base import BaseEstimator
from pyod.models.knn import KNN
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
from scipy.stats import pearsonr
import joblib
import os
from pathlib import Path
import sys

from logging import getLogger, Logger, DEBUG, INFO, Formatter, StreamHandler
# Configure module logger
logger = getLogger(__name__)
logger.propagate = False
if not logger.handlers:
    handler = StreamHandler()
    handler.setLevel(DEBUG)
    formatter = Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class IncrementalAnomalyModel(BaseEstimator):
    def __init__(self, stock: str, contamination=0.1, direction="both"):
        self.stock = stock
        if not 0 < contamination <= 1:
            raise ValueError(f"Contamination must be between 0 and 1, got {contamination}")
        self.contamination = contamination
        self.direction = direction
        self.scaler = RobustScaler()
        self.base_model = None
        self.meta_learner = SGDClassifier(loss='hinge', random_state=999, warm_start=True, class_weight='balanced')

    def _create_base_model(self):
        return KNN(contamination=self.contamination, n_neighbors=20)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        if not isinstance(X, pd.DataFrame) or not isinstance(y, pd.Series):
            raise TypeError("X must be a DataFrame and y must be a Series")
        if X.empty or y.empty:
            raise ValueError("Input X or y is empty")
        self.feature_names = list(X.columns)
        
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=999, stratify=y)
        X_scaled = self.scaler.fit_transform(X_train)
        self.base_model = self._create_base_model()
        self.base_model.fit(X_scaled)
        
        base_preds = self.base_model.decision_scores_.reshape(-1, 1)
        anom_score = base_preds[y_train == 1].mean()
        norm_score = base_preds[y_train == 0].mean()
        invert_scores = anom_score < norm_score
        if invert_scores:
            logger.warning("Anomalies have lower scores than normal data. Adjusting threshold.")
        
        base_preds_normalized = (base_preds - base_preds.min()) / (base_preds.max() - base_preds.min())
        self.meta_learner.fit(base_preds_normalized, y_train)
        
        X_val_scaled = self.scaler.transform(X_val)
        val_preds = self.base_model.decision_function(X_val_scaled).reshape(-1, 1)
        val_preds_normalized = (val_preds - val_preds.min()) / (val_preds.max() - val_preds.min())
        val_scores = self.meta_learner.decision_function(val_preds_normalized)
        self.threshold = np.percentile(val_scores, 100 * (self.contamination if invert_scores else (1 - self.contamination)))
        logger.debug(f"Validation threshold: {self.threshold}")

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a DataFrame")
        if not all(col in X.columns for col in self.feature_names):raise ValueError("Input X is missing some features present during training")
        X_scaled = self.scaler.transform(X[self.feature_names])
        base_preds = self.base_model.decision_function(X_scaled).reshape(-1, 1)
        base_preds_normalized = (base_preds - base_preds.min()) / (base_preds.max() - base_preds.min())
        scores = self.meta_learner.decision_function(base_preds_normalized)
        return pd.Series(np.where(scores > self.threshold, 1, 0), index=X.index, name='anomaly')

    def save_model(self, directory: str = "bin/anom/models/saved"):
        filepath = os.path.join(directory, f"anomaly_model_{self.stock}_{self.direction}.pkl")
        os.makedirs(directory, exist_ok=True)
        joblib.dump({
            'feature_names': self.feature_names,
            'scaler': self.scaler,
            'base_model': self.base_model,
            'meta_learner': self.meta_learner,
            'threshold': self.threshold,
            'stock': self.stock,
            'direction': self.direction
        }, filepath)
        logger.info(f"Model saved to {filepath}")

    def load_model(self, directory: str = "bin/anom/models/saved"):
        filepath = os.path.join(directory, f"anomaly_model_{self.stock}_{self.direction}.pkl")
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No model found at {filepath}")
        data = joblib.load(filepath)
        self.feature_names = data['feature_names']
        self.scaler = data['scaler']
        self.base_model = data['base_model']
        self.meta_learner = data['meta_learner']
        self.threshold = data['threshold']
        self.stock = data['stock']
        self.direction = data['direction']

