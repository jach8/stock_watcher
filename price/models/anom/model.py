"""
Stacked anomaly detection model using scikit-learn pipelines and neural network meta-learner
"""
from __future__ import annotations
import os
import logging
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from logging import getLogger, Logger, DEBUG, INFO, Formatter, StreamHandler
from typing import Union, Optional, Dict, List, Tuple, Any
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA, FastICA
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from numpy.typing import NDArray

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from connect import setup

# Type aliases
ArrayLike = Union[np.ndarray, pd.DataFrame, pd.Series]
ModelType = Union[Pipeline, IsolationForest, OneClassSVM, LocalOutlierFactor, KMeans]
PredictionType = Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]

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

class BaseAnomalyDetector(BaseEstimator, TransformerMixin):
    """Base class for anomaly detectors to ensure consistent interface"""
    def __init__(self, contamination=0.05):
        self.contamination = contamination
        
    def fit(self, X, y=None):
        return self
        
    def predict(self, X):
        return np.ones(X.shape[0])

class PCAKMeansDetector(BaseAnomalyDetector):
    """PCA with K-means anomaly detection"""
    def __init__(self, n_components=2, n_clusters=3, contamination=0.05):
        super().__init__(contamination)
        self.n_components = n_components
        self.n_clusters = n_clusters
        self.pca = PCA(n_components=n_components)
        self.kmeans = KMeans(n_clusters=n_clusters, n_init=10)
        
    def fit(self, X, y=None):
        transformed = self.pca.fit_transform(X)
        self.kmeans.fit(transformed)
        self.distances_ = self._compute_distances(transformed)
        self.threshold_ = np.percentile(self.distances_, 100 * (1 - self.contamination))
        return self
    
    def _compute_distances(self, X):
        labels = self.kmeans.predict(X)
        distances = np.array([
            np.linalg.norm(X[i] - self.kmeans.cluster_centers_[labels[i]])
            for i in range(X.shape[0])
        ])
        return distances
    
    def predict(self, X):
        transformed = self.pca.transform(X)
        distances = self._compute_distances(transformed)
        return np.where(distances >= self.threshold_, -1, 1)

class ICAKMeansDetector(BaseAnomalyDetector):
    """ICA with K-means anomaly detection"""
    def __init__(self, n_components=2, n_clusters=3, contamination=0.05):
        super().__init__(contamination)
        self.n_components = n_components
        self.n_clusters = n_clusters
        self.ica = FastICA(n_components=n_components)
        self.kmeans = KMeans(n_clusters=n_clusters, n_init=10)
        
    def fit(self, X, y=None):
        transformed = self.ica.fit_transform(X)
        self.kmeans.fit(transformed)
        self.distances_ = self._compute_distances(transformed)
        self.threshold_ = np.percentile(self.distances_, 100 * (1 - self.contamination))
        return self
    
    def _compute_distances(self, X):
        labels = self.kmeans.predict(X)
        distances = np.array([
            np.linalg.norm(X[i] - self.kmeans.cluster_centers_[labels[i]])
            for i in range(X.shape[0])
        ])
        return distances
    
    def predict(self, X):
        transformed = self.ica.transform(X)
        distances = self._compute_distances(transformed)
        return np.where(distances >= self.threshold_, -1, 1)

class StackedAnomalyModel(setup):
    """Stacked anomaly detection model using scikit-learn pipelines"""
    
    def __init__(
            self,
            df: pd.DataFrame,
            feature_names: List[str],
            target_names: List[str],
            stock: str,
            verbose: bool = False,
            **kwargs
        ) -> None:
        """
        Initialize the stacked anomaly detection model
        Args:
            df: DataFrame containing the data
            feature_names: List of feature column names
            target_names: List of target column names
            stock: Stock symbol
            verbose: Control logging verbosity
        """
        super().__init__(df, feature_names, target_names, stock)
        self.verbose = verbose
        self._setup_logging()
        
        self.base_models: Dict[str, Pipeline] = {}
        self.meta_learner: Optional[MLPClassifier] = None
        self.model_training: Dict[str, NDArray[np.float64]] = {}
        self.training_preds: Dict[str, pd.DataFrame] = {}
        self.test_preds: Dict[str, pd.DataFrame] = {}

    def _setup_logging(self) -> None:
        """Configure logging based on verbosity"""
        log_level = DEBUG if self.verbose else INFO
        logger.setLevel(log_level)

    def _create_base_models(self) -> Dict[str, Pipeline]:
        """Create base model pipelines"""
        models = {
            'IsolationForest': make_pipeline(
                StandardScaler(),
                IsolationForest(contamination=0.05, random_state=999)
            ),
            'OneClassSVM': make_pipeline(
                StandardScaler(),
                OneClassSVM(kernel='rbf', nu=0.05)
            ),
            'LOF': make_pipeline(
                StandardScaler(),
                LocalOutlierFactor(n_neighbors=20, contamination=0.05, novelty=True)
            ),
            'PCA_KMeans': make_pipeline(
                StandardScaler(),
                PCAKMeansDetector(n_components=2, n_clusters=3)
            ),
            'ICA_KMeans': make_pipeline(
                StandardScaler(),
                ICAKMeansDetector(n_components=2, n_clusters=3)
            )
        }
        return models

    def fit(self, **kwargs) -> None:
        """Fit base models and meta-learner"""
        logger.debug("Initializing model instance")
        self.initialize(**kwargs)
        
        # Initialize and fit base models
        self.base_models = self._create_base_models()
        base_predictions = []
        
        for name, model in self.base_models.items():
            logger.debug(f"Fitting {name}...")
            model.fit(self.xtrain)
            train_pred = model.predict(self.xtrain)
            test_pred = model.predict(self.xtest)
            
            self.training_preds[name], self.test_preds[name] = \
                self.merge_preds(train_pred, test_pred, model_name=name)
            
            base_predictions.append(train_pred.reshape(-1, 1))
            
        # Stack predictions for meta-learner
        X_meta = np.hstack(base_predictions)
        
        # Initialize and fit meta-learner
        self.meta_learner = MLPClassifier(
            hidden_layer_sizes=(10, 5),
            activation='relu',
            solver='adam',
            random_state=999
        )
        self.meta_learner.fit(X_meta, self.ytrain.values.ravel())
        
        logger.info("Model fitting complete")

    def predict(self, X: ArrayLike) -> pd.DataFrame:
        """Generate predictions using the stacked model"""
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            
        base_predictions = []
        for name, model in self.base_models.items():
            pred = model.predict(X)
            base_predictions.append(pred.reshape(-1, 1))
            
        X_meta = np.hstack(base_predictions)
        final_predictions = self.meta_learner.predict(X_meta)
        
        return pd.DataFrame(final_predictions, index=X.index, columns=['anomaly'])
    
    def merge_preds(self, trainpred, testpred, model_name='anomaly'):
        """Merge predictions with price data"""
        if not isinstance(trainpred, (pd.Series, pd.DataFrame)):
            trainpred = pd.DataFrame(trainpred, index=self.ytrain.index, columns=[model_name])
            testpred = pd.DataFrame(testpred, index=self.ytest.index, columns=[model_name])
            
        trainpred = self.price_data.loc[self.xtrain.index].join(trainpred)
        testpred = self.price_data.loc[self.xtest.index].join(testpred)
        return trainpred, testpred
    
    def pca_pred(self, pred):
        """Handle dimensionality reduction predictions"""
        if not isinstance(pred, (pd.Series, pd.DataFrame)):
            pred = pd.DataFrame(pred, index=self.features_scaled.index)
            
        trainpred = self.price_data.loc[self.xtrain.index].join(pred.loc[self.xtrain.index])
        testpred = self.price_data.loc[self.xtest.index].join(pred.loc[self.xtest.index])
        return trainpred, testpred
        
    def save_models(self, directory: str = "bin/price/models/anom/saved") -> None:
        """Save all fitted models to disk"""
        os.makedirs(directory, exist_ok=True)
        filepath = os.path.join(directory, f"anomaly_model_{self.stock}.pkl")
        
        joblib.dump({
            'scaler': self.scaler,
            'base_models': self.base_models,
            'meta_learner': self.meta_learner,
            'feature_names': self.feature_names,
            'target_names': self.target_names,
            'stock': self.stock
        }, filepath)
        
        logger.info(f"Saved models to {filepath}")

    def load_and_predict(self, new_data: pd.DataFrame, stock: str = None, directory: str = "bin/price/models/anom/saved") -> pd.DataFrame:
        """Load saved models and predict on new data"""
        filepath = os.path.join(directory, f"anomaly_model_{self.stock if stock is None else stock}.pkl")
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No model found at {filepath}")
        
        saved_data = joblib.load(filepath)
        self.base_models = saved_data['base_models']
        self.meta_learner = saved_data['meta_learner']
        self.feature_names = saved_data['feature_names']
        self.target_names = saved_data['target_names']
        self.stock = saved_data['stock']
        scaler = saved_data['scaler']
                
        # Validate New Data
        if new_data.empty:
            raise ValueError("No valid features found in new data")
        if new_data.shape[1] != len(self.feature_names):
            print(self.feature_names)
            print()
            print(list(new_data.columns))
            raise ValueError(f"New data must have {len(self.feature_names)} features, found {new_data.shape[1]}")
        
        if new_data.isnull().values.any():
            raise ValueError("New data contains NaN values, please clean the data before prediction")
        if new_data.shape[0] == 0:
            raise ValueError("New data is empty, please provide valid data for prediction")
        
        # Scale new data
        new_data[self.feature_names] = scaler.transform(new_data[self.feature_names])
        new_data = new_data[self.feature_names]
        
        # Predict
        return self.predict(new_data[self.feature_names])