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
from sklearn.naive_bayes import CategoricalNB
from sklearn.base import BaseEstimator, TransformerMixin
from numpy.typing import NDArray

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))


# Type aliases
ArrayLike = Union[np.ndarray, pd.DataFrame, pd.Series]
ModelType = Union[Pipeline, IsolationForest, OneClassSVM, LocalOutlierFactor, KMeans]
PredictionType = Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]

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

class StackedAnomalyModel:
    """Stacked anomaly detection model using scikit-learn pipelines"""
    def __init__(
            self,
            verbose: bool = False,
            **kwargs
        ) -> None:
        """
        Initialize the stacked anomaly detection model
        Args:
            X: Feature DataFrame
            y: Target Series
            kwargs: Additional arguments for model initialization
            feature_names: List of feature column names
            target_names: List of target column names
            stock: Stock symbol
            verbose: Control logging verbosity
        """
        self.verbose = verbose
        self._setup_logging()
        self.base_models: Dict[str, Pipeline] = {}
        self.meta_learner: Optional[CategoricalNB] = None
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
                IsolationForest(contamination= self.contamination, random_state=999)
            ),
            'OneClassSVM': make_pipeline(
                StandardScaler(),
                OneClassSVM(kernel='rbf', nu=0.05)
            ),
            'LOF': make_pipeline(
                StandardScaler(),
                LocalOutlierFactor(n_neighbors=20, contamination= self.contamination, novelty=True)
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

    def _calibrate_probabilities(self, probs: np.ndarray, threshold: float = 0.7) -> np.ndarray:
        """Calibrate probabilities to reduce false positives"""
        return np.where(probs >= threshold, 1, 0)
    
    def _prepare_meta_features(self, X: np.ndarray) -> np.ndarray:
        """Convert base model predictions to categorical features with adaptive thresholds"""
        X_categorical = np.zeros_like(X, dtype=np.int32)
        
        for i in range(X.shape[1]):
            scores = X[:, i]
            
            # Calculate adaptive thresholds based on distribution
            q25, q75 = np.percentile(scores, [25, 75])
            iqr = q75 - q25
            lower_bound = q25 - 1.5 * iqr
            upper_bound = q75 + 1.5 * iqr
            
            # More conservative categorization:
            # 0: definite normal (below lower bound)
            # 1: probably normal (between lower bound and median)
            # 2: probably anomaly (between median and upper bound)
            # 3: definite anomaly (above upper bound)
            median = np.median(scores)
            
            X_categorical[scores <= lower_bound, i] = 0
            X_categorical[(scores > lower_bound) & (scores <= median), i] = 1
            X_categorical[(scores > median) & (scores <= upper_bound), i] = 2
            X_categorical[scores > upper_bound, i] = 3
            
        return X_categorical

    def _fine_tuned_meta_learner(self, X: np.ndarray, y: np.ndarray) -> CategoricalNB:
        """Fine-tune the CategoricalNB meta-learner using GridSearchCV with incremental learning"""
        # Convert features to categorical
        X_cat = self._prepare_meta_features(X)
        
        # Calculate class weights
        unique_classes, class_counts = np.unique(y, return_counts=True)
        class_weights = dict(zip(unique_classes, len(y) / (len(unique_classes) * class_counts)))
        logger.info(f"Class distribution: {dict(zip(unique_classes, class_counts))}")
        logger.info(f"Class weights: {class_weights}")
        
        param_grid = {
            'alpha': [0.01, 0.1, 0.5, 1.0],
            'fit_prior': [True, False],
            'min_categories': [None],  # Let the model determine categories from data
            'force_alpha': [True]  # Ensure smoothing is applied even with min_categories=None
        }
        
        grid_search = GridSearchCV(
            CategoricalNB(),
            param_grid,
            scoring='f1',  # Use F1 score instead of accuracy for imbalanced data
            cv=5,
            n_jobs=-1
        )
        
        # Initial fit to find best parameters
        grid_search.fit(X_cat, y)
        
        # Create a new model with best parameters
        best_params = grid_search.best_params_
        logger.debug(f"Best parameters for CategoricalNB: {best_params}")
        logger.debug(f"Best cross-validation score: {grid_search.best_score_:.3f}")
        
        # Initialize model with best parameters
        best_model = CategoricalNB(**best_params)
        
        # Initialize with partial_fit for incremental learning
        unique_classes = np.unique(y)
        best_model.partial_fit(X_cat, y, classes=unique_classes)
        
        return best_model

    def initialize_training(self,
                            x: pd.DataFrame,
                            y: pd.Series,
                            stock: Optional[str] = None,
                            contamination = 0.05,
                            **kwargs) -> None:
        """
        Initialize training data for the model, This will use all of the x and y data for training purposes 
        Args:
            x: pd.DataFrame: Feature DataFrame
            y: pd.Series: Target Series, Already in the binary format 
            kwargs: Additional arguments for model initialization
        
        """
        if not isinstance(x, pd.DataFrame):
            x = pd.DataFrame(x)
        if not isinstance(y, (pd.Series, pd.DataFrame)):
            y = pd.Series(y)
        # Ensure x and y have the same index
        if not x.index.equals(y.index):
            raise ValueError("x and y must have the same index")
        # Ensure x and y are sorted by index
        x, y = x.sort_index(), y.sort_index()
        # Remove datetime column if it exists
        if 'gatherdate' in x.columns:
            x = x.drop('gatherdate', axis=1)
        if stock is not None: 
            self.stock = stock
        else: 
            raise ValueError("Stock symbol must be provided")
        
        self.contamination = contamination  # Set contamination level for the model

        # Scale
        scaler = kwargs.get('scaler', StandardScaler())
        x = pd.DataFrame(scaler.fit_transform(x), index=x.index, columns=x.columns)
        self.training_dates = x.index
        
        self.xtrain = x.to_numpy()
        self.ytrain = y.to_numpy()
        self.feature_names = list(x.columns)
        self.target_names = list(y.name) if isinstance(y, pd.Series) else list(y.columns)
    
    def train(self,stock:str,x:pd.DataFrame, y:pd.Series, contamination:float = 0.05) -> None: 
        """Train the model using the training data"""
        logger.debug("Training model...")
        self.initialize_training(stock = stock, x = x, y = y, contamination=contamination)
        self.base_models = self._create_base_models()
        base_predictions = []
        for name, model in self.base_models.items():
            logger.debug(f"Training {name}...")
            model.fit(self.xtrain, self.ytrain)
            train_pred = model.predict(self.xtrain)
            self.training_preds[name] = pd.DataFrame(train_pred, index=self.training_dates, columns=[name])
            base_predictions.append(train_pred.reshape(-1, 1))

        # Stack predictions for meta-learner
        X_meta = np.hstack(base_predictions)
        
        # Initialize meta-learner with tuned parameters and initial partial fit
        self.meta_learner = self._fine_tuned_meta_learner(X_meta, self.ytrain.ravel())
        
        # Store classes for future partial_fit calls
        self.classes_ = np.unique(self.ytrain)
        
        logger.info("Model training complete")
        self.save_models(direction=None, directory="bin/models/anom/saved")
        
        # Get ensemble vote
        ensemble_votes = self._ensemble_vote(X_meta)
        logger.info(f"Ensemble vote distribution: {pd.Series(ensemble_votes).value_counts()}")
        
        # Get meta-learner probabilities using prepared features
        X_meta_prepared = self._prepare_meta_features(X_meta)
        probas = self.meta_learner.predict_proba(X_meta_prepared)
        anomaly_probs = probas[:, 1] if probas.shape[1] == 2 else probas.max(axis=1)
        
        # Use multiple probability thresholds
        thresholds = [0.6, 0.7, 0.8]
        threshold_votes = [(anomaly_probs >= t).astype(int) for t in thresholds]
        threshold_vote = (sum(threshold_votes) >= len(thresholds)/2).astype(int)
        
        # Combine ensemble vote and probability threshold vote
        final_predictions = ((ensemble_votes + threshold_vote) >= 1).astype(int)
        
        # Log detailed prediction information
        unique_preds, pred_counts = np.unique(final_predictions, return_counts=True)
        logger.info(f"Training prediction distribution: {dict(zip(unique_preds, pred_counts))}")
        logger.info(f"Mean probability: {anomaly_probs.mean():.3f}")
        logger.info(f"Mean ensemble vote ratio: {ensemble_votes.mean():.3f}")
        
        return pd.DataFrame({
            'anomaly': final_predictions,
            'probability': anomaly_probs,
            'ensemble_vote': ensemble_votes
        }, index=self.training_dates)

    def partial_fit(self, x: pd.DataFrame, y: pd.Series) -> pd.Series:
        """Incrementally train the model with new data"""
        if not isinstance(x, pd.DataFrame):
            x = pd.DataFrame(x)
        if not isinstance(y, (pd.Series, pd.DataFrame)):
            y = pd.Series(y)

        # Process new data through base models
        base_predictions = []
        for name, model in self.base_models.items():
            pred = model.predict(x)
            base_predictions.append(pred.reshape(-1, 1))
            
        X_meta = np.hstack(base_predictions)
        
        # Prepare features for incremental learning
        X_meta_prepared = self._prepare_meta_features(X_meta)
        
        # Incrementally update meta-learner
        self.meta_learner.partial_fit(X_meta_prepared, y.ravel(), classes=self.classes_)
        
        # Get ensemble vote
        ensemble_votes = self._ensemble_vote(X_meta)
        
        # Get meta-learner probabilities
        probas = self.meta_learner.predict_proba(X_meta_prepared)
        anomaly_probs = probas[:, 1] if probas.shape[1] == 2 else probas.max(axis=1)
        
        # Use multiple probability thresholds
        thresholds = [0.6, 0.7, 0.8]
        threshold_votes = [(anomaly_probs >= t).astype(int) for t in thresholds]
        threshold_vote = (sum(threshold_votes) >= len(thresholds)/2).astype(int)
        
        # Combine ensemble vote and probability threshold vote
        final_predictions = ((ensemble_votes + threshold_vote) >= 1).astype(int)
        
        # Log detailed prediction information
        unique_preds, pred_counts = np.unique(final_predictions, return_counts=True)
        logger.info(f"Partial fit prediction distribution: {dict(zip(unique_preds, pred_counts))}")
        logger.info(f"Partial fit mean probability: {anomaly_probs.mean():.3f}")
        logger.info(f"Partial fit mean ensemble vote ratio: {ensemble_votes.mean():.3f}")
        
        return pd.DataFrame({
            'anomaly': final_predictions,
            'probability': anomaly_probs,
            'ensemble_vote': ensemble_votes
        }, index=x.index)

    def initialize_testing(self, x: pd.DataFrame, stock:str = None, contamination: float = 0.05, **kwargs) -> None:
        """
        Initialize testing data for the model
        Args:
            x: pd.DataFrame: Feature DataFrame for testing
            kwargs: Additional arguments for model initialization
        """
        if not isinstance(x, pd.DataFrame):
            x = pd.DataFrame(x)
        # Ensure x is sorted by index
        x = x.sort_index()
        # Remove datetime column if it exists
        if 'gatherdate' in x.columns:
            x = x.drop('gatherdate', axis=1)
        if stock is not None:
            self.stock = stock
        
        self.contamination = contamination  # Set contamination level for the model

        scaler = kwargs.get('scaler', StandardScaler())
        x = pd.DataFrame(scaler.transform(x), index=x.index, columns=x.columns)

        self.testing_dates = x.index
        self.xtest = x.to_numpy()
        self.feature_names = list(x.columns)

    def _ensemble_vote(self, base_preds: np.ndarray, threshold: float = 0.6) -> np.ndarray:
        """Get ensemble vote from base models"""
        # Convert -1/1 to 0/1 for voting
        votes = (base_preds == -1).astype(int)
        # Calculate proportion of models voting for anomaly
        vote_ratio = votes.mean(axis=1)
        return (vote_ratio >= threshold).astype(int)

    def predict(self, X: ArrayLike) -> pd.DataFrame:
        """Generate predictions using the stacked model with probability calibration and ensemble voting"""
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            
        # Get base model predictions
        base_predictions = []
        for name, model in self.base_models.items():
            pred = model.predict(X)
            base_predictions.append(pred.reshape(-1, 1))
            
        X_meta = np.hstack(base_predictions)
        
        # Get ensemble vote
        ensemble_votes = self._ensemble_vote(X_meta)
        
        # Get meta-learner probabilities
        X_meta_prepared = self._prepare_meta_features(X_meta)
        probas = self.meta_learner.predict_proba(X_meta_prepared)
        anomaly_probs = probas[:, 1] if probas.shape[1] == 2 else probas.max(axis=1)
        
        # Use multiple probability thresholds
        thresholds = [0.6, 0.7, 0.8]
        threshold_votes = [(anomaly_probs >= t).astype(int) for t in thresholds]
        threshold_vote = (sum(threshold_votes) >= len(thresholds)/2).astype(int)
        
        # Combine ensemble vote and probability threshold vote
        final_predictions = ((ensemble_votes + threshold_vote) >= 1).astype(int)
        
        # Log detailed prediction information
        unique_preds, pred_counts = np.unique(final_predictions, return_counts=True)
        logger.info(f"Final prediction distribution: {dict(zip(unique_preds, pred_counts))}")
        logger.info(f"Mean probability: {anomaly_probs.mean():.3f}")
        logger.info(f"Mean ensemble vote ratio: {ensemble_votes.mean():.3f}")
        
        return pd.DataFrame({
            'anomaly': final_predictions,
            'probability': anomaly_probs,
            'ensemble_vote': ensemble_votes
        }, index=self.testing_dates)
        
    def save_models(self, direction: str = None, directory: str = "bin/models/anom/saved") -> None:
        """Save all fitted models to disk"""
        if self.stock is None:
            raise ValueError("Stock symbol not set. Initialize with stock parameter.")
            
        os.makedirs(directory, exist_ok=True)
        if direction is not None:
            filepath = os.path.join(directory, f"stacked_anomaly_model_{direction}_{self.stock}.pkl")
        else:
            filepath = os.path.join(directory, f"stacked_anomaly_model_{self.stock}.pkl")
        
        # Get the scaler from the first base model's pipeline
        scaler = self.base_models[list(self.base_models.keys())[0]].named_steps['standardscaler']
        
        joblib.dump({
            'scaler': scaler,
            'base_models': self.base_models,
            'meta_learner': self.meta_learner,
            'feature_names': self.feature_names,
            'target_names': self.target_names,
            'stock': self.stock
        }, filepath)
        
        logger.info(f"Saved models to {filepath}")

    def load_and_predict(self, new_data: pd.DataFrame, stock: str = None, direction:str = None,  directory: str = "bin/models/anom/saved") -> pd.DataFrame:
        """Load saved models and predict on new data"""
        if direction is not None:
            filepath = os.path.join(directory, f"stacked_anomaly_model_{direction}_{self.stock if stock is None else stock}.pkl")
        else:
            filepath = os.path.join(directory, f"stacked_anomaly_model_{self.stock if stock is None else stock}.pkl")
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No model found at {filepath}")
        
        saved_data = joblib.load(filepath)
        self.base_models = saved_data['base_models']
        self.meta_learner = saved_data['meta_learner']
        self.feature_names = saved_data['feature_names']
        self.target_names = saved_data['target_names']
        self.stock = saved_data['stock']
        scaler = saved_data['scaler']
        
        # Initialize testing data
        new_data = new_data[self.feature_names]        
        self.initialize_testing(new_data, scaler=scaler)
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
    

if __name__ == "__main__":
    from pathlib import Path
    import sys
    sys.path.append(str(Path(__file__).resolve().parents[3]))
    from bin.models.option_stats_model_setup import data
    from bin.main import get_path
    
    # Set up logging
    logger.setLevel(DEBUG)
    
    # # Load data for training
    d = data(connections = get_path())
    x, y = d._returnxy('spy', end_date = "2024-10-01")
    y = pd.Series(np.where(np.abs(y) > 0.01, 1, 0), index=y.index, name='target')  # Convert target to binary
    print(y.value_counts())
    print("\nInitializing and fitting model...")
    model = StackedAnomalyModel()
    preds = model.train(x=x, y=y, contamination=0.05, stock='spy')
    print(preds.value_counts())
    
    # # Make predictions
    d = data(connections = get_path())
    x, y = d._returnxy('spy', start_date = "2024-10-02")
    y = pd.Series(np.where(np.abs(y) > 0.01, 1, 0), index=y.index, name='target')  # Convert target to binary
    print(y.value_counts())
    model = StackedAnomalyModel()
    preds = model.load_and_predict(x, stock='spy')
    print(preds.value_counts())