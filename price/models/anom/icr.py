import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.base import BaseEstimator
from pyod.models.knn import KNN
import joblib

class IncrementalAnomalyModel(BaseEstimator):
    def __init__(self, stock: str, contamination=0.05, max_feature_sets: int = 10):
        self.stock = stock
        self.contamination = contamination
        self.max_feature_sets = max_feature_sets  # Max number of feature sets to anticipate
        self.feature_sets = {}  # {set_name: [feature_names]}
        self.models = {}       # {set_name: {model_name: model}}
        self.scalers = {}      # {set_name: scaler}
        self.meta_learner = SGDClassifier(loss='hinge', random_state=999, warm_start=True)
        self.n_feature_sets_trained = 0  # Track number of feature sets added

    def _create_base_model(self):
        return KNN(contamination=self.contamination, n_neighbors=20)

    def fit(self, X: pd.DataFrame, y: pd.Series, feature_set_name: str = 'original'):
        """Initial training on a feature set"""
        feature_names = list(X.columns)
        self.feature_sets[feature_set_name] = feature_names
        self.scalers[feature_set_name] = StandardScaler()
        
        # Scale and train base model
        X_scaled = self.scalers[feature_set_name].fit_transform(X)
        base_model = self._create_base_model()
        base_model.fit(X_scaled)
        self.models[feature_set_name] = {'KNN': base_model}
        
        # Prepare meta-learner input with padding for future feature sets
        base_preds = base_model.decision_scores_.reshape(-1, 1)
        padded_preds = np.pad(base_preds, ((0, 0), (0, self.max_feature_sets - 1)), mode='constant', constant_values=0)
        self.meta_learner.fit(padded_preds, y)
        self.n_feature_sets_trained = 1

    def update_with_new_features(self, X: pd.DataFrame, y: pd.Series, feature_set_name: str = 'new'):
        """Add new features and update model"""
        if feature_set_name in self.feature_sets:
            raise ValueError(f"Feature set '{feature_set_name}' already exists")
        
        feature_names = list(X.columns)
        self.feature_sets[feature_set_name] = feature_names
        self.scalers[feature_set_name] = StandardScaler()
        
        # Scale and train new base model
        X_scaled = self.scalers[feature_set_name].fit_transform(X)
        base_model = self._create_base_model()
        base_model.fit(X_scaled)
        self.models[feature_set_name] = {'KNN': base_model}
        
        # Update meta-learner with all predictions
        all_preds = self._get_all_predictions(self._prepare_full_X(X))
        padded_preds = np.pad(all_preds, ((0, 0), (0, self.max_feature_sets - all_preds.shape[1])), mode='constant', constant_values=0)
        self.meta_learner.partial_fit(padded_preds, y)
        self.n_feature_sets_trained += 1

    def _prepare_full_X(self, X: pd.DataFrame) -> pd.DataFrame:
        """Combine all feature sets into a single DataFrame"""
        full_X = pd.DataFrame(index=X.index)
        for set_name, features in self.feature_sets.items():
            if set(features).issubset(X.columns):
                full_X = full_X.join(X[features])
        return full_X

    def _get_all_predictions(self, X: pd.DataFrame) -> np.ndarray:
        """Get predictions from all base models"""
        base_preds = []
        for set_name, features in self.feature_sets.items():
            if set(features).issubset(X.columns):
                X_subset = X[features]
                X_scaled = self.scalers[set_name].transform(X_subset)
                model = self.models[set_name]['KNN']
                preds = model.decision_function(X_scaled).reshape(-1, 1)  # Decision scores
                base_preds.append(preds)
        return np.hstack(base_preds) if base_preds else np.zeros((X.shape[0], 1))

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Predict anomalies"""
        all_preds = self._get_all_predictions(X)
        padded_preds = np.pad(all_preds, ((0, 0), (0, self.max_feature_sets - all_preds.shape[1])), mode='constant', constant_values=0)
        return pd.Series(self.meta_learner.predict(padded_preds), index=X.index, name='anomaly')

    def save_model(self, directory: str = "models"):
        filepath = os.path.join(directory, f"anomaly_model_{self.stock}.pkl")
        os.makedirs(directory, exist_ok=True)
        joblib.dump({
            'feature_sets': self.feature_sets,
            'models': self.models,
            'scalers': self.scalers,
            'meta_learner': self.meta_learner,
            'n_feature_sets_trained': self.n_feature_sets_trained,
            'stock': self.stock
        }, filepath)

    def load_model(self, directory: str = "models"):
        filepath = os.path.join(directory, f"anomaly_model_{self.stock}.pkl")
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No model found at {filepath}")
        data = joblib.load(filepath)
        self.feature_sets = data['feature_sets']
        self.models = data['models']
        self.scalers = data['scalers']
        self.meta_learner = data['meta_learner']
        self.n_feature_sets_trained = data['n_feature_sets_trained']
        self.stock = data['stock']

# Example Usage
if __name__ == "__main__":
    # Initial training
    df = pd.DataFrame({
        'Close': np.random.randn(100),
        'EMA_10': np.random.randn(100),
        'RSI': np.random.randn(100)
    })
    y = pd.Series(np.where(np.random.rand(100) > 0.95, -1, 1))  # Simulated anomalies
    model = IncrementalAnomalyModel(stock='spy', max_feature_sets=10)
    model.fit(df, y, feature_set_name='original')

    # Add new features
    df_new = df.copy()
    df_new['MACD'] = np.random.randn(100)
    df_new['Volume'] = np.random.randn(100)
    model.update_with_new_features(df_new[['MACD', 'Volume']], y, feature_set_name='new')

    # Predict
    preds = model.predict(df_new)
    print(preds.value_counts())
# Example Usage
if __name__ == "__main__":
    ############################################################################################################
    from pathlib import Path 
    import sys
    sys.path.append(str(Path(__file__).resolve().parents[3]))
    from main import Manager, get_path
    
    ############################################################################################################
    # Get data
    get_path = get_path()
    m = Manager(get_path)
    
    # Initialize the Anomaly Model
    sys.path.append(str(Path(__file__).resolve().parent))

    # stock = np.random.choice(m.Optionsdb.all_stocks)
    stock = 'spy'
    d = m.Pricedb.model_preparation(stock, start_date = "2000-01-01", end_date = "2025-03-13")
    m.close_connection()
    # keys:  df, features names, target names, and stock name 
    df = d['df']
    features = d['features']
    target = d['target']
    df[target] = np.where(np.abs(df[target]) > 0.02, 1, 0)  # Convert target to binary for anomaly detection
    # stock = d['stock']

    print(df[features])
    print(df[target])

    ic = IncrementalAnomalyModel(stock=stock, contamination=0.05, max_feature_sets=10)
    
    # Fit the model on the initial feature set
    ic.fit(df[features], df[target], feature_set_name='original')

    # get the predictions
    preds = ic.predict(df[features])
    df['anomaly'] = preds
    print(df[['anomaly']].head())

