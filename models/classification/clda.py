import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegressionCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import logging
import uuid
from typing import Tuple, Dict, List, Union
import warnings

class ClassificationModel:
    """A generalized classification model for binary classification tasks.

    This class handles data preprocessing, model training, evaluation, and prediction on unseen data
    with proper error handling and configurable logging.
    """
    
    def __init__(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray], 
                 categorical_cols: List[str] = None, numerical_cols: List[str] = None,
                 test_size: float = 0.3, random_state: int = 0, verbose: int = 2):
        """Initialize the classification model.

        Args:
            X (Union[pd.DataFrame, np.ndarray]): Feature matrix
            y (Union[pd.Series, np.ndarray]): Target vector
            categorical_cols (List[str], optional): List of categorical column names
            numerical_cols (List[str], optional): List of numerical column names
            test_size (float): Proportion of data to use for testing (default: 0.3)
            random_state (int): Random seed for reproducibility (default: 0)
            verbose (int): Logging verbosity level (0=off, 1=errors, 2=info, 3=debug)
        """
        # Configure logging based on verbose level
        self.logger = logging.getLogger(__name__)
        self.logger.handlers = []  # Clear any existing handlers
        
        if verbose == 0:
            self.logger.addHandler(logging.NullHandler())
            self.logger.setLevel(logging.NOTSET)
        else:
            log_levels = {1: logging.ERROR, 2: logging.INFO, 3: logging.DEBUG}
            level = log_levels.get(verbose, logging.INFO)
            self.logger.setLevel(level)
            
            file_handler = logging.FileHandler('classification_model.log')
            file_handler.setLevel(level)
            stream_handler = logging.StreamHandler()
            stream_handler.setLevel(level)
            
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            stream_handler.setFormatter(formatter)
            
            self.logger.addHandler(file_handler)
            self.logger.addHandler(stream_handler)
        
        self.logger.debug("Initializing ClassificationModel with verbose level %d", verbose)
        
        # Convert X to DataFrame with appropriate column names
        if isinstance(X, pd.DataFrame):
            self.X = X
        else:
            columns = [f'feature_{i}' for i in range(X.shape[1])]
            self.X = pd.DataFrame(X, columns=columns)
        
        self.y = y if isinstance(y, pd.Series) else pd.Series(y, name='target')
        
        # Infer numerical and categorical columns if not provided
        if categorical_cols is None and numerical_cols is None:
            self.numerical_cols = self.X.select_dtypes(include=['float64', 'int64']).columns.tolist()
            self.categorical_cols = self.X.select_dtypes(include=['object', 'category']).columns.tolist()
            self.logger.info("Inferred %d numerical and %d categorical columns", 
                            len(self.numerical_cols), len(self.categorical_cols))
        else:
            self.categorical_cols = categorical_cols if categorical_cols else []
            self.numerical_cols = numerical_cols if numerical_cols else [
                col for col in self.X.columns if col not in self.categorical_cols
            ]
        
        # Validate column names
        invalid_numerical = [col for col in self.numerical_cols if col not in self.X.columns]
        invalid_categorical = [col for col in self.categorical_cols if col not in self.X.columns]
        if invalid_numerical or invalid_categorical:
            raise ValueError(f"Invalid columns: numerical {invalid_numerical}, categorical {invalid_categorical}")
        
        # Validate numerical columns are numeric
        non_numeric = [col for col in self.numerical_cols 
                      if col in self.X.columns and not pd.api.types.is_numeric_dtype(self.X[col])]
        if non_numeric:
            self.logger.warning("Non-numeric columns in numerical_cols: %s. Treating as categorical.", non_numeric)
            self.categorical_cols.extend(non_numeric)
            self.numerical_cols = [col for col in self.numerical_cols if col not in non_numeric]
        
        self.test_size = test_size
        self.random_state = random_state
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_train_pca = None
        self.X_test_pca = None
        self.y_train_pca = None
        self.y_test_pca = None
        self.scaler = StandardScaler()
        self.pca = PCA()
        self.label_encoders = {}  # Store LabelEncoder instances for categorical columns
        self.selected_cols = None  # Store selected columns for preprocessing
        self.results = pd.DataFrame(
            columns=['Accuracy', 'Precision', 'Sensitivity', 'Specificity'],
            index=['LDA', 'QDA', 'Naive Bayes', 'KNN', 'PCA KNN', 
                   'Logistic Regression', 'Neural Network']
        )
        self.models = {}
        self.logger.info("ClassificationModel initialized with X shape: %s", str(self.X.shape))

    def preprocess_data(self) -> None:
        """Preprocess the input dataset.

        Handles categorical variable encoding and feature scaling.
        """
        try:
            self.logger.info("Starting data preprocessing")
            X_processed = self.X.copy()
            
            # Encode categorical variables
            for col in self.categorical_cols:
                if col in X_processed.columns:
                    self.logger.debug("Encoding categorical column: %s", col)
                    try:
                        le = LabelEncoder()
                        X_processed[col] = le.fit_transform(X_processed[col].astype(str))
                        self.label_encoders[col] = le  # Store encoder
                    except Exception as e:
                        self.logger.warning("Failed to encode column %s: %s. Skipping.", col, str(e))
                        X_processed = X_processed.drop(columns=[col])
                        self.categorical_cols.remove(col)
                else:
                    self.logger.warning("Categorical column %s not found in data", col)
            
            # Ensure only specified columns are used
            self.selected_cols = list(set(self.numerical_cols + self.categorical_cols))
            if not self.selected_cols:
                raise ValueError("No valid columns selected for preprocessing")
            self.logger.debug("Selected columns for preprocessing: %s", self.selected_cols)
            X_processed = X_processed[self.selected_cols]
            
            # Handle missing values
            for col in self.numerical_cols:
                if col in X_processed.columns:
                    self.logger.debug("Imputing missing values for numerical column: %s", col)
                    X_processed[col] = X_processed[col].fillna(X_processed[col].mean())
            for col in self.categorical_cols:
                if col in X_processed.columns:
                    self.logger.debug("Imputing missing values for categorical column: %s", col)
                    X_processed[col] = X_processed[col].fillna(X_processed[col].mode()[0])
            
            # Scale features
            self.logger.debug("Scaling features")
            X_scaled = self.scaler.fit_transform(X_processed)
            
            # Split data
            self.logger.debug("Splitting data into train/test sets")
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X_scaled, self.y, test_size=self.test_size, random_state=self.random_state
            )
            
            # Prepare PCA data
            self._prepare_pca_data(X_scaled)
            
            self.logger.info("Data preprocessing completed. Training set shape: %s", 
                            str(self.X_train.shape))
            
        except Exception as e:
            self.logger.error("Error in data preprocessing: %s", str(e))
            raise

    def _prepare_pca_data(self, X_scaled: np.ndarray) -> None:
        """Prepare PCA-transformed data.

        Args:
            X_scaled (np.ndarray): Scaled feature matrix
        """
        try:
            self.logger.debug("Starting PCA transformation")
            self.pca.fit(X_scaled)
            n_components = min(3, np.where(np.cumsum(self.pca.explained_variance_ratio_) >= 0.8)[0][0] + 1)
            pca_data = self.pca.transform(X_scaled)[:, :n_components]
            pca_data = pd.DataFrame(pca_data, columns=[f'PC{i+1}' for i in range(n_components)])
            pca_data = pd.concat([pca_data, self.y.reset_index(drop=True)], axis=1)
            
            X_pca = pca_data.drop([self.y.name], axis=1)
            y_pca = pca_data[self.y.name]
            
            self.X_train_pca, self.X_test_pca, self.y_train_pca, self.y_test_pca = train_test_split(
                X_pca, y_pca, test_size=self.test_size, random_state=self.random_state
            )
            
            self.logger.info("PCA data preparation completed with %d components", n_components)
            
        except Exception as e:
            self.logger.error("Error in PCA data preparation: %s", str(e))
            raise

    def predict_new_data(self, X_new: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        """Predict on new, unseen data using all trained models.

        Args:
            X_new (Union[pd.DataFrame, np.ndarray]): New feature matrix

        Returns:
            pd.DataFrame: Predictions from each model, with columns named by model
        """
        try:
            self.logger.info("Predicting on new data with shape: %s", str(X_new.shape))
            
            if not self.models:
                raise ValueError("No models have been trained. Call train_models() first.")
            
            # Convert X_new to DataFrame
            if isinstance(X_new, pd.DataFrame):
                X_new_processed = X_new.copy()
            else:
                if X_new.shape[1] != len(self.selected_cols):
                    raise ValueError(f"X_new has {X_new.shape[1]} features, expected {len(self.selected_cols)}")
                X_new_processed = pd.DataFrame(X_new, columns=self.selected_cols)
            
            # Validate columns
            missing_cols = [col for col in self.selected_cols if col not in X_new_processed.columns]
            if missing_cols:
                raise ValueError(f"Missing columns in X_new: {missing_cols}")
            X_new_processed = X_new_processed[self.selected_cols]
            
            # Encode categorical variables
            for col in self.categorical_cols:
                if col in X_new_processed.columns:
                    self.logger.debug("Encoding new data categorical column: %s", col)
                    if col in self.label_encoders:
                        try:
                            # Transform using stored encoder, map unseen values to a default
                            X_new_processed[col] = X_new_processed[col].astype(str).map(
                                lambda x: self.label_encoders[col].transform([x])[0]
                                if x in self.label_encoders[col].classes_ else -1
                            )
                        except Exception as e:
                            self.logger.warning("Failed to encode new data column %s: %s. Filling with -1.", col, str(e))
                            X_new_processed[col] = -1
                    else:
                        self.logger.warning("No encoder found for column %s. Filling with -1.", col)
                        X_new_processed[col] = -1
            
            # Handle missing values
            for col in self.numerical_cols:
                if col in X_new_processed.columns:
                    self.logger.debug("Imputing missing values for new data numerical column: %s", col)
                    X_new_processed[col] = X_new_processed[col].fillna(X_new_processed[col].mean())
            for col in self.categorical_cols:
                if col in X_new_processed.columns:
                    self.logger.debug("Imputing missing values for new data categorical column: %s", col)
                    X_new_processed[col] = X_new_processed[col].fillna(X_new_processed[col].mode()[0])
            
            # Scale features
            self.logger.debug("Scaling new data features")
            X_new_scaled = self.scaler.transform(X_new_processed)
            
            # Prepare PCA data for PCA KNN
            self.logger.debug("Applying PCA transformation to new data")
            X_new_pca = self.pca.transform(X_new_scaled)[:, :self.X_train_pca.shape[1]]
            
            # Generate predictions
            predictions = {}
            for model_name, model in self.models.items():
                self.logger.debug("Generating predictions for %s", model_name)
                if model_name == 'PCA KNN':
                    predictions[model_name] = model.predict(X_new_pca)
                else:
                    predictions[model_name] = model.predict(X_new_scaled)
            
            # Convert to DataFrame
            pred_df = pd.DataFrame(predictions)
            self.logger.info("Predictions generated for %d samples", len(pred_df))
            self.logger.debug("Predictions: %s", pred_df.head().to_dict())
            
            return pred_df
            
        except Exception as e:
            self.logger.error("Error predicting on new data: %s", str(e))
            raise

    def sensitivity_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate sensitivity (recall for positive class).

        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels

        Returns:
            float: Sensitivity score
        """
        try:
            score = recall_score(y_true, y_pred, pos_label=1)
            self.logger.debug("Sensitivity score: %.3f", score)
            return score
        except ValueError as e:
            self.logger.warning("Error calculating sensitivity: %s. Returning 0.0", str(e))
            return 0.0

    def specificity_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate specificity (recall for negative class).

        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels

        Returns:
            float: Specificity score
        """
        try:
            score = recall_score(y_true, y_pred, pos_label=0)
            self.logger.debug("Specificity score: %.3f", score)
            return score
        except ValueError as e:
            self.logger.warning("Error calculating specificity: %s. Returning 0.0", str(e))
            return 0.0

    def evaluate_model(self, model_name: str, model, X_test: np.ndarray, 
                      y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate a model's performance.

        Args:
            model_name (str): Name of the model
            model: Trained model instance
            X_test (np.ndarray): Test features
            y_test (np.ndarray): Test labels

        Returns:
            Dict[str, float]: Dictionary of performance metrics
        """
        try:
            self.logger.info("Evaluating %s", model_name)
            y_pred = model.predict(X_test)
            
            metrics = {
                'Accuracy': accuracy_score(y_test, y_pred),
                'Precision': precision_score(y_test, y_pred, zero_division=0),
                'Sensitivity': self.sensitivity_score(y_test, y_pred),
                'Specificity': self.specificity_score(y_test, y_pred)
            }
            
            self.results.loc[model_name] = metrics
            self.logger.info("%s evaluation completed. Accuracy: %.3f", 
                            model_name, metrics['Accuracy'])
            self.logger.debug("Metrics for %s: %s", model_name, metrics)
            
            return metrics
            
        except Exception as e:
            self.logger.error("Error evaluating %s: %s", model_name, str(e))
            raise

    def train_models(self) -> None:
        """Train all classification models."""
        try:
            # LDA
            self.logger.debug("Training LDA")
            lda = LinearDiscriminantAnalysis()
            lda.fit(self.X_train, self.y_train)
            self.models['LDA'] = lda
            self.evaluate_model('LDA', lda, self.X_test, self.y_test)

            # QDA
            self.logger.debug("Training QDA")
            qda = QuadraticDiscriminantAnalysis()
            qda.fit(self.X_train, self.y_train)
            self.models['QDA'] = qda
            self.evaluate_model('QDA', qda, self.X_test, self.y_test)

            # Naive Bayes
            self.logger.debug("Training Naive Bayes")
            nb = GaussianNB()
            nb.fit(self.X_train, self.y_train)
            self.models['Naive Bayes'] = nb
            self.evaluate_model('Naive Bayes', nb, self.X_test, self.y_test)

            # KNN with GridSearch
            self.logger.debug("Training KNN with GridSearch")
            neighbor_range = np.arange(2, 10)
            param_grid = {'n_neighbors': neighbor_range}
            cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=self.random_state)
            grid = GridSearchCV(KNeighborsClassifier(), param_grid=param_grid, cv=cv)
            grid.fit(self.X_train, self.y_train)
            self.logger.info("Best KNN neighbors: %d", grid.best_params_['n_neighbors'])
            knn = KNeighborsClassifier(n_neighbors=grid.best_params_['n_neighbors'])
            knn.fit(self.X_train, self.y_train)
            self.models['KNN'] = knn
            self.evaluate_model('KNN', knn, self.X_test, self.y_test)

            # PCA KNN
            self.logger.debug("Training PCA KNN with GridSearch")
            grid.fit(self.X_train_pca, self.y_train_pca)
            self.logger.info("Best PCA KNN neighbors: %d", grid.best_params_['n_neighbors'])
            knn_pca = KNeighborsClassifier(n_neighbors=grid.best_params_['n_neighbors'])
            knn_pca.fit(self.X_train_pca, self.y_train_pca)
            self.models['PCA KNN'] = knn_pca
            self.evaluate_model('PCA KNN', knn_pca, self.X_test_pca, self.y_test_pca)

            # Logistic Regression
            self.logger.debug("Training Logistic Regression")
            lr = LogisticRegressionCV(cv=5)
            lr.fit(self.X_train, self.y_train)
            self.models['Logistic Regression'] = lr
            self.evaluate_model('Logistic Regression', lr, self.X_test, self.y_test)

            # Neural Network
            self.logger.debug("Training Neural Network")
            nn = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=2000)

            nn.fit(self.X_train, self.y_train)
            self.models['Neural Network'] = nn
            self.evaluate_model('Neural Network', nn, self.X_test, self.y_test)

            self.logger.info("All models trained successfully")
            
        except Exception as e:
            self.logger.error("Error in training models: %s", str(e))
            raise

    def get_results(self) -> pd.DataFrame:
        """Return the results DataFrame.

        Returns:
            pd.DataFrame: Model performance metrics
        """
        self.logger.debug("Returning model results")
        return self.results

if __name__ == "__main__":
    try:
        # Example usage with synthetic data
        from sklearn.datasets import make_classification
        
        # Generate synthetic dataset
        X, y = make_classification(
            n_samples=1000,
            n_features=20,
            n_informative=15,
            n_redundant=5,
            random_state=0,
            n_classes=2
        )
        
        # Split into training and "unseen" data
        X_train, X_new, y_train, y_new = train_test_split(
            X, y, test_size=0.2, random_state=0
        )
        
        # Initialize and train model
        model = ClassificationModel(X_train, y_train, verbose=2)
        model.preprocess_data()
        model.train_models()
        
        # Display training results
        results = model.get_results()
        print("\nModel Performance Results on Test Set:")
        print(results)
        
        # Predict on unseen data
        predictions = model.predict_new_data(X_new)
        print("\nPredictions on Unseen Data:")
        print(predictions.head())
        
        # Save results and predictions
        results.to_csv('model_results.csv')
        predictions.to_csv('new_data_predictions.csv')
        model.logger.info("Results and predictions saved to CSV files")
        
    except Exception as e:
        print(f"Main execution failed: {str(e)}")
        raise