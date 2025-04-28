"""
Enhanced visualization module for stacked anomaly detection model
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from logging import getLogger
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap
from scipy.stats import gaussian_kde

from .model import StackedAnomalyModel

# Configure module logger
logger = getLogger(__name__)

class EnhancedViewer(StackedAnomalyModel):
    """Enhanced visualization class for stacked anomaly detection results"""
    
    def __init__(
        self, 
        df: pd.DataFrame, 
        feature_names: List[str], 
        target_names: List[str], 
        stock: str, 
        verbose: bool = False
    ) -> None:
        """
        Initialize the enhanced anomaly visualization class
        Args:
            df: DataFrame containing the data
            feature_names: List of feature column names
            target_names: List of target column names
            stock: Stock symbol
            verbose: Control logging verbosity
        """
        super().__init__(df, feature_names, target_names, stock, verbose)
        logger.debug(f"Initializing enhanced viewer for stock {stock}")
        
        # Custom color maps
        self.colors = ['#2ecc71', '#e74c3c']  # Green for normal, Red for anomaly
        self.cmap = ListedColormap(self.colors)
        
    def plot_individual_models(self, data: Optional[pd.DataFrame] = None) -> None:
        """
        Plot predictions from individual base models
        Args:
            data: Optional test data to visualize (uses test set if None)
        """
        if data is None:
            data = self.xtest
            
        n_models = len(self.base_models)
        fig = plt.figure(figsize=(15, 3 * ((n_models + 1) // 2)))
        gs = gridspec.GridSpec(((n_models + 1) // 2), 2)
        
        for i, (name, model) in enumerate(self.base_models.items()):
            ax = plt.subplot(gs[i // 2, i % 2])
            
            # Get predictions
            preds = model.predict(data)
            scores = None
            
            # Get decision scores if available
            try:
                if hasattr(model, 'decision_function'):
                    scores = model.decision_function(data)
                elif hasattr(model, 'score_samples'):
                    scores = model.score_samples(data)
            except:
                pass
                
            close = self.price_data.loc[data.index, 'close'].values
            # Plot time series with predictions
            scatter = ax.scatter(data.index, close, 
                               c=preds, cmap=self.cmap, 
                               alpha=0.6, s=50)
            ax.plot(data.index, close, color='black', alpha=0.2)
            
            # Add confidence band if scores available
            if scores is not None:
                scores = StandardScaler().fit_transform(scores.reshape(-1, 1)).ravel()
                ax.fill_between(data.index, close, 
                              close + scores, 
                              alpha=0.2, color='gray')
                
            ax.set_title(f'{name} Predictions')
            ax.set_xlabel('Date')
            ax.set_ylabel('Price')
            ax.legend(*scatter.legend_elements(), title="Predictions", loc='upper left')
            
        plt.tight_layout()
        plt.show()
        
    def plot_meta_learner_decision_boundary(self) -> None:
        """Visualize meta-learner decision boundaries using PCA"""
        if not self.meta_learner:
            logger.error("Meta-learner not trained yet")
            return
            
        # Get base model predictions
        base_preds = []
        for name, model in self.base_models.items():
            preds = model.predict(self.xtest)
            base_preds.append(preds.reshape(-1, 1))
            
        X_meta = np.hstack(base_preds)
        
        # Reduce to 2D for visualization
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X_meta)
        
        # Create mesh grid
        x_min, x_max = X_2d[:, 0].min() - 0.5, X_2d[:, 0].max() + 0.5
        y_min, y_max = X_2d[:, 1].min() - 0.5, X_2d[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                            np.arange(y_min, y_max, 0.02))
        
        # Get predictions for mesh grid
        Z = self.meta_learner.predict(pca.inverse_transform(
            np.c_[xx.ravel(), yy.ravel()]
        ))
        Z = Z.reshape(xx.shape)
        
        # Plot decision boundary
        colors = np.where(self.ytest == 1, 1, -1)  # Map to -1 and 1 for coloring
        plt.figure(figsize=(10, 8))
        plt.contourf(xx, yy, Z, cmap=self.cmap, alpha=0.4)
        plt.scatter(X_2d[:, 0], X_2d[:, 1], c=colors, alpha=0.8, cmap=self.cmap, edgecolor='k', s=50)
        plt.title("Meta-Learner Decision Boundary")
        plt.xlabel(f"First Principal Component ({pca.explained_variance_ratio_[0]:.2%})")
        plt.ylabel(f"Second Principal Component ({pca.explained_variance_ratio_[1]:.2%})")
        plt.colorbar(ticks=[-1, 1])
        plt.show()
        
    def plot_confidence_distributions(self) -> None:
        """Plot confidence score distributions for each model"""
        scores_dict = {}
        
        # Collect confidence scores from each model's final layer
        for name, model in self.base_models.items():
            try:
                if hasattr(model, 'decision_function'):
                    scores = model.decision_function(self.xtest)
                elif hasattr(model, 'score_samples'):
                    scores = model.score_samples(self.xtest)
                else:
                    continue
                    
                # Ensure scores are 1D
                scores = np.asarray(scores).ravel()
                scores_dict[name] = scores
                
            except Exception as e:
                logger.warning(f"Could not get confidence scores for {name}: {str(e)}")
                continue
                
        if not scores_dict:
            logger.error("No confidence scores available")
            return
            
        # Plot distributions
        n_models = len(scores_dict)
        fig, axes = plt.subplots(n_models, 1, figsize=(12, 2*n_models))
        if n_models == 1:
            axes = [axes]
            
        for ax, (name, scores) in zip(axes, scores_dict.items()):
            # Separate scores by class
            normal_scores = [scores[x] for x, i in enumerate(self.ytest.target) if i == 0]
            anomaly_scores = [scores[x] for x, i in enumerate(self.ytest.target) if i == 1]
            print(f"Normal scores: {len(normal_scores)}, Anomaly scores: {len(anomaly_scores)}")
            
            # Plot KDE for each class if we have data
            for scores_set, label, color in [
                (normal_scores, 'Normal', self.colors[0]),
                (anomaly_scores, 'Anomaly', self.colors[1])
            ]:
                if len(scores_set) > 0:
                    
                    try:
                        kde = gaussian_kde(scores_set)
                        x_range = np.linspace(min(scores), max(scores), 200)
                        density = kde(x_range)
                        ax.plot(x_range, density, label=label, color=color)
                    except Exception as e:
                        logger.warning(f"KDE failed for {name} {label}: {str(e)}")
                        continue
                    
            ax.set_title(f'{name} Confidence Score Distribution')
            ax.set_xlabel('Confidence Score')
            ax.set_ylabel('Density')
            # ax.legend()
            
        plt.tight_layout()
        plt.show()
        
    def plot_ensemble_performance(self) -> None:
        """Compare performance metrics across ensemble models"""
        metrics = {}
        
        true_preds = np.where(self.ytest == -1, -1, 1)  # Map to -1 and 1 for consistency
        # Calculate metrics for each model
        for name, model in self.base_models.items():
            preds = model.predict(self.xtest)
            
            # Metrics
            metrics[name] = {
                'Accuracy': (preds == true_preds).mean(),
                'Anomaly Rate': (preds == -1).mean(),
                'F1': f1_score(true_preds, preds, pos_label=-1)
            }
            
        # Add meta-learner metrics
        meta_preds = self.predict(self.xtest).values
        metrics['Meta-Learner'] = {
            'Accuracy': (meta_preds == true_preds).mean(),
            'Anomaly Rate': (meta_preds == -1).mean(),
            'F1': f1_score(true_preds, meta_preds, average = 'micro')
        }
        
        # Create comparison plot
        metrics_df = pd.DataFrame(metrics).T

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for i, metric in enumerate(metrics_df.columns):
            # axes[i].bar(range(len(metrics_df)), metrics_df[metric], color=self.colors[0], alpha=0.7)
            # axes[i].bar(metrics_df.index, metrics_df[metric], color=self.colors[0], alpha=0.7)
            axes[i].bar(metrics_df.index, metrics_df[metric])
            axes[i].set_xticks(range(len(metrics_df)))
            axes[i].set_xticklabels(metrics_df.index, rotation=45)
            axes[i].set_title(f'{metric} by Model')
            axes[i].set_ylim(0, 1)
            
        plt.tight_layout()
        plt.show()
        
    def plot_temporal_agreement(self, window: int = 30) -> None:
        """
        Plot temporal model agreement and disagreement
        Args:
            window: Rolling window size for agreement calculation
        """
        # Get predictions from all models
        all_preds = []
        for name, model in self.base_models.items():
            preds = model.predict(self.xtest)
            all_preds.append(pd.Series(preds, index=self.xtest.index, name=name))
            
        meta_preds = self.predict(self.xtest)
        all_preds.append(pd.Series(meta_preds.values.ravel(), index=self.xtest.index, name='Meta-Learner'))
        
        # Calculate rolling agreement
        preds_df = pd.concat(all_preds, axis=1)
        agreement = preds_df.apply(lambda x: (x == x.mode().iloc[0]).mean(), axis=1)
        rolling_agreement = agreement.rolling(window=window).mean()
        
        # Plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # Plot price with agreement
        ax1.plot(self.xtest.index, self.price_data['close'].loc[self.xtest.index], color='black', alpha=0.5)
        ax1.set_ylabel('Price')
        ax2.plot(rolling_agreement.index, rolling_agreement, color='blue')
        ax2.set_ylabel('Model Agreement')
        # ax2.set_ylim(0, 1)
        
        plt.title(f'Model Agreement (Rolling {window}-day window)')
        plt.tight_layout()
        plt.show()

    def plot_all(self, data: Optional[pd.DataFrame] = None) -> None:
        """Generate all visualization plots"""
        logger.info("Generating all visualization plots")
        
        self.plot_individual_models(data)
        self.plot_meta_learner_decision_boundary()
        self.plot_confidence_distributions()
        self.plot_ensemble_performance()
        self.plot_temporal_agreement()