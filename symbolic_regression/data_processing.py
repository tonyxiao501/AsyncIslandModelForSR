"""
Data Processing for MIMO Symbolic Regression
This module provides multi-scale fitness evaluation and minimal preprocessing for physics data.
"""

import numpy as np
from typing import Tuple, Dict, Optional, Callable, List, Union, Any
from sklearn.metrics import r2_score
import warnings


class DataScaler:
    """
    REMOVED: Data scaling functionality has been completely removed.
    
    This class now provides minimal compatibility stubs for existing code.
    All methods return input data unchanged to preserve physical meaning.
    """
    
    def __init__(self, input_scaling: str = 'none', output_scaling: str = 'none',
                 target_range: Tuple[float, float] = (-5.0, 5.0)):
        """Initialize stub scaler - all parameters ignored."""
        pass
    
    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return data unchanged."""
        return X.copy(), y.copy()
    
    def transform_input(self, X: np.ndarray) -> np.ndarray:
        """Return input data unchanged."""
        return X.copy()
    
    def transform_output(self, y: np.ndarray) -> np.ndarray:
        """Return output data unchanged."""
        return y.copy()
    
    def inverse_transform_output(self, y: np.ndarray) -> np.ndarray:
        """Return data unchanged."""
        return y.copy()
    
    def get_scaling_info(self) -> Dict[str, Any]:
        """Return empty scaling info."""
        return {
            'input_transforms': [],
            'output_transform': 'none',
            'target_range': (-5.0, 5.0),
            'input_log_offsets': [],
            'output_log_offset': 0.0
        }


class MultiScaleFitnessEvaluator:
    """
    Enhanced fitness evaluator for extreme-scale problems.
    Provides multiple robust metrics beyond simple R².
    """
    
    def __init__(self, 
                 use_log_space: bool = True,
                 use_relative_metrics: bool = True,
                 extreme_value_threshold: float = 1e6):
        """
        Initialize multi-scale fitness evaluator.
        
        Args:
            use_log_space: Evaluate fitness in log space for extreme values
            use_relative_metrics: Use relative error metrics (MAPE, SMAPE)
            extreme_value_threshold: Threshold for switching to log-space evaluation
        """
        self.use_log_space = use_log_space
        self.use_relative_metrics = use_relative_metrics
        self.extreme_value_threshold = extreme_value_threshold
    
    def evaluate_fitness(self, y_true: np.ndarray, y_pred: np.ndarray, 
                        complexity: float = 0.0, parsimony_coefficient: float = 0.0) -> float:
        """
        Comprehensive fitness evaluation that handles extreme scales.
        
        Returns:
            Combined fitness score (higher is better)
        """
        # Ensure arrays are flattened
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        
        # Handle invalid predictions
        if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
            return -10.0  # Large negative R² score for invalid predictions
        
        # Determine if we're dealing with extreme scales
        max_magnitude = max(np.max(np.abs(y_true)), np.max(np.abs(y_pred)))
        min_magnitude = min(np.min(np.abs(y_true[y_true != 0])), 
                           np.min(np.abs(y_pred[y_pred != 0]))) if np.any(y_true != 0) and np.any(y_pred != 0) else 1.0
        
        is_extreme_scale = (max_magnitude > self.extreme_value_threshold or 
                           min_magnitude < 1.0/self.extreme_value_threshold or
                           (max_magnitude / min_magnitude) > 1e6)
        
        # Choose appropriate fitness metric
        if is_extreme_scale and self.use_log_space:
            fitness = self._log_space_fitness(y_true, y_pred)
        elif self.use_relative_metrics:
            fitness = self._relative_error_fitness(y_true, y_pred)
        else:
            fitness = self._standard_r2_fitness(y_true, y_pred)
        
        # Apply parsimony penalty
        fitness -= parsimony_coefficient * complexity
        
        return fitness
    
    def _log_space_fitness(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Evaluate fitness in log space for extreme-scale problems.
        Returns R² equivalent score.
        """
        try:
            # Handle negative values by shifting to positive range
            min_val = min(np.min(y_true), np.min(y_pred))
            if min_val <= 0:
                shift = abs(min_val) + 1e-12
                y_true_shifted = y_true + shift
                y_pred_shifted = y_pred + shift
            else:
                y_true_shifted = y_true
                y_pred_shifted = y_pred
            
            # Handle zero values to avoid log(0)
            y_true_shifted = np.where(y_true_shifted <= 1e-15, 1e-15, y_true_shifted)
            y_pred_shifted = np.where(y_pred_shifted <= 1e-15, 1e-15, y_pred_shifted)
            
            # Convert to log space
            log_true = np.log10(y_true_shifted)
            log_pred = np.log10(y_pred_shifted)
            
            # Check for invalid values
            if np.any(np.isnan(log_true)) or np.any(np.isnan(log_pred)) or \
               np.any(np.isinf(log_true)) or np.any(np.isinf(log_pred)):
                return self._relative_error_fitness(y_true, y_pred)
            
            # Calculate R² in log space using scikit-learn
            r2_log = r2_score(log_true, log_pred)
            
            # Also calculate relative error for robustness and convert to R² equivalent
            relative_error = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-12)))
            # Convert relative error to R² equivalent (1 - normalized error)
            relative_r2 = max(0.0, float(1.0 - relative_error))
            
            # Combine both R² metrics (weighted towards log-space R²)
            combined_r2 = 0.7 * r2_log + 0.3 * relative_r2
            
            return combined_r2
            
        except Exception as e:
            warnings.warn(f"Log space fitness calculation failed: {e}")
            return self._standard_r2_fitness(y_true, y_pred)
    
    def _relative_error_fitness(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate fitness based on relative error metrics, converted to R² equivalent.
        """
        try:
            # Symmetric Mean Absolute Percentage Error (SMAPE)
            denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
            # Avoid division by zero
            denominator = np.where(denominator < 1e-12, 1e-12, denominator)
            smape = np.mean(np.abs(y_true - y_pred) / denominator)
            
            # Convert SMAPE to R² equivalent (1 - normalized error)
            smape_r2 = max(0.0, float(1.0 - smape))
            
            # Also calculate normalized RMSE and convert to R² equivalent
            rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
            mean_magnitude = np.mean(np.abs(y_true))
            normalized_rmse = rmse / (mean_magnitude + 1e-12)
            rmse_r2 = max(0.0, float(1.0 - normalized_rmse))
            
            # Combine R² equivalent metrics
            combined_r2 = 0.6 * smape_r2 + 0.4 * rmse_r2
            
            return combined_r2
            
        except Exception as e:
            warnings.warn(f"Relative error fitness calculation failed: {e}")
            return self._standard_r2_fitness(y_true, y_pred)
    
    def _standard_r2_fitness(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Standard R² fitness calculation using scikit-learn.
        """
        try:
            # Handle edge cases
            if np.var(y_true) == 0:
                # If true values have no variance, check if predictions match
                if np.allclose(y_pred, y_true):
                    return 1.0
                else:
                    return 0.0
            
            # Use scikit-learn's R² implementation
            r2 = r2_score(y_true, y_pred)
            
            # Clamp to reasonable range
            return max(-10.0, min(1.0, r2))
            
        except Exception as e:
            warnings.warn(f"Standard R² calculation failed: {e}")
            return 0.0
    
    def get_detailed_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Get detailed metrics for analysis"""
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        
        metrics = {}
        
        try:
            # Standard R²
            metrics['r2'] = float(r2_score(y_true, y_pred))
            
            # RMSE
            metrics['rmse'] = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
            
            # Mean Absolute Error
            metrics['mae'] = float(np.mean(np.abs(y_true - y_pred)))
            
            # Relative error metrics
            relative_error = np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-12))
            metrics['mean_relative_error'] = float(np.mean(relative_error))
            
            # Log-space R² if appropriate
            if np.all(y_true > 0) and np.all(y_pred > 0):
                log_true = np.log10(y_true)
                log_pred = np.log10(y_pred)
                metrics['log_r2'] = float(r2_score(log_true, log_pred))
            
        except Exception as e:
            warnings.warn(f"Detailed metrics calculation failed: {e}")
        
        return metrics


def create_robust_fitness_function(use_multi_scale: bool = True):
    """
    Create a robust fitness function for symbolic regression.
    
    Args:
        use_multi_scale: Whether to use multi-scale fitness evaluation
        
    Returns:
        Fitness function that can be used in symbolic regression
    """
    if use_multi_scale:
        evaluator = MultiScaleFitnessEvaluator()
        
        def fitness_function(y_true: np.ndarray, y_pred: np.ndarray, 
                           complexity: float = 0.0, parsimony_coefficient: float = 0.0) -> float:
            return evaluator.evaluate_fitness(y_true, y_pred, complexity, parsimony_coefficient)
    else:
        def fitness_function(y_true: np.ndarray, y_pred: np.ndarray,
                           complexity: float = 0.0, parsimony_coefficient: float = 0.0) -> float:
            try:
                r2 = r2_score(y_true.flatten(), y_pred.flatten())
                return r2 - parsimony_coefficient * complexity
            except:
                return -10.0
    
    return fitness_function


# RECOMMENDED: Raw data preprocessing for physics-aware symbolic regression
def prepare_physics_data(X: np.ndarray, y: np.ndarray, 
                        remove_outliers: bool = False, 
                        outlier_threshold: float = 5.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Minimal preprocessing for physics data that preserves relationships.
    
    Args:
        X: Input features
        y: Target values  
        remove_outliers: Whether to remove statistical outliers
        outlier_threshold: Z-score threshold for outlier removal
        
    Returns:
        Cleaned X and y (minimally processed)
    """
    X_clean = X.copy()
    y_clean = y.copy()
    
    # Remove NaN/Inf values
    valid_mask = np.isfinite(X_clean).all(axis=1) & np.isfinite(y_clean.flatten())
    X_clean = X_clean[valid_mask]
    y_clean = y_clean[valid_mask]
    
    # Optional outlier removal (preserves physics relationships)
    if remove_outliers and len(y_clean) > 10:
        from scipy.stats import zscore
        z_scores = np.abs(zscore(y_clean.flatten()))
        outlier_mask = z_scores < outlier_threshold
        X_clean = X_clean[outlier_mask]
        y_clean = y_clean[outlier_mask]
    
    return X_clean, y_clean


# Minimal compatibility stubs for deprecated classes
class PhysicsDataPreprocessor:
    """Simplified physics data preprocessor"""
    
    def __init__(self):
        pass
    
    def preprocess(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess physics data"""
        return prepare_physics_data(X, y, remove_outliers=True)


class PhysicsAwareScaler:
    """Compatibility stub - does no scaling"""
    
    def __init__(self, **kwargs):
        pass
    
    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return X.copy(), y.copy()
    
    def transform_input(self, X: np.ndarray) -> np.ndarray:
        return X.copy()
    
    def inverse_transform_output(self, y: np.ndarray) -> np.ndarray:
        return y.copy()
