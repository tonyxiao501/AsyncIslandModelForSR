"""
Data Processing for MIMO Symbolic Regression
This module provides minimal preprocessing for physics data while preserving physical meaning.
"""

import numpy as np
from typing import Tuple
from sklearn.metrics import r2_score


class DataScaler:
    """
    REMOVED: Data scaling functionality has been completely removed.
    
    This class provides minimal compatibility stubs for existing code.
    All methods return input data unchanged to preserve physical meaning.
    """
    
    def __init__(self, **kwargs):
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


def standard_fitness_function(y_true: np.ndarray, y_pred: np.ndarray,
                            complexity: float = 0.0, parsimony_coefficient: float = 0.0) -> float:
    """
    Enhanced R² fitness function with PySR-style complexity penalty.
    
    Args:
        y_true: True target values
        y_pred: Predicted values
        complexity: Expression complexity (for parsimony penalty)
        parsimony_coefficient: Weight of parsimony penalty
        
    Returns:
        R² score minus PySR-style parsimony penalty
    """
    try:
        # Handle invalid predictions
        if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
            return -10.0
        
        # Ensure arrays are flattened
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
        
        # Handle edge case where true values have no variance
        if np.var(y_true_flat) == 0:
            if np.allclose(y_pred_flat, y_true_flat):
                return 1.0
            else:
                return 0.0
        
        # Standard R² calculation
        r2 = r2_score(y_true_flat, y_pred_flat)
        
        # Apply PySR-style parsimony penalty (includes depth and size considerations)
        # Note: This simplified version uses the provided complexity directly
        # In practice, the full PySR-style calculation is done in adaptive_parsimony.py
        parsimony_penalty = parsimony_coefficient * complexity
        fitness = r2 - parsimony_penalty
        
        # Clamp to reasonable range
        return max(-10.0, min(1.0, fitness))
        
    except Exception:
        return -10.0


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
        try:
            from scipy.stats import zscore
            z_scores = np.abs(zscore(y_clean.flatten()))
            outlier_mask = z_scores < outlier_threshold
            X_clean = X_clean[outlier_mask]
            y_clean = y_clean[outlier_mask]
        except ImportError:
            # If scipy not available, skip outlier removal
            pass
    
    return X_clean, y_clean
