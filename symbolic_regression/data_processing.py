"""
Data Processing for MIMO Symbolic Regression
This module provides minimal preprocessing for physics data while preserving physical meaning.
"""

import numpy as np
from typing import Tuple


def r2_score(y_true, y_pred):
    """
    Robust R² score calculation that handles numerical issues better than sklearn.
    
    Key improvements:
    1. Better handling of extreme values (very small/large numbers common in physics)
    2. More robust numerical computation using double precision
    3. Explicit handling of edge cases
    4. Clamping to prevent overflow/underflow issues
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    # Handle edge cases
    if len(y_true) == 0:
        return 0.0
    
    # Check for invalid predictions
    if np.any(~np.isfinite(y_pred)):
        return -10.0  # Large negative score for invalid predictions
    
    # Calculate mean
    y_mean = np.mean(y_true)
    
    # Handle case where true values have no variance
    y_var = np.var(y_true)
    if y_var == 0 or y_var < 1e-30:  # Very small variance
        if np.allclose(y_pred, y_true, rtol=1e-10):
            return 1.0
        else:
            return 0.0
    
    # Calculate sums of squares with numerical stability and overflow protection
    try:
        # Use double precision for intermediate calculations
        y_true_dp = y_true.astype(np.float64)
        y_pred_dp = y_pred.astype(np.float64)
        y_mean_dp = np.float64(y_mean)
        
        # Check for extreme values that might cause overflow
        diff_pred = y_true_dp - y_pred_dp
        diff_mean = y_true_dp - y_mean_dp
        
        # Check if differences are too large (would cause overflow when squared)
        max_safe_value = 1e100  # Conservative threshold
        if np.any(np.abs(diff_pred) > max_safe_value) or np.any(np.abs(diff_mean) > max_safe_value):
            return -10.0
        
        # Compute residual and total sums of squares with overflow protection
        with np.errstate(over='raise', invalid='raise'):
            ss_res = np.sum(diff_pred ** 2)
            ss_tot = np.sum(diff_mean ** 2)
        
        # Handle division by zero
        if ss_tot == 0.0 or ss_tot < 1e-30:
            return 1.0 if ss_res == 0.0 or ss_res < 1e-30 else 0.0
        
        # Compute R² with overflow protection
        with np.errstate(over='raise', invalid='raise'):
            r2 = 1.0 - (ss_res / ss_tot)
        
        # Clamp to reasonable range to prevent extreme values
        return float(np.clip(r2, -10.0, 1.0))
        
    except (OverflowError, RuntimeWarning, FloatingPointError, Warning):
        # If calculation fails due to numerical issues, return very negative score
        return -10.0


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


# --- Additional robust metrics for fitness options ---
def mae_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error with basic sanitization."""
    yt = np.asarray(y_true, dtype=float).flatten()
    yp = np.asarray(y_pred, dtype=float).flatten()
    if yt.size == 0:
        return float('inf')
    if np.any(~np.isfinite(yp)):
        return 1e12
    diff = np.abs(yt - yp)
    return float(np.mean(np.nan_to_num(diff, nan=1e12, posinf=1e12, neginf=1e12)))


def huber_loss(y_true: np.ndarray, y_pred: np.ndarray, delta: float = 1.0) -> float:
    """Huber loss (smooth L1) with parameter delta."""
    yt = np.asarray(y_true, dtype=float).flatten()
    yp = np.asarray(y_pred, dtype=float).flatten()
    if yt.size == 0:
        return float('inf')
    if np.any(~np.isfinite(yp)):
        return 1e12
    d = np.abs(yt - yp)
    quad = np.minimum(d, delta)
    lin = d - quad
    loss = 0.5 * (quad ** 2) + delta * lin
    return float(np.mean(np.nan_to_num(loss, nan=1e12, posinf=1e12, neginf=1e12)))


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
