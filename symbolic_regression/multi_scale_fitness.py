"""
Multi-Scale Fitness Evaluation Module

This module provides robust fitness metrics that work well across extreme magnitude ranges,
specifically designed for physics problems with very small or very large values.
All fitness evaluations are unified to return R² scores using scikit-learn's implementation.
"""

import numpy as np
from typing import Union, Optional
import warnings
from sklearn.metrics import r2_score


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
                        parsimony_penalty: float = 0.0) -> float:
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
        fitness -= parsimony_penalty
        
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
        Standard R² fitness calculation using scikit-learn's implementation.
        """
        try:
            # Use scikit-learn's optimized R² implementation
            r2 = r2_score(y_true, y_pred)
            
            # Clip to reasonable range for extreme cases  
            return max(-10.0, min(1.0, r2))  # R² scores should be between -10.0 and 1.0
            
        except Exception:
            return -10.0  # Large negative R² score for calculation errors
    
    def get_detailed_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """
        Get a comprehensive set of metrics for analysis.
        """
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        
        metrics = {}
        
        # Standard metrics using scikit-learn
        try:
            metrics['r2'] = r2_score(y_true, y_pred)
            metrics['rmse'] = np.sqrt(np.mean((y_true - y_pred) ** 2))
            metrics['mae'] = np.mean(np.abs(y_true - y_pred))
        except Exception as e:
            metrics['r2'] = -10.0  # Large negative R² score for errors
            metrics['rmse'] = 1e6
            metrics['mae'] = 1e6
        
        # Relative metrics
        try:
            denominator = np.abs(y_true) + 1e-12
            mape = np.mean(np.abs((y_true - y_pred) / denominator))
            metrics['mape'] = mape
            
            smape_denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0 + 1e-12
            smape = np.mean(np.abs(y_true - y_pred) / smape_denominator)
            metrics['smape'] = smape
        except Exception:
            metrics['mape'] = 1e6
            metrics['smape'] = 1e6
        
        # Log-space metrics (if applicable) using scikit-learn
        try:
            if np.all(y_true > 0) and np.all(y_pred > 0):
                log_true = np.log10(y_true)
                log_pred = np.log10(y_pred)
                
                # Use scikit-learn for log-space R²
                metrics['r2_log'] = r2_score(log_true, log_pred)
                metrics['rmse_log'] = np.sqrt(np.mean((log_true - log_pred) ** 2))
            else:
                metrics['r2_log'] = None
                metrics['rmse_log'] = None
        except Exception:
            metrics['r2_log'] = None
            metrics['rmse_log'] = None
        
        # Scale characteristics
        max_magnitude = max(np.max(np.abs(y_true)), np.max(np.abs(y_pred)))
        min_magnitude = min(np.min(np.abs(y_true[y_true != 0])), 
                           np.min(np.abs(y_pred[y_pred != 0]))) if np.any(y_true != 0) and np.any(y_pred != 0) else 1.0
        
        metrics['magnitude_range'] = max_magnitude / min_magnitude if min_magnitude > 0 else float('inf')
        metrics['max_magnitude'] = max_magnitude
        metrics['min_magnitude'] = min_magnitude
        
        return metrics


def create_robust_fitness_function(use_multi_scale: bool = True):
    """
    Factory function to create a robust fitness function.
    
    Returns:
        Fitness function compatible with existing population evaluation code
    """
    if use_multi_scale:
        evaluator = MultiScaleFitnessEvaluator()
        
        def fitness_function(y_true: np.ndarray, y_pred: np.ndarray, 
                           parsimony_coefficient: float = 0.0) -> float:
            return evaluator.evaluate_fitness(y_true, y_pred, parsimony_coefficient)
    else:
        def fitness_function(y_true: np.ndarray, y_pred: np.ndarray,
                           parsimony_coefficient: float = 0.0) -> float:
            try:
                # Use scikit-learn's R² implementation for consistency
                r2 = r2_score(y_true, y_pred)
                
                # Apply parsimony penalty to R² score
                return r2 - parsimony_coefficient
            except Exception:
                return -10.0  # Large negative R² score for calculation errors
    
    return fitness_function
