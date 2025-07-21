"""
Multi-Scale Fitness Evaluation Module

This module provides robust fitness metrics that work well across extreme magnitude ranges,
specifically designed for physics problems with very small or very large values.
"""

import numpy as np
from typing import Union, Optional
import warnings


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
            return -1e6
        
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
            
            # Calculate R² in log space
            ss_res = np.sum((log_true - log_pred) ** 2)
            ss_tot = np.sum((log_true - np.mean(log_true)) ** 2)
            
            if ss_tot == 0:
                return 1.0 if ss_res == 0 else 0.0
            
            r2_log = 1.0 - (ss_res / ss_tot)
            
            # Also calculate relative error for robustness
            relative_error = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-12)))
            relative_fitness = 1.0 / (1.0 + relative_error)
            
            # Combine both metrics
            combined_fitness = 0.7 * r2_log + 0.3 * relative_fitness
            
            return combined_fitness
            
        except Exception as e:
            warnings.warn(f"Log space fitness calculation failed: {e}")
            return self._standard_r2_fitness(y_true, y_pred)
    
    def _relative_error_fitness(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate fitness based on relative error metrics (MAPE/SMAPE).
        """
        try:
            # Symmetric Mean Absolute Percentage Error (SMAPE)
            denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
            # Avoid division by zero
            denominator = np.where(denominator < 1e-12, 1e-12, denominator)
            smape = np.mean(np.abs(y_true - y_pred) / denominator)
            
            # Convert to fitness (higher is better)
            smape_fitness = 1.0 / (1.0 + smape)
            
            # Also calculate normalized RMSE for additional robustness
            rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
            mean_magnitude = np.mean(np.abs(y_true))
            normalized_rmse = rmse / (mean_magnitude + 1e-12)
            rmse_fitness = 1.0 / (1.0 + normalized_rmse)
            
            # Combine metrics
            combined_fitness = 0.6 * smape_fitness + 0.4 * rmse_fitness
            
            return combined_fitness
            
        except Exception as e:
            warnings.warn(f"Relative error fitness calculation failed: {e}")
            return self._standard_r2_fitness(y_true, y_pred)
    
    def _standard_r2_fitness(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Standard R² fitness calculation.
        """
        try:
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            
            if ss_tot == 0:
                return 1.0 if ss_res == 0 else 0.0
            
            r2 = 1.0 - (ss_res / ss_tot)
            
            # Clip to reasonable range
            return max(-1e6, min(1.0, r2))
            
        except Exception:
            return -1e6
    
    def get_detailed_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """
        Get a comprehensive set of metrics for analysis.
        """
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        
        metrics = {}
        
        # Standard metrics
        try:
            metrics['r2'] = self._standard_r2_fitness(y_true, y_pred)
            metrics['rmse'] = np.sqrt(np.mean((y_true - y_pred) ** 2))
            metrics['mae'] = np.mean(np.abs(y_true - y_pred))
        except Exception as e:
            metrics['r2'] = -1e6
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
        
        # Log-space metrics (if applicable)
        try:
            if np.all(y_true > 0) and np.all(y_pred > 0):
                log_true = np.log10(y_true)
                log_pred = np.log10(y_pred)
                
                ss_res_log = np.sum((log_true - log_pred) ** 2)
                ss_tot_log = np.sum((log_true - np.mean(log_true)) ** 2)
                
                if ss_tot_log > 0:
                    metrics['r2_log'] = 1.0 - (ss_res_log / ss_tot_log)
                else:
                    metrics['r2_log'] = 1.0 if ss_res_log == 0 else 0.0
                    
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
                ss_res = np.sum((y_true - y_pred) ** 2)
                ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
                
                if ss_tot == 0:
                    r2 = 1.0 if ss_res == 0 else 0.0
                else:
                    r2 = 1.0 - (ss_res / ss_tot)
                
                return r2 - parsimony_coefficient
            except Exception:
                return -1e6
    
    return fitness_function
