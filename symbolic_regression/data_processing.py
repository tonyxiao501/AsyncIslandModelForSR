"""
Data Processing for MIMO Symbolic Regression
This module consolidates data scaling, preprocessing, and multi-scale fitness evaluation.
"""

import numpy as np
from typing import Tuple, Dict, Optional, Callable, List, Union, Any
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import r2_score
import warnings


class DataScaler:
    """
    Advanced data scaling for symbolic regression with automatic scale detection
    and inverse transformation of discovered expressions.
    """
    
    def __init__(self, input_scaling: str = 'auto', output_scaling: str = 'auto',
                 target_range: Tuple[float, float] = (-5.0, 5.0)):
        """
        Initialize the data scaler.
        
        Args:
            input_scaling: 'standard', 'minmax', 'robust', 'log', 'auto'
            output_scaling: 'standard', 'minmax', 'robust', 'log', 'auto'  
            target_range: Target range for scaled data (expanded for better GP performance)
        """
        self.input_scaling = input_scaling
        self.output_scaling = output_scaling
        self.target_range = target_range
        
        # Scalers will be fitted during preprocessing
        self.input_scalers: List[Optional[Any]] = []
        self.output_scaler: Optional[Any] = None
        
        # Transformation metadata
        self.input_transforms: List[str] = []
        self.output_transform: str = 'none'
        self.input_log_offsets: List[float] = []
        self.output_log_offset: float = 0.0
        
        # Original data statistics
        self.original_X_stats: Dict = {}
        self.original_y_stats: Dict = {}
        
    def _detect_optimal_scaling(self, data: np.ndarray, is_output: bool = False) -> str:
        """
        Detect optimal scaling method for physics data with extreme values.
        More aggressive scaling for better physics law discovery.
        
        Args:
            data: Input data array
            is_output: Whether this is output data (different heuristics)
            
        Returns:
            Optimal scaling method name
        """
        # Check for negative values (affects log scaling)
        has_negative = np.any(data < 0)
        
        # Calculate statistical properties
        data_range = np.ptp(data)  # peak-to-peak (max - min)
        data_std = np.std(data)
        data_mean = np.mean(data)
        
        # Avoid scaling if data is already well-behaved
        if data_range == 0 or data_std == 0:
            return 'none'
        
        # More aggressive approach for physics problems
        magnitude_range = 0
        nonzero_data = np.abs(data[data != 0])
        if len(nonzero_data) > 0:
            max_val = nonzero_data.max()
            min_val = nonzero_data.min()
            if max_val > 0 and min_val > 0:
                magnitude_range = np.log10(max_val) - np.log10(min_val)
            
            # Much more aggressive log scaling for physics constants
            if magnitude_range > 4:  # Raised threshold - less aggressive scaling
                if not has_negative and np.all(data > 0):
                    return 'log'
                else:
                    return 'robust'
                    
            # Scale for smaller extreme values in physics
            min_magnitude = np.log10(min_val) if min_val > 0 else 0
            max_magnitude = np.log10(max_val) if max_val > 0 else 0
            if min_magnitude < -6 or max_magnitude > 6:  # Less aggressive for physics
                if not has_negative and np.all(data > 0):
                    return 'log'
                else:
                    return 'robust'
        
        # Check for highly skewed distributions (less sensitive)
        if data_range > 10 * data_std:  # Higher threshold - less aggressive scaling
            return 'robust'
        
        # For moderately large values, use standard scaling (more aggressive)
        if abs(data_mean) > 3 * data_std:  # Lower threshold
            return 'standard'
        
        # For reasonably scaled data, use minimal scaling
        if data_range > 10:  # Much lower threshold for physics
            return 'minmax'
        
        # Default: robust scaling for physics (more robust than standard)
        return 'robust'
    
    def _apply_log_transform(self, data: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Apply log transformation with automatic offset for negative/zero values.
        
        Returns:
            Transformed data and the offset used
        """
        if np.any(data <= 0):
            offset = abs(np.min(data)) + 1e-8
        else:
            offset = 0.0
            
        return np.log(data + offset), offset
    
    def _inverse_log_transform(self, data: np.ndarray, offset: float) -> np.ndarray:
        """Inverse log transformation."""
        return np.exp(data) - offset
    
    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit scalers and transform the data.
        
        Args:
            X: Input features [n_samples, n_features]
            y: Target values [n_samples, n_outputs]
            
        Returns:
            Scaled X and y
        """
        # Store original statistics
        self.original_X_stats = {
            'mean': np.mean(X, axis=0),
            'std': np.std(X, axis=0),
            'min': np.min(X, axis=0),
            'max': np.max(X, axis=0)
        }
        
        self.original_y_stats = {
            'mean': np.mean(y),
            'std': np.std(y),
            'min': np.min(y),
            'max': np.max(y)
        }
        
        # Transform inputs (per feature)
        X_scaled = X.copy()
        self.input_scalers = []
        self.input_transforms = []
        self.input_log_offsets = []
        
        for i in range(X.shape[1]):
            feature = X[:, i]
            
            # Determine scaling method
            if self.input_scaling == 'auto':
                scale_method = self._detect_optimal_scaling(feature)
            else:
                scale_method = self.input_scaling
            
            self.input_transforms.append(scale_method)
            
            # Apply transformation
            if scale_method == 'log':
                feature_transformed, offset = self._apply_log_transform(feature)
                self.input_log_offsets.append(offset)
                # Then apply standard scaling to log-transformed data
                scaler = StandardScaler()
                X_scaled[:, i] = scaler.fit_transform(feature_transformed.reshape(-1, 1)).flatten()
            else:
                self.input_log_offsets.append(0.0)
                if scale_method == 'standard':
                    scaler = StandardScaler()
                elif scale_method == 'minmax':
                    scaler = MinMaxScaler(feature_range=self.target_range)  # type: ignore
                elif scale_method == 'robust':
                    scaler = RobustScaler()
                elif scale_method == 'none':
                    scaler = None
                else:
                    scaler = None
                
                if scaler:
                    X_scaled[:, i] = scaler.fit_transform(feature.reshape(-1, 1)).flatten()
                # If no scaler, keep original values
            
            self.input_scalers.append(scaler)
        
        # Transform output
        y_scaled = y.copy()
        
        # Determine output scaling method
        if self.output_scaling == 'auto':
            self.output_transform = self._detect_optimal_scaling(y.flatten(), is_output=True)
        else:
            self.output_transform = self.output_scaling
        
        # Apply output transformation
        if self.output_transform == 'log':
            y_transformed, self.output_log_offset = self._apply_log_transform(y.flatten())
            self.output_scaler = StandardScaler()
            y_scaled = self.output_scaler.fit_transform(y_transformed.reshape(-1, 1))
        else:
            self.output_log_offset = 0.0
            if self.output_transform == 'standard':
                self.output_scaler = StandardScaler()
            elif self.output_transform == 'minmax':
                self.output_scaler = MinMaxScaler(feature_range=self.target_range)  # type: ignore
            elif self.output_transform == 'robust':
                self.output_scaler = RobustScaler()
            elif self.output_transform == 'none':
                self.output_scaler = None
            else:
                self.output_scaler = None
            
            if self.output_scaler:
                y_scaled = self.output_scaler.fit_transform(y.reshape(-1, 1))
            # If no scaler, keep original values
        
        return X_scaled, y_scaled
    
    def transform_input(self, X: np.ndarray) -> np.ndarray:
        """Transform input data using fitted scalers"""
        if not hasattr(self, 'input_transforms') or not self.input_scalers:
            return X
        
        X_scaled = X.copy()
        
        for i, (transform, scaler, offset) in enumerate(zip(self.input_transforms, self.input_scalers, self.input_log_offsets)):
            if scaler is not None:
                col_data = X_scaled[:, i:i+1]
                if transform == 'log':
                    # Apply log transform with offset
                    col_data_positive = np.maximum(col_data + offset, 1e-15)
                    col_data = np.log(col_data_positive)
                    col_data = scaler.transform(col_data)
                else:
                    col_data = scaler.transform(col_data)
                X_scaled[:, i:i+1] = col_data
        
        return X_scaled
    
    def transform_output(self, y: np.ndarray) -> np.ndarray:
        """Transform output data using fitted scaler"""
        if not hasattr(self, 'output_transform') or self.output_scaler is None:
            return y
        
        y_scaled = y.copy()
        
        try:
            if self.output_transform == 'log':
                # Apply log transform with offset
                y_positive = np.maximum(y_scaled + self.output_log_offset, 1e-15)
                y_scaled = np.log(y_positive)
                if self.output_scaler:
                    y_scaled = self.output_scaler.transform(y_scaled.reshape(-1, 1))
            elif self.output_transform in ['standard', 'minmax', 'robust']:
                if self.output_scaler:
                    y_scaled = self.output_scaler.transform(y.reshape(-1, 1))
            # If no scaling, keep original values
        except Exception as e:
            print(f"Warning: Output scaling failed: {e}")
            return y
        
        return y_scaled.reshape(y.shape)
    
    def inverse_transform_output(self, y: np.ndarray) -> np.ndarray:
        """Inverse transform output predictions to original scale"""
        if not hasattr(self, 'output_transform') or self.output_scaler is None:
            return y
        
        y_original = y.copy()
        
        try:
            if self.output_transform == 'log':
                # First inverse the standard scaler if applied
                if hasattr(self.output_scaler, 'inverse_transform'):
                    y_original = self.output_scaler.inverse_transform(y_original.reshape(-1, 1))
                else:
                    y_original = y_original.reshape(-1, 1)
                
                # Then inverse the log transform with safety bounds
                # Clip extreme values to prevent overflow
                y_original = np.clip(y_original, -50, 50)  # Prevent exp(50)+ values
                y_original = np.exp(y_original) - self.output_log_offset
                
                # Additional safety check for extreme results
                if np.any(np.abs(y_original) > 1e30):
                    y_original = np.where(
                        np.abs(y_original) > 1e30,
                        np.sign(y_original) * 1e30,
                        y_original
                    )
                    
            elif self.output_transform in ['standard', 'minmax', 'robust']:
                if hasattr(self.output_scaler, 'inverse_transform'):
                    y_original = self.output_scaler.inverse_transform(y_original.reshape(-1, 1))
            # If no scaling, keep original values
        except Exception as e:
            print(f"Warning: Output inverse scaling failed: {e}")
            return y
        
        return y_original.reshape(y.shape)
    
    def get_scaling_info(self) -> Dict[str, Any]:
        """Get information about applied scaling transformations"""
        return {
            'input_transforms': self.input_transforms,
            'output_transform': self.output_transform,
            'target_range': self.target_range,
            'input_log_offsets': self.input_log_offsets,
            'output_log_offset': self.output_log_offset
        }
    
    def get_scaling_transformation_expressions(self, n_inputs: int) -> Tuple[List[str], str]:
        """Get symbolic expressions for the scaling transformations"""
        input_expressions = []
        
        # Generate input transformation expressions
        for i in range(n_inputs):
            if i < len(self.input_transforms):
                transform = self.input_transforms[i]
                offset = self.input_log_offsets[i] if i < len(self.input_log_offsets) else 0.0
                
                if transform == 'log' and offset > 0:
                    if offset < 1e-6:
                        input_expressions.append(f"log(x{i} + {offset:.2e})")
                    else:
                        input_expressions.append(f"log(x{i} + {offset:.3f})")
                elif transform == 'log':
                    input_expressions.append(f"log(x{i})")
                elif transform == 'standard' and i < len(self.input_scalers) and self.input_scalers[i] is not None:
                    scaler = self.input_scalers[i]
                    if scaler is not None and hasattr(scaler, 'mean_') and hasattr(scaler, 'scale_'):
                        mean = scaler.mean_[0]
                        scale = scaler.scale_[0]
                        input_expressions.append(f"(x{i} - {mean:.3f}) / {scale:.3f}")
                    else:
                        input_expressions.append(f"x{i}")
                else:
                    input_expressions.append(f"x{i}")
            else:
                input_expressions.append(f"x{i}")
        
        # Generate output transformation expression
        if self.output_transform == 'log' and self.output_log_offset > 0:
            if self.output_log_offset < 1e-6:
                output_expression = f"log(y + {self.output_log_offset:.2e})"
            else:
                output_expression = f"log(y + {self.output_log_offset:.3f})"
        elif self.output_transform == 'log':
            output_expression = "log(y)"
        else:
            output_expression = "y'"
        
        return input_expressions, output_expression


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


# Physics Data Preprocessor placeholder (simplified version)
class PhysicsDataPreprocessor:
    """Simplified physics data preprocessor"""
    
    def __init__(self):
        self.data_scaler = DataScaler()
    
    def preprocess(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess physics data"""
        return self.data_scaler.fit_transform(X, y)


class PhysicsAwareScaler:
    """Alias for DataScaler with physics-aware defaults"""
    
    def __init__(self, **kwargs):
        self.scaler = DataScaler(**kwargs)
    
    def __getattr__(self, name):
        return getattr(self.scaler, name)
