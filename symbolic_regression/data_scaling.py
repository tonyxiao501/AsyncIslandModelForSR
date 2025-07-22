"""
Data Scaling and Transformation Module for Symbolic Regression

This module provides comprehensive data preprocessing and postprocessing
capabilities to improve symbolic regression performance on multi-scale problems.
"""

import numpy as np
from typing import Tuple, Dict, Optional, Callable, List, Union, Any
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
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
                        np.sign(y_original) * 1e30,  # Cap at 1e30
                        y_original
                    )
                    
            elif self.output_transform in ['standard', 'minmax', 'robust']:
                y_original = self.output_scaler.inverse_transform(y_original.reshape(-1, 1))
                
        except Exception as e:
            # If inverse transformation fails, return clipped input
            print(f"Warning: Inverse transformation failed: {e}")
            return np.clip(y, -1e10, 1e10)
        
        return y_original.reshape(y.shape)
    
    def get_scaling_transformation_expressions(self, n_inputs: int) -> Tuple[List[str], str]:
        """
        Get clean mathematical expressions for the scaling transformations.
        
        Args:
            n_inputs: Number of input variables
            
        Returns:
            Tuple of (input_transform_expressions, output_transform_expression)
        """
        input_expressions = []
        
        # Generate input transformation expressions
        for i in range(n_inputs):
            if i < len(self.input_transforms):
                transform = self.input_transforms[i]
                
                if transform == 'log':
                    offset = self.input_log_offsets[i]
                    if offset > 0:
                        if offset < 1e-6:
                            input_expressions.append(f"log(x{i} + {offset:.2e})")
                        else:
                            input_expressions.append(f"log(x{i} + {offset:.6f})")
                    else:
                        input_expressions.append(f"log(x{i})")
                elif transform == 'standard':
                    if i < len(self.input_scalers) and self.input_scalers[i] is not None:
                        scaler = self.input_scalers[i]
                        mean = scaler.mean_[0]  # type: ignore
                        scale = scaler.scale_[0]  # type: ignore
                        input_expressions.append(f"(x{i} - {mean:.6f}) / {scale:.6f}")
                    else:
                        input_expressions.append(f"x{i}")
                elif transform == 'minmax':
                    if i < len(self.input_scalers) and self.input_scalers[i] is not None:
                        scaler = self.input_scalers[i]
                        data_min = scaler.data_min_[0]  # type: ignore
                        data_range = scaler.data_range_[0]  # type: ignore
                        min_range, max_range = self.target_range
                        input_expressions.append(f"{min_range:.1f} + {max_range - min_range:.1f} * (x{i} - {data_min:.6f}) / {data_range:.6f}")
                    else:
                        input_expressions.append(f"x{i}")
                elif transform == 'robust':
                    if i < len(self.input_scalers) and self.input_scalers[i] is not None:
                        scaler = self.input_scalers[i]
                        center = scaler.center_[0]  # type: ignore
                        scale = scaler.scale_[0]  # type: ignore
                        input_expressions.append(f"(x{i} - {center:.6f}) / {scale:.6f}")
                    else:
                        input_expressions.append(f"x{i}")
                else:  # 'none' or unknown
                    input_expressions.append(f"x{i}")
            else:
                input_expressions.append(f"x{i}")
        
        # Generate output transformation expression
        if self.output_transform == 'log':
            if self.output_log_offset > 0:
                if self.output_log_offset < 1e-6:
                    output_expr = f"exp(y_scaled) - {self.output_log_offset:.2e}"
                else:
                    output_expr = f"exp(y_scaled) - {self.output_log_offset:.6f}"
            else:
                output_expr = "exp(y_scaled)"
        elif self.output_transform == 'standard':
            if self.output_scaler is not None:
                mean = self.output_scaler.mean_[0]  # type: ignore
                scale = self.output_scaler.scale_[0]  # type: ignore
                output_expr = f"y_scaled * {scale:.6f} + {mean:.6f}"
            else:
                output_expr = "y_scaled"
        elif self.output_transform == 'minmax':
            if self.output_scaler is not None:
                data_min = self.output_scaler.data_min_[0]  # type: ignore
                data_range = self.output_scaler.data_range_[0]  # type: ignore
                min_range, max_range = self.target_range
                output_expr = f"(y_scaled - {min_range:.1f}) * {data_range:.6f} / {max_range - min_range:.1f} + {data_min:.6f}"
            else:
                output_expr = "y_scaled"
        elif self.output_transform == 'robust':
            if self.output_scaler is not None:
                center = self.output_scaler.center_[0]  # type: ignore
                scale = self.output_scaler.scale_[0]  # type: ignore
                output_expr = f"y_scaled * {scale:.6f} + {center:.6f}"
            else:
                output_expr = "y_scaled"
        else:  # 'none' or unknown
            output_expr = "y_scaled"
        
        return input_expressions, output_expr

    def get_scaling_info(self) -> Dict:
        """Get information about the applied scaling transformations."""
        return {
            'input_transforms': self.input_transforms,
            'output_transform': self.output_transform,
            'input_log_offsets': self.input_log_offsets,
            'output_log_offset': self.output_log_offset,
            'original_X_stats': self.original_X_stats,
            'original_y_stats': self.original_y_stats,
            'target_range': self.target_range
        }
