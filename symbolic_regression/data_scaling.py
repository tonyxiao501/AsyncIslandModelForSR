"""
Data Scaling and Transformation Module for Symbolic Regression

This             # Enhanced thresholds for extreme physics scales
            if magnitude_range > 4:  # Reduced threshold for earlier log detection
                if not has_negative and np.all(data > 0):
                    return 'log'
                else:
                    return 'robust'  # Use robust for extreme ranges with negatives
                    
            # Detect very small constants (< 1e-6) or very large values (> 1e6)
            if min_magnitude < -6 or max_magnitude > 6:
                if not has_negative and np.all(data > 0):
                    return 'log'
                else:
                    return 'robust'
        else:
            magnitude_range = 0
        
        # Check for skewed distributions
        if data_range > 10 * data_std:  # Highly skewed
            return 'robust'
        
        # For moderate ranges, use standard scaling
        if abs(data_mean) > 3 * data_std:  # Mean far from zero
            return 'standard'
        
        # Default to minmax for bounded, well-behaved data
        return 'minmax' comprehensive data preprocessing and postprocessing
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
        Automatically detect the optimal scaling method for the data.
        Enhanced for extreme-scale physics problems.
        
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
        
        # Enhanced magnitude range calculation for extreme scales
        nonzero_data = np.abs(data[data != 0])
        if len(nonzero_data) > 0:
            magnitude_range = np.log10(nonzero_data.max()) - np.log10(nonzero_data.min())
            
            # Check for very small values (e.g., physical constants)
            min_magnitude = np.log10(nonzero_data.min())
            max_magnitude = np.log10(nonzero_data.max())
            
            # Enhanced thresholds for extreme physics scales
            if magnitude_range > 4:  # Reduced threshold for earlier log detection
                if not has_negative and np.all(data > 0):
                    return 'log'
                else:
                    return 'robust'  # Use robust for extreme ranges with negatives
                    
            # Detect very small constants (< 1e-6) or very large values (> 1e6)
            if min_magnitude < -6 or max_magnitude > 6:
                if not has_negative and np.all(data > 0):
                    return 'log'
                else:
                    return 'robust'
        else:
            magnitude_range = 0
        
        # Check for skewed distributions
        if data_range > 10 * data_std:  # Highly skewed
            return 'robust'
        
        # For moderate ranges, use standard scaling
        if abs(data_mean) > 3 * data_std:  # Mean far from zero
            return 'standard'
        
        # Default to minmax for bounded, well-behaved data
        return 'minmax'
    
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
                else:  # 'none'
                    scaler = None
                
                if scaler:
                    X_scaled[:, i] = scaler.fit_transform(feature.reshape(-1, 1)).flatten()
                else:
                    pass  # No scaling
            
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
            else:  # 'none'
                self.output_scaler = None
            
            if self.output_scaler:
                y_scaled = self.output_scaler.fit_transform(y.reshape(-1, 1))
        
        return X_scaled, y_scaled
    
    def transform_input(self, X: np.ndarray) -> np.ndarray:
        """Transform input data using fitted scalers"""
        if not hasattr(self, 'input_transforms') or self.input_scalers is None:
            return X
        
        X_scaled = X.copy()
        
        for i, (transform, scaler) in enumerate(zip(self.input_transforms, self.input_scalers)):
            if scaler is not None:
                col_data = X_scaled[:, i:i+1]
                if transform == 'log':
                    # Apply log transform with offset
                    offset = getattr(self, f'input_log_offset_{i}', 0.0)
                    col_data_positive = np.maximum(col_data + offset, 1e-15)
                    col_data = np.log10(col_data_positive)
                    if hasattr(scaler, 'transform'):
                        col_data = scaler.transform(col_data)
                else:
                    col_data = scaler.transform(col_data)
                X_scaled[:, i:i+1] = col_data
        
        return X_scaled
    
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
                offset = getattr(self, 'output_log_offset', 0.0)
                
                # Clip extreme values to prevent overflow
                y_original = np.clip(y_original, -50, 50)  # Prevent 10^50+ values
                
                y_original = np.power(10, y_original) - offset
                
                # Additional safety check for extreme results
                if np.any(np.abs(y_original) > 1e30):
                    # If we get extreme values, fall back to a safer approach
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
    
    def get_scaled_expression_with_indicators(self, expression_str: str, n_inputs: int) -> str:
        """
        Add scaling indicators to expression variables instead of symbolic transformation.
        
        Args:
            expression_str: Expression in terms of scaled variables
            n_inputs: Number of input variables
            
        Returns:
            Expression with scaling indicators on variables
        """
        result = expression_str
        
        # Add scaling indicators to each variable
        for i in range(n_inputs):
            if i < len(self.input_transforms):
                transform = self.input_transforms[i]
                
                if transform == 'log':
                    offset = self.input_log_offsets[i]
                    if offset > 0:
                        indicator = f"X{i}_log(+{offset:.2e})"
                    else:
                        indicator = f"X{i}_log"
                elif transform == 'standard':
                    indicator = f"X{i}_std"
                elif transform == 'minmax':
                    indicator = f"X{i}_minmax[{self.target_range[0]:.1f},{self.target_range[1]:.1f}]"
                elif transform == 'robust':
                    indicator = f"X{i}_robust"
                else:
                    indicator = f"X{i}_raw"
                
                # Replace X{i} with the indicator
                result = result.replace(f'X{i}', indicator)
        
        # Add output scaling indicator
        output_indicator = ""
        if self.output_transform == 'log':
            if self.output_log_offset > 0:
                output_indicator = f" → Y_log(+{self.output_log_offset:.2e})"
            else:
                output_indicator = f" → Y_log"
        elif self.output_transform == 'standard':
            output_indicator = f" → Y_std"
        elif self.output_transform == 'minmax':
            output_indicator = f" → Y_minmax[{self.target_range[0]:.1f},{self.target_range[1]:.1f}]"
        elif self.output_transform == 'robust':
            output_indicator = f" → Y_robust"
        else:
            output_indicator = f" → Y_raw"
        
        return result + output_indicator
    
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
