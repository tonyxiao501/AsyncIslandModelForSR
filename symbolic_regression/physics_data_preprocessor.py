"""
Physics-Specific Data Preprocessing Module

This module provides specialized preprocessing for extreme-scale physics problems,
including automatic detection of physical constants and intelligent scaling strategies.
"""

import numpy as np
from typing import Tuple, Dict, List, Optional
import warnings


class PhysicsDataPreprocessor:
    """Specialized data preprocessor for physics problems with extreme scales"""
    
    def __init__(self, 
                 aggressive_log_scaling: bool = True,
                 constant_detection_threshold: float = 1e-10,
                 large_value_threshold: float = 1e10):
        """
        Initialize the physics data preprocessor.
        
        Args:
            aggressive_log_scaling: Use more aggressive log scaling for extreme values
            constant_detection_threshold: Threshold for detecting physical constants
            large_value_threshold: Threshold for detecting very large values
        """
        self.aggressive_log_scaling = aggressive_log_scaling
        self.constant_detection_threshold = constant_detection_threshold
        self.large_value_threshold = large_value_threshold
        
        # Store transformation metadata
        self.scaling_info: Dict = {}
        
    def detect_physics_scaling_needs(self, X: np.ndarray, y: np.ndarray) -> Dict[str, str]:
        """
        Analyze data to detect physics-specific scaling needs.
        
        Returns:
            Dictionary with recommended scaling methods for inputs and output
        """
        scaling_recommendations = {
            'input_scalings': [],
            'output_scaling': 'standard',
            'detected_patterns': []
        }
        
        # Analyze each input feature
        for i in range(X.shape[1]):
            feature = X[:, i]
            scaling_rec = self._analyze_feature_for_physics(feature, f"X{i}")
            scaling_recommendations['input_scalings'].append(scaling_rec)
            
        # Analyze output
        output_rec = self._analyze_feature_for_physics(y.flatten(), "y")
        scaling_recommendations['output_scaling'] = output_rec['scaling']
        
        return scaling_recommendations
    
    def _analyze_feature_for_physics(self, data: np.ndarray, name: str) -> Dict:
        """Analyze a single feature for physics-specific patterns"""
        analysis = {
            'name': name,
            'scaling': 'standard',
            'patterns_detected': [],
            'statistics': {}
        }
        
        # Basic statistics
        data_nonzero = data[data != 0]
        if len(data_nonzero) == 0:
            return analysis
            
        min_val = np.min(np.abs(data_nonzero))
        max_val = np.max(np.abs(data_nonzero))
        mean_val = np.mean(data)
        std_val = np.std(data)
        
        analysis['statistics'] = {
            'min_abs': min_val,
            'max_abs': max_val,
            'mean': mean_val,
            'std': std_val,
            'magnitude_range': np.log10(max_val) - np.log10(min_val),
            'has_negative': np.any(data < 0)
        }
        
        # Physics pattern detection
        magnitude_range = analysis['statistics']['magnitude_range']
        
        # Detect physical constants (very small values)
        if min_val < self.constant_detection_threshold:
            analysis['patterns_detected'].append('physical_constant')
            
        # Detect astronomical scales (very large values)  
        if max_val > self.large_value_threshold:
            analysis['patterns_detected'].append('astronomical_scale')
            
        # Detect inverse square law patterns (1/r²)
        if magnitude_range > 8 and not analysis['statistics']['has_negative']:
            analysis['patterns_detected'].append('inverse_power_law')
            
        # Recommend scaling based on patterns
        if magnitude_range > 6 or 'physical_constant' in analysis['patterns_detected']:
            if not analysis['statistics']['has_negative'] and np.all(data > 0):
                analysis['scaling'] = 'log'
            else:
                analysis['scaling'] = 'robust'
        elif magnitude_range > 3:
            analysis['scaling'] = 'robust'
        elif abs(mean_val) > 5 * std_val:
            analysis['scaling'] = 'standard'
        else:
            analysis['scaling'] = 'minmax'
            
        return analysis
    
    def create_physics_aware_scaler(self, X: np.ndarray, y: np.ndarray) -> 'PhysicsAwareScaler':
        """Create a physics-aware scaler based on data analysis"""
        recommendations = self.detect_physics_scaling_needs(X, y)
        
        return PhysicsAwareScaler(
            input_scalings=recommendations['input_scalings'],
            output_scaling=recommendations['output_scaling'],
            target_range=(-6.0, 6.0)  # Wider range for extreme physics scales
        )


class PhysicsAwareScaler:
    """
    Enhanced scaler specifically designed for physics problems with extreme scales.
    """
    
    def __init__(self, 
                 input_scalings: List[Dict],
                 output_scaling: str,
                 target_range: Tuple[float, float] = (-6.0, 6.0)):
        self.input_scalings = input_scalings
        self.output_scaling = output_scaling
        self.target_range = target_range
        
        # Will be populated during fit
        self.input_transformers = []
        self.output_transformer = None
        self.is_fitted = False
        
    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Fit the scaler and transform the data"""
        from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
        
        X_scaled = X.copy()
        y_scaled = y.copy()
        
        # Transform inputs
        for i, scaling_info in enumerate(self.input_scalings):
            feature = X[:, i]
            
            if scaling_info['scaling'] == 'log':
                # Apply log transformation for extreme scales
                if np.any(feature <= 0):
                    offset = abs(np.min(feature)) + 1e-12
                    feature_log = np.log(feature + offset)
                else:
                    offset = 0
                    feature_log = np.log(feature)
                
                # Then apply standard scaling to log-transformed data
                scaler = StandardScaler()
                X_scaled[:, i] = scaler.fit_transform(feature_log.reshape(-1, 1)).flatten()
                self.input_transformers.append({'type': 'log', 'scaler': scaler, 'offset': offset})
                
            elif scaling_info['scaling'] == 'robust':
                scaler = RobustScaler()
                X_scaled[:, i] = scaler.fit_transform(feature.reshape(-1, 1)).flatten()
                self.input_transformers.append({'type': 'robust', 'scaler': scaler})
                
            elif scaling_info['scaling'] == 'minmax':
                scaler = MinMaxScaler(feature_range=self.target_range)
                X_scaled[:, i] = scaler.fit_transform(feature.reshape(-1, 1)).flatten()
                self.input_transformers.append({'type': 'minmax', 'scaler': scaler})
                
            else:  # standard
                scaler = StandardScaler()
                X_scaled[:, i] = scaler.fit_transform(feature.reshape(-1, 1)).flatten()
                self.input_transformers.append({'type': 'standard', 'scaler': scaler})
        
        # Transform output
        output_data = y.flatten()
        
        if self.output_scaling == 'log' and np.all(output_data > 0):
            output_log = np.log(output_data)
            self.output_transformer = {'type': 'log', 'scaler': StandardScaler()}
            y_scaled = self.output_transformer['scaler'].fit_transform(output_log.reshape(-1, 1))
        elif self.output_scaling == 'robust':
            self.output_transformer = {'type': 'robust', 'scaler': RobustScaler()}
            y_scaled = self.output_transformer['scaler'].fit_transform(output_data.reshape(-1, 1))
        elif self.output_scaling == 'minmax':
            self.output_transformer = {'type': 'minmax', 'scaler': MinMaxScaler(feature_range=self.target_range)}
            y_scaled = self.output_transformer['scaler'].fit_transform(output_data.reshape(-1, 1))
        else:  # standard
            self.output_transformer = {'type': 'standard', 'scaler': StandardScaler()}
            y_scaled = self.output_transformer['scaler'].fit_transform(output_data.reshape(-1, 1))
        
        self.is_fitted = True
        return X_scaled, y_scaled
    
    def get_scaled_expression_with_indicators(self, expression_str: str, n_inputs: int) -> str:
        """
        Add scaling indicators to expression variables instead of symbolic transformation.
        
        This avoids the complex symbolic transformation that can explode with nested functions.
        """
        if not self.is_fitted:
            warnings.warn("Scaler not fitted. Cannot add scaling indicators.")
            return expression_str
            
        result = expression_str
        
        # Add scaling indicators to each variable
        for i in range(n_inputs):
            if i < len(self.input_transformers):
                transformer = self.input_transformers[i]
                transform_type = transformer.get('type', 'none')
                
                if transform_type == 'log':
                    offset = transformer.get('offset', 0)
                    if offset > 0:
                        indicator = f"X{i}_log(+{offset:.2e})"
                    else:
                        indicator = f"X{i}_log"
                elif transform_type == 'standard':
                    indicator = f"X{i}_std"
                elif transform_type == 'minmax':
                    indicator = f"X{i}_minmax[{self.target_range[0]:.1f},{self.target_range[1]:.1f}]"
                elif transform_type == 'robust':
                    indicator = f"X{i}_robust"
                else:
                    indicator = f"X{i}_raw"
            else:
                indicator = f"X{i}_raw"
            
            # Replace X{i} with the indicator
            result = result.replace(f'X{i}', indicator)
        
        # Add output scaling indicator
        output_indicator = ""
        if self.output_transformer is not None:
            transform_type = self.output_transformer.get('type', 'none')
            if transform_type == 'log':
                output_indicator = f" → Y_log"
            elif transform_type == 'standard':
                output_indicator = f" → Y_std"
            elif transform_type == 'minmax':
                output_indicator = f" → Y_minmax[{self.target_range[0]:.1f},{self.target_range[1]:.1f}]"
            elif transform_type == 'robust':
                output_indicator = f" → Y_robust"
            else:
                output_indicator = f" → Y_raw"
        else:
            output_indicator = f" → Y_raw"
        
        return result + output_indicator
    
    def get_scaling_summary(self) -> Dict:
        """Get a summary of the scaling transformations applied"""
        return {
            'input_transformations': [t.get('type', 'none') for t in self.input_transformers],
            'output_transformation': self.output_transformer.get('type', 'none') if self.output_transformer else 'none',
            'target_range': self.target_range,
            'is_fitted': self.is_fitted
        }
