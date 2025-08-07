"""
Adaptive parsimony coefficient system for symbolic regression.
Implements PySR-style adaptive complexity penalties and domain-specific operator weighting.
"""

import numpy as np
from typing import Dict, Optional


class AdaptiveParsimonySystem:
    """
    PySR-inspired adaptive parsimony system that adjusts complexity penalties
    based on generation progress and population diversity.
    """
    
    def __init__(self, base_coefficient: float = 0.003, domain_type: str = "general"):
        self.base_coefficient = base_coefficient
        self.domain_type = domain_type
        self.custom_weights = CustomOperatorWeights(domain_type)
        
    def get_adaptive_coefficient(self, generation: int, max_generations: int, 
                               population_diversity: float) -> float:
        """
        Calculate adaptive parsimony coefficient based on evolution progress.
        
        Args:
            generation: Current generation number
            max_generations: Maximum generations planned
            population_diversity: Current population diversity score (0-1)
            
        Returns:
            Adjusted parsimony coefficient
        """
        # Increase parsimony pressure over generations (but not too aggressively)
        generation_factor = 1.0 + 0.5 * (generation / max(max_generations, 1))
        
        # Reduce parsimony when diversity is low to encourage exploration
        diversity_factor = max(0.5, population_diversity)
        
        # Apply non-linear scaling to prevent extreme values
        adaptive_coeff = self.base_coefficient * generation_factor * diversity_factor
        
        # Clamp to reasonable bounds
        return np.clip(adaptive_coeff, 0.0001, 0.02)
    
    def get_domain_weights(self) -> Dict[str, float]:
        """Get domain-specific operator weights."""
        return self.custom_weights.get_weights(self.domain_type)


class CustomOperatorWeights:
    """
    Domain-specific operator weighting system for different application areas.
    """
    
    def __init__(self, domain_type: str = "general"):
        self.domain_weights = {
            "physics": {
                'reciprocal': 0.8,    # Very important for physics (1/r, 1/t)
                'inv_square': 0.9,    # Inverse square laws
                'sqrt': 0.9,          # Square roots common (energy relations)
                'square': 0.8,        # Very common (kinetic energy, etc.)
                '^': 0.9,             # Power laws important
                'sin': 0.9, 'cos': 0.9,  # Oscillations, waves
                '/': 0.9,             # Ratios, rates common
                'exp': 1.0,           # Decay/growth processes
                'log': 1.0,           # Scaling relationships
            },
            "engineering": {
                'exp': 0.9,           # Exponential decay/growth
                'log': 0.9,           # Logarithmic scales
                '/': 0.85,            # Ratios important
                '*': 0.9,             # Products common
                '^': 1.0,             # Power relationships
                'sqrt': 0.95,         # Common in engineering formulas
                'sin': 1.0, 'cos': 1.0,  # Periodic behavior
            },
            "biology": {
                'exp': 0.85,          # Growth/decay very common
                'log': 0.9,           # Scaling laws (allometry)
                'sqrt': 0.9,          # Allometric relationships
                '^': 0.95,            # Power laws in biology
                '/': 0.9,             # Ratios (concentration, etc.)
                'reciprocal': 1.0,    # Less common than in physics
            },
            "finance": {
                'exp': 0.85,          # Compound interest, growth
                'log': 0.9,           # Log returns
                '*': 0.9,             # Product calculations
                '/': 0.85,            # Ratios very common
                '^': 1.0,             # Power relationships
                'sqrt': 1.0,          # Volatility measures
            },
            "general": {}  # No adjustments for general case
        }
        
    def get_weights(self, domain_type: str) -> Dict[str, float]:
        """
        Get adjusted complexity weights for specific domain.
        
        Args:
            domain_type: Domain type ("physics", "engineering", "biology", "finance", "general")
            
        Returns:
            Dictionary of operator weights (multipliers for base complexity)
        """
        # Import here to avoid circular imports
        try:
            from .expression_tree.core.node import COMPLEXITY_WEIGHTS
        except ImportError:
            # Fallback default weights if import fails
            COMPLEXITY_WEIGHTS = {
                '+': 1.0, '-': 1.0, '*': 1.1, '/': 1.5, '^': 2.0,
                'sin': 1.2, 'cos': 1.2, 'tan': 1.6, 'exp': 1.8, 'log': 1.6,
                'sqrt': 1.2, 'abs': 1.05, 'neg': 1.0, 'square': 1.0, 'cube': 1.1,
                'reciprocal': 1.3, 'inv_square': 1.5, 'variable': 1.0, 'constant': 1.0
            }
        
        base_weights = COMPLEXITY_WEIGHTS.copy()
        domain_adjustments = self.domain_weights.get(domain_type, {})
        
        # Apply domain-specific multipliers
        for op, multiplier in domain_adjustments.items():
            if op in base_weights:
                base_weights[op] *= multiplier
                
        return base_weights


class PySRStyleComplexity:
    """
    PySR-style complexity calculation with depth weighting and expression size normalization.
    """
    
    @staticmethod
    def calculate_complexity(expression, domain_weights: Optional[Dict[str, float]] = None) -> float:
        """
        Calculate PySR-style complexity with depth consideration.
        
        Args:
            expression: Expression object to evaluate
            domain_weights: Optional domain-specific weight adjustments
            
        Returns:
            Adjusted complexity score
        """
        from .expression_tree.utils.tree_utils import calculate_tree_depth
        
        # Base complexity from node weights
        base_complexity = expression.complexity()
        
        # Apply domain weights if provided
        if domain_weights:
            base_complexity = PySRStyleComplexity._apply_domain_weights(
                expression, domain_weights, base_complexity
            )
        
        # PySR-style depth penalty (logarithmic scaling)
        try:
            from .expression_tree.utils.tree_utils import calculate_tree_depth
            depth = calculate_tree_depth(expression.root)
        except ImportError:
            # Fallback if tree_utils not available
            depth = expression.size() / 5.0  # Rough approximation
            
        depth_penalty = 0.1 * np.log(1 + depth)
        
        # Size normalization (prevents bias toward very large expressions)
        size = expression.size()
        size_penalty = 0.05 * np.log(1 + size)
        
        return base_complexity + depth_penalty + size_penalty
    
    @staticmethod
    def _apply_domain_weights(expression, domain_weights: Dict[str, float], 
                            base_complexity: float) -> float:
        """Apply domain-specific weights to complexity calculation."""
        # This is a simplified approach - could be made more sophisticated
        # by walking the tree and applying weights per operator
        return base_complexity
    
    @staticmethod
    def get_parsimony_penalty(expression, parsimony_coefficient: float, 
                            domain_weights: Optional[Dict[str, float]] = None) -> float:
        """
        Calculate PySR-style parsimony penalty.
        
        Args:
            expression: Expression to evaluate
            parsimony_coefficient: Base parsimony coefficient
            domain_weights: Optional domain-specific weights
            
        Returns:
            Parsimony penalty value
        """
        complexity = PySRStyleComplexity.calculate_complexity(expression, domain_weights)
        return parsimony_coefficient * complexity
