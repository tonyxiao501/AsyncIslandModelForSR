"""
Genetic Operations Module - Backward Compatibility

This module provides backward compatibility by re-exporting the modular genetic operations.
The implementation has been refactored into separate modules for better organization.

For new code, consider importing directly from the genetic_ops submodules:
- genetic_ops.GeneticOperations: Main interface
- genetic_ops.MutationStrategies: Mutation strategies
- genetic_ops.CrossoverOperations: Crossover operations  
- genetic_ops.ExpressionContextAnalyzer: Context analysis
- genetic_ops.DiversityMetrics: Diversity measurements
"""

# Re-export the main class for backward compatibility
from .genetic_ops import GeneticOperations

# Also export component classes for advanced usage
from .genetic_ops import (
    MutationStrategies,
    CrossoverOperations, 
    ExpressionContextAnalyzer,
    DiversityMetrics
)

__all__ = [
    'GeneticOperations',
    'MutationStrategies',
    'CrossoverOperations', 
    'ExpressionContextAnalyzer',
    'DiversityMetrics'
]