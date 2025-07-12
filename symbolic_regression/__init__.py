# Python

"""Symbolic Regression Package

A genetic programming approach to symbolic regression for MIMO systems.
"""

from .expression_tree import (
  Expression, Node, VariableNode, ConstantNode,
  BinaryOpNode, UnaryOpNode
)
from .generator import ExpressionGenerator, BiasedExpressionGenerator
from .genetic_ops import GeneticOperations
from .mimo_regressor import MIMOSymbolicRegressor
from .ensemble_regressor import EnsembleMIMORegressor
from .population import (
  PopulationManager, generate_diverse_population_optimized, inject_diversity_optimized,
  evaluate_population_enhanced_optimized,
  generate_simple_combination_optimized, generate_high_diversity_expression_optimized,
  generate_targeted_diverse_expression_optimized, generate_complex_diverse_expression_optimized,
  generate_polynomial_expression_optimized, generate_mixed_expression_optimized,
  generate_constant_heavy_optimized
)
from .selection import enhanced_selection, diversity_selection, tournament_selection
from .utils import (
  string_similarity, calculate_expression_uniqueness,
  calculate_population_diversity
)
from .adaptive_evolution import update_adaptive_parameters, restart_population_enhanced
from .expression_utils import to_sympy_expression, optimize_constants
from .evolution_stats import get_evolution_stats, get_detailed_expressions

__version__ = "0.1.0"
__all__ = [
  "Expression", "Node", "VariableNode", "ConstantNode",
  "BinaryOpNode", "UnaryOpNode",
  "ExpressionGenerator", "BiasedExpressionGenerator",
  "GeneticOperations", "MIMOSymbolicRegressor", "EnsembleMIMORegressor",
  "PopulationManager", "generate_diverse_population_optimized", "inject_diversity_optimized",
  "evaluate_population_enhanced_optimized",
  "generate_simple_combination_optimized", "generate_high_diversity_expression_optimized",
  "generate_targeted_diverse_expression_optimized", "generate_complex_diverse_expression_optimized",
  "generate_polynomial_expression_optimized", "generate_mixed_expression_optimized",
  "generate_constant_heavy_optimized",
  "enhanced_selection", "diversity_selection", "tournament_selection",
  "string_similarity", "calculate_expression_uniqueness",
  "calculate_population_diversity",
  "update_adaptive_parameters", "restart_population_enhanced",
  "to_sympy_expression", "optimize_constants",
  "get_evolution_stats", "get_detailed_expressions"
]