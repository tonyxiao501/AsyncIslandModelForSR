import random
from typing import List, Optional
from .expression_tree import Expression, Node, VariableNode, ConstantNode, BinaryOpNode, UnaryOpNode, ScalingOpNode
from .expression_tree.utils.simplifier import ExpressionSimplifier
import numpy as np


class ExpressionGenerator:
  """Enhanced expression generator with complexity control"""

  def __init__(self, n_inputs: int, max_depth: int = 6, scaling_range: int = 3):
    self.n_inputs = n_inputs
    self.max_depth = max_depth
    self.binary_ops = ['+', '-', '*', '/', '^']  # Re-added power operations for physics
    self.scaling_range = scaling_range
    # Enhanced unary operations for physics laws
    self.unary_ops = [
        'sin', 'cos', 'tan',           # Trigonometric functions
        'exp', 'log',                  # Exponential and logarithmic
        'sqrt',                        # Square root
        'abs',                         # Absolute value
        'square', 'cube',              # Basic powers
        'reciprocal',                  # 1/x - critical for inverse laws
        'neg',                         # -x (unary minus)
        'sqrt_abs',                    # sqrt(|x|) - for safe square roots
        'log_abs',                     # log(|x|) - for safe logarithms
        'inv_square',                  # 1/x^2 - direct inverse square law
        'cbrt',                        # cube root (x^(1/3))
        'fourth_root',                 # fourth root (x^(1/4))
        'sinh', 'cosh', 'tanh'         # Hyperbolic functions for advanced physics
    ]
    
    # Physics-biased operator weights
    self.physics_unary_weights = {
        'reciprocal': 3.0,      # Critical for physics laws
        'inv_square': 3.0,      # Critical for inverse square laws
        'sqrt': 2.5,            # Important for many physics relations
        'square': 2.5,          # Important for energy laws
        'log': 2.0,             # Important for exponential phenomena
        'exp': 2.0,             # Important for decay/growth
        'abs': 2.0,             # Safe operations
        'sqrt_abs': 2.0,        # Safe square root
        'log_abs': 2.0,         # Safe logarithm
        'sin': 1.5, 'cos': 1.5, 'tan': 1.0,  # Trig functions
        'cube': 1.5, 'cbrt': 1.5,    # Cubic operations
        'neg': 1.5,             # Sign changes
        'fourth_root': 1.0,     # Less common roots
        'sinh': 1.0, 'cosh': 1.0, 'tanh': 1.0  # Hyperbolic functions
    }

  def generate_random_expression(self, depth: int = 0, max_depth: Optional[int] = None) -> Node:
    """Generate a validated random expression"""
    if max_depth is None:
      max_depth = self.max_depth

    max_attempts = 20

    for _ in range(max_attempts):
      node = self._generate_node(depth, max_depth)
      simplified = ExpressionSimplifier.simplify_expression(node)
      if simplified:
        return simplified

    # Fallback to simple terminal node
    if random.random() < 0.7:
      return VariableNode(random.randint(0, self.n_inputs - 1))
    return ConstantNode(random.uniform(-3, 3))

  def _generate_node(self, depth: int, max_depth: int) -> Node:
    """Internal node generation with stricter depth control"""
    # More aggressive terminal probability
    terminal_prob = 0.3 + (depth / max_depth) * 0.7

    if depth >= max_depth or random.random() < terminal_prob:
      # Terminal nodes with expanded constant range for physics
      if random.random() < 0.65:  # Favor variables
        return VariableNode(random.randint(0, self.n_inputs - 1))
      else:
        # Enhanced constant generation for physics
        if random.random() < 0.3:  # 30% chance for common physics constants
          physics_constants = [
            0.5, 1.0, 2.0, 3.0, 4.0, 5.0,     # Common small integers
            0.1, 0.2, 0.25, 0.33, 0.67,       # Common fractions
            1.41, 1.73, 2.72, 3.14,           # sqrt(2), sqrt(3), e, pi approximations
            -1.0, -0.5, -2.0                   # Common negative values
          ]
          return ConstantNode(random.choice(physics_constants))
        else:
          # Random constants with physics-appropriate range
          return ConstantNode(random.uniform(-5, 5))  # Most common range

    # Function nodes with reduced complexity
    if depth < max_depth - 1:
      node_type = random.choices(['binary', 'unary', 'scale'], weights=[0.7, 0.2, 0.1])[0]

      if node_type == 'binary':
        op = random.choice(self.binary_ops)
        left = self._generate_node(depth + 1, max_depth)
        right = self._generate_node(depth + 1, max_depth)
        return BinaryOpNode(op, left, right)
      
      elif node_type == 'unary':
        # Use physics-biased unary operations
        if depth < max_depth - 2:
          # Choose unary operator with physics bias
          if hasattr(self, 'physics_unary_weights'):
            weights = [self.physics_unary_weights.get(op, 1.0) for op in self.unary_ops]
            op = random.choices(self.unary_ops, weights=weights)[0]
          else:
            op = random.choice(self.unary_ops)
          operand = self._generate_node(depth + 1, max_depth)
          return UnaryOpNode(op, operand)
          
      elif node_type == 'scale':
        if depth < max_depth - 2:
            power = random.randint(-self.scaling_range, self.scaling_range)
            operand = self._generate_node(depth + 1, max_depth)
            return ScalingOpNode(power, operand)

    # Fallback to terminal with enhanced physics constants (ExpressionGenerator)
    if random.random() < 0.65:
      return VariableNode(random.randint(0, self.n_inputs - 1))
    else:
      # Enhanced constant generation for physics (ExpressionGenerator fallback)
      if random.random() < 0.3:  # 30% chance for common physics constants
        physics_constants = [
          0.5, 1.0, 2.0, 3.0, 4.0, 5.0,     # Common small integers
          0.1, 0.2, 0.25, 0.33, 0.67,       # Common fractions
          1.41, 1.73, 2.72, 3.14,           # sqrt(2), sqrt(3), e, pi approximations
          -1.0, -0.5, -2.0                   # Common negative values
        ]
        return ConstantNode(random.choice(physics_constants))
      else:
        # Random constants with physics-appropriate range
        return ConstantNode(random.uniform(-5, 5))  # Most common range

  def generate_population(self, population_size: int) -> List[Expression]:
    """Generate a validated population"""
    population = []
    max_attempts = population_size * 3

    for _ in range(max_attempts):
      if len(population) >= population_size:
        break

      expr = Expression(self.generate_random_expression())
      if expr.size() <= 20:  # Complexity limit
        population.append(expr)

    # Fill remaining slots with simple expressions if needed
    while len(population) < population_size:
      if random.random() < 0.8:
        node = VariableNode(random.randint(0, self.n_inputs - 1))
      else:
        node = ConstantNode(random.uniform(-2, 2))
      population.append(Expression(node))

    return population


class BiasedExpressionGenerator(ExpressionGenerator):
  """Expression generator with biased operator selection based on diversity needs"""

  def __init__(self, n_inputs: int, max_depth: int = 6, operator_rates: Optional[dict] = None):
    super().__init__(n_inputs, max_depth)
    self.operator_rates = operator_rates or {}

    # Create weighted operator lists
    self.weighted_binary_ops = self._create_weighted_operator_list()
    
    # Inherit physics unary weights from parent
    if not hasattr(self, 'physics_unary_weights'):
      self.physics_unary_weights = {
          'reciprocal': 3.0, 'inv_square': 3.0, 'sqrt': 2.5, 'square': 2.5,
          'log': 2.0, 'exp': 2.0, 'abs': 2.0, 'sqrt_abs': 2.0, 'log_abs': 2.0,
          'sin': 1.5, 'cos': 1.5, 'tan': 1.0, 'cube': 1.5, 'cbrt': 1.5,
          'neg': 1.5, 'fourth_root': 1.0, 'sinh': 1.0, 'cosh': 1.0, 'tanh': 1.0
      }

  def _create_weighted_operator_list(self) -> List[str]:
    """Create weighted list of binary operators based on generation rates"""
    weighted_ops = []

    for op in self.binary_ops:
      # Get rate for this operator (default to equal if not specified)
      rate = self.operator_rates.get(op, 0.2)
      # Convert rate to count (multiply by 100 for reasonable granularity)
      count = max(1, int(rate * 100))
      weighted_ops.extend([op] * count)

    return weighted_ops if weighted_ops else self.binary_ops

  def generate_biased_expression(self, depth: int = 0, max_depth: Optional[int] = None) -> Node:
    """Generate expression with biased operator selection"""
    if max_depth is None:
      max_depth = self.max_depth

    max_attempts = 20

    for _ in range(max_attempts):
      node = self._generate_biased_node(depth, max_depth)
      simplified = ExpressionSimplifier.simplify_expression(node)
      if simplified:
        return simplified

    # Fallback
    if random.random() < 0.7:
      return VariableNode(random.randint(0, self.n_inputs - 1))
    return ConstantNode(random.uniform(-3, 3))

  def _generate_biased_node(self, depth: int, max_depth: int) -> Node:
    """Generate node with biased operator selection"""
    # More aggressive terminal probability
    terminal_prob = 0.3 + (depth / max_depth) * 0.7

    if depth >= max_depth or random.random() < terminal_prob:
      # Terminal nodes
      if random.random() < 0.65:  # Favor variables
        return VariableNode(random.randint(0, self.n_inputs - 1))
      else:
        return ConstantNode(random.uniform(-3, 3))

    # Function nodes with biased operator selection
    if depth < max_depth - 1:
      node_type = random.choices(['binary', 'unary', 'scale'], weights=[0.7, 0.25, 0.05])[0]

      if node_type == 'binary':
        # Use weighted operator selection
        op = random.choice(self.weighted_binary_ops)
        left = self._generate_biased_node(depth + 1, max_depth)
        right = self._generate_biased_node(depth + 1, max_depth)
        return BinaryOpNode(op, left, right)
      
      elif node_type == 'unary':
        # Unary operations with physics bias
        if depth < max_depth - 2:
          # Choose unary operator with physics bias
          if hasattr(self, 'physics_unary_weights'):
            weights = [self.physics_unary_weights.get(op, 1.0) for op in self.unary_ops]
            op = random.choices(self.unary_ops, weights=weights)[0]
          else:
            op = random.choice(self.unary_ops)
          operand = self._generate_biased_node(depth + 1, max_depth)
          return UnaryOpNode(op, operand)

      elif node_type == 'scale':
        if depth < max_depth - 2:
            power = random.randint(-3, 3)
            operand = self._generate_biased_node(depth + 1, max_depth)
            return ScalingOpNode(power, operand)

    # Fallback to terminal with enhanced physics constants (BiasedExpressionGenerator)
    if random.random() < 0.65:
      return VariableNode(random.randint(0, self.n_inputs - 1))
    else:
      # Enhanced constant generation for physics (BiasedExpressionGenerator fallback)
      if random.random() < 0.3:  # 30% chance for common physics constants
        physics_constants = [
          0.5, 1.0, 2.0, 3.0, 4.0, 5.0,     # Common small integers
          0.1, 0.2, 0.25, 0.33, 0.67,       # Common fractions
          1.41, 1.73, 2.72, 3.14,           # sqrt(2), sqrt(3), e, pi approximations
          -1.0, -0.5, -2.0                   # Common negative values
        ]
        return ConstantNode(random.choice(physics_constants))
      else:
        return ConstantNode(random.uniform(-5, 5))  # Most common range
