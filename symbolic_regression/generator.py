import random
from typing import List, Optional
from .expression_tree import Expression, Node, VariableNode, ConstantNode, BinaryOpNode, UnaryOpNode
from .expression_tree.utils.simplifier import ExpressionSimplifier


class ExpressionGenerator:
  """Enhanced expression generator with complexity control"""

  def __init__(self, n_inputs: int, max_depth: int = 6):
    self.n_inputs = n_inputs
    self.max_depth = max_depth
    self.binary_ops = ['+', '-', '*', '/']  # Removed '^' to reduce complexity
    self.unary_ops = ['sin', 'cos', 'exp', 'log', 'sqrt']

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
      # Terminal nodes
      if random.random() < 0.65:  # Favor variables
        return VariableNode(random.randint(0, self.n_inputs - 1))
      else:
        return ConstantNode(random.uniform(-3, 3))  # Smaller constant range

    # Function nodes with reduced complexity
    if depth < max_depth - 1:
      if random.random() < 0.75:  # Favor binary ops
        op = random.choice(self.binary_ops)
        left = self._generate_node(depth + 1, max_depth)
        right = self._generate_node(depth + 1, max_depth)
        return BinaryOpNode(op, left, right)
      else:
        # Limit unary operations at deeper levels
        if depth < max_depth - 2:
          op = random.choice(self.unary_ops)
          operand = self._generate_node(depth + 1, max_depth)
          return UnaryOpNode(op, operand)

    # Fallback to terminal
    if random.random() < 0.65:
      return VariableNode(random.randint(0, self.n_inputs - 1))
    return ConstantNode(random.uniform(-3, 3))

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

  def __init__(self, n_inputs: int, max_depth: int = 6, operator_rates: dict = None):
    super().__init__(n_inputs, max_depth)
    self.operator_rates = operator_rates or {}

    # Create weighted operator lists
    self.weighted_binary_ops = self._create_weighted_operator_list()

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
      if random.random() < 0.75:  # Favor binary ops
        # Use weighted operator selection
        op = random.choice(self.weighted_binary_ops)
        left = self._generate_biased_node(depth + 1, max_depth)
        right = self._generate_biased_node(depth + 1, max_depth)
        return BinaryOpNode(op, left, right)
      else:
        # Unary operations (unchanged)
        if depth < max_depth - 2:
          op = random.choice(self.unary_ops)
          operand = self._generate_biased_node(depth + 1, max_depth)
          return UnaryOpNode(op, operand)

    # Fallback to terminal
    if random.random() < 0.65:
      return VariableNode(random.randint(0, self.n_inputs - 1))
    return ConstantNode(random.uniform(-3, 3))
