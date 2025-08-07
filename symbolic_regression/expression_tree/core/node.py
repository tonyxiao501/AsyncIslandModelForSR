import numpy as np
import sympy as sp
from abc import ABC, abstractmethod
from typing import Optional, Dict, Generator, List
from .operators import (
  NodeType, OpType, BINARY_OP_MAP, UNARY_OP_MAP,
  evaluate_variable, evaluate_constant, evaluate_binary_op, evaluate_unary_op,
  evaluate_binary_op_fast, evaluate_unary_op_fast
)
from ..optimization.memory_pool import get_global_pool

# Optimized complexity weights based on research and physics applications
COMPLEXITY_WEIGHTS: Dict[str, float] = {
  # Binary operations - balanced for physics applications
  '+': 1.0,
  '-': 1.0,
  '*': 1.1,  # Slightly more expensive than addition
  '/': 1.5,  # Reduced - division is common in physics
  '^': 2.0,  # Reduced - power laws are important in physics

  # Core transcendental functions - essential for physics
  'sin': 1.2,  # Reduced - very common in physics
  'cos': 1.2,  # Reduced - very common in physics
  'tan': 1.6,  # Increased due to singularities
  'exp': 1.8,  # Reduced - common in physics (decay, growth)
  'log': 1.6,  # Reduced - common in physics (scaling laws)
  'sqrt': 1.2, # Reduced - very common in physics
  'abs': 1.05, # Reduced - simple operation
  'neg': 1.0,  # Unary minus - trivial
  
  # Scaling operation
  'scale': 2.2, # Increased - this shouldn't dominate basic expressions
  
  # Power-related operations
  'square': 1.0,  # Very common, should be cheap
  'cube': 1.5,    # More expensive - less common
  'cbrt': 4.0,    # VERY expensive - should rarely be used for simple functions
  'fourth_root': 4.5,
  
  # Physics-critical operations
  'reciprocal': 1.3,  # Reduced - very common in physics (1/r, 1/t, etc.)
  'inv_square': 1.5,  # Reduced - common (inverse square law)
  
  # Safe variants - slight penalty for redundancy
  'sqrt_abs': 1.3,
  'log_abs': 1.7,
  
  # Hyperbolic functions - less common
  'sinh': 1.5,
  'cosh': 1.5,
  'tanh': 1.4,  # Bounded, so slightly simpler

  # Terminal nodes
  'variable': 1.0,
  'constant': 1.0,
}

# Reduced penalty multipliers for dangerous combinations (additive approach)
COMBINATION_PENALTIES: Dict[tuple, float] = {
  # Nested transcendental functions - smaller penalties
  ('sin', 'sin'): 0.3,
  ('cos', 'cos'): 0.3,
  ('sin', 'cos'): 0.2,
  ('cos', 'sin'): 0.2,

  # Exponential/logarithm combinations - moderate penalties
  ('exp', 'log'): 0.5,  # exp(log(x)) = x, but numerically unstable
  ('log', 'exp'): 0.5,  # log(exp(x)) = x, but can overflow
  ('exp', 'exp'): 0.8,  # Nested exponentials are dangerous
  ('log', 'log'): 0.5,  # Nested logs can be unstable

  # Power combinations - moderate penalties
  ('^', '^'): 0.6,  # x^(y^z) grows extremely fast
  ('^', 'exp'): 0.8,  # x^exp(y) is explosive
  ('exp', '^'): 0.8,  # exp(x^y) is explosive

  # Division chains - small penalties
  ('/', '/'): 0.3,  # Nested divisions can amplify errors
  ('/', '^'): 0.4,  # Division with powers
  ('^', '/'): 0.4,
  
  # Root operations with trigonometric functions - discourage exotic combinations
  ('cbrt', 'sin'): 0.8,  # cbrt(sin(x)) is usually unnecessary
  ('cbrt', 'cos'): 0.8,  # cbrt(cos(x)) is usually unnecessary  
  ('cbrt', '*'): 0.5,    # cbrt of products often overcomplicates
  ('cbrt', '+'): 0.5,    # cbrt of sums often overcomplicates
  
  # Scaling chains - small penalty
  ('scale', 'scale'): 0.4  # Nested scaling operations
}


class Node(ABC):
  """Base node class with weighted complexity caching"""

  __slots__ = ('_hash_cache', '_size_cache', '_complexity_cache')

  def __init__(self):
    self._hash_cache: Optional[int] = None
    self._size_cache: Optional[int] = None
    self._complexity_cache: Optional[float] = None

  def _clear_cache(self):
    self._hash_cache = None
    self._size_cache = None
    self._complexity_cache = None

  @abstractmethod
  def evaluate(self, X: np.ndarray) -> np.ndarray:
    pass

  @abstractmethod
  def to_string(self) -> str:
    pass

  @abstractmethod
  def copy(self) -> 'Node':
    pass
  
  @abstractmethod
  def to_sympy(self, c_generator: Generator) -> sp.Expr:
    pass
  
  def get_constants(self, constant_list: List[float]):
    pass
  
  def set_constants(self, constant_list: List[float]):
    pass

  def size(self) -> int:
    """Original node count"""
    if self._size_cache is None:
      self._size_cache = self._compute_size()
    return self._size_cache

  def complexity(self) -> float:
    """Weighted complexity score"""
    if self._complexity_cache is None:
      self._complexity_cache = self._compute_complexity()
    return self._complexity_cache

  @abstractmethod
  def _compute_size(self) -> int:
    pass

  @abstractmethod
  def _compute_complexity(self) -> float:
    pass

  @abstractmethod
  def compress_constants(self):
    pass

  def __hash__(self) -> int:
    if self._hash_cache is None:
      self._hash_cache = self._compute_hash()
    return self._hash_cache

  @abstractmethod
  def _compute_hash(self) -> int:
    pass


class VariableNode(Node):
  __slots__ = ('index',)

  def __init__(self, index: int):
    super().__init__()
    self.index = index

  def evaluate(self, X: np.ndarray) -> np.ndarray:
    return evaluate_variable(X, self.index)

  def to_string(self) -> str:
    return f"X{self.index}"

  def copy(self) -> 'VariableNode':
    return get_global_pool().get_variable_node(self.index)

  def _compute_size(self) -> int:
    return 1

  def _compute_complexity(self) -> float:
    return COMPLEXITY_WEIGHTS['variable']

  def _compute_hash(self) -> int:
    return hash((NodeType.VARIABLE, self.index))

  def compress_constants(self):
    return self
  
  def to_sympy(self, c_generator):
    return sp.Symbol(f'x{self.index}')


class ConstantNode(Node):
  __slots__ = ('value',)

  def __init__(self, value: float):
    super().__init__()
    self.value = float(value)

  def evaluate(self, X: np.ndarray) -> np.ndarray:
    return evaluate_constant(X.shape[0], self.value)

  def to_string(self) -> str:
    return f"{self.value:.3f}"

  def copy(self) -> 'ConstantNode':
    return get_global_pool().get_constant_node(self.value)

  def _compute_size(self) -> int:
    return 1

  def _compute_complexity(self) -> float:
    return COMPLEXITY_WEIGHTS['constant']

  def _compute_hash(self) -> int:
    return hash((NodeType.CONSTANT, self.value))

  def compress_constants(self):
    return self
  
  def to_sympy(self, c_generator):
    return next(c_generator)
  
  def get_constants(self, constant_list):
    constant_list.append(self.value)
    
  def set_constants(self, constant_list):
    self.value = constant_list[0]
    constant_list.pop(0)


class BinaryOpNode(Node):
  __slots__ = ('operator', 'left', 'right')

  def __init__(self, operator: str, left: Node, right: Node):
    super().__init__()
    self.operator = operator
    self.left = left
    self.right = right

  def evaluate(self, X: np.ndarray) -> np.ndarray:
    left_val = self.left.evaluate(X)
    right_val = self.right.evaluate(X)
    try:
      result = evaluate_binary_op(left_val, right_val, self.operator)
      result = np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)
      return result.astype(np.float64)
    except Exception:
      return np.zeros(X.shape[0], dtype=np.float64)

  def to_string(self) -> str:
    return f"({self.left.to_string()} {self.operator} {self.right.to_string()})"

  def copy(self) -> 'BinaryOpNode':
    return get_global_pool().get_binary_node(self.operator, self.left.copy(), self.right.copy())

  def _compute_size(self) -> int:
    return 1 + self.left.size() + self.right.size()

  def _compute_complexity(self) -> float:
    """PySR-style complexity calculation with additive penalties"""
    base_complexity = COMPLEXITY_WEIGHTS.get(self.operator, 1.0)
    left_complexity = self.left.complexity()
    right_complexity = self.right.complexity()

    # Additive complexity (PySR style) instead of multiplicative
    complexity = base_complexity + left_complexity + right_complexity

    # Small additive penalties for dangerous combinations
    penalty = 0.0
    if isinstance(self.left, (BinaryOpNode, UnaryOpNode)):
      left_op = self.left.operator
      combo_key = (self.operator, left_op)
      penalty += COMBINATION_PENALTIES.get(combo_key, 0.0)

    if isinstance(self.right, (BinaryOpNode, UnaryOpNode)):
      right_op = self.right.operator
      combo_key = (self.operator, right_op)
      penalty += COMBINATION_PENALTIES.get(combo_key, 0.0)

    return complexity + penalty

  def _compute_hash(self) -> int:
    return hash((NodeType.BINARY_OP, self.operator, hash(self.left), hash(self.right)))

  def compress_constants(self):
    left_c = self.left.compress_constants()
    right_c = self.right.compress_constants()

    if isinstance(left_c, ConstantNode) and isinstance(right_c, ConstantNode):
      val = evaluate_binary_op(np.array([left_c.value]), np.array([right_c.value]), self.operator)[0]
      return get_global_pool().get_constant_node(val)

    if left_c is None:
      left_c = self.left
    if right_c is None:
      right_c = self.right

    return get_global_pool().get_binary_node(self.operator, left_c, right_c)
  
  def to_sympy(self, c_generator):
    if self.operator == '+':
      return sp.Add(self.left.to_sympy(c_generator), self.right.to_sympy(c_generator))
    elif self.operator == '-':
      return sp.Add(self.left.to_sympy(c_generator), sp.Mul(-1, self.right.to_sympy(c_generator)))
    elif self.operator == '*':
      #return self.left.to_sympy(c_generator) * self.right.to_sympy(c_generator)
      return sp.Mul(self.left.to_sympy(c_generator), self.right.to_sympy(c_generator))
    elif self.operator == '/':
      return sp.Mul(self.left.to_sympy(c_generator), sp.Pow(self.right.to_sympy(c_generator), -1))
    elif self.operator == '^':
      return sp.Pow(self.left.to_sympy(c_generator), self.right.to_sympy(c_generator))
    else:
      raise RuntimeWarning(f"to_sympy reached unexpected operation at node {type(self)}")
    
  def get_constants(self, constant_list):
    self.left.get_constants(constant_list)
    self.right.get_constants(constant_list)

  def set_constants(self, constant_list):
    self.left.set_constants(constant_list)
    self.right.set_constants(constant_list)

class UnaryOpNode(Node):
  __slots__ = ('operator', 'operand')

  def __init__(self, operator: str, operand: Node):
    super().__init__()
    self.operator = operator
    self.operand = operand

  def evaluate(self, X: np.ndarray) -> np.ndarray:
    operand_val = self.operand.evaluate(X)
    try:
      result = evaluate_unary_op(operand_val, self.operator)
      result = np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)
      return result.astype(np.float64)
    except Exception:
      return np.zeros(X.shape[0], dtype=np.float64)

  def to_string(self) -> str:
    return f"{self.operator}({self.operand.to_string()})"

  def copy(self) -> 'UnaryOpNode':
    return get_global_pool().get_unary_node(self.operator, self.operand.copy())

  def _compute_size(self) -> int:
    return 1 + self.operand.size()

  def _compute_complexity(self) -> float:
    """PySR-style complexity calculation with additive penalties"""
    base_complexity = COMPLEXITY_WEIGHTS.get(self.operator, 1.0)
    operand_complexity = self.operand.complexity()

    # Additive complexity (PySR style) instead of multiplicative
    complexity = base_complexity + operand_complexity

    # Small additive penalty for dangerous combinations
    penalty = 0.0
    if isinstance(self.operand, (BinaryOpNode, UnaryOpNode)):
      operand_op = self.operand.operator
      combo_key = (self.operator, operand_op)
      penalty += COMBINATION_PENALTIES.get(combo_key, 0.0)

    return complexity + penalty

  def _compute_hash(self) -> int:
    return hash((NodeType.UNARY_OP, self.operator, hash(self.operand)))

  def compress_constants(self):
    operand_c = self.operand.compress_constants()
    if operand_c is None:
      operand_c = self.operand
    if isinstance(operand_c, ConstantNode):
      val = evaluate_unary_op(np.array([operand_c.value]), self.operator)[0]
      return get_global_pool().get_constant_node(val)
    return get_global_pool().get_unary_node(self.operator, operand_c)
  
  def to_sympy(self, c_generator):
    operand_sympy = self.operand.to_sympy(c_generator)
    
    if self.operator == 'sin':
      return sp.sin(operand_sympy)
    elif self.operator == 'cos':
      return sp.cos(operand_sympy)
    elif self.operator == 'tan':
      return sp.tan(operand_sympy)
    elif self.operator == 'sqrt':
      return sp.sqrt(operand_sympy)
    elif self.operator == 'log':
      return sp.log(operand_sympy)
    elif self.operator == 'exp':
      return sp.exp(operand_sympy)
    elif self.operator == 'abs':
      return sp.Abs(operand_sympy)
    elif self.operator == 'neg':
      return -operand_sympy
    elif self.operator == 'square':
      return operand_sympy**2
    elif self.operator == 'cube':
      return operand_sympy**3
    elif self.operator == 'reciprocal':
      return sp.Pow(operand_sympy, -1)
    elif self.operator == 'sqrt_abs':
      return sp.sqrt(sp.Abs(operand_sympy))
    elif self.operator == 'log_abs':
      return sp.log(sp.Abs(operand_sympy))
    elif self.operator == 'inv_square':
      return sp.Pow(operand_sympy, -2)
    elif self.operator == 'cbrt':
      return operand_sympy**(sp.Rational(1, 3))
    elif self.operator == 'fourth_root':
      return operand_sympy**(sp.Rational(1, 4))
    elif self.operator == 'sinh':
      return sp.sinh(operand_sympy)
    elif self.operator == 'cosh':
      return sp.cosh(operand_sympy)
    elif self.operator == 'tanh':
      return sp.tanh(operand_sympy)
    else:
      raise RuntimeWarning(f"to_sympy reached unexpected unary operation: {self.operator}")

  def get_constants(self, constant_list):
    self.operand.get_constants(constant_list)

  def set_constants(self, constant_list):
    self.operand.set_constants(constant_list)

class ScalingOpNode(Node):
  __slots__ = ('power', 'operand')
  
  def __init__(self, power: int, operand: Node):
    super().__init__()
    self.power = power
    self.operand = operand
    
  def evaluate(self, X: np.ndarray):
    operand_val = self.operand.evaluate(X)
    try:
      # Limit power to reasonable range to avoid extreme values
      safe_power = np.clip(self.power, -10, 10)
      scale_factor = pow(10., safe_power)
      result = operand_val * scale_factor
      
      # More aggressive clipping for stability
      result = np.nan_to_num(result, nan=0.0, posinf=1e10, neginf=-1e10)
      result = np.clip(result, -1e10, 1e10)
      return result.astype(np.float64)
    except Exception:
      return np.zeros(X.shape[0], dtype=np.float64)

  def to_string(self) -> str:
    # Use a unique format that can be parsed back: scale(operand, power)
    return f"scale({self.operand.to_string()}, {self.power})"

  def copy(self) -> 'ScalingOpNode':
    # Assuming get_scaling_node will be added to the memory pool
    return get_global_pool().get_scaling_node(self.power, self.operand.copy())

  def _compute_size(self) -> int:
    return 1 + self.operand.size()

  def _compute_complexity(self) -> float:
    """PySR-style complexity calculation with additive penalties"""
    base_complexity = COMPLEXITY_WEIGHTS.get('scale', 1.8)
    operand_complexity = self.operand.complexity()

    # Additive complexity (PySR style)
    complexity = base_complexity + operand_complexity

    # Small additive penalty for nested scaling
    penalty = 0.0
    if isinstance(self.operand, ScalingOpNode):
        penalty += COMBINATION_PENALTIES.get(('scale', 'scale'), 0.4)

    return complexity + penalty

  def _compute_hash(self) -> int:
    # Use proper NodeType for scaling operations
    return hash((NodeType.SCALING_OP, 'scale', self.power, hash(self.operand)))

  def compress_constants(self):
    operand_c = self.operand.compress_constants()
    if operand_c is None:
        operand_c = self.operand
    if isinstance(operand_c, ConstantNode):
        val = operand_c.value * pow(10., self.power)
        return get_global_pool().get_constant_node(val)
    # Assuming get_scaling_node will be added
    return get_global_pool().get_scaling_node(self.power, operand_c)

  def to_sympy(self, c_generator):
    operand_sympy = self.operand.to_sympy(c_generator)
    return sp.Mul(operand_sympy, sp.Pow(sp.Integer(10), self.power))

  def get_constants(self, constant_list):
    self.operand.get_constants(constant_list)

  def set_constants(self, constant_list):
    self.operand.set_constants(constant_list)
