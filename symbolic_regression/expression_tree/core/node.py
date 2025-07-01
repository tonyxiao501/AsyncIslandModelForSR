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

# Define complexity weights for different operations
COMPLEXITY_WEIGHTS: Dict[str, float] = {
  # Binary operations
  '+': 1.0,
  '-': 1.0,
  '*': 1.2,
  '/': 1.8,  # Division is more complex due to potential divide-by-zero
  '^': 2.5,  # Power operations are very complex

  # Unary operations
  'sin': 1.3,
  'cos': 1.3,
  'sqrt': 1.4,
  'log': 2.0,  # Logarithm can be unstable
  'exp': 2.2,  # Exponential can explode quickly

  # Terminal nodes
  'variable': 1.0,
  'constant': 1.0,
}

# Penalty multipliers for dangerous combinations
COMBINATION_PENALTIES: Dict[tuple, float] = {
  # Nested transcendental functions
  ('sin', 'sin'): 2.0,
  ('cos', 'cos'): 2.0,
  ('sin', 'cos'): 1.5,
  ('cos', 'sin'): 1.5,

  # Exponential/logarithm combinations
  ('exp', 'log'): 3.0,  # exp(log(x)) = x, but numerically unstable
  ('log', 'exp'): 3.0,  # log(exp(x)) = x, but can overflow
  ('exp', 'exp'): 4.0,  # Nested exponentials are very dangerous
  ('log', 'log'): 3.0,  # Nested logs can be unstable

  # Power combinations
  ('^', '^'): 3.5,  # x^(y^z) grows extremely fast
  ('^', 'exp'): 4.0,  # x^exp(y) is explosive
  ('exp', '^'): 4.0,  # exp(x^y) is explosive

  # Division chains
  ('/', '/'): 2.0,  # Nested divisions can amplify errors
  ('/', '^'): 2.5,  # Division with powers
  ('^', '/'): 2.5,
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
    base_complexity = COMPLEXITY_WEIGHTS.get(self.operator, 1.0)
    left_complexity = self.left.complexity()
    right_complexity = self.right.complexity()

    # Check for dangerous combinations
    penalty = 1.0

    # Check if operands are operations that could cause issues
    if isinstance(self.left, (BinaryOpNode, UnaryOpNode)):
      left_op = self.left.operator
      combo_key = (self.operator, left_op)
      penalty *= COMBINATION_PENALTIES.get(combo_key, 1.0)

    if isinstance(self.right, (BinaryOpNode, UnaryOpNode)):
      right_op = self.right.operator
      combo_key = (self.operator, right_op)
      penalty *= COMBINATION_PENALTIES.get(combo_key, 1.0)

    # Multiplicative complexity with penalty
    return base_complexity * (1 + left_complexity + right_complexity) * penalty

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
      return self.left.to_sympy(c_generator) + self.right.to_sympy(c_generator)
    elif self.operator == '-':
      return self.left.to_sympy(c_generator) - self.right.to_sympy(c_generator)
    elif self.operator == '*':
      #return self.left.to_sympy(c_generator) * self.right.to_sympy(c_generator)
      return sp.Mul(self.left.to_sympy(c_generator), self.right.to_sympy(c_generator))
    elif self.operator == '/':
      return self.left.to_sympy(c_generator) / self.right.to_sympy(c_generator)
    elif self.operator == '^':
      return sp.Pow(self.left.to_sympy(c_generator), self.right.to_sympy(c_generator))
    else:
      raise RuntimeWarning(f"to_sympy reached unexpected operation at node {type(self)}")
    
  def get_constants(self, constant_list):
    self.left.get_constants(constant_list)
    self.right.get_constants(constant_list)


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
    base_complexity = COMPLEXITY_WEIGHTS.get(self.operator, 1.0)
    operand_complexity = self.operand.complexity()

    # Check for dangerous combinations
    penalty = 1.0
    if isinstance(self.operand, (BinaryOpNode, UnaryOpNode)):
      operand_op = self.operand.operator
      combo_key = (self.operator, operand_op)
      penalty *= COMBINATION_PENALTIES.get(combo_key, 1.0)

    # Multiplicative complexity with penalty
    return base_complexity * (1 + operand_complexity) * penalty

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
    if self.operator == 'sin':
      return sp.sin(self.operand.to_sympy(c_generator))
    elif self.operator == 'cos':
      return sp.cos(self.operand.to_sympy(c_generator))
    elif self.operator == 'sqrt':
      return sp.sqrt(self.operand.to_sympy(c_generator))
    elif self.operator == 'log':
      return sp.log(self.operand.to_sympy(c_generator))
    elif self.operator == 'exp':
      return sp.exp(self.operand.to_sympy(c_generator))

  def get_constants(self, constant_list):
    self.operand.get_constants(constant_list)
