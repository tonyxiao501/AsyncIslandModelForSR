import numpy as np
import numba
from abc import ABC, abstractmethod

# JIT-compiled helper functions
@numba.njit
def evaluate_variable(X, index):
  return X[:, index].astype(np.float64)

@numba.njit
def evaluate_constant(n_samples, value):
  return np.full(n_samples, value, dtype=np.float64)

@numba.njit
def evaluate_binary_op(left_val, right_val, operator):
  if operator == '+':
    return left_val + right_val
  elif operator == '-':
    return left_val - right_val
  elif operator == '*':
    return left_val * right_val
  elif operator == '/':
    out = np.ones_like(left_val)
    mask = right_val != 0
    out[mask] = left_val[mask] / right_val[mask]
    return out
  elif operator == '^':
    return np.power(left_val, np.clip(right_val, -10, 10))
  return np.zeros_like(left_val)

@numba.njit
def evaluate_unary_op(operand_val, operator):
  if operator == 'sin':
    return np.sin(operand_val)
  elif operator == 'cos':
    return np.cos(operand_val)
  elif operator == 'exp':
    return np.exp(np.clip(operand_val, -10, 10))
  elif operator == 'log':
    return np.log(np.abs(operand_val) + 1e-8)
  elif operator == 'sqrt':
    return np.sqrt(np.abs(operand_val))
  return np.zeros_like(operand_val)

class Node(ABC):
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
  def size(self) -> int:
    pass

  @abstractmethod
  def compress_constants(self):
    pass


class VariableNode(Node):
  def __init__(self, index: int):
    self.index = index

  def evaluate(self, X: np.ndarray) -> np.ndarray:
    return evaluate_variable(X, self.index)

  def to_string(self) -> str:
    return f"X{self.index}"

  def copy(self) -> 'VariableNode':
    return VariableNode(self.index)

  def size(self) -> int:
    return 1
  def compress_constants(self) -> Node:
    return self  # Variables do not compress constants

class ConstantNode(Node):
  def __init__(self, value: float):
    self.value = float(value)

  def evaluate(self, X: np.ndarray) -> np.ndarray:
    return evaluate_constant(X.shape[0], self.value)

  def to_string(self) -> str:
    return f"{self.value:.3f}"

  def copy(self) -> 'ConstantNode':
    return ConstantNode(self.value)

  def size(self) -> int:
    return 1
  def compress_constants(self) -> Node:
    return self  # Constants do not compress further

class BinaryOpNode(Node):
  def __init__(self, operator: str, left: Node, right: Node):
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
    return BinaryOpNode(self.operator, self.left.copy(), self.right.copy())

  def size(self) -> int:
    return 1 + self.left.size() + self.right.size()
  def compress_constants(self) -> Node:
    left_c = self.left.compress_constants()
    right_c = self.right.compress_constants()
    if isinstance(left_c, ConstantNode) and isinstance(right_c, ConstantNode):
      val = evaluate_binary_op(np.array([left_c.value]), np.array([right_c.value]), self.operator)[0]
      return ConstantNode(val)
    if left_c is None:
      left_c = self.left
    if right_c is None:
      right_c = self.right
    return BinaryOpNode(self.operator, left_c, right_c)

class UnaryOpNode(Node):
  def __init__(self, operator: str, operand: Node):
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
    return UnaryOpNode(self.operator, self.operand.copy())

  def size(self) -> int:
    return 1 + self.operand.size()
  def compress_constants(self) -> Node:
    operand_c = self.operand.compress_constants()
    if operand_c is None:
      operand_c = self.operand
    if isinstance(operand_c, ConstantNode):
      val = evaluate_unary_op(np.array([operand_c.value]), self.operator)[0]
      return ConstantNode(val)
    return UnaryOpNode(self.operator, operand_c)

class Expression:
  def __init__(self, root: Node):
    self.root = root

  def evaluate(self, X: np.ndarray) -> np.ndarray:
    return self.root.evaluate(X)

  def to_string(self) -> str:
    return self.root.to_string()

  def copy(self) -> 'Expression':
    return Expression(self.root.copy())

  def size(self) -> int:
    return self.root.size()
  def compress_constants(self) -> 'Expression':
    node = self.root.compress_constants()
    if node is None:
      node = self.root
    return Expression(node)
