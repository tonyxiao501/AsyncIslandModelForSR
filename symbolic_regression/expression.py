import numpy as np
from abc import ABC, abstractmethod

class Node(ABC):
  """Abstract base class for expression tree nodes"""

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

class VariableNode(Node):
  """Represents input variables (X0, X1, X2, ...)"""

  def __init__(self, index: int):
    self.index = index

  def evaluate(self, X: np.ndarray) -> np.ndarray:
    return X[:, self.index].astype(np.float64)

  def to_string(self) -> str:
    return f"X{self.index}"

  def copy(self) -> 'VariableNode':
    return VariableNode(self.index)

  def size(self) -> int:
    return 1

class ConstantNode(Node):
  """Represents constant values"""

  def __init__(self, value: float):
    self.value = float(value)

  def evaluate(self, X: np.ndarray) -> np.ndarray:
    return np.full(X.shape[0], self.value, dtype=np.float64)

  def to_string(self) -> str:
    return f"{self.value:.3f}"

  def copy(self) -> 'ConstantNode':
    return ConstantNode(self.value)

  def size(self) -> int:
    return 1

class BinaryOpNode(Node):
  """Represents binary operations (+, -, *, /, ^)"""

  def __init__(self, operator: str, left: Node, right: Node):
    self.operator = operator
    self.left = left
    self.right = right

    self.ops = {
      '+': lambda a, b: a + b,
      '-': lambda a, b: a - b,
      '*': lambda a, b: a * b,
      '/': lambda a, b: np.divide(a, b, out=np.ones_like(a), where=b!=0),
      '^': lambda a, b: np.power(a, np.clip(b, -10, 10))
    }

  def evaluate(self, X: np.ndarray) -> np.ndarray:
    left_val = self.left.evaluate(X)
    right_val = self.right.evaluate(X)

    try:
      with np.errstate(invalid='ignore', divide='ignore'):
        result = self.ops[self.operator](left_val, right_val)
      # Handle invalid results
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

class UnaryOpNode(Node):
  """Represents unary operations (sin, cos, exp, log, sqrt)"""

  def __init__(self, operator: str, operand: Node):
    self.operator = operator
    self.operand = operand

    self.ops = {
      'sin': lambda x: np.sin(x),
      'cos': lambda x: np.cos(x),
      'exp': lambda x: np.exp(np.clip(x, -10, 10)),
      'log': lambda x: np.log(np.abs(x) + 1e-8),
      'sqrt': lambda x: np.sqrt(np.abs(x))
    }

  def evaluate(self, X: np.ndarray) -> np.ndarray:
    operand_val = self.operand.evaluate(X)

    try:
      result = self.ops[self.operator](operand_val)
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

class Expression:
  """Represents a complete mathematical expression"""

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