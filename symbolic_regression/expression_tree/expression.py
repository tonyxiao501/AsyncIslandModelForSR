import numpy as np
from typing import Optional
from expression_tree.core.node import Node
import sympy as sp
from typing import Callable


class Expression:
  """Expression class with weighted complexity caching"""

  __slots__ = ('root', '_string_cache')

  def __init__(self, root: Node):
    self.root = root
    self._string_cache: Optional[str] = None

  def evaluate(self, X: np.ndarray) -> np.ndarray:
    return self.root.evaluate(X)

  def to_string(self) -> str:
    if self._string_cache is None:
      self._string_cache = self.root.to_string()
    return self._string_cache

  def copy(self) -> 'Expression':
    return Expression(self.root.copy())

  def size(self) -> int:
    """Node count (original complexity measure)"""
    return self.root.size()

  def complexity(self) -> float:
    """Weighted complexity score"""
    return self.root.complexity()

  def compress_constants(self) -> 'Expression':
    node = self.root.compress_constants()
    if node is None:
      node = self.root
    return Expression(node)

  def clear_cache(self):
    """Clear cached values"""
    self._string_cache = None
    
  def to_sympy(self) -> sp.Expr:   
    return self.root.to_sympy(sp.numbered_symbols(prefix='c'))

  def get_constants(self) -> tuple:
    const_list = []
    self.root.get_constants(const_list)
    return const_list

# Function: lambda X, *params -> Y
  def vector_lambdify(self) -> Callable:
    constants = self.get_constants()
    sp_expr = self.to_sympy()
    c_dim = len(constants)
    x_dim = len(sp_expr.free_symbols) - c_dim
    symbols = sp.symbols(f'x0:{x_dim}')
    if c_dim > 0:
        symbols += sp.symbols(f'c0:{c_dim}')
    lambda_func = sp.lambdify(symbols, sp_expr, modules='numpy')
    # wrapper for unpack parameters
    def wrapper(X, *param):
        X_arr = np.array(X)
        if len(X_arr.shape) == 0:
            return lambda_func(X, *param)
        else:
            return lambda_func(*X, *param)
    return np.vectorize(wrapper)

  def __hash__(self) -> int:
    return hash(self.root)

  def __eq__(self, other) -> bool:
    if not isinstance(other, Expression):
      return False
    return hash(self) == hash(other)