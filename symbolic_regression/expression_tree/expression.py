import numpy as np
from typing import Optional
from expression_tree.core.node import Node


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

  def __hash__(self) -> int:
    return hash(self.root)

  def __eq__(self, other) -> bool:
    if not isinstance(other, Expression):
      return False
    return hash(self) == hash(other)