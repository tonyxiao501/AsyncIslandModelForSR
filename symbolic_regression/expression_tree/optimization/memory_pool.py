from typing import List, TYPE_CHECKING, Optional
import threading

if TYPE_CHECKING:
  from ..core.node import Node, VariableNode, ConstantNode, BinaryOpNode, UnaryOpNode


class NodePool:
  """High-performance memory pool for node allocation"""

  def __init__(self, initial_size: int = 1000):
    self.variable_pool: List['VariableNode'] = []
    self.constant_pool: List['ConstantNode'] = []
    self.binary_pool: List['BinaryOpNode'] = []
    self.unary_pool: List['UnaryOpNode'] = []
    self._preallocate(initial_size)

  def _preallocate(self, size: int):
    # Import here to avoid circular imports
    from ..core.node import VariableNode, ConstantNode, BinaryOpNode, UnaryOpNode

    quarter = size // 4
    for _ in range(quarter):
      self.variable_pool.append(VariableNode.__new__(VariableNode))
      self.constant_pool.append(ConstantNode.__new__(ConstantNode))
      self.binary_pool.append(BinaryOpNode.__new__(BinaryOpNode))
      self.unary_pool.append(UnaryOpNode.__new__(UnaryOpNode))

  def get_variable_node(self, index: int) -> 'VariableNode':
    from ..core.node import VariableNode
    if self.variable_pool:
      node = self.variable_pool.pop()
      node.__init__(index)
      return node
    return VariableNode(index)

  def get_constant_node(self, value: float) -> 'ConstantNode':
    from ..core.node import ConstantNode
    if self.constant_pool:
      node = self.constant_pool.pop()
      node.__init__(value)
      return node
    return ConstantNode(value)

  def get_binary_node(self, operator: str, left: 'Node', right: 'Node') -> 'BinaryOpNode':
    from ..core.node import BinaryOpNode
    if self.binary_pool:
      node = self.binary_pool.pop()
      node.__init__(operator, left, right)
      return node
    return BinaryOpNode(operator, left, right)

  def get_unary_node(self, operator: str, operand: 'Node') -> 'UnaryOpNode':
    from ..core.node import UnaryOpNode
    if self.unary_pool:
      node = self.unary_pool.pop()
      node.__init__(operator, operand)
      return node
    return UnaryOpNode(operator, operand)

  def return_node(self, node: 'Node'):
    """Return node to pool for reuse"""
    from ..core.node import VariableNode, ConstantNode, BinaryOpNode, UnaryOpNode

    if hasattr(node, '_clear_cache'):
      node._clear_cache()

    if isinstance(node, VariableNode) and len(self.variable_pool) < 500:
      self.variable_pool.append(node)
    elif isinstance(node, ConstantNode) and len(self.constant_pool) < 500:
      self.constant_pool.append(node)
    elif isinstance(node, BinaryOpNode) and len(self.binary_pool) < 500:
      self.binary_pool.append(node)
    elif isinstance(node, UnaryOpNode) and len(self.unary_pool) < 500:
      self.unary_pool.append(node)

  def get_stats(self) -> dict:
    """Get pool statistics"""
    return {
      'variable_pool_size': len(self.variable_pool),
      'constant_pool_size': len(self.constant_pool),
      'binary_pool_size': len(self.binary_pool),
      'unary_pool_size': len(self.unary_pool)
    }

  def clear(self):
    """Clear all pools"""
    self.variable_pool.clear()
    self.constant_pool.clear()
    self.binary_pool.clear()
    self.unary_pool.clear()


# Global instance - process-local initialization
_GLOBAL_POOL: Optional[NodePool] = None
_POOL_LOCK = threading.Lock()


def get_global_pool() -> NodePool:
  global _GLOBAL_POOL
  if _GLOBAL_POOL is None:
    with _POOL_LOCK:
      if _GLOBAL_POOL is None:  # Double-check locking
        _GLOBAL_POOL = NodePool()
  return _GLOBAL_POOL


def clear_global_pool():
  global _GLOBAL_POOL
  with _POOL_LOCK:
    if _GLOBAL_POOL is not None:
      _GLOBAL_POOL.clear()
    _GLOBAL_POOL = None


def reset_global_pool():
  """Reset the global pool - useful for multiprocessing"""
  global _GLOBAL_POOL
  _GLOBAL_POOL = None