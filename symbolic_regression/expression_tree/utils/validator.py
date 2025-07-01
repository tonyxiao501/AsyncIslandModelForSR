import numpy as np
from typing import Optional
from ..core.node import Node, ConstantNode, BinaryOpNode, UnaryOpNode


class ExpressionValidator:

  @staticmethod
  def is_valid_expression(node: Node, X: Optional[np.ndarray] = None) -> bool:
    try:
      if not ExpressionValidator._is_structurally_valid(node):
        return False

      if X is not None:
        return ExpressionValidator._test_evaluation(node, X)

      return True

    except Exception:
      return False

  @staticmethod
  def _is_structurally_valid(node: Node) -> bool:
    if isinstance(node, ConstantNode):
      return np.isfinite(node.value)

    elif isinstance(node, BinaryOpNode):
      if not (ExpressionValidator._is_structurally_valid(node.left) and
              ExpressionValidator._is_structurally_valid(node.right)):
        return False

      if node.operator == '/' and isinstance(node.right, ConstantNode):
        if abs(node.right.value) < 1e-12:
          return False

      if node.operator == '^' and isinstance(node.right, ConstantNode):
        if node.right.value > 10 or node.right.value < -10:
          return False

      return True

    elif isinstance(node, UnaryOpNode):
      if not ExpressionValidator._is_structurally_valid(node.operand):
        return False

      if node.operator == 'log' and isinstance(node.operand, ConstantNode):
        if node.operand.value <= 0:
          return False

      if node.operator == 'sqrt' and isinstance(node.operand, ConstantNode):
        if node.operand.value < 0:
          return False

      return True

    return True

  @staticmethod
  def _test_evaluation(node: Node, X: np.ndarray, sample_size: int = 10) -> bool:
    try:
      sample_indices = np.random.choice(len(X), min(sample_size, len(X)), replace=False)
      X_sample = X[sample_indices]

      result = node.evaluate(X_sample)

      if not isinstance(result, np.ndarray):
        return False

      if np.any(~np.isfinite(result)):
        return False

      if np.any(np.abs(result) > 1e10):
        return False

      return True

    except (ZeroDivisionError, ValueError, OverflowError, RuntimeWarning):
      return False
    except Exception:
      return False