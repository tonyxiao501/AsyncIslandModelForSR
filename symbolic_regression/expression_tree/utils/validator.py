import numpy as np
from typing import Optional
from ..core.node import Node, ConstantNode, BinaryOpNode, UnaryOpNode, ScalingOpNode, VariableNode


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
    return ExpressionValidator._is_structurally_valid_recursive(node, is_root=True)

  @staticmethod
  def _is_structurally_valid_recursive(node: Node, is_root: bool = False) -> bool:
    if isinstance(node, ConstantNode):
        return np.isfinite(node.value)

    elif isinstance(node, BinaryOpNode):
        if not (ExpressionValidator._is_structurally_valid_recursive(node.left, is_root=False) and
                ExpressionValidator._is_structurally_valid_recursive(node.right, is_root=False)):
            return False

        if node.operator == '/' and isinstance(node.right, ConstantNode):
            if abs(node.right.value) < 1e-12:
                return False

        if node.operator == '^' and isinstance(node.right, ConstantNode):
            if node.right.value > 10 or node.right.value < -10:
                return False
        
        return True

    elif isinstance(node, UnaryOpNode):
        if not ExpressionValidator._is_structurally_valid_recursive(node.operand, is_root=False):
            return False

        if node.operator == 'log' and isinstance(node.operand, ConstantNode):
            if node.operand.value <= 0:
                return False

        if node.operator == 'sqrt' and isinstance(node.operand, ConstantNode):
            if node.operand.value < 0:
                return False

        return True

    elif isinstance(node, ScalingOpNode):
        # Rule: scaling node can only be parent of constants or variables, or the root of the tree.
        # This means if a node is a ScalingOpNode, its child must be a leaf, OR it is the root.
        # This implementation detail is subtle. If a scaling node is NOT the root, its child MUST be a leaf.
        if not is_root:
            if not isinstance(node.operand, (ConstantNode, VariableNode)):
                return False
        # Now, regardless of whether it's a root or not, its operand must be structurally valid.
        return ExpressionValidator._is_structurally_valid_recursive(node.operand, is_root=False)
    
    elif isinstance(node, VariableNode):
        return True

    return False
  

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
