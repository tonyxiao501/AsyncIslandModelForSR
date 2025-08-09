import numpy as np
from typing import Optional
from ..core.node import Node, VariableNode, ConstantNode, BinaryOpNode, UnaryOpNode, ScalingOpNode
from ..optimization.memory_pool import get_global_pool
from .tree_utils import count_nested_depth


class ExpressionSimplifier:
  """Simplifies and validates expressions"""

  @staticmethod
  def simplify_expression(node: Node) -> Optional[Node]:
    """Simplify expression and return None if invalid"""
    if ExpressionSimplifier._is_invalid_combination(node):
      return None

    simplified = ExpressionSimplifier._apply_simplification_rules(node)
    return simplified if simplified else node

  @staticmethod
  def _is_invalid_combination(node: Node) -> bool:
    """Check for meaningless combinations"""
    if isinstance(node, UnaryOpNode):
      operand = node.operand

      # sin(sin(x)), cos(cos(x)) - redundant nested trig
      if (node.operator in ['sin', 'cos'] and
          isinstance(operand, UnaryOpNode) and
          operand.operator in ['sin', 'cos']):
        return True

      # log(exp(x)), exp(log(x)) - cancel each other
      if ((node.operator == 'log' and isinstance(operand, UnaryOpNode) and operand.operator == 'exp') or
          (node.operator == 'exp' and isinstance(operand, UnaryOpNode) and operand.operator == 'log')):
        return True

      # log1p(expm1(x)) and expm1(log1p(x)) approximately cancel for small x
      if ((node.operator == 'log1p' and isinstance(operand, UnaryOpNode) and operand.operator == 'expm1') or
          (node.operator == 'expm1' and isinstance(operand, UnaryOpNode) and operand.operator == 'log1p')):
        return True

      if (node.operator == 'sqrt' and isinstance(operand, UnaryOpNode) and operand.operator == 'sqrt'):
        return True

      if isinstance(operand, UnaryOpNode):
        nested_depth = count_nested_depth(operand)
        if nested_depth > 2:
          return True

    elif isinstance(node, ScalingOpNode):
        if isinstance(node.operand, ScalingOpNode):
            return True # scale(scale(x)) is redundant

    elif isinstance(node, BinaryOpNode):
      if (ExpressionSimplifier._is_invalid_combination(node.left) or
          ExpressionSimplifier._is_invalid_combination(node.right)):
        return True

    return False

  @staticmethod
  def _apply_simplification_rules(node: Node) -> Optional[Node]:
    if isinstance(node, BinaryOpNode):
      left = ExpressionSimplifier._apply_simplification_rules(node.left) or node.left
      right = ExpressionSimplifier._apply_simplification_rules(node.right) or node.right

      if isinstance(left, ConstantNode) and isinstance(right, ConstantNode):
        try:
          from ..core.operators import evaluate_binary_op
          result = evaluate_binary_op(np.array([left.value]), np.array([right.value]), node.operator)[0]
          if np.isfinite(result):
            return get_global_pool().get_constant_node(result)
        except:
          pass

      if node.operator == '+':
        if isinstance(right, ConstantNode) and abs(right.value) < 1e-10:
          return left  # x + 0 = x
        if isinstance(left, ConstantNode) and abs(left.value) < 1e-10:
          return right  # 0 + x = x

      elif node.operator == '*':
        if isinstance(right, ConstantNode):
          if abs(right.value) < 1e-10:
            return get_global_pool().get_constant_node(0.0)  # x * 0 = 0
          if abs(right.value - 1.0) < 1e-10:
            return left  # x * 1 = x
        if isinstance(left, ConstantNode):
          if abs(left.value) < 1e-10:
            return get_global_pool().get_constant_node(0.0)  # 0 * x = 0
          if abs(left.value - 1.0) < 1e-10:
            return right  # 1 * x = x

      return get_global_pool().get_binary_node(node.operator, left, right)

    elif isinstance(node, UnaryOpNode):
      operand = ExpressionSimplifier._apply_simplification_rules(node.operand) or node.operand

      if isinstance(operand, ConstantNode):
        try:
          from ..core.operators import evaluate_unary_op
          result = evaluate_unary_op(np.array([operand.value]), node.operator)[0]
          if np.isfinite(result):
            return get_global_pool().get_constant_node(result)
        except:
          pass

      return get_global_pool().get_unary_node(node.operator, operand)

    elif isinstance(node, ScalingOpNode):
        operand = ExpressionSimplifier._apply_simplification_rules(node.operand) or node.operand

        if node.power == 0:
            return operand # scale(x, 0) = x

        if isinstance(operand, ScalingOpNode):
            new_power = node.power + operand.power
            return get_global_pool().get_scaling_node(new_power, operand.operand)

        return get_global_pool().get_scaling_node(node.power, operand)

    return None
