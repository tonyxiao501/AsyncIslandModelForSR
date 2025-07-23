"""Expression Tree Module

Core expression tree functionality for symbolic regression.
"""

from .expression import Expression
from .core.node import (
    Node,
    VariableNode,
    ConstantNode,
    BinaryOpNode,
    UnaryOpNode,
    ScalingOpNode
)
from .core.operators import (
    NodeType,
    OpType,
    BINARY_OP_MAP,
    UNARY_OP_MAP,
    evaluate_variable,
    evaluate_constant,
    evaluate_binary_op,
    evaluate_unary_op,
    evaluate_binary_op_fast,
    evaluate_unary_op_fast
)
from .optimization import NodePool, get_global_pool, clear_global_pool
from .utils import SymPySimplifier, ExpressionValidator

__all__ = [
    "Expression",
    "Node", "VariableNode", "ConstantNode", "BinaryOpNode", "UnaryOpNode", "ScalingOpNode",
    "NodeType", "OpType",
    "BINARY_OP_MAP", "UNARY_OP_MAP",
    "evaluate_variable", "evaluate_constant", "evaluate_binary_op", "evaluate_unary_op",
    "evaluate_binary_op_fast", "evaluate_unary_op_fast",
    "NodePool", "get_global_pool", "clear_global_pool",
    "SymPySimplifier", "ExpressionValidator"
]
