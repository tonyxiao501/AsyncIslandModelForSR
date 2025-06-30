"""Expression Tree Module

Core expression tree functionality for symbolic regression.
"""

from .expression import Expression
from .core.node import (
    Node,
    VariableNode,
    ConstantNode,
    BinaryOpNode,
    UnaryOpNode
)
from .core.operators import (
    NodeType,
    OpType,
    BINARY_OP_MAP,
    UNARY_OP_MAP,
    evaluate_variable,
    evaluate_constant,
    evaluate_binary_op,
    evaluate_unary_op
)

__all__ = [
    'Expression',
    'Node',
    'VariableNode', 
    'ConstantNode',
    'BinaryOpNode',
    'UnaryOpNode',
    'NodeType',
    'OpType',
    'BINARY_OP_MAP',
    'UNARY_OP_MAP',
    'evaluate_variable',
    'evaluate_constant', 
    'evaluate_binary_op',
    'evaluate_unary_op'
]