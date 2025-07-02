# Python

"""Symbolic Regression Package

A genetic programming approach to symbolic regression for MIMO systems.
"""

from .expression_tree import (
  Expression, Node, VariableNode, ConstantNode,
  BinaryOpNode, UnaryOpNode
)
from .generator import ExpressionGenerator
from .genetic_ops import GeneticOperations
from .mimo_regressor import MIMOSymbolicRegressor

__version__ = "0.1.0"
__all__ = [
  "Expression", "Node", "VariableNode", "ConstantNode",
  "BinaryOpNode", "UnaryOpNode",
  "ExpressionGenerator", "GeneticOperations",
  "MIMOSymbolicRegressor"
]