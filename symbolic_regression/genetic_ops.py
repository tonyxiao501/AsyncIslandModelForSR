import numpy as np
import random
from typing import List, Tuple, Optional
from symbolic_regression.expression import (
  Expression,
  Node,
  BinaryOpNode,
  UnaryOpNode,
  ConstantNode
)
from symbolic_regression.generator import ExpressionGenerator

class GeneticOperations:
  """Implements genetic operations for symbolic regression"""

  def __init__(self, n_inputs: int):
    self.n_inputs = n_inputs
    self.generator = ExpressionGenerator(n_inputs)

  def mutate(self, expression: Expression, mutation_rate: float = 0.1) -> Expression:
    """Mutate an expression by randomly changing nodes"""
    mutated = expression.copy()
    self._mutate_node(mutated.root, mutation_rate)
    return mutated

  def _mutate_node(self, node: Node, mutation_rate: float) -> None:
    """Recursively mutate nodes in the tree"""
    if random.random() < mutation_rate:
      # Replace this node with a random subtree
      new_node = self.generator.generate_random_expression(depth=0)
      if isinstance(node, BinaryOpNode):
        if random.random() < 0.5:
          node.left = new_node
        else:
          node.right = new_node
      elif isinstance(node, UnaryOpNode):
        node.operand = new_node
      elif isinstance(node, ConstantNode):
        node.value += random.gauss(0, 0.1)

    # Recursively mutate children
    if isinstance(node, BinaryOpNode):
      self._mutate_node(node.left, mutation_rate)
      self._mutate_node(node.right, mutation_rate)
    elif isinstance(node, UnaryOpNode):
      self._mutate_node(node.operand, mutation_rate)

  def crossover(self, parent1: Expression, parent2: Expression) -> Tuple[Expression, Expression]:
    """Perform crossover between two expressions"""
    child1 = parent1.copy()
    child2 = parent2.copy()

    # Select random nodes for crossover
    nodes1 = self._get_all_nodes(child1.root)
    nodes2 = self._get_all_nodes(child2.root)

    if len(nodes1) > 1 and len(nodes2) > 1:
      # Swap random subtrees
      node1 = random.choice(nodes1[1:])  # Don't select root
      node2 = random.choice(nodes2[1:])

      # This is a simplified crossover - in practice, you'd need
      # to track parent relationships for proper swapping
      if isinstance(node1, ConstantNode) and isinstance(node2, ConstantNode):
        node1.value, node2.value = node2.value, node1.value

    return child1, child2

  def _get_all_nodes(self, node: Node) -> List[Node]:
    """Get all nodes in the tree"""
    nodes = [node]
    if isinstance(node, BinaryOpNode):
      nodes.extend(self._get_all_nodes(node.left))
      nodes.extend(self._get_all_nodes(node.right))
    elif isinstance(node, UnaryOpNode):
      nodes.extend(self._get_all_nodes(node.operand))
    return nodes