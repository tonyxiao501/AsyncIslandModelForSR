import numpy as np
import random
from typing import List, Tuple, Optional
from .expression_tree import Expression, Node, BinaryOpNode, UnaryOpNode, ConstantNode, VariableNode
from .expression_tree.utils.simplifier import ExpressionSimplifier
from .generator import ExpressionGenerator


class GeneticOperations:
  """Enhanced genetic operations with better diversity preservation"""

  def __init__(self, n_inputs: int, max_complexity: int = 20):
    self.n_inputs = n_inputs
    self.max_complexity = max_complexity
    self.generator = ExpressionGenerator(n_inputs)

  def mutate(self, expression: Expression, mutation_rate: float = 0.1) -> Expression:
    """Mutate an expression with better exploration"""
    # Try different mutation strategies
    strategies = [
      self._point_mutation,
      self._subtree_mutation,
      self._insert_mutation,
      self._simplify_mutation
    ]

    for strategy in strategies:
      mutated = strategy(expression, mutation_rate)
      if mutated and mutated.complexity() <= self.max_complexity:
        return mutated

    # Fallback to original with small constant perturbation
    return self._safe_constant_mutation(expression)

  def _point_mutation(self, expression: Expression, rate: float) -> Optional[Expression]:
    """Point mutation - modify constants and operators"""
    mutated = expression.copy()
    nodes = self._get_all_nodes(mutated.root)

    changed = False
    for node in nodes:
      if random.random() < rate:
        if isinstance(node, ConstantNode):
          # More aggressive constant mutation
          if random.random() < 0.5:
            node.value += random.gauss(0, abs(node.value) * 0.3 + 0.1)
          else:
            node.value = random.uniform(-5, 5)
          changed = True
        elif isinstance(node, BinaryOpNode):
          # Change operator
          old_op = node.operator
          node.operator = random.choice(['+', '-', '*', '/'])
          if node.operator != old_op:
            changed = True
        elif isinstance(node, UnaryOpNode):
          # Change unary operator
          old_op = node.operator
          node.operator = random.choice(['sin', 'cos', 'exp', 'log', 'sqrt'])
          if node.operator != old_op:
            changed = True

    if changed:
      mutated.clear_cache()
      return mutated
    return None

  def _subtree_mutation(self, expression: Expression, rate: float) -> Optional[Expression]:
    """Replace a subtree with a new random subtree"""
    mutated = expression.copy()
    nodes = self._get_all_nodes(mutated.root)

    if len(nodes) <= 1:
      return None

    # Select a non-root node to replace
    target_node = random.choice(nodes[1:])

    # Generate replacement subtree
    max_depth = min(3, self.max_complexity // 4)
    new_subtree = self.generator._generate_node(0, max_depth)

    # Replace the subtree
    if self._replace_node_in_tree(mutated.root, target_node, new_subtree):
      mutated.clear_cache()
      return mutated
    return None

  def _insert_mutation(self, expression: Expression, rate: float) -> Optional[Expression]:
    """Insert a new operation around an existing node"""
    if expression.complexity() >= self.max_complexity - 2:
      return None

    mutated = expression.copy()
    nodes = self._get_all_nodes(mutated.root)

    target_node = random.choice(nodes)

    # Create new operation with target as operand
    if random.random() < 0.7:  # Binary operation
      op = random.choice(['+', '-', '*', '/'])
      other_operand = self.generator._generate_node(0, 2)

      if random.random() < 0.5:
        new_node = BinaryOpNode(op, target_node.copy(), other_operand)
      else:
        new_node = BinaryOpNode(op, other_operand, target_node.copy())
    else:  # Unary operation
      op = random.choice(['sin', 'cos', 'sqrt'])  # Safer unary ops
      new_node = UnaryOpNode(op, target_node.copy())

    # Replace target with new node
    if self._replace_node_in_tree(mutated.root, target_node, new_node):
      mutated.clear_cache()
      return mutated
    return None

  def _simplify_mutation(self, expression: Expression, rate: float) -> Optional[Expression]:
    """Apply simplification rules"""
    simplified_root = ExpressionSimplifier.simplify_expression(expression.root)
    if simplified_root and simplified_root != expression.root:
      return Expression(simplified_root)
    return None

  def _safe_constant_mutation(self, expression: Expression) -> Expression:
    """Safe fallback mutation that only changes constants slightly"""
    mutated = expression.copy()
    nodes = self._get_all_nodes(mutated.root)

    constant_nodes = [n for n in nodes if isinstance(n, ConstantNode)]
    if constant_nodes:
      node = random.choice(constant_nodes)
      node.value += random.gauss(0, 0.1)
      mutated.clear_cache()

    return mutated

  def crossover(self, parent1: Expression, parent2: Expression) -> Tuple[Expression, Expression]:
    """Enhanced crossover with proper subtree exchange"""
    # Try structural crossover first
    for _ in range(3):
      child1, child2 = self._structural_crossover(parent1, parent2)
      if (child1.complexity() <= self.max_complexity and
          child2.complexity() <= self.max_complexity):
        return child1, child2

    # Fallback to simpler crossover
    return self._simple_crossover(parent1, parent2)

  def _structural_crossover(self, parent1: Expression, parent2: Expression) -> Tuple[Expression, Expression]:
    """Exchange random subtrees between parents"""
    child1 = parent1.copy()
    child2 = parent2.copy()

    nodes1 = self._get_all_nodes(child1.root)
    nodes2 = self._get_all_nodes(child2.root)

    if len(nodes1) > 1 and len(nodes2) > 1:
      # Select crossover points (avoid root for diversity)
      idx1 = random.randint(1, len(nodes1) - 1)
      idx2 = random.randint(1, len(nodes2) - 1)

      node1 = nodes1[idx1]
      node2 = nodes2[idx2]

      # Exchange subtrees
      subtree1 = node1.copy()
      subtree2 = node2.copy()

      self._replace_node_in_tree(child1.root, node1, subtree2)
      self._replace_node_in_tree(child2.root, node2, subtree1)

      child1.clear_cache()
      child2.clear_cache()

    return child1, child2

  def _simple_crossover(self, parent1: Expression, parent2: Expression) -> Tuple[Expression, Expression]:
    """Simple parameter exchange crossover"""
    child1 = parent1.copy()
    child2 = parent2.copy()

    # Exchange constants
    constants1 = [n for n in self._get_all_nodes(child1.root) if isinstance(n, ConstantNode)]
    constants2 = [n for n in self._get_all_nodes(child2.root) if isinstance(n, ConstantNode)]

    if constants1 and constants2:
      # Random constant exchange
      for _ in range(min(len(constants1), len(constants2), 3)):
        if random.random() < 0.5:
          c1 = random.choice(constants1)
          c2 = random.choice(constants2)
          c1.value, c2.value = c2.value, c1.value

    child1.clear_cache()
    child2.clear_cache()
    return child1, child2

  def _replace_node_in_tree(self, root: Node, target: Node, replacement: Node) -> bool:
    """Replace target node with replacement in tree"""
    if root == target:
      return False  # Cannot replace root

    if isinstance(root, BinaryOpNode):
      if root.left == target:
        root.left = replacement
        return True
      elif root.right == target:
        root.right = replacement
        return True
      else:
        return (self._replace_node_in_tree(root.left, target, replacement) or
                self._replace_node_in_tree(root.right, target, replacement))
    elif isinstance(root, UnaryOpNode):
      if root.operand == target:
        root.operand = replacement
        return True
      else:
        return self._replace_node_in_tree(root.operand, target, replacement)

    return False

  def _get_all_nodes(self, node: Node) -> List[Node]:
    """Get all nodes in the tree (breadth-first)"""
    nodes = [node]
    if isinstance(node, BinaryOpNode):
      nodes.extend(self._get_all_nodes(node.left))
      nodes.extend(self._get_all_nodes(node.right))
    elif isinstance(node, UnaryOpNode):
      nodes.extend(self._get_all_nodes(node.operand))
    return nodes

  def generate_replacement(self, population: List[Expression], fitness_scores: List[float]) -> Expression:
    """Generate a replacement expression with guided diversity"""
    if len(population) >= 3:
      # Select diverse parents based on fitness and structure
      sorted_indices = np.argsort(fitness_scores)

      # Mix good and diverse individuals
      good_indices = sorted_indices[:len(population) // 3]
      diverse_indices = self._get_diverse_individuals(population, 3)

      parent_indices = list(set(good_indices) | set(diverse_indices))
      parents = [population[i] for i in parent_indices[:3]]

      # Create hybrid offspring
      if len(parents) >= 2:
        child1, child2 = self.crossover(parents[0], parents[1])
        child = self.mutate(child1, 0.3)  # Higher mutation for replacement
        return child

    # Fallback to new random expression
    return Expression(self.generator.generate_random_expression(max_depth=4))

  def _get_diverse_individuals(self, population: List[Expression], count: int) -> List[int]:
    """Select diverse individuals based on structure differences"""
    if len(population) <= count:
      return list(range(len(population)))

    selected = [0]  # Start with first individual

    for _ in range(count - 1):
      best_candidate = -1
      max_diversity = -1

      for i in range(len(population)):
        if i in selected:
          continue

        # Calculate diversity score (structural difference)
        diversity = sum(self._structural_distance(population[i], population[j])
                        for j in selected)

        if diversity > max_diversity:
          max_diversity = diversity
          best_candidate = i

      if best_candidate >= 0:
        selected.append(best_candidate)

    return selected

  def _structural_distance(self, expr1: Expression, expr2: Expression) -> float:
    """Calculate structural distance between two expressions"""
    # Simple structural distance based on size and string comparison
    size_diff = abs(expr1.complexity() - expr2.complexity())
    string_diff = len(set(expr1.to_string()) ^ set(expr2.to_string()))
    return size_diff + string_diff * 0.1