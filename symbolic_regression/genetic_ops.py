import numpy as np
import random
from typing import List, Tuple, Optional, Dict, Callable

from .quality_assessment import calculate_subtree_qualities
from .expression_tree import Expression, Node, BinaryOpNode, UnaryOpNode, ConstantNode, VariableNode
from .expression_tree.utils.simplifier import ExpressionSimplifier
from .generator import ExpressionGenerator


class GeneticOperations:
  """Enhanced genetic operations with Grammatical Evolution-inspired improvements"""

  def __init__(self, n_inputs: int, max_complexity: int = 20):
    self.n_inputs = n_inputs
    self.max_complexity = max_complexity
    self.generator = ExpressionGenerator(n_inputs)
    
    # Context-aware mutation tracking
    self.mutation_success_rates = {
      'point': 0.5,
      'subtree': 0.3,
      'insert': 0.2,
      'context_aware': 0.4,
      'semantic_preserving': 0.3,
      'legacy_simplify': 0.2
    }
    self.mutation_attempts = {key: 1 for key in self.mutation_success_rates}
    self.mutation_successes = {key: 1 for key in self.mutation_success_rates}

  def mutate(self, expression: Expression, mutation_rate: float = 0.1, X: Optional[np.ndarray] = None, 
             y: Optional[np.ndarray] = None) -> Expression:
    """Enhanced mutation with Grammatical Evolution-inspired context awareness"""
    
    # Update success rates for adaptive strategy selection
    self._update_success_rates()
    
    # Context-aware strategy selection based on expression characteristics
    strategies = self._select_mutation_strategies(expression, X, y)
    
    # Try strategies in order of predicted success
    for strategy_name, strategy_func in strategies:
      self.mutation_attempts[strategy_name] += 1
      
      mutated = strategy_func(expression, mutation_rate, X, y)
      if mutated and mutated.complexity() <= self.max_complexity:
        # Verify the mutation improves or maintains fitness
        if self._is_beneficial_mutation(expression, mutated, X, y):
          self.mutation_successes[strategy_name] += 1
          return mutated
    
    # Fallback to safe constant mutation
    return self._safe_constant_mutation(expression)
  
  def _select_mutation_strategies(self, expression: Expression, X: Optional[np.ndarray] = None, 
                                y: Optional[np.ndarray] = None) -> List[Tuple[str, Callable]]:
    """Select and order mutation strategies based on context and success rates"""
    
    # Analyze expression characteristics
    context = self._analyze_expression_context(expression)
    
    # Build strategy list with priorities
    strategies = []
    
    # High complexity expressions benefit from simplification
    if context['complexity'] > self.max_complexity * 0.7:
      strategies.append(('semantic_preserving', self._semantic_preserving_mutation))
      strategies.append(('legacy_simplify', self._legacy_simplify_mutation))
    
    # Low diversity expressions need more radical changes
    if context['has_repeated_patterns']:
      strategies.append(('context_aware', self._context_aware_mutation))
      strategies.append(('subtree', self._legacy_subtree_mutation))
    
    # Expressions with many constants benefit from constant optimization
    if context['constant_ratio'] > 0.3:
      strategies.append(('point', self._legacy_point_mutation))
    
    # Always include standard strategies, ordered by success rate
    standard_strategies = [
      ('context_aware', self._context_aware_mutation),
      ('point', self._legacy_point_mutation),
      ('subtree', self._legacy_subtree_mutation),
      ('insert', self._legacy_insert_mutation)
    ]
    
    # Sort by success rate
    standard_strategies.sort(key=lambda x: self.mutation_success_rates.get(x[0], 0.1), reverse=True)
    strategies.extend(standard_strategies)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_strategies = []
    for name, func in strategies:
      if name not in seen:
        unique_strategies.append((name, func))
        seen.add(name)
    
    return unique_strategies
  
  def _analyze_expression_context(self, expression: Expression) -> Dict:
    """Analyze expression to determine context for mutation strategy selection"""
    nodes = self._get_all_nodes(expression.root)
    
    # Count node types
    constants = [n for n in nodes if isinstance(n, ConstantNode)]
    variables = [n for n in nodes if isinstance(n, VariableNode)]
    operators = [n for n in nodes if isinstance(n, (BinaryOpNode, UnaryOpNode))]
    
    # Detect patterns
    expr_str = expression.to_string()
    has_repeated_patterns = len(set(expr_str.split())) < len(expr_str.split()) * 0.7
    
    return {
      'complexity': expression.complexity(),
      'depth': self._calculate_depth(expression.root),
      'constant_ratio': len(constants) / len(nodes) if nodes else 0,
      'variable_ratio': len(variables) / len(nodes) if nodes else 0,
      'operator_ratio': len(operators) / len(nodes) if nodes else 0,
      'has_repeated_patterns': has_repeated_patterns,
      'node_count': len(nodes)
    }
  
  def _context_aware_mutation(self, expression: Expression, rate: float, 
                            X: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None) -> Optional[Expression]:
    """Context-aware mutation that considers the semantic role of nodes"""
    mutated = expression.copy()
    nodes = self._get_all_nodes(mutated.root)
    
    if len(nodes) <= 1:
      return None
    
    # Analyze node importance if data is available
    node_importance = {}
    if X is not None and y is not None:
      try:
        # Calculate gradient/sensitivity for each node
        for i, node in enumerate(nodes):
          if isinstance(node, ConstantNode):
            # Test sensitivity to constant changes
            original_value = node.value
            node.value += 0.01
            pred_plus = expression.evaluate(X)
            node.value = original_value - 0.01  
            pred_minus = expression.evaluate(X)
            node.value = original_value
            
            sensitivity = np.mean(np.abs(pred_plus - pred_minus))
            node_importance[i] = sensitivity
          else:
            node_importance[i] = 1.0  # Default importance
      except:
        # Fallback to uniform importance
        node_importance = {i: 1.0 for i in range(len(nodes))}
    else:
      node_importance = {i: 1.0 for i in range(len(nodes))}
    
    # Select nodes for mutation based on importance (lower importance = higher mutation probability)
    total_importance = sum(node_importance.values())
    mutation_probs = {i: (total_importance - imp) / (total_importance * len(nodes)) 
                     for i, imp in node_importance.items()}
    
    changed = False
    for i, node in enumerate(nodes):
      if random.random() < mutation_probs.get(i, rate):
        if isinstance(node, ConstantNode):
          # Smart constant mutation based on sensitivity
          sensitivity = node_importance.get(i, 1.0)
          mutation_scale = 0.1 / (sensitivity + 0.01)  # Less sensitive nodes get larger mutations
          
          if random.random() < 0.7:
            node.value += random.gauss(0, abs(node.value) * mutation_scale + 0.1)
          else:
            node.value = random.uniform(-3, 3)
          changed = True
          
        elif isinstance(node, (BinaryOpNode, UnaryOpNode)):
          # Grammar-aware operator mutation
          if isinstance(node, BinaryOpNode):
            # Group operators by semantic similarity
            arithmetic_ops = ['+', '-', '*', '/']
            current_op = node.operator
            if current_op in arithmetic_ops:
              # Prefer semantically similar operators
              if random.random() < 0.7:
                similar_ops = [op for op in arithmetic_ops if op != current_op]
                node.operator = random.choice(similar_ops) if similar_ops else current_op
              else:
                node.operator = random.choice(arithmetic_ops)
              changed = True
          
          elif isinstance(node, UnaryOpNode):
            # Group unary operators
            trig_ops = ['sin', 'cos']
            other_ops = ['exp', 'log', 'sqrt']
            current_op = node.operator
            
            if current_op in trig_ops and random.random() < 0.8:
              new_ops = [op for op in trig_ops if op != current_op]
              if new_ops:
                node.operator = random.choice(new_ops)
                changed = True
            elif current_op in other_ops and random.random() < 0.6:
              new_ops = [op for op in other_ops if op != current_op]
              if new_ops:
                node.operator = random.choice(new_ops)
                changed = True
            else:
              # Cross-group mutation with lower probability
              all_ops = ['sin', 'cos', 'exp', 'log', 'sqrt']
              new_ops = [op for op in all_ops if op != current_op]
              if new_ops:
                node.operator = random.choice(new_ops)
                changed = True
    
    if changed:
      mutated.clear_cache()
      return mutated
    return None
  
  def _semantic_preserving_mutation(self, expression: Expression, rate: float,
                                  X: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None) -> Optional[Expression]:
    """Mutation that attempts to preserve semantic meaning while changing structure"""
    mutated = expression.copy()
    nodes = self._get_all_nodes(mutated.root)
    
    if len(nodes) <= 2:
      return None
    
    # Try to find semantically equivalent transformations
    for _ in range(3):  # Try multiple transformations
      # Look for patterns that can be simplified/transformed
      target_node = random.choice(nodes[1:])  # Skip root
      
      if isinstance(target_node, BinaryOpNode):
        # Apply algebraic transformations
        if target_node.operator == '+' and isinstance(target_node.right, ConstantNode):
          if target_node.right.value == 0:
            # Remove addition of zero
            if self._replace_node_in_tree(mutated.root, target_node, target_node.left):
              mutated.clear_cache()
              return mutated
        
        elif target_node.operator == '*' and isinstance(target_node.right, ConstantNode):
          if abs(target_node.right.value - 1.0) < 1e-6:
            # Remove multiplication by one
            if self._replace_node_in_tree(mutated.root, target_node, target_node.left):
              mutated.clear_cache()
              return mutated
          elif abs(target_node.right.value) < 1e-6:
            # Replace multiplication by zero with zero
            zero_node = ConstantNode(0)
            if self._replace_node_in_tree(mutated.root, target_node, zero_node):
              mutated.clear_cache()
              return mutated
      
      elif isinstance(target_node, UnaryOpNode):
        # Apply trigonometric identities with low probability
        if target_node.operator == 'sin' and random.random() < 0.3:
          # sin(x) can sometimes be replaced with cos(x - π/2) but this is complex
          # For now, just do simple operator swaps
          if random.random() < 0.5:
            target_node.operator = 'cos'
            mutated.clear_cache()
            return mutated
    
    return None
  
  def _calculate_depth(self, node: Node) -> int:
    """Calculate the depth of a node tree"""
    if isinstance(node, (ConstantNode, VariableNode)):
      return 1
    elif isinstance(node, UnaryOpNode):
      return 1 + self._calculate_depth(node.operand)
    elif isinstance(node, BinaryOpNode):
      return 1 + max(self._calculate_depth(node.left), self._calculate_depth(node.right))
    return 1
  
  def _is_beneficial_mutation(self, original: Expression, mutated: Expression,
                            X: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None) -> bool:
    """Check if a mutation is beneficial (maintains or improves fitness)"""
    if X is None or y is None:
      return True  # Accept if we can't evaluate
    
    try:
      original_fitness = self._evaluate_fitness(original, X, y)
      mutated_fitness = self._evaluate_fitness(mutated, X, y)
      
      # Accept if fitness is maintained or improved, or with small probability for exploration
      return mutated_fitness >= original_fitness - 0.01 or random.random() < 0.1
    except:
      return True  # Accept if evaluation fails
  
  def _update_success_rates(self):
    """Update success rates for adaptive strategy selection"""
    for strategy in self.mutation_success_rates:
      if self.mutation_attempts[strategy] > 0:
        success_rate = self.mutation_successes[strategy] / self.mutation_attempts[strategy]
        # Exponential moving average
        self.mutation_success_rates[strategy] = 0.9 * self.mutation_success_rates[strategy] + 0.1 * success_rate

  # Legacy methods preserved for compatibility
  def _legacy_point_mutation(self, expression: Expression, rate: float, 
                            X: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None) -> Optional[Expression]:
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

  def _legacy_subtree_mutation(self, expression: Expression, rate: float,
                              X: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None) -> Optional[Expression]:
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

  def _legacy_insert_mutation(self, expression: Expression, rate: float,
                             X: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None) -> Optional[Expression]:
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

  def _legacy_simplify_mutation(self, expression: Expression, rate: float,
                               X: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None) -> Optional[Expression]:
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

  def _evaluate_fitness(self, expr: Expression, X: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None, parsimony_coefficient: float = 0.001) -> float:
    """
    Evaluate the fitness of a single expression.
    If X and y are not provided, returns a large negative value.
    """
    if X is None or y is None:
      return -1e8

    try:
      predictions = expr.evaluate(X)
      if predictions.ndim == 1:
        predictions = predictions.reshape(-1, 1)
      mse = np.mean((y - predictions) ** 2)
      complexity_penalty = parsimony_coefficient * expr.complexity()
      stability_penalty = 0.0
      max_abs_pred = np.max(np.abs(predictions))
      if max_abs_pred > 1e6:
        stability_penalty = 0.5
      elif max_abs_pred > 1e4:
        stability_penalty = 0.1
      if np.any(~np.isfinite(predictions)):
        stability_penalty += 1.0
      fitness = -mse - complexity_penalty - stability_penalty
      return float(fitness)
    except Exception:
      return -1e8

  def quality_guided_crossover(self, parent1: Expression, parent2: Expression, X: np.ndarray, residuals1: np.ndarray, residuals2: np.ndarray) -> Tuple[Expression, Expression]:
    """
    Performs crossover by probabilistically swapping high-quality subtrees.
    """
    child1 = parent1.copy()
    child2 = parent2.copy()

    # 1. Get quality scores for all subtrees
    qualities1 = calculate_subtree_qualities(child1, X, residuals1)
    qualities2 = calculate_subtree_qualities(child2, X, residuals2)

    nodes1 = list(qualities1.keys())
    scores1 = np.array(list(qualities1.values()), dtype=np.float64)

    nodes2 = list(qualities2.keys())
    scores2 = np.array(list(qualities2.values()), dtype=np.float64)

    # 2. Fallback to simple structural crossover if quality calculation fails or is trivial
    if np.sum(scores1) < 1e-6 or np.sum(scores2) < 1e-6 or len(nodes1) <= 1 or len(nodes2) <= 1:
        return self._structural_crossover(parent1, parent2)

    # 3. Create probability distributions from quality scores
    probs1 = scores1 / np.sum(scores1)
    probs2 = scores2 / np.sum(scores2)

    # 4. Probabilistically select crossover points
    idx1 = np.random.choice(len(nodes1), p=probs1)
    idx2 = np.random.choice(len(nodes2), p=probs2)
    crossover_point1 = nodes1[idx1]
    crossover_point2 = nodes2[idx2]

    # 5. Avoid swapping the entire tree (root node)
    if crossover_point1 is child1.root or crossover_point2 is child2.root:
        return self._structural_crossover(parent1, parent2) # Fallback

    # 6. Perform the swap
    subtree1 = crossover_point1.copy()
    subtree2 = crossover_point2.copy()

    self._replace_node_in_tree(child1.root, crossover_point1, subtree2)
    self._replace_node_in_tree(child2.root, crossover_point2, subtree1)

    child1.clear_cache()
    child2.clear_cache()

    return child1, child2
  
  def adaptive_mutate_with_feedback(self, expression: Expression, mutation_rate: float = 0.1, 
                                   X: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None,
                                   generation: int = 0, stagnation_count: int = 0) -> Expression:
    """Enhanced mutation with adaptive strategy selection based on evolution state"""
    
    # Adjust mutation aggressiveness based on stagnation
    if stagnation_count > 15:
      # High stagnation - use more aggressive mutations
      mutation_rate *= 1.5
      aggressive_strategies = [
        ('context_aware', self._context_aware_mutation),
        ('subtree', self._legacy_subtree_mutation),
        ('semantic_preserving', self._semantic_preserving_mutation)
      ]
      
      for strategy_name, strategy_func in aggressive_strategies:
        mutated = strategy_func(expression, mutation_rate, X, y)
        if mutated and mutated.complexity() <= self.max_complexity:
          return mutated
    
    elif stagnation_count > 8:
      # Medium stagnation - prefer structure-changing mutations
      mutation_rate *= 1.2
      medium_strategies = [
        ('context_aware', self._context_aware_mutation),
        ('insert', self._legacy_insert_mutation),
        ('subtree', self._legacy_subtree_mutation)
      ]
      
      for strategy_name, strategy_func in medium_strategies:
        mutated = strategy_func(expression, mutation_rate, X, y)
        if mutated and mutated.complexity() <= self.max_complexity:
          return mutated
    
    # Normal mutation using the standard adaptive approach
    return self.mutate(expression, mutation_rate, X, y)
  
  def get_mutation_statistics(self) -> Dict[str, float]:
    """Get statistics about mutation strategy success rates"""
    stats = {}
    for strategy in self.mutation_success_rates:
      attempts = self.mutation_attempts.get(strategy, 1)
      successes = self.mutation_successes.get(strategy, 0)
      stats[f"{strategy}_success_rate"] = successes / attempts
      stats[f"{strategy}_attempts"] = attempts
    return stats