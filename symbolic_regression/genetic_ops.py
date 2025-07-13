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
    """Advanced expression context analysis for intelligent mutation strategy selection"""
    nodes = self._get_all_nodes(expression.root)
    
    if not nodes:
      return {'complexity': 0, 'depth': 0, 'constant_ratio': 0, 'variable_ratio': 0,
              'operator_ratio': 0, 'has_repeated_patterns': False, 'node_count': 0,
              'operator_diversity': 0, 'structural_balance': 0, 'nonlinearity_score': 0,
              'symmetry_score': 0, 'redundancy_score': 0}
    
    # Count node types
    constants = [n for n in nodes if isinstance(n, ConstantNode)]
    variables = [n for n in nodes if isinstance(n, VariableNode)]
    binary_ops = [n for n in nodes if isinstance(n, BinaryOpNode)]
    unary_ops = [n for n in nodes if isinstance(n, UnaryOpNode)]
    
    # Advanced pattern detection
    expr_str = expression.to_string()
    
    # 1. Operator diversity analysis
    operator_types = set()
    for node in binary_ops + unary_ops:
      operator_types.add(node.operator)
    operator_diversity = len(operator_types) / max(1, len(binary_ops) + len(unary_ops))
    
    # 2. Structural balance (tree symmetry and balance)
    def calculate_subtree_sizes(node):
      if isinstance(node, (ConstantNode, VariableNode)):
        return 1
      elif isinstance(node, UnaryOpNode):
        return 1 + calculate_subtree_sizes(node.operand)
      elif isinstance(node, BinaryOpNode):
        left_size = calculate_subtree_sizes(node.left)
        right_size = calculate_subtree_sizes(node.right)
        return 1 + left_size + right_size
      return 1
    
    structural_balance = 0.5  # Default for single nodes
    if isinstance(expression.root, BinaryOpNode):
      left_size = calculate_subtree_sizes(expression.root.left)
      right_size = calculate_subtree_sizes(expression.root.right)
      total_size = left_size + right_size
      if total_size > 0:
        structural_balance = 1.0 - abs(left_size - right_size) / total_size
    
    # 3. Nonlinearity score (presence of nonlinear functions)
    nonlinear_ops = {'sin', 'cos', 'exp', 'log', 'sqrt', 'tan', 'pow'}
    nonlinear_count = sum(1 for node in binary_ops + unary_ops if node.operator in nonlinear_ops)
    nonlinearity_score = nonlinear_count / max(1, len(binary_ops) + len(unary_ops))
    
    # 4. Improved redundancy detection using AST patterns
    redundancy_score = self._calculate_redundancy_score(expression.root)
    
    # 5. Variable usage symmetry
    var_usage = {}
    for node in variables:
      var_usage[node.index] = var_usage.get(node.index, 0) + 1
    
    if var_usage:
      usage_values = list(var_usage.values())
      mean_usage = sum(usage_values) / len(usage_values)
      variance = sum((v - mean_usage) ** 2 for v in usage_values) / len(usage_values)
      symmetry_score = 1.0 / (1.0 + variance)  # Higher score for balanced variable usage
    else:
      symmetry_score = 1.0
    
    # 6. Complexity distribution analysis
    depth = self._calculate_depth(expression.root)
    complexity = expression.complexity()
    
    return {
      'complexity': complexity,
      'depth': depth,
      'constant_ratio': len(constants) / len(nodes),
      'variable_ratio': len(variables) / len(nodes),
      'operator_ratio': (len(binary_ops) + len(unary_ops)) / len(nodes),
      'has_repeated_patterns': redundancy_score > 0.3,
      'node_count': len(nodes),
      'operator_diversity': operator_diversity,
      'structural_balance': structural_balance,
      'nonlinearity_score': nonlinearity_score,
      'symmetry_score': symmetry_score,
      'redundancy_score': redundancy_score,
      'is_linear': nonlinearity_score < 0.1,
      'is_balanced': structural_balance > 0.7,
      'is_diverse': operator_diversity > 0.5
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
    """Advanced mutation that preserves semantic meaning while changing structure"""
    mutated = expression.copy()
    nodes = self._get_all_nodes(mutated.root)
    
    if len(nodes) <= 2:
      return None
    
    # Try sophisticated semantic transformations
    for _ in range(5):  # More attempts for better success rate
      target_node = random.choice(nodes[1:])  # Skip root
      
      if isinstance(target_node, BinaryOpNode):
        # Advanced algebraic transformations
        transformation_applied = False
        
        # Identity transformations
        if target_node.operator == '+' and isinstance(target_node.right, ConstantNode):
          if abs(target_node.right.value) < 1e-6:
            # Remove addition of zero: x + 0 → x
            if self._replace_node_in_tree(mutated.root, target_node, target_node.left):
              transformation_applied = True
        
        elif target_node.operator == '*':
          if isinstance(target_node.right, ConstantNode):
            if abs(target_node.right.value - 1.0) < 1e-6:
              # Remove multiplication by one: x * 1 → x
              if self._replace_node_in_tree(mutated.root, target_node, target_node.left):
                transformation_applied = True
            elif abs(target_node.right.value) < 1e-6:
              # Multiplication by zero: x * 0 → 0
              zero_node = ConstantNode(0)
              if self._replace_node_in_tree(mutated.root, target_node, zero_node):
                transformation_applied = True
            elif abs(target_node.right.value + 1.0) < 1e-6:
              # x * (-1) → -x (convert to unary minus if supported)
              neg_node = UnaryOpNode('-', target_node.left.copy()) if hasattr(UnaryOpNode, '__init__') else None
              if neg_node and self._replace_node_in_tree(mutated.root, target_node, neg_node):
                transformation_applied = True
        
        elif target_node.operator == '/':
          if isinstance(target_node.right, ConstantNode):
            if abs(target_node.right.value - 1.0) < 1e-6:
              # Division by one: x / 1 → x
              if self._replace_node_in_tree(mutated.root, target_node, target_node.left):
                transformation_applied = True
        
        elif target_node.operator == '-':
          if isinstance(target_node.right, ConstantNode) and abs(target_node.right.value) < 1e-6:
            # Subtraction of zero: x - 0 → x
            if self._replace_node_in_tree(mutated.root, target_node, target_node.left):
              transformation_applied = True
        
        # Commutative property transformations
        if not transformation_applied and target_node.operator in ['+', '*']:
          # Swap operands for commutative operators: a + b → b + a
          if random.random() < 0.3:
            target_node.left, target_node.right = target_node.right, target_node.left
            transformation_applied = True
        
        # Distributive property (limited cases)
        if not transformation_applied and target_node.operator == '*':
          # Look for patterns like a * (b + c) → a*b + a*c (with probability)
          if (isinstance(target_node.right, BinaryOpNode) and 
              target_node.right.operator == '+' and random.random() < 0.2):
            # Apply distributive property
            a = target_node.left.copy()
            b = target_node.right.left.copy()
            c = target_node.right.right.copy()
            
            # Create a*b + a*c
            ab = BinaryOpNode('*', a.copy(), b)
            ac = BinaryOpNode('*', a.copy(), c) 
            distributed = BinaryOpNode('+', ab, ac)
            
            if self._replace_node_in_tree(mutated.root, target_node, distributed):
              transformation_applied = True
        
        if transformation_applied:
          mutated.clear_cache()
          return mutated
      
      elif isinstance(target_node, UnaryOpNode):
        # Advanced trigonometric and exponential identities
        transformation_applied = False
        
        if target_node.operator == 'sin':
          if random.random() < 0.2:
            # sin(x) → cos(x - π/2) - complex, so use simpler transformations
            # sin(x) ≈ x for small x (first-order approximation, with low probability)
            if isinstance(target_node.operand, VariableNode) and random.random() < 0.1:
              if self._replace_node_in_tree(mutated.root, target_node, target_node.operand):
                transformation_applied = True
        
        elif target_node.operator == 'cos':
          if random.random() < 0.2:
            # cos(x) → sin(x + π/2) - use sin instead with low probability
            sin_node = UnaryOpNode('sin', target_node.operand.copy())
            if self._replace_node_in_tree(mutated.root, target_node, sin_node):
              transformation_applied = True
        
        elif target_node.operator == 'exp':
          # exp(0) = 1
          if isinstance(target_node.operand, ConstantNode) and abs(target_node.operand.value) < 1e-6:
            one_node = ConstantNode(1.0)
            if self._replace_node_in_tree(mutated.root, target_node, one_node):
              transformation_applied = True
        
        elif target_node.operator == 'log':
          # log(1) = 0
          if isinstance(target_node.operand, ConstantNode) and abs(target_node.operand.value - 1.0) < 1e-6:
            zero_node = ConstantNode(0.0)
            if self._replace_node_in_tree(mutated.root, target_node, zero_node):
              transformation_applied = True
        
        elif target_node.operator == 'sqrt':
          # sqrt(x^2) → |x| ≈ x (simplified)
          if (isinstance(target_node.operand, BinaryOpNode) and 
              target_node.operand.operator == '*' and
              target_node.operand.left == target_node.operand.right):
            # sqrt(x*x) → x
            if self._replace_node_in_tree(mutated.root, target_node, target_node.operand.left):
              transformation_applied = True
        
        if transformation_applied:
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
    """Calculate advanced structural distance between two expressions"""
    try:
      # 1. Size-based distance
      size1, size2 = expr1.complexity(), expr2.complexity()
      size_diff = abs(size1 - size2) / max(1, max(size1, size2))
      
      # 2. Depth-based distance  
      depth1 = self._calculate_depth(expr1.root)
      depth2 = self._calculate_depth(expr2.root)
      depth_diff = abs(depth1 - depth2) / max(1, max(depth1, depth2))
      
      # 3. Operator signature distance
      ops1 = self._get_operator_signature(expr1.root)
      ops2 = self._get_operator_signature(expr2.root)
      
      # Calculate Jaccard distance for operator sets
      union = set(ops1.keys()) | set(ops2.keys())
      intersection_count = sum(min(ops1.get(op, 0), ops2.get(op, 0)) for op in union)
      union_count = sum(max(ops1.get(op, 0), ops2.get(op, 0)) for op in union)
      
      operator_distance = 1.0 - (intersection_count / max(1, union_count))
      
      # 4. Tree shape distance (using tree edit distance approximation)
      shape_distance = self._approximate_tree_edit_distance(expr1.root, expr2.root)
      
      # 5. Semantic distance (if expressions are different but evaluate similarly)
      semantic_distance = 0.0
      try:
        test_points = np.random.randn(10, self.n_inputs)
        vals1 = expr1.evaluate(test_points)
        vals2 = expr2.evaluate(test_points)
        if vals1 is not None and vals2 is not None:
          correlation = np.corrcoef(vals1.flatten(), vals2.flatten())[0, 1]
          if not np.isnan(correlation):
            semantic_distance = 1.0 - abs(correlation)
      except:
        semantic_distance = 1.0  # Default high distance if evaluation fails
      
      # Combine distances with weights
      total_distance = (
        0.2 * size_diff +
        0.15 * depth_diff + 
        0.3 * operator_distance +
        0.25 * shape_distance +
        0.1 * semantic_distance
      )
      
      return min(1.0, total_distance)
      
    except Exception:
      # Fallback to simple distance
      return abs(expr1.complexity() - expr2.complexity()) / max(1, max(expr1.complexity(), expr2.complexity()))

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

  def _calculate_redundancy_score(self, root: Node) -> float:
    """Calculate redundancy score based on repeated subtree patterns"""
    try:
      # Get all subtrees and their string representations
      subtrees = []
      
      def collect_subtrees(node):
        if isinstance(node, (ConstantNode, VariableNode)):
          subtrees.append(node.to_string() if hasattr(node, 'to_string') else str(node))
        elif isinstance(node, UnaryOpNode):
          subtree_str = f"{node.operator}({node.operand.to_string() if hasattr(node.operand, 'to_string') else str(node.operand)})"
          subtrees.append(subtree_str)
          collect_subtrees(node.operand)
        elif isinstance(node, BinaryOpNode):
          left_str = node.left.to_string() if hasattr(node.left, 'to_string') else str(node.left)
          right_str = node.right.to_string() if hasattr(node.right, 'to_string') else str(node.right)
          subtree_str = f"({left_str} {node.operator} {right_str})"
          subtrees.append(subtree_str)
          collect_subtrees(node.left)
          collect_subtrees(node.right)
      
      collect_subtrees(root)
      
      if len(subtrees) <= 1:
        return 0.0
      
      # Count frequency of each subtree pattern
      from collections import Counter
      subtree_counts = Counter(subtrees)
      
      # Calculate redundancy as ratio of repeated subtrees
      total_subtrees = len(subtrees)
      unique_subtrees = len(subtree_counts)
      repeated_count = sum(count - 1 for count in subtree_counts.values() if count > 1)
      
      redundancy_score = repeated_count / max(1, total_subtrees)
      return min(1.0, redundancy_score)
      
    except Exception:
      # Fallback to simple string-based redundancy
      expr_str = str(root)
      tokens = expr_str.split()
      if len(tokens) <= 1:
        return 0.0
      unique_tokens = len(set(tokens))
      redundancy = 1.0 - (unique_tokens / len(tokens))
      return max(0.0, redundancy)

  def _get_operator_signature(self, node: Node) -> Dict[str, int]:
    """Get operator frequency signature for a tree"""
    signature = {}
    
    def collect_operators(n):
      if isinstance(n, BinaryOpNode):
        signature[f"binary_{n.operator}"] = signature.get(f"binary_{n.operator}", 0) + 1
        collect_operators(n.left)
        collect_operators(n.right)
      elif isinstance(n, UnaryOpNode):
        signature[f"unary_{n.operator}"] = signature.get(f"unary_{n.operator}", 0) + 1
        collect_operators(n.operand)
      elif isinstance(n, ConstantNode):
        signature["constant"] = signature.get("constant", 0) + 1
      elif isinstance(n, VariableNode):
        signature[f"var_{n.index}"] = signature.get(f"var_{n.index}", 0) + 1
    
    collect_operators(node)
    return signature
  
  def _approximate_tree_edit_distance(self, node1: Node, node2: Node) -> float:
    """Approximate tree edit distance using recursive structure comparison"""
    try:
      # Base cases
      if type(node1) != type(node2):
        return 1.0
      
      if isinstance(node1, ConstantNode) and isinstance(node2, ConstantNode):
        # Normalized difference between constants
        diff = abs(node1.value - node2.value)
        return min(1.0, diff / (1.0 + max(abs(node1.value), abs(node2.value))))
      
      if isinstance(node1, VariableNode) and isinstance(node2, VariableNode):
        return 0.0 if node1.index == node2.index else 1.0
      
      if isinstance(node1, BinaryOpNode) and isinstance(node2, BinaryOpNode):
        op_diff = 0.0 if node1.operator == node2.operator else 0.5
        left_diff = self._approximate_tree_edit_distance(node1.left, node2.left)
        right_diff = self._approximate_tree_edit_distance(node1.right, node2.right)
        return (op_diff + left_diff + right_diff) / 3.0
      
      if isinstance(node1, UnaryOpNode) and isinstance(node2, UnaryOpNode):
        op_diff = 0.0 if node1.operator == node2.operator else 0.5
        operand_diff = self._approximate_tree_edit_distance(node1.operand, node2.operand)
        return (op_diff + operand_diff) / 2.0
      
      return 1.0  # Different node types
      
    except Exception:
      return 1.0  # Default high distance for errors