"""
Mutation Strategies Module

Provides various mutation strategies for symbolic regression including
context-aware, semantic-preserving, and legacy mutation operations.
"""

import numpy as np
import random
from typing import Optional, Dict, List

from ..expression_tree import Expression, Node, BinaryOpNode, UnaryOpNode, ConstantNode, VariableNode, ScalingOpNode
from ..expression_tree.utils.simplifier import ExpressionSimplifier
from ..generator import ExpressionGenerator
from .context_analysis import ExpressionContextAnalyzer


class MutationStrategies:
    """Collection of advanced mutation strategies for genetic programming"""
    
    def __init__(self, n_inputs: int, max_complexity: int = 20):
        self.n_inputs = n_inputs
        self.max_complexity = max_complexity
        self.generator = ExpressionGenerator(n_inputs)
        self.context_analyzer = ExpressionContextAnalyzer(n_inputs)
    
    def context_aware_mutation(self, expression: Expression, rate: float, 
                              X: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None) -> Optional[Expression]:
        """Context-aware mutation that considers the semantic role of nodes"""
        mutated = expression.copy()
        nodes = self._get_all_nodes(mutated.root)
        
        if len(nodes) <= 1:
            return None
        
        # **PERFORMANCE FIX**: Only analyze node importance if data is available and expression is complex
        if X is not None and y is not None and len(nodes) > 5:
            try:
                node_importance = self.context_analyzer.calculate_node_importance(expression, X, y)
            except:
                # Fallback to uniform importance if analysis fails
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
                    
                elif isinstance(node, (BinaryOpNode, UnaryOpNode, ScalingOpNode)):
                    # Grammar-aware operator mutation
                    if isinstance(node, BinaryOpNode):
                        changed = self._mutate_binary_operator(node) or changed
                    elif isinstance(node, UnaryOpNode):
                        changed = self._mutate_unary_operator(node) or changed
                    elif isinstance(node, ScalingOpNode):
                        # TODO: BETTER MUTSTION
                        node.power += random.choice([-1, 1])
                        changed = True
        
        if changed:
            mutated.clear_cache()
            return mutated
        return None
    
    def _mutate_binary_operator(self, node: BinaryOpNode) -> bool:
        """Mutate binary operator with semantic grouping"""
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
            return True
        return False
    
    def _mutate_unary_operator(self, node: UnaryOpNode) -> bool:
        """Mutate unary operator with semantic grouping and physics bias"""
        # Enhanced operator groups for physics
        trig_ops = ['sin', 'cos', 'tan']
        exp_log_ops = ['exp', 'log', 'log_abs']
        power_ops = ['sqrt', 'cbrt', 'fourth_root', 'square', 'cube']
        reciprocal_ops = ['reciprocal', 'inv_square']  # Critical for physics
        hyperbolic_ops = ['sinh', 'cosh', 'tanh']
        safe_ops = ['sqrt_abs', 'log_abs', 'abs', 'neg']
        
        current_op = node.operator
        
        # Physics-biased mutation with higher probabilities for critical operators
        if current_op in trig_ops and random.random() < 0.8:
            new_ops = [op for op in trig_ops if op != current_op]
            if new_ops:
                node.operator = random.choice(new_ops)
                return True
        elif current_op in exp_log_ops and random.random() < 0.7:
            new_ops = [op for op in exp_log_ops if op != current_op]
            if new_ops:
                node.operator = random.choice(new_ops)
                return True
        elif current_op in power_ops and random.random() < 0.8:
            new_ops = [op for op in power_ops if op != current_op]
            if new_ops:
                node.operator = random.choice(new_ops)
                return True
        elif current_op in reciprocal_ops and random.random() < 0.9:  # High preservation for physics
            new_ops = [op for op in reciprocal_ops if op != current_op]
            if new_ops:
                node.operator = random.choice(new_ops)
                return True
        elif current_op in hyperbolic_ops and random.random() < 0.7:
            new_ops = [op for op in hyperbolic_ops if op != current_op]
            if new_ops:
                node.operator = random.choice(new_ops)
                return True
        elif current_op in safe_ops and random.random() < 0.6:
            new_ops = [op for op in safe_ops if op != current_op]
            if new_ops:
                node.operator = random.choice(new_ops)
                return True
        else:
            # Cross-group mutation with physics bias towards reciprocal operations
            all_ops = trig_ops + exp_log_ops + power_ops + reciprocal_ops + hyperbolic_ops + safe_ops
            new_ops = [op for op in all_ops if op != current_op]
            if new_ops:
                # Bias towards reciprocal operations for physics laws
                if random.random() < 0.3:  # 30% chance to choose reciprocal
                    physics_critical = reciprocal_ops + ['sqrt', 'square', 'log_abs']
                    available_critical = [op for op in physics_critical if op != current_op]
                    if available_critical:
                        node.operator = random.choice(available_critical)
                        return True
                
                node.operator = random.choice(new_ops)
                return True
        return False
    
    def semantic_preserving_mutation(self, expression: Expression, rate: float,
                                   X: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None) -> Optional[Expression]:
        """Advanced mutation that preserves semantic meaning while changing structure"""
        mutated = expression.copy()
        nodes = self._get_all_nodes(mutated.root)
        
        if len(nodes) <= 2:
            return None
        
        # **PERFORMANCE FIX**: Further reduce attempts and focus on simpler transformations
        max_attempts = 1  # Reduced from 2 to 1 for maximum performance
        
        # Try sophisticated semantic transformations
        for _ in range(max_attempts):
            target_node = random.choice(nodes[1:])  # Skip root
            
            if isinstance(target_node, BinaryOpNode):
                if self._apply_binary_transformation(mutated.root, target_node):
                    mutated.clear_cache()
                    return mutated
            
            elif isinstance(target_node, UnaryOpNode):
                if self._apply_unary_transformation(mutated.root, target_node):
                    mutated.clear_cache()
                    return mutated
        
        return None
    
    def _apply_binary_transformation(self, root: Node, target_node: BinaryOpNode) -> bool:
        """Apply algebraic transformations to binary operators"""
        # Identity transformations
        if target_node.operator == '+' and isinstance(target_node.right, ConstantNode):
            if abs(target_node.right.value) < 1e-6:
                # Remove addition of zero: x + 0 → x
                return self._replace_node_in_tree(root, target_node, target_node.left)
        
        elif target_node.operator == '*':
            if isinstance(target_node.right, ConstantNode):
                if abs(target_node.right.value - 1.0) < 1e-6:
                    # Remove multiplication by one: x * 1 → x
                    return self._replace_node_in_tree(root, target_node, target_node.left)
                elif abs(target_node.right.value) < 1e-6:
                    # Multiplication by zero: x * 0 → 0
                    zero_node = ConstantNode(0)
                    return self._replace_node_in_tree(root, target_node, zero_node)
                elif abs(target_node.right.value + 1.0) < 1e-6:
                    # x * (-1) → -x (convert to unary minus if supported)
                    neg_node = UnaryOpNode('-', target_node.left.copy()) if hasattr(UnaryOpNode, '__init__') else None
                    if neg_node:
                        return self._replace_node_in_tree(root, target_node, neg_node)
        
        elif target_node.operator == '/':
            if isinstance(target_node.right, ConstantNode):
                if abs(target_node.right.value - 1.0) < 1e-6:
                    # Division by one: x / 1 → x
                    return self._replace_node_in_tree(root, target_node, target_node.left)
        
        elif target_node.operator == '-':
            if isinstance(target_node.right, ConstantNode) and abs(target_node.right.value) < 1e-6:
                # Subtraction of zero: x - 0 → x
                return self._replace_node_in_tree(root, target_node, target_node.left)
        
        # Commutative property transformations
        if target_node.operator in ['+', '*'] and random.random() < 0.3:
            # Swap operands for commutative operators: a + b → b + a
            target_node.left, target_node.right = target_node.right, target_node.left
            return True
        
        # Distributive property (limited cases)
        if (target_node.operator == '*' and isinstance(target_node.right, BinaryOpNode) and 
            target_node.right.operator == '+' and random.random() < 0.2):
            # Apply distributive property: a * (b + c) → a*b + a*c
            a = target_node.left.copy()
            b = target_node.right.left.copy()
            c = target_node.right.right.copy()
            
            # Create a*b + a*c
            ab = BinaryOpNode('*', a.copy(), b)
            ac = BinaryOpNode('*', a.copy(), c) 
            distributed = BinaryOpNode('+', ab, ac)
            
            return self._replace_node_in_tree(root, target_node, distributed)
        
        return False
    
    def _apply_unary_transformation(self, root: Node, target_node: UnaryOpNode) -> bool:
        """Apply transformations to unary operators"""
        if target_node.operator == 'sin' and random.random() < 0.2:
            # sin(x) ≈ x for small x (first-order approximation, with low probability)
            if isinstance(target_node.operand, VariableNode) and random.random() < 0.1:
                return self._replace_node_in_tree(root, target_node, target_node.operand)
        
        elif target_node.operator == 'cos' and random.random() < 0.2:
            # cos(x) → sin(x + π/2) - use sin instead with low probability
            sin_node = UnaryOpNode('sin', target_node.operand.copy())
            return self._replace_node_in_tree(root, target_node, sin_node)
        
        elif target_node.operator == 'exp':
            # exp(0) = 1
            if isinstance(target_node.operand, ConstantNode) and abs(target_node.operand.value) < 1e-6:
                one_node = ConstantNode(1.0)
                return self._replace_node_in_tree(root, target_node, one_node)
        
        elif target_node.operator == 'log':
            # log(1) = 0
            if isinstance(target_node.operand, ConstantNode) and abs(target_node.operand.value - 1.0) < 1e-6:
                zero_node = ConstantNode(0.0)
                return self._replace_node_in_tree(root, target_node, zero_node)
        
        elif target_node.operator == 'sqrt':
            # sqrt(x^2) → |x| ≈ x (simplified)
            if (isinstance(target_node.operand, BinaryOpNode) and 
                target_node.operand.operator == '*' and
                target_node.operand.left == target_node.operand.right):
                # sqrt(x*x) → x
                return self._replace_node_in_tree(root, target_node, target_node.operand.left)
        
        return False
    
    def point_mutation(self, expression: Expression, rate: float, 
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
                elif isinstance(node, ScalingOpNode):
                    node.power = random.randint(-3, 3)
                    changed = True

        if changed:
            mutated.clear_cache()
            return mutated
        return None

    def subtree_mutation(self, expression: Expression, rate: float,
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

    def insert_mutation(self, expression: Expression, rate: float,
                       X: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None) -> Optional[Expression]:
        """Insert a new operation around an existing node"""
        if expression.complexity() >= self.max_complexity - 2:
            return None

        mutated = expression.copy()
        nodes = self._get_all_nodes(mutated.root)

        target_node = random.choice(nodes)

        # Create new operation with target as operand
        node_type = random.choices(['binary', 'unary', 'scale'], weights=[0.6, 0.3, 0.1])[0]
        if node_type == 'binary':  # Binary operation
            op = random.choice(['+', '-', '*', '/'])
            other_operand = self.generator._generate_node(0, 2)

            if random.random() < 0.5:
                new_node = BinaryOpNode(op, target_node.copy(), other_operand)
            else:
                new_node = BinaryOpNode(op, other_operand, target_node.copy())
        elif node_type == 'unary':  # Unary operation
            op = random.choice(['sin', 'cos', 'sqrt'])  # Safer unary ops
            new_node = UnaryOpNode(op, target_node.copy())
        else: # Scaling operation
            power = random.randint(-3, 3)
            new_node = ScalingOpNode(power, target_node.copy())

        # Replace target with new node
        if self._replace_node_in_tree(mutated.root, target_node, new_node):
            mutated.clear_cache()
            return mutated
        return None

    def simplify_mutation(self, expression: Expression, rate: float,
                         X: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None) -> Optional[Expression]:
        """Apply simplification rules"""
        simplified_root = ExpressionSimplifier.simplify_expression(expression.root)
        if simplified_root and simplified_root != expression.root:
            return Expression(simplified_root)
        return None

    def safe_constant_mutation(self, expression: Expression) -> Expression:
        """Safe fallback mutation that only changes constants slightly"""
        mutated = expression.copy()
        nodes = self._get_all_nodes(mutated.root)

        constant_nodes = [n for n in nodes if isinstance(n, ConstantNode)]
        if constant_nodes:
            node = random.choice(constant_nodes)
            node.value += random.gauss(0, 0.1)
            mutated.clear_cache()

        return mutated
    
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
        elif isinstance(root, ScalingOpNode):
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
        elif isinstance(node, ScalingOpNode):
            nodes.extend(self._get_all_nodes(node.operand))
        return nodes
