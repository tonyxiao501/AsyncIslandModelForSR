"""
Context Analysis Module for Expression Structure Analysis

Provides advanced context analysis capabilities for expression trees,
including structural metrics, redundancy detection, and complexity assessment.
"""

import numpy as np
from typing import Dict, List
from collections import Counter

from ..expression_tree import Expression, Node, BinaryOpNode, UnaryOpNode, ConstantNode, VariableNode


class ExpressionContextAnalyzer:
    """Advanced expression context analysis for intelligent strategy selection"""
    
    def __init__(self, n_inputs: int):
        self.n_inputs = n_inputs
    
    def analyze_expression_context(self, expression: Expression) -> Dict:
        """Advanced expression context analysis for intelligent mutation strategy selection"""
        nodes = self._get_all_nodes(expression.root)
        
        if not nodes:
            return self._empty_context()
        
        # Count node types
        constants = [n for n in nodes if isinstance(n, ConstantNode)]
        variables = [n for n in nodes if isinstance(n, VariableNode)]
        binary_ops = [n for n in nodes if isinstance(n, BinaryOpNode)]
        unary_ops = [n for n in nodes if isinstance(n, UnaryOpNode)]
        
        # Advanced pattern detection
        expr_str = expression.to_string()
        
        # 1. Operator diversity analysis
        operator_diversity = self._calculate_operator_diversity(binary_ops, unary_ops)
        
        # 2. Structural balance (tree symmetry and balance)
        structural_balance = self._calculate_structural_balance(expression.root)
        
        # 3. Nonlinearity score (presence of nonlinear functions)
        nonlinearity_score = self._calculate_nonlinearity_score(binary_ops, unary_ops)
        
        # 4. Improved redundancy detection using AST patterns
        redundancy_score = self._calculate_redundancy_score(expression.root)
        
        # 5. Variable usage symmetry
        symmetry_score = self._calculate_variable_symmetry(variables)
        
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
    
    def _empty_context(self) -> Dict:
        """Return empty context for trivial expressions"""
        return {
            'complexity': 0, 'depth': 0, 'constant_ratio': 0, 'variable_ratio': 0,
            'operator_ratio': 0, 'has_repeated_patterns': False, 'node_count': 0,
            'operator_diversity': 0, 'structural_balance': 0, 'nonlinearity_score': 0,
            'symmetry_score': 0, 'redundancy_score': 0, 'is_linear': True,
            'is_balanced': True, 'is_diverse': False
        }
    
    def _calculate_operator_diversity(self, binary_ops: List[BinaryOpNode], unary_ops: List[UnaryOpNode]) -> float:
        """Calculate operator diversity score"""
        operator_types = set()
        for node in binary_ops + unary_ops:
            operator_types.add(node.operator)
        return len(operator_types) / max(1, len(binary_ops) + len(unary_ops))
    
    def _calculate_structural_balance(self, root: Node) -> float:
        """Calculate structural balance (tree symmetry)"""
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
        if isinstance(root, BinaryOpNode):
            left_size = calculate_subtree_sizes(root.left)
            right_size = calculate_subtree_sizes(root.right)
            total_size = left_size + right_size
            if total_size > 0:
                structural_balance = 1.0 - abs(left_size - right_size) / total_size
        
        return structural_balance
    
    def _calculate_nonlinearity_score(self, binary_ops: List[BinaryOpNode], unary_ops: List[UnaryOpNode]) -> float:
        """Calculate nonlinearity score based on presence of nonlinear functions"""
        nonlinear_ops = {'sin', 'cos', 'exp', 'log', 'sqrt', 'tan', 'pow'}
        nonlinear_count = sum(1 for node in binary_ops + unary_ops if node.operator in nonlinear_ops)
        return nonlinear_count / max(1, len(binary_ops) + len(unary_ops))
    
    def _calculate_variable_symmetry(self, variables: List[VariableNode]) -> float:
        """Calculate variable usage symmetry"""
        var_usage = {}
        for node in variables:
            var_usage[node.index] = var_usage.get(node.index, 0) + 1
        
        if var_usage:
            usage_values = list(var_usage.values())
            mean_usage = sum(usage_values) / len(usage_values)
            variance = sum((v - mean_usage) ** 2 for v in usage_values) / len(usage_values)
            return 1.0 / (1.0 + variance)  # Higher score for balanced variable usage
        else:
            return 1.0
    
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
            subtree_counts = Counter(subtrees)
            
            # Calculate redundancy as ratio of repeated subtrees
            total_subtrees = len(subtrees)
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
    
    def _calculate_depth(self, node: Node) -> int:
        """Calculate the depth of a node tree"""
        if isinstance(node, (ConstantNode, VariableNode)):
            return 1
        elif isinstance(node, UnaryOpNode):
            return 1 + self._calculate_depth(node.operand)
        elif isinstance(node, BinaryOpNode):
            return 1 + max(self._calculate_depth(node.left), self._calculate_depth(node.right))
        return 1
    
    def _get_all_nodes(self, node: Node) -> List[Node]:
        """Get all nodes in the tree (breadth-first)"""
        nodes = [node]
        if isinstance(node, BinaryOpNode):
            nodes.extend(self._get_all_nodes(node.left))
            nodes.extend(self._get_all_nodes(node.right))
        elif isinstance(node, UnaryOpNode):
            nodes.extend(self._get_all_nodes(node.operand))
        return nodes
    
    def calculate_node_importance(self, expression: Expression, X: np.ndarray, y: np.ndarray) -> Dict[int, float]:
        """Calculate importance/sensitivity of each node for data-driven mutations"""
        nodes = self._get_all_nodes(expression.root)
        node_importance = {}
        
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
        
        return node_importance
