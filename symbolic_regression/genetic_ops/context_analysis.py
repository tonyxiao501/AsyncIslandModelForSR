"""
Context Analysis Module for Expression Structure Analysis

Provides advanced context analysis capabilities for expression trees,
including structural metrics, redundancy detection, and complexity assessment.
"""

import numpy as np
from typing import Dict, List
from collections import Counter

from ..expression_tree import Expression, Node, ConstantNode, VariableNode, BinaryOpNode, UnaryOpNode, ScalingOpNode
from ..expression_tree.utils.tree_utils import (
    get_all_nodes, calculate_tree_depth, calculate_subtree_sizes,
    collect_subtree_patterns, calculate_redundancy_score,
    calculate_structural_balance, get_variable_usage_counts
)


class ExpressionContextAnalyzer:
    """Advanced expression context analysis for intelligent strategy selection"""
    
    def __init__(self, n_inputs: int):
        self.n_inputs = n_inputs
    
    def analyze_expression_context(self, expression: Expression) -> Dict:
        """Advanced expression context analysis for intelligent mutation strategy selection"""
        nodes = get_all_nodes(expression.root)
        
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
        structural_balance = calculate_structural_balance(expression.root)
        
        # 3. Nonlinearity score (presence of nonlinear functions)
        nonlinearity_score = self._calculate_nonlinearity_score(binary_ops, unary_ops)
        
        # 4. Improved redundancy detection using AST patterns
        redundancy_score = calculate_redundancy_score(expression.root)
        
        # 5. Variable usage symmetry
        symmetry_score = self._calculate_variable_symmetry(variables)
        
        # 6. Complexity distribution analysis
        depth = calculate_tree_depth(expression.root)
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
    
    # Note: _calculate_depth, _get_all_nodes, _calculate_structural_balance, 
    # and _calculate_redundancy_score methods have been removed.
    # Use centralized tree_utils functions instead.
    
    def calculate_node_importance(self, expression: Expression, X: np.ndarray, y: np.ndarray) -> Dict[int, float]:
        """Calculate importance/sensitivity of each node for data-driven mutations"""
        nodes = get_all_nodes(expression.root)
        node_importance = {}
        
        # **PERFORMANCE FIX**: Use sampling for large datasets and limit constant nodes evaluated
        max_data_points = min(50, len(X))  # Sample data for speed
        sample_indices = np.random.choice(len(X), max_data_points, replace=False) if len(X) > max_data_points else slice(None)
        X_sample = X[sample_indices]
        
        try:
            # **PERFORMANCE FIX**: Only evaluate importance for a subset of constant nodes
            constant_nodes = [(i, node) for i, node in enumerate(nodes) if isinstance(node, ConstantNode)]
            max_constants_to_evaluate = min(5, len(constant_nodes))  # Limit to 5 most important constants
            
            # Calculate gradient/sensitivity for limited constant nodes
            for i, node in constant_nodes[:max_constants_to_evaluate]:
                try:
                    # Use smaller perturbation for faster calculation
                    original_value = node.value
                    perturbation = max(0.01, abs(original_value) * 0.01)  # Adaptive perturbation
                    
                    node.value += perturbation
                    pred_plus = expression.evaluate(X_sample)
                    node.value = original_value - perturbation
                    pred_minus = expression.evaluate(X_sample)
                    node.value = original_value
                    
                    # Quick sensitivity calculation
                    if pred_plus.shape == pred_minus.shape:
                        sensitivity = np.mean(np.abs(pred_plus - pred_minus))
                        # Normalize by perturbation size
                        node_importance[i] = sensitivity / (2 * perturbation)
                    else:
                        node_importance[i] = 1.0
                except:
                    node_importance[i] = 1.0
            
            # Set remaining nodes to default importance
            for i, node in enumerate(nodes):
                if i not in node_importance:
                    node_importance[i] = 1.0  # Default importance
                    
        except Exception:
            # Fallback to uniform importance
            node_importance = {i: 1.0 for i in range(len(nodes))}
        
        return node_importance
