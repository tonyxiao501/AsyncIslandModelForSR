"""Utilities for expression trees."""

from .simplifier import ExpressionSimplifier
from .sympy_utils import SymPySimplifier
from .tree_utils import (
    get_all_nodes, calculate_tree_depth, calculate_subtree_sizes,
    collect_subtree_patterns, calculate_redundancy_score,
    find_nodes_by_type, find_nodes_by_operator, replace_node_in_tree,
    count_nested_depth, calculate_structural_balance,
    get_variable_usage_counts, apply_to_all_nodes, clone_tree,
    validate_tree_structure, get_constants, get_variables,
    get_binary_ops, get_unary_ops, get_scaling_ops,
    # Enhanced node replacement utilities
    swap_binary_operands, replace_child_node, find_parent_node,
    replace_subtree_at_path, bulk_replace_nodes, get_node_path,
    create_node_replacement_context
)

__all__ = [
    'ExpressionSimplifier', 'SymPySimplifier',
    'get_all_nodes', 'calculate_tree_depth', 'calculate_subtree_sizes',
    'collect_subtree_patterns', 'calculate_redundancy_score',
    'find_nodes_by_type', 'find_nodes_by_operator', 'replace_node_in_tree',
    'count_nested_depth', 'calculate_structural_balance',
    'get_variable_usage_counts', 'apply_to_all_nodes', 'clone_tree',
    'validate_tree_structure', 'get_constants', 'get_variables',
    'get_binary_ops', 'get_unary_ops', 'get_scaling_ops',
    # Enhanced node replacement utilities
    'swap_binary_operands', 'replace_child_node', 'find_parent_node',
    'replace_subtree_at_path', 'bulk_replace_nodes', 'get_node_path',
    'create_node_replacement_context'
]
