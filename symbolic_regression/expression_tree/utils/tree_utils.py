"""
Tree Utility Functions

Consolidated tree traversal and analysis utilities for expression trees.
This module eliminates redundancy by providing centralized implementations
of common tree operations used across the codebase.
"""

import numpy as np
from typing import List, Dict, Set, Optional, Any, Callable, TypeVar, cast
from collections import Counter

from ..core.node import Node, BinaryOpNode, UnaryOpNode, ConstantNode, VariableNode, ScalingOpNode

T = TypeVar('T', bound=Node)


def get_all_nodes(node: Node, traversal_order: str = 'breadth_first') -> List[Node]:
    """
    Get all nodes in the tree using specified traversal order.
    
    Args:
        node: Root node of the tree
        traversal_order: 'breadth_first' (default) or 'depth_first'
    
    Returns:
        List of all nodes in the tree
    """
    if traversal_order == 'breadth_first':
        return _breadth_first_traversal(node)
    elif traversal_order == 'depth_first':
        return _depth_first_traversal(node)
    else:
        raise ValueError(f"Invalid traversal_order: {traversal_order}")


def _breadth_first_traversal(node: Node) -> List[Node]:
    """Breadth-first traversal (iterative, non-recursive)"""
    nodes_to_visit = [node]
    all_nodes = []
    
    while nodes_to_visit:
        current_node = nodes_to_visit.pop(0)  # FIFO for breadth-first
        all_nodes.append(current_node)
        
        if isinstance(current_node, BinaryOpNode):
            nodes_to_visit.append(current_node.left)
            nodes_to_visit.append(current_node.right)
        elif isinstance(current_node, UnaryOpNode):
            nodes_to_visit.append(current_node.operand)
        elif isinstance(current_node, ScalingOpNode):
            nodes_to_visit.append(current_node.operand)
    
    return all_nodes


def _depth_first_traversal(node: Node) -> List[Node]:
    """Depth-first traversal (recursive)"""
    nodes = [node]
    
    if isinstance(node, BinaryOpNode):
        nodes.extend(_depth_first_traversal(node.left))
        nodes.extend(_depth_first_traversal(node.right))
    elif isinstance(node, UnaryOpNode):
        nodes.extend(_depth_first_traversal(node.operand))
    elif isinstance(node, ScalingOpNode):
        nodes.extend(_depth_first_traversal(node.operand))
    
    return nodes


def calculate_tree_depth(node: Node) -> int:
    """
    Calculate the maximum depth of the tree.
    
    Args:
        node: Root node of the tree
    
    Returns:
        Maximum depth (leaf nodes have depth 1)
    """
    if isinstance(node, (ConstantNode, VariableNode)):
        return 1
    elif isinstance(node, UnaryOpNode):
        return 1 + calculate_tree_depth(node.operand)
    elif isinstance(node, BinaryOpNode):
        left_depth = calculate_tree_depth(node.left)
        right_depth = calculate_tree_depth(node.right)
        return 1 + max(left_depth, right_depth)
    elif isinstance(node, ScalingOpNode):
        return 1 + calculate_tree_depth(node.operand)
    else:
        return 1


def calculate_subtree_sizes(node: Node) -> Dict[Node, int]:
    """
    Calculate the size (node count) of each subtree.
    
    Args:
        node: Root node of the tree
    
    Returns:
        Dictionary mapping each node to its subtree size
    """
    sizes = {}
    
    def _calculate_size(current_node: Node) -> int:
        if isinstance(current_node, (ConstantNode, VariableNode)):
            size = 1
        elif isinstance(current_node, UnaryOpNode):
            size = 1 + _calculate_size(current_node.operand)
        elif isinstance(current_node, BinaryOpNode):
            left_size = _calculate_size(current_node.left)
            right_size = _calculate_size(current_node.right)
            size = 1 + left_size + right_size
        elif isinstance(current_node, ScalingOpNode):
            size = 1 + _calculate_size(current_node.operand)
        else:
            size = 1
        
        sizes[current_node] = size
        return size
    
    _calculate_size(node)
    return sizes


def collect_subtree_patterns(node: Node) -> List[str]:
    """
    Collect string representations of all subtrees for redundancy analysis.
    
    Args:
        node: Root node of the tree
    
    Returns:
        List of subtree string representations
    """
    subtrees = []
    
    def _collect_subtrees(current_node: Node):
        if isinstance(current_node, (ConstantNode, VariableNode)):
            subtrees.append(current_node.to_string())
        elif isinstance(current_node, UnaryOpNode):
            operand_str = current_node.operand.to_string()
            subtree_str = f"{current_node.operator}({operand_str})"
            subtrees.append(subtree_str)
            _collect_subtrees(current_node.operand)
        elif isinstance(current_node, BinaryOpNode):
            left_str = current_node.left.to_string()
            right_str = current_node.right.to_string()
            subtree_str = f"({left_str} {current_node.operator} {right_str})"
            subtrees.append(subtree_str)
            _collect_subtrees(current_node.left)
            _collect_subtrees(current_node.right)
        elif isinstance(current_node, ScalingOpNode):
            operand_str = current_node.operand.to_string()
            subtree_str = f"scale({operand_str}, {current_node.power})"
            subtrees.append(subtree_str)
            _collect_subtrees(current_node.operand)
    
    _collect_subtrees(node)
    return subtrees


def calculate_redundancy_score(node: Node) -> float:
    """
    Calculate redundancy score based on repeated subtree patterns.
    
    Args:
        node: Root node of the tree
    
    Returns:
        Redundancy score between 0.0 (no redundancy) and 1.0 (high redundancy)
    """
    subtrees = collect_subtree_patterns(node)
    
    if len(subtrees) <= 1:
        return 0.0
    
    # Count frequency of each subtree pattern
    subtree_counts = Counter(subtrees)
    
    # Calculate redundancy as ratio of repeated subtrees
    total_subtrees = len(subtrees)
    repeated_count = sum(count - 1 for count in subtree_counts.values() if count > 1)
    
    return repeated_count / total_subtrees if total_subtrees > 0 else 0.0


def find_nodes_by_type(node: Node, node_type: type) -> List[Node]:
    """
    Find all nodes of a specific type in the tree.
    
    Args:
        node: Root node of the tree
        node_type: Type of nodes to find (e.g., ConstantNode, VariableNode)
    
    Returns:
        List of nodes matching the specified type
    """
    all_nodes = get_all_nodes(node)
    return [n for n in all_nodes if isinstance(n, node_type)]


def find_nodes_by_operator(node: Node, operator: str) -> List[Node]:
    """
    Find all operator nodes with a specific operator.
    
    Args:
        node: Root node of the tree
        operator: Operator string to search for
    
    Returns:
        List of nodes with the specified operator
    """
    all_nodes = get_all_nodes(node)
    matching_nodes = []
    
    for n in all_nodes:
        if isinstance(n, (BinaryOpNode, UnaryOpNode)) and n.operator == operator:
            matching_nodes.append(n)
    
    return matching_nodes


def replace_node_in_tree(root: Node, target: Node, replacement: Node) -> bool:
    """
    Replace a target node with a replacement node in the tree.
    
    Args:
        root: Root node of the tree
        target: Node to be replaced
        replacement: Node to replace the target
    
    Returns:
        True if replacement was successful, False otherwise
    """
    if root == target:
        # Cannot replace root node directly - would need to return new root
        return False
    
    if isinstance(root, BinaryOpNode):
        if root.left == target:
            root.left = replacement
            return True
        elif root.right == target:
            root.right = replacement
            return True
        else:
            # Recursively search in subtrees
            return (replace_node_in_tree(root.left, target, replacement) or
                    replace_node_in_tree(root.right, target, replacement))
    
    elif isinstance(root, UnaryOpNode):
        if root.operand == target:
            root.operand = replacement
            return True
        else:
            return replace_node_in_tree(root.operand, target, replacement)
    
    elif isinstance(root, ScalingOpNode):
        if root.operand == target:
            root.operand = replacement
            return True
        else:
            return replace_node_in_tree(root.operand, target, replacement)
    
    return False


def count_nested_depth(node: Node) -> int:
    """
    Count the maximum nested depth for complexity analysis.
    
    Args:
        node: Root node of the tree
    
    Returns:
        Maximum nested depth (0 for terminal nodes)
    """
    if isinstance(node, (ConstantNode, VariableNode)):
        return 0
    elif isinstance(node, (UnaryOpNode, ScalingOpNode)):
        return 1 + count_nested_depth(node.operand)
    elif isinstance(node, BinaryOpNode):
        left_depth = count_nested_depth(node.left)
        right_depth = count_nested_depth(node.right)
        return max(left_depth, right_depth)
    else:
        return 0


def calculate_structural_balance(node: Node) -> float:
    """
    Calculate structural balance (tree symmetry) score.
    
    Args:
        node: Root node of the tree
    
    Returns:
        Balance score between 0.0 (unbalanced) and 1.0 (perfectly balanced)
    """
    if isinstance(node, BinaryOpNode):
        subtree_sizes = calculate_subtree_sizes(node)
        left_size = subtree_sizes.get(node.left, 0)
        right_size = subtree_sizes.get(node.right, 0)
        total_size = left_size + right_size
        
        if total_size > 0:
            return 1.0 - abs(left_size - right_size) / total_size
        else:
            return 1.0
    else:
        return 1.0  # Non-binary nodes are considered balanced


def get_variable_usage_counts(node: Node) -> Dict[int, int]:
    """
    Count the usage frequency of each variable in the tree.
    
    Args:
        node: Root node of the tree
    
    Returns:
        Dictionary mapping variable indices to their usage counts
    """
    variable_nodes = find_nodes_by_type(node, VariableNode)
    usage_counts = {}
    
    for var_node in variable_nodes:
        if isinstance(var_node, VariableNode):  # Type guard
            var_index = var_node.index
            usage_counts[var_index] = usage_counts.get(var_index, 0) + 1
    
    return usage_counts


def apply_to_all_nodes(node: Node, func: Callable[[Node], Any], 
                      filter_type: Optional[type] = None) -> List[Any]:
    """
    Apply a function to all nodes (optionally filtered by type).
    
    Args:
        node: Root node of the tree
        func: Function to apply to each node
        filter_type: Optional type filter (only apply to nodes of this type)
    
    Returns:
        List of function results
    """
    all_nodes = get_all_nodes(node)
    
    if filter_type is not None:
        all_nodes = [n for n in all_nodes if isinstance(n, filter_type)]
    
    return [func(n) for n in all_nodes]


def clone_tree(node: Node) -> Node:
    """
    Create a deep copy of the entire tree.
    
    Args:
        node: Root node of the tree to clone
    
    Returns:
        Deep copy of the tree
    """
    return node.copy()


def validate_tree_structure(node: Node) -> bool:
    """
    Validate that the tree structure is consistent and well-formed.
    
    Args:
        node: Root node of the tree
    
    Returns:
        True if tree structure is valid, False otherwise
    """
    try:
        # Check that all nodes have required attributes
        if isinstance(node, BinaryOpNode):
            if not hasattr(node, 'left') or not hasattr(node, 'right') or not hasattr(node, 'operator'):
                return False
            return validate_tree_structure(node.left) and validate_tree_structure(node.right)
        
        elif isinstance(node, UnaryOpNode):
            if not hasattr(node, 'operand') or not hasattr(node, 'operator'):
                return False
            return validate_tree_structure(node.operand)
        
        elif isinstance(node, ScalingOpNode):
            if not hasattr(node, 'operand') or not hasattr(node, 'power'):
                return False
            return validate_tree_structure(node.operand)
        
        elif isinstance(node, VariableNode):
            return hasattr(node, 'index')
        
        elif isinstance(node, ConstantNode):
            return hasattr(node, 'value')
        
        else:
            # Unknown node type
            return False
    
    except Exception:
        return False


# Convenience functions for common operations
def get_constants(node: Node) -> List[ConstantNode]:
    """Get all constant nodes in the tree."""
    return cast(List[ConstantNode], find_nodes_by_type(node, ConstantNode))


def get_variables(node: Node) -> List[VariableNode]:
    """Get all variable nodes in the tree."""
    return cast(List[VariableNode], find_nodes_by_type(node, VariableNode))


def get_binary_ops(node: Node) -> List[BinaryOpNode]:
    """Get all binary operation nodes in the tree."""
    return cast(List[BinaryOpNode], find_nodes_by_type(node, BinaryOpNode))


def get_unary_ops(node: Node) -> List[UnaryOpNode]:
    """Get all unary operation nodes in the tree."""
    return cast(List[UnaryOpNode], find_nodes_by_type(node, UnaryOpNode))


def get_scaling_ops(node: Node) -> List[ScalingOpNode]:
    """Get all scaling operation nodes in the tree."""
    return cast(List[ScalingOpNode], find_nodes_by_type(node, ScalingOpNode))
