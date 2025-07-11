import numpy as np
from .expression_tree import Expression, Node
from .expression_tree.core.node import BinaryOpNode, UnaryOpNode

def get_all_nodes(node: Node) -> list[Node]:
    """Helper to get all nodes in a tree using a non-recursive traversal."""
    nodes_to_visit = [node]
    all_nodes = []
    while nodes_to_visit:
        current_node = nodes_to_visit.pop()
        all_nodes.append(current_node)
        if isinstance(current_node, BinaryOpNode):
            nodes_to_visit.append(current_node.left)
            nodes_to_visit.append(current_node.right)
        elif isinstance(current_node, UnaryOpNode):
            nodes_to_visit.append(current_node.operand)
    return all_nodes

def calculate_subtree_qualities(expression: Expression, X: np.ndarray, residuals: np.ndarray) -> dict[Node, float]:
    qualities = {}
    nodes = get_all_nodes(expression.root)

    for node in nodes:
        try:
            subtree_output = node.evaluate(X)
            
            if np.std(subtree_output) < 1e-6 or np.any(~np.isfinite(subtree_output)):
                qualities[node] = 0.0
                continue

            correlation = np.corrcoef(subtree_output, residuals)[0, 1]
           
            # Convert correlation to a quality score (0 to 1)
            # A perfect negative correlation (-1) gives a quality of 0, while no correlation 
            quality_score = (1.0 - correlation) / 2.0
            qualities[node] = quality_score
            
        except Exception:
            qualities[node] = 0.0 # Assign zero quality if any error occurs
            
    return qualities