"""
Crossover Operations Module

Provides various crossover strategies for symbolic regression including
structural crossover, quality-guided crossover, and parameter exchange.
"""

import numpy as np
import random
from typing import Tuple, List

from ..expression_tree import Expression, Node, BinaryOpNode, UnaryOpNode, ConstantNode, VariableNode
from ..expression_tree.utils.tree_utils import get_all_nodes, replace_node_in_tree
from ..quality_assessment import calculate_subtree_qualities


class CrossoverOperations:
    """Collection of crossover operations for genetic programming"""
    
    def __init__(self, max_complexity: int = 20):
        self.max_complexity = max_complexity
    
    def enhanced_crossover(self, parent1: Expression, parent2: Expression) -> Tuple[Expression, Expression]:
        """Enhanced crossover with proper subtree exchange"""
        # Try structural crossover first
        for _ in range(3):
            child1, child2 = self.structural_crossover(parent1, parent2)
            if (child1.complexity() <= self.max_complexity and
                child2.complexity() <= self.max_complexity):
                return child1, child2

        # Fallback to simpler crossover
        return self.simple_crossover(parent1, parent2)
    
    def structural_crossover(self, parent1: Expression, parent2: Expression) -> Tuple[Expression, Expression]:
        """Exchange random subtrees between parents"""
        child1 = parent1.copy()
        child2 = parent2.copy()

        nodes1 = get_all_nodes(child1.root)
        nodes2 = get_all_nodes(child2.root)

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
    
    def simple_crossover(self, parent1: Expression, parent2: Expression) -> Tuple[Expression, Expression]:
        """Simple parameter exchange crossover"""
        child1 = parent1.copy()
        child2 = parent2.copy()

        # Exchange constants
        constants1 = [n for n in get_all_nodes(child1.root) if isinstance(n, ConstantNode)]
        constants2 = [n for n in get_all_nodes(child2.root) if isinstance(n, ConstantNode)]

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
    
    def quality_guided_crossover(self, parent1: Expression, parent2: Expression, 
                                X: np.ndarray, residuals1: np.ndarray, residuals2: np.ndarray) -> Tuple[Expression, Expression]:
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
            return self.structural_crossover(parent1, parent2)

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
            return self.structural_crossover(parent1, parent2)  # Fallback

        # 6. Perform the swap
        subtree1 = crossover_point1.copy()
        subtree2 = crossover_point2.copy()

        self._replace_node_in_tree(child1.root, crossover_point1, subtree2)
        self._replace_node_in_tree(child2.root, crossover_point2, subtree1)

        child1.clear_cache()
        child2.clear_cache()

        return child1, child2
    
    def semantic_crossover(self, parent1: Expression, parent2: Expression, 
                          X: np.ndarray, blend_ratio: float = 0.5) -> Expression:
        """
        Semantic crossover that blends the outputs of two expressions
        Creates: child = blend_ratio * parent1 + (1 - blend_ratio) * parent2
        """
        # Create a new expression that represents the blend
        # This is a simplified implementation - a full implementation would 
        # create a new tree structure representing the blend
        
        # For now, we'll use structural crossover as fallback
        # In a complete implementation, this would create:
        # BinaryOpNode('+', 
        #   BinaryOpNode('*', ConstantNode(blend_ratio), parent1.root.copy()),
        #   BinaryOpNode('*', ConstantNode(1-blend_ratio), parent2.root.copy())
        # )
        
        child1, child2 = self.structural_crossover(parent1, parent2)
        return child1  # Return one of the children
    
    def uniform_crossover(self, parent1: Expression, parent2: Expression, 
                         crossover_rate: float = 0.5) -> Tuple[Expression, Expression]:
        """
        Uniform crossover where each node is independently selected from either parent
        """
        child1 = parent1.copy()
        child2 = parent2.copy()
        
        nodes1 = get_all_nodes(child1.root)
        nodes2 = get_all_nodes(child2.root)
        
        # Only perform crossover if both trees have similar structure
        if len(nodes1) == len(nodes2):
            for i in range(len(nodes1)):
                if random.random() < crossover_rate:
                    # Swap nodes if they are compatible types
                    node1, node2 = nodes1[i], nodes2[i]
                    if type(node1) == type(node2):
                        if isinstance(node1, ConstantNode) and isinstance(node2, ConstantNode):
                            node1.value, node2.value = node2.value, node1.value
                        elif isinstance(node1, BinaryOpNode) and isinstance(node2, BinaryOpNode):
                            node1.operator, node2.operator = node2.operator, node1.operator
                        elif isinstance(node1, UnaryOpNode) and isinstance(node2, UnaryOpNode):
                            node1.operator, node2.operator = node2.operator, node1.operator
                        elif isinstance(node1, VariableNode) and isinstance(node2, VariableNode):
                            node1.index, node2.index = node2.index, node1.index
        
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
                return replace_node_in_tree(root.operand, target, replacement)

        return False

    # Note: _get_all_nodes method has been removed.
    # Use centralized tree_utils.get_all_nodes instead.
