"""
Diversity Metrics Module

Provides metrics and utilities for measuring and maintaining diversity
in symbolic regression populations.
"""

import numpy as np
from typing import List, Dict
from collections import Counter

from ..expression_tree import Expression, Node, BinaryOpNode, UnaryOpNode, ConstantNode, VariableNode


class DiversityMetrics:
    """Collection of diversity measurement and maintenance utilities"""
    
    def __init__(self, n_inputs: int):
        self.n_inputs = n_inputs
    
    def calculate_population_diversity(self, population: List[Expression]) -> Dict[str, float]:
        """Calculate various diversity metrics for a population"""
        if not population:
            return {'structural_diversity': 0.0, 'operator_diversity': 0.0, 
                   'complexity_diversity': 0.0, 'semantic_diversity': 0.0}
        
        # Structural diversity
        structural_diversity = self._calculate_structural_diversity(population)
        
        # Operator diversity
        operator_diversity = self._calculate_operator_diversity(population)
        
        # Complexity diversity
        complexity_diversity = self._calculate_complexity_diversity(population)
        
        # Semantic diversity (if evaluation data is available)
        semantic_diversity = 0.0  # Placeholder - requires evaluation data
        
        return {
            'structural_diversity': structural_diversity,
            'operator_diversity': operator_diversity,
            'complexity_diversity': complexity_diversity,
            'semantic_diversity': semantic_diversity,
            'overall_diversity': (structural_diversity + operator_diversity + complexity_diversity) / 3.0
        }
    
    def _calculate_structural_diversity(self, population: List[Expression]) -> float:
        """Calculate structural diversity based on tree shapes"""
        if len(population) <= 1:
            return 0.0
        
        # Calculate pairwise structural distances
        total_distance = 0.0
        comparisons = 0
        
        for i in range(len(population)):
            for j in range(i + 1, len(population)):
                distance = self.structural_distance(population[i], population[j])
                total_distance += distance
                comparisons += 1
        
        return total_distance / max(1, comparisons)
    
    def _calculate_operator_diversity(self, population: List[Expression]) -> float:
        """Calculate operator diversity across the population"""
        all_operators = []
        
        for expr in population:
            operators = self._get_operator_signature(expr.root)
            all_operators.extend(operators.keys())
        
        if not all_operators:
            return 0.0
        
        # Calculate entropy of operator distribution
        operator_counts = Counter(all_operators)
        total_ops = len(all_operators)
        
        entropy = 0.0
        for count in operator_counts.values():
            p = count / total_ops
            if p > 0:
                entropy -= p * np.log2(p)
        
        # Normalize by maximum possible entropy
        max_entropy = np.log2(len(operator_counts)) if len(operator_counts) > 1 else 1.0
        return entropy / max_entropy if max_entropy > 0 else 0.0
    
    def _calculate_complexity_diversity(self, population: List[Expression]) -> float:
        """Calculate diversity in expression complexity"""
        if not population:
            return 0.0
        
        complexities = [expr.complexity() for expr in population]
        
        if len(set(complexities)) <= 1:
            return 0.0
        
        # Calculate coefficient of variation
        mean_complexity = np.mean(complexities)
        std_complexity = np.std(complexities)
        
        if mean_complexity == 0:
            return 0.0
        
        # Normalize to [0, 1] range
        cv = std_complexity / mean_complexity
        return min(1.0, float(cv))
    
    def structural_distance(self, expr1: Expression, expr2: Expression) -> float:
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
            
            # Combine distances with weights
            total_distance = (
                0.25 * size_diff +
                0.15 * depth_diff + 
                0.35 * operator_distance +
                0.25 * shape_distance
            )
            
            return min(1.0, total_distance)
            
        except Exception:
            # Fallback to simple distance
            return abs(expr1.complexity() - expr2.complexity()) / max(1, max(expr1.complexity(), expr2.complexity()))
    
    def semantic_distance(self, expr1: Expression, expr2: Expression, X: np.ndarray) -> float:
        """Calculate semantic distance between expressions based on their outputs"""
        try:
            vals1 = expr1.evaluate(X)
            vals2 = expr2.evaluate(X)
            
            if vals1 is None or vals2 is None:
                return 1.0
            
            # Flatten arrays for correlation calculation
            vals1_flat = vals1.flatten()
            vals2_flat = vals2.flatten()
            
            # Calculate correlation
            correlation = np.corrcoef(vals1_flat, vals2_flat)[0, 1]
            
            if np.isnan(correlation):
                return 1.0
            
            # Convert correlation to distance (1 - |correlation|)
            return 1.0 - abs(correlation)
            
        except Exception:
            return 1.0  # Maximum distance if evaluation fails
    
    def get_diverse_individuals(self, population: List[Expression], count: int) -> List[int]:
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
                diversity = sum(self.structural_distance(population[i], population[j])
                              for j in selected)

                if diversity > max_diversity:
                    max_diversity = diversity
                    best_candidate = i

            if best_candidate >= 0:
                selected.append(best_candidate)

        return selected
    
    def maintain_diversity(self, population: List[Expression], target_diversity: float = 0.5) -> List[Expression]:
        """Maintain population diversity by replacing similar individuals"""
        if len(population) <= 2:
            return population
        
        current_diversity = self.calculate_population_diversity(population)['overall_diversity']
        
        if current_diversity >= target_diversity:
            return population
        
        # Find most similar pairs
        similarity_threshold = 0.1
        to_replace = []
        
        for i in range(len(population)):
            for j in range(i + 1, len(population)):
                distance = self.structural_distance(population[i], population[j])
                if distance < similarity_threshold:
                    # Mark the less complex one for replacement
                    if population[i].complexity() <= population[j].complexity():
                        to_replace.append(i)
                    else:
                        to_replace.append(j)
        
        # Remove duplicates and return indices to replace
        to_replace = list(set(to_replace))
        
        # For now, just return the population as-is
        # In a complete implementation, this would generate new diverse individuals
        return population
    
    def _calculate_depth(self, node: Node) -> int:
        """Calculate the depth of a node tree"""
        if isinstance(node, (ConstantNode, VariableNode)):
            return 1
        elif isinstance(node, UnaryOpNode):
            return 1 + self._calculate_depth(node.operand)
        elif isinstance(node, BinaryOpNode):
            return 1 + max(self._calculate_depth(node.left), self._calculate_depth(node.right))
        return 1
    
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
