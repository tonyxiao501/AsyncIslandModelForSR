"""
Diversity Metrics Module

Provides metrics and utilities for measuring and maintaining diversity
in symbolic regression populations.
"""

import numpy as np
from typing import List, Dict, Optional
from collections import Counter

from ..expression_tree import Expression, Node, BinaryOpNode, UnaryOpNode, ConstantNode, VariableNode
from ..expression_tree.utils.tree_utils import calculate_tree_depth


class DiversityMetrics:
    """Collection of diversity measurement and maintenance utilities"""
    
    def __init__(self, n_inputs: int):
        self.n_inputs = n_inputs
        self.diversity_history = []
        self.archive = set()  # Archive of seen expressions for novelty
        self.crowding_threshold = 0.15
    
    def calculate_population_diversity(self, population: List[Expression]) -> Dict[str, float]:
        """Calculate various diversity metrics for a population with enhanced measures"""
        if not population:
            return {'structural_diversity': 0.0, 'operator_diversity': 0.0, 
                   'complexity_diversity': 0.0, 'semantic_diversity': 0.0,
                   'behavioral_diversity': 0.0, 'novelty_score': 0.0}
        
        # Structural diversity
        structural_diversity = self._calculate_structural_diversity(population)
        
        # Operator diversity
        operator_diversity = self._calculate_operator_diversity(population)
        
        # Complexity diversity
        complexity_diversity = self._calculate_complexity_diversity(population)
        
        # Behavioral diversity using functional signatures
        behavioral_diversity = self._calculate_behavioral_diversity(population)
        
        # Novelty score based on archive
        novelty_score = self._calculate_novelty_score(population)
        
        # Semantic diversity (if evaluation data is available)
        semantic_diversity = 0.0  # Placeholder - requires evaluation data
        
        # Track diversity history for trend analysis
        current_diversity = (structural_diversity + operator_diversity + 
                           complexity_diversity + behavioral_diversity) / 4.0
        self.diversity_history.append(current_diversity)
        if len(self.diversity_history) > 100:  # Keep last 100 generations
            self.diversity_history.pop(0)
        
        return {
            'structural_diversity': structural_diversity,
            'operator_diversity': operator_diversity,
            'complexity_diversity': complexity_diversity,
            'semantic_diversity': semantic_diversity,
            'behavioral_diversity': behavioral_diversity,
            'novelty_score': novelty_score,
            'overall_diversity': current_diversity,
            'diversity_trend': self._calculate_diversity_trend()
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
    
    def _calculate_behavioral_diversity(self, population: List[Expression]) -> float:
        """Calculate behavioral diversity using functional signatures"""
        if len(population) <= 1:
            return 0.0
        
        # Create test points for behavioral evaluation
        test_points = np.random.randn(10, self.n_inputs) * 2.0
        behavioral_signatures = []
        
        for expr in population:
            try:
                # Evaluate on test points to get behavioral signature
                outputs = expr.evaluate(test_points)
                if outputs is not None:
                    # Create signature from output patterns
                    signature = self._create_behavioral_signature(outputs)
                    behavioral_signatures.append(signature)
                else:
                    behavioral_signatures.append(tuple([0.0] * 10))
            except:
                behavioral_signatures.append(tuple([0.0] * 10))
        
        # Calculate pairwise behavioral distances
        total_distance = 0.0
        comparisons = 0
        
        for i in range(len(behavioral_signatures)):
            for j in range(i + 1, len(behavioral_signatures)):
                distance = self._behavioral_signature_distance(
                    behavioral_signatures[i], behavioral_signatures[j])
                total_distance += distance
                comparisons += 1
        
        return total_distance / max(1, comparisons)
    
    def _calculate_novelty_score(self, population: List[Expression]) -> float:
        """Calculate novelty score based on expression archive"""
        if not population:
            return 0.0
        
        novelty_scores = []
        for expr in population:
            expr_sig = self._get_expression_signature(expr)
            
            # Calculate distance to nearest neighbors in archive
            if self.archive:
                distances = [self._signature_distance(expr_sig, archived_sig) 
                           for archived_sig in self.archive]
                avg_distance = np.mean(sorted(distances)[:min(5, len(distances))])
                novelty_scores.append(avg_distance)
            else:
                novelty_scores.append(1.0)  # High novelty if no archive
            
            # Add to archive (maintain size limit)
            self.archive.add(expr_sig)
            if len(self.archive) > 1000:
                # Remove oldest entries (simple FIFO for now)
                self.archive = set(list(self.archive)[100:])
        
        return float(np.mean(novelty_scores)) if novelty_scores else 0.0
    
    def _calculate_diversity_trend(self) -> float:
        """Calculate diversity trend over recent generations"""
        if len(self.diversity_history) < 5:
            return 0.0
        
        # Simple linear trend calculation
        recent_history = self.diversity_history[-10:]
        x = np.arange(len(recent_history))
        slope = np.polyfit(x, recent_history, 1)[0]
        
        # Normalize slope to [-1, 1] range
        return np.tanh(slope * 10)
    
    def _create_behavioral_signature(self, outputs: np.ndarray) -> tuple:
        """Create behavioral signature from expression outputs"""
        if outputs is None or len(outputs) == 0:
            return tuple([0.0] * 10)
        
        # Extract statistical features as behavioral signature
        flat_outputs = outputs.flatten()
        try:
            features = [
                float(np.mean(flat_outputs)),
                float(np.std(flat_outputs)),
                float(np.min(flat_outputs)),
                float(np.max(flat_outputs)),
                float(np.median(flat_outputs)),
                float(np.percentile(flat_outputs, 25)),
                float(np.percentile(flat_outputs, 75)),
                float(np.sum(flat_outputs > 0) / len(flat_outputs)),  # Positive ratio
                float(np.sum(np.abs(flat_outputs) < 1e-6) / len(flat_outputs)),  # Zero ratio
                float(np.sum(np.abs(flat_outputs)) / len(flat_outputs))  # Mean absolute value
            ]
            return tuple(features)
        except:
            return tuple([0.0] * 10)
    
    def _behavioral_signature_distance(self, sig1: tuple, sig2: tuple) -> float:
        """Calculate distance between behavioral signatures"""
        try:
            sig1_arr = np.array(sig1)
            sig2_arr = np.array(sig2)
            
            # Normalize signatures
            sig1_norm = sig1_arr / (np.linalg.norm(sig1_arr) + 1e-10)
            sig2_norm = sig2_arr / (np.linalg.norm(sig2_arr) + 1e-10)
            
            # Euclidean distance between normalized signatures
            return float(np.linalg.norm(sig1_norm - sig2_norm))
        except:
            return 1.0
    
    def _get_expression_signature(self, expr: Expression) -> str:
        """Get a string signature for expression archival"""
        try:
            # Use simplified expression string as signature
            return expr.to_string()[:100]  # Limit length
        except:
            return f"expr_{id(expr)}"
    
    def _signature_distance(self, sig1: str, sig2: str) -> float:
        """Calculate distance between string signatures"""
        # Simple edit distance approximation
        if sig1 == sig2:
            return 0.0
        
        # Jaccard distance on character n-grams
        def get_ngrams(s, n=3):
            return set(s[i:i+n] for i in range(len(s) - n + 1))
        
        ngrams1 = get_ngrams(sig1)
        ngrams2 = get_ngrams(sig2)
        
        if not ngrams1 and not ngrams2:
            return 0.0
        if not ngrams1 or not ngrams2:
            return 1.0
        
        intersection = len(ngrams1.intersection(ngrams2))
        union = len(ngrams1.union(ngrams2))
        
        return 1.0 - (intersection / union)
    
    def structural_distance(self, expr1: Expression, expr2: Expression) -> float:
        """Calculate advanced structural distance between two expressions"""
        try:
            # 1. Size-based distance
            size1, size2 = expr1.complexity(), expr2.complexity()
            size_diff = abs(size1 - size2) / max(1, max(size1, size2))
            
            # 2. Depth-based distance  
            depth1 = calculate_tree_depth(expr1.root)
            depth2 = calculate_tree_depth(expr2.root)
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
        """Calculate enhanced semantic distance between expressions"""
        try:
            vals1 = expr1.evaluate(X)
            vals2 = expr2.evaluate(X)
            
            if vals1 is None or vals2 is None:
                return 1.0
            
            # Flatten arrays for comparison
            vals1_flat = vals1.flatten()
            vals2_flat = vals2.flatten()
            
            # Multiple semantic distance metrics
            distances = []
            
            # 1. Correlation-based distance
            correlation = np.corrcoef(vals1_flat, vals2_flat)[0, 1]
            if not np.isnan(correlation):
                distances.append(1.0 - abs(correlation))
            
            # 2. Normalized RMSE
            rmse = np.sqrt(np.mean((vals1_flat - vals2_flat) ** 2))
            vals_range = max(float(np.std(vals1_flat)), float(np.std(vals2_flat)), 1e-10)
            normalized_rmse = min(1.0, rmse / vals_range)
            distances.append(normalized_rmse)
            
            # 3. Rank correlation (Spearman)
            try:
                from scipy.stats import spearmanr
                spear_corr, _ = spearmanr(vals1_flat, vals2_flat)
                if not np.isnan(spear_corr):
                    distances.append(1.0 - abs(spear_corr))
            except:
                pass
            
            # 4. Distribution-based distance (Kolmogorov-Smirnov)
            try:
                from scipy.stats import ks_2samp
                ks_stat, _ = ks_2samp(vals1_flat, vals2_flat)
                distances.append(float(ks_stat))
            except:
                pass
            
            # Return average of available distance measures
            return float(np.mean(distances)) if distances else 1.0
            
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
        """Maintain population diversity using crowding distance and novelty pressure"""
        if len(population) <= 2:
            return population
        
        current_diversity = self.calculate_population_diversity(population)['overall_diversity']
        
        if current_diversity >= target_diversity:
            return population
        
        # Calculate crowding distances for all individuals
        crowding_distances = self._calculate_crowding_distances(population)
        
        # Find individuals to replace (those with low crowding distance)
        sorted_indices = np.argsort(crowding_distances)
        replacement_count = min(len(population) // 4, 
                              sum(1 for d in crowding_distances if d < self.crowding_threshold))
        
        if replacement_count == 0:
            return population
        
        # Mark individuals for replacement
        to_replace = sorted_indices[:replacement_count]
        
        # For actual implementation, this would generate new diverse individuals
        # For now, we return the original population with diversity analysis
        return population
    
    def _calculate_crowding_distances(self, population: List[Expression]) -> List[float]:
        """Calculate crowding distances for diversity-based selection"""
        if len(population) <= 2:
            return [1.0] * len(population)
        
        distances = [0.0] * len(population)
        
        # Calculate distances in multiple objective spaces
        objectives = [
            [expr.complexity() for expr in population],
            [calculate_tree_depth(expr.root) for expr in population],
            [len(self._get_operator_signature(expr.root)) for expr in population]
        ]
        
        for obj_values in objectives:
            # Sort population by this objective
            sorted_indices = np.argsort(obj_values)
            
            # Boundary solutions get infinite distance
            distances[sorted_indices[0]] = float('inf')
            distances[sorted_indices[-1]] = float('inf')
            
            # Calculate crowding distance for intermediate solutions
            obj_range = max(obj_values) - min(obj_values)
            if obj_range > 0:
                for i in range(1, len(sorted_indices) - 1):
                    idx = sorted_indices[i]
                    prev_idx = sorted_indices[i - 1]
                    next_idx = sorted_indices[i + 1]
                    
                    distance_contribution = (obj_values[next_idx] - obj_values[prev_idx]) / obj_range
                    distances[idx] += distance_contribution
        
        return distances
    
    def diversity_tournament_selection(self, population: List[Expression], 
                                     fitness_scores: List[float], 
                                     tournament_size: int = 3) -> int:
        """Tournament selection with diversity pressure"""
        if not population or len(population) != len(fitness_scores):
            return 0
        
        # Select random tournament participants
        tournament_indices = np.random.choice(len(population), 
                                            min(tournament_size, len(population)), 
                                            replace=False)
        
        # Calculate diversity scores for tournament participants
        diversity_scores = []
        for idx in tournament_indices:
            # Calculate average distance to other population members
            distances = [self.structural_distance(population[idx], population[j]) 
                        for j in range(len(population)) if j != idx]
            avg_distance = np.mean(distances) if distances else 0.0
            diversity_scores.append(avg_distance)
        
        # Combine fitness and diversity (weighted sum)
        diversity_weight = 0.3  # Adjust based on diversity pressure needed
        fitness_weight = 1.0 - diversity_weight
        
        combined_scores = []
        for i, idx in enumerate(tournament_indices):
            # Normalize fitness (higher is better)
            normalized_fitness = fitness_scores[idx]
            # Diversity score (higher is better)
            diversity_score = diversity_scores[i]
            
            combined_score = (fitness_weight * normalized_fitness + 
                            diversity_weight * diversity_score)
            combined_scores.append(combined_score)
        
        # Select best from tournament
        best_tournament_idx = np.argmax(combined_scores)
        return tournament_indices[best_tournament_idx]
    
    def adaptive_diversity_pressure(self, generation: int, max_generations: int) -> float:
        """Calculate adaptive diversity pressure based on search progress"""
        # Higher diversity pressure early in search, lower later
        progress = generation / max_generations if max_generations > 0 else 0.0
        
        # Use sigmoid function for smooth transition
        base_pressure = 0.5
        adaptive_component = 0.3 * (1.0 - 2.0 / (1.0 + np.exp(-5 * (progress - 0.5))))
        
        return base_pressure + adaptive_component
    
    def detect_diversity_crisis(self, threshold: float = 0.1) -> bool:
        """Detect if population diversity has critically low levels"""
        if len(self.diversity_history) < 5:
            return False
        
        recent_diversity = self.diversity_history[-5:]
        return all(d < threshold for d in recent_diversity)
    
    def estimate_effective_population_size(self, population: List[Expression]) -> float:
        """Estimate effective population size based on diversity"""
        if len(population) <= 1:
            return float(len(population))
        
        # Calculate pairwise similarities
        similarities = []
        for i in range(len(population)):
            for j in range(i + 1, len(population)):
                distance = self.structural_distance(population[i], population[j])
                similarity = 1.0 - distance
                similarities.append(similarity)
        
        if not similarities:
            return float(len(population))
        
        # Effective size is inversely related to average similarity
        avg_similarity = np.mean(similarities)
        effective_size = len(population) * (1.0 - avg_similarity)
        
        return max(1.0, float(effective_size))
    
    def multi_objective_diversity_selection(self, population: List[Expression], 
                                          fitness_scores: List[float],
                                          X: Optional[np.ndarray] = None,
                                          selection_count: Optional[int] = None) -> List[int]:
        """Multi-objective selection considering fitness, diversity, and novelty"""
        if not population:
            return []
        
        if selection_count is None:
            selection_count = len(population) // 2
        
        # Calculate multiple objectives
        objectives = []
        
        # Objective 1: Fitness (to maximize)
        objectives.append(fitness_scores)
        
        # Objective 2: Structural diversity (to maximize)
        diversity_scores = []
        for i, expr in enumerate(population):
            avg_distance = np.mean([self.structural_distance(expr, population[j]) 
                                  for j in range(len(population)) if j != i])
            diversity_scores.append(avg_distance)
        objectives.append(diversity_scores)
        
        # Objective 3: Novelty (to maximize)
        novelty_scores = []
        for expr in population:
            expr_sig = self._get_expression_signature(expr)
            if self.archive:
                distances = [self._signature_distance(expr_sig, archived_sig) 
                           for archived_sig in self.archive]
                novelty = np.mean(sorted(distances)[:min(5, len(distances))])
            else:
                novelty = 1.0
            novelty_scores.append(novelty)
        objectives.append(novelty_scores)
        
        # Objective 4: Semantic diversity (if data available)
        if X is not None:
            semantic_scores = []
            for i, expr in enumerate(population):
                semantic_distances = []
                for j, other_expr in enumerate(population):
                    if i != j:
                        sem_dist = self.semantic_distance(expr, other_expr, X)
                        semantic_distances.append(sem_dist)
                avg_semantic = np.mean(semantic_distances) if semantic_distances else 0.0
                semantic_scores.append(avg_semantic)
            objectives.append(semantic_scores)
        
        # Perform non-dominated sorting
        fronts = self._non_dominated_sort(objectives)
        
        # Select individuals from fronts
        selected_indices = []
        for front in fronts:
            if len(selected_indices) + len(front) <= selection_count:
                selected_indices.extend(front)
            else:
                # Calculate crowding distances for this front
                remaining_slots = selection_count - len(selected_indices)
                front_crowding = self._calculate_front_crowding_distances(front, objectives)
                
                # Select individuals with highest crowding distance
                front_with_crowding = list(zip(front, front_crowding))
                front_with_crowding.sort(key=lambda x: x[1], reverse=True)
                selected_indices.extend([idx for idx, _ in front_with_crowding[:remaining_slots]])
                break
        
        return selected_indices[:selection_count]
    
    def _non_dominated_sort(self, objectives: List[List[float]]) -> List[List[int]]:
        """Perform non-dominated sorting for multi-objective optimization"""
        n_individuals = len(objectives[0])
        
        # Initialize domination structures
        dominated_count = [0] * n_individuals
        dominated_solutions = [[] for _ in range(n_individuals)]
        fronts = [[]]
        
        # For each individual
        for i in range(n_individuals):
            for j in range(n_individuals):
                if i != j:
                    if self._dominates(i, j, objectives):
                        dominated_solutions[i].append(j)
                    elif self._dominates(j, i, objectives):
                        dominated_count[i] += 1
            
            # If no one dominates this individual, it's in the first front
            if dominated_count[i] == 0:
                fronts[0].append(i)
        
        # Generate subsequent fronts
        front_idx = 0
        while fronts[front_idx]:
            next_front = []
            for i in fronts[front_idx]:
                for j in dominated_solutions[i]:
                    dominated_count[j] -= 1
                    if dominated_count[j] == 0:
                        next_front.append(j)
            
            front_idx += 1
            if next_front:
                fronts.append(next_front)
            else:
                break
        
        return [front for front in fronts if front]
    
    def _dominates(self, i: int, j: int, objectives: List[List[float]]) -> bool:
        """Check if individual i dominates individual j (higher is better for all objectives)"""
        better_in_any = False
        for obj_values in objectives:
            if obj_values[i] < obj_values[j]:
                return False
            elif obj_values[i] > obj_values[j]:
                better_in_any = True
        return better_in_any
    
    def _calculate_front_crowding_distances(self, front: List[int], 
                                          objectives: List[List[float]]) -> List[float]:
        """Calculate crowding distances for individuals in a front"""
        n_individuals = len(front)
        distances = [0.0] * n_individuals
        
        if n_individuals <= 2:
            return [float('inf')] * n_individuals
        
        for obj_values in objectives:
            # Sort front by this objective
            sorted_indices = sorted(range(n_individuals), 
                                  key=lambda x: obj_values[front[x]])
            
            # Boundary solutions get infinite distance
            distances[sorted_indices[0]] = float('inf')
            distances[sorted_indices[-1]] = float('inf')
            
            # Calculate distances for intermediate solutions
            obj_range = obj_values[front[sorted_indices[-1]]] - obj_values[front[sorted_indices[0]]]
            if obj_range > 0:
                for i in range(1, n_individuals - 1):
                    prev_val = obj_values[front[sorted_indices[i - 1]]]
                    next_val = obj_values[front[sorted_indices[i + 1]]]
                    distances[sorted_indices[i]] += (next_val - prev_val) / obj_range
        
        return distances
    
    # Note: _calculate_depth method has been removed.
    # Use centralized tree_utils.calculate_tree_depth instead.
    
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
