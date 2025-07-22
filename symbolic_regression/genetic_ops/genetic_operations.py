"""
Main Genetic Operations Module

Integrates all genetic operations including mutations, crossover, and diversity
management for symbolic regression with enhanced adaptive capabilities.
"""

import numpy as np
import random
from typing import List, Tuple, Optional, Dict, Callable

from ..expression_tree import Expression
from ..generator import ExpressionGenerator
from .mutation_strategies import MutationStrategies
from .crossover_operations import CrossoverOperations
from .context_analysis import ExpressionContextAnalyzer
from .diversity_metrics import DiversityMetrics


class GeneticOperations:
    """Enhanced genetic operations with Grammatical Evolution-inspired improvements"""

    def __init__(self, n_inputs: int, max_complexity: int = 20):
        self.n_inputs = n_inputs
        self.max_complexity = max_complexity
        self.generator = ExpressionGenerator(n_inputs)
        
        # Initialize component modules
        self.mutation_strategies = MutationStrategies(n_inputs, max_complexity)
        self.crossover_ops = CrossoverOperations(max_complexity)
        self.context_analyzer = ExpressionContextAnalyzer(n_inputs)
        self.diversity_metrics = DiversityMetrics(n_inputs)
        
        # Context-aware mutation tracking
        self.mutation_success_rates = {
            'point': 0.5,
            'subtree': 0.3,
            'insert': 0.2,
            'context_aware': 0.4,
            'semantic_preserving': 0.3,
            'simplify': 0.2
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
        return self.mutation_strategies.safe_constant_mutation(expression)
    
    def crossover(self, parent1: Expression, parent2: Expression) -> Tuple[Expression, Expression]:
        """Enhanced crossover with proper subtree exchange"""
        return self.crossover_ops.enhanced_crossover(parent1, parent2)
    
    def quality_guided_crossover(self, parent1: Expression, parent2: Expression, 
                                X: np.ndarray, residuals1: np.ndarray, residuals2: np.ndarray) -> Tuple[Expression, Expression]:
        """Quality-guided crossover using subtree quality assessment"""
        return self.crossover_ops.quality_guided_crossover(parent1, parent2, X, residuals1, residuals2)
    
    def adaptive_mutate_with_feedback(self, expression: Expression, mutation_rate: float = 0.1, 
                                     X: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None,
                                     generation: int = 0, stagnation_count: int = 0) -> Expression:
        """Enhanced mutation with adaptive strategy selection based on evolution state"""
        
        # Adjust mutation aggressiveness based on stagnation
        if stagnation_count > 15:
            # High stagnation - use more aggressive mutations
            mutation_rate *= 1.5
            aggressive_strategies = [
                ('context_aware', self.mutation_strategies.context_aware_mutation),
                ('subtree', self.mutation_strategies.subtree_mutation),
                ('semantic_preserving', self.mutation_strategies.semantic_preserving_mutation)
            ]
            
            for strategy_name, strategy_func in aggressive_strategies:
                mutated = strategy_func(expression, mutation_rate, X, y)
                if mutated and mutated.complexity() <= self.max_complexity:
                    return mutated
        
        elif stagnation_count > 8:
            # Medium stagnation - prefer structure-changing mutations
            mutation_rate *= 1.2
            medium_strategies = [
                ('context_aware', self.mutation_strategies.context_aware_mutation),
                ('insert', self.mutation_strategies.insert_mutation),
                ('subtree', self.mutation_strategies.subtree_mutation)
            ]
            
            for strategy_name, strategy_func in medium_strategies:
                mutated = strategy_func(expression, mutation_rate, X, y)
                if mutated and mutated.complexity() <= self.max_complexity:
                    return mutated
        
        # Normal mutation using the standard adaptive approach
        return self.mutate(expression, mutation_rate, X, y)
    
    def generate_replacement(self, population: List[Expression], fitness_scores: List[float]) -> Expression:
        """Generate a replacement expression with guided diversity"""
        if len(population) >= 3:
            # Select diverse parents based on fitness and structure
            sorted_indices = np.argsort(fitness_scores)

            # Mix good and diverse individuals
            good_indices = sorted_indices[:len(population) // 3]
            diverse_indices = self.diversity_metrics.get_diverse_individuals(population, 3)

            parent_indices = list(set(good_indices) | set(diverse_indices))
            parents = [population[i] for i in parent_indices[:3]]

            # Create hybrid offspring
            if len(parents) >= 2:
                child1, child2 = self.crossover(parents[0], parents[1])
                child = self.mutate(child1, 0.3)  # Higher mutation for replacement
                return child

        # Fallback to new random expression
        return Expression(self.generator.generate_random_expression(max_depth=4))
    
    def get_mutation_statistics(self) -> Dict[str, float]:
        """Get statistics about mutation strategy success rates"""
        stats = {}
        for strategy in self.mutation_success_rates:
            attempts = self.mutation_attempts.get(strategy, 1)
            successes = self.mutation_successes.get(strategy, 0)
            stats[f"{strategy}_success_rate"] = successes / attempts
            stats[f"{strategy}_attempts"] = attempts
        return stats
    
    def analyze_population_diversity(self, population: List[Expression]) -> Dict[str, float]:
        """Analyze diversity metrics for the current population"""
        return self.diversity_metrics.calculate_population_diversity(population)
    
    def maintain_population_diversity(self, population: List[Expression], target_diversity: float = 0.5) -> List[Expression]:
        """Maintain population diversity by replacing similar individuals"""
        return self.diversity_metrics.maintain_diversity(population, target_diversity)
    
    def _select_mutation_strategies(self, expression: Expression, X: Optional[np.ndarray] = None, 
                                  y: Optional[np.ndarray] = None) -> List[Tuple[str, Callable]]:
        """Select and order mutation strategies based on context and success rates"""
        
        # Analyze expression characteristics
        context = self.context_analyzer.analyze_expression_context(expression)
        
        # Build strategy list with priorities
        strategies = []
        
        # High complexity expressions benefit from simplification
        if context['complexity'] > self.max_complexity * 0.7:
            strategies.append(('semantic_preserving', self.mutation_strategies.semantic_preserving_mutation))
            strategies.append(('simplify', self.mutation_strategies.simplify_mutation))
        
        # Low diversity expressions need more radical changes
        if context['has_repeated_patterns']:
            strategies.append(('context_aware', self.mutation_strategies.context_aware_mutation))
            strategies.append(('subtree', self.mutation_strategies.subtree_mutation))
        
        # Expressions with many constants benefit from constant optimization
        if context['constant_ratio'] > 0.3:
            strategies.append(('point', self.mutation_strategies.point_mutation))
        
        # Always include standard strategies, ordered by success rate
        standard_strategies = [
            ('context_aware', self.mutation_strategies.context_aware_mutation),
            ('point', self.mutation_strategies.point_mutation),
            ('subtree', self.mutation_strategies.subtree_mutation),
            ('insert', self.mutation_strategies.insert_mutation)
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
    
    def _evaluate_fitness(self, expr: Expression, X: Optional[np.ndarray] = None, 
                         y: Optional[np.ndarray] = None, parsimony_coefficient: float = 0.001) -> float:
        """
        Evaluate the fitness of a single expression using R² score.
        If X and y are not provided, returns a large negative value.
        """
        if X is None or y is None:
            return -1e8

        try:
            from sklearn.metrics import r2_score
            
            predictions = expr.evaluate(X)
            if predictions.ndim == 1:
                predictions = predictions.reshape(-1, 1)
            
            # Calculate R² score using scikit-learn
            try:
                r2 = r2_score(y.flatten(), predictions.flatten())
            except Exception:
                # Fallback calculation for edge cases
                ss_res = np.sum((y - predictions) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                if ss_tot == 0:
                    r2 = 1.0 if ss_res == 0 else 0.0
                else:
                    r2 = 1.0 - (ss_res / ss_tot)
            
            # Apply penalties to R² score
            complexity_penalty = parsimony_coefficient * expr.complexity()
            stability_penalty = 0.0
            max_abs_pred = np.max(np.abs(predictions))
            
            if max_abs_pred > 1e6:
                stability_penalty = 0.3  # Adjust to R² scale
            elif max_abs_pred > 1e4:
                stability_penalty = 0.1
            
            if np.any(~np.isfinite(predictions)):
                stability_penalty += 0.5  # Heavily penalize infinite/NaN
            
            fitness = r2 - complexity_penalty - stability_penalty
            return float(fitness)
        except Exception:
            return -1e8
