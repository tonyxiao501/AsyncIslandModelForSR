"""
Main Genetic Operations Module

Integrates all genetic operations including mutations, crossover, and diversity
management for symbolic regression with enhanced adaptive capabilities.

ARCHITECTURAL RESTORATION (July 2025):
- Removed fitness pre-filtering in mutations to restore natural evolution
- Increased mutation attempts from 2 to 4 for better exploration
- Lowered complexity thresholds for context-aware mutations (8→3)
- Increased probabilities for intelligent mutations while preserving exploration
- Restored full dataset evaluation instead of sampling for accuracy
- Allow neutral mutations (30% worse fitness threshold vs 5% previously)
- Increased exploration probability (25% vs 5%) to maintain diversity
- Balanced mutation success rates to not over-favor any single strategy
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

    def __init__(self, n_inputs: int, max_complexity: int = 20, scaling_range: int = 3):
        self.n_inputs = n_inputs
        self.max_complexity = max_complexity
        self.generator = ExpressionGenerator(n_inputs)
        
        # Initialize component modules
        self.mutation_strategies = MutationStrategies(n_inputs, max_complexity, scaling_range)
        self.crossover_ops = CrossoverOperations(max_complexity)
        self.context_analyzer = ExpressionContextAnalyzer(n_inputs)
        self.diversity_metrics = DiversityMetrics(n_inputs)
        
        # Context-aware mutation tracking with balanced initial rates
        self.mutation_success_rates = {
            'point': 0.4,           # Reduced from 0.5 to balance with intelligent mutations
            'subtree': 0.35,        # Slightly increased from 0.3
            'insert': 0.3,          # Increased from 0.2
            'context_aware': 0.35,  # Reduced from 0.4 to not over-favor
            'semantic_preserving': 0.25,  # Reduced from 0.3
            'simplify': 0.3         # Increased from 0.2
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
        
        # Restore natural evolution: Allow more mutation attempts and remove fitness pre-filtering
        max_attempts = min(4, len(strategies))  # Increased back to allow more exploration
        
        # Try strategies in order of predicted success
        for i, (strategy_name, strategy_func) in enumerate(strategies[:max_attempts]):
            self.mutation_attempts[strategy_name] += 1
            
            mutated = strategy_func(expression, mutation_rate, X, y)
            if mutated and mutated.complexity() <= self.max_complexity:
                # Remove fitness pre-filtering - let natural selection handle quality control
                # Only do basic validity checks, not fitness evaluation
                try:
                    # Quick stability check only
                    if X is not None:
                        test_pred = mutated.evaluate(X[:min(10, len(X))])  # Small sample for stability
                        if np.any(~np.isfinite(test_pred)) or np.max(np.abs(test_pred)) > 1e10:
                            continue  # Only reject clearly invalid mutations
                except:
                    continue  # Reject if evaluation fails
                    
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
        """Generate a replacement expression with guided diversity and exploration balance"""
        if len(population) >= 3:
            # Select diverse parents based on fitness and structure
            sorted_indices = np.argsort(fitness_scores)

            # Balance between good and diverse individuals (increased diversity weight)
            good_indices = sorted_indices[:len(population) // 4]  # Top 25% instead of 33%
            diverse_indices = self.diversity_metrics.get_diverse_individuals(population, 5)  # More diverse candidates

            parent_indices = list(set(good_indices) | set(diverse_indices))
            parents = [population[i] for i in parent_indices[:3]]

            # Create hybrid offspring with higher mutation for exploration
            if len(parents) >= 2:
                child1, child2 = self.crossover(parents[0], parents[1])
                child = self.mutate(child1, 0.4)  # Increased from 0.3 for more exploration
                return child

        # Fallback to new random expression with slightly deeper trees
        return Expression(self.generator.generate_random_expression(max_depth=5))
    
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
        """Select mutation strategies based on expression characteristics and performance"""
        
        # Restore balanced strategy selection - mix exploration with intelligence
        strategies = []
        
        # Start with basic exploration strategies (always available)
        basic_strategies = [
            ('point', self.mutation_strategies.point_mutation),
            ('subtree', self.mutation_strategies.subtree_mutation),
            ('insert', self.mutation_strategies.insert_mutation),
        ]
        
        # Sort by success rate but don't exclude any
        basic_strategies.sort(key=lambda x: self.mutation_success_rates.get(x[0], 0.3), reverse=True)
        strategies.extend(basic_strategies)
        
        # Lower complexity threshold for context-aware mutations (was > 8, now > 3)
        complexity = expression.complexity()
        
        if complexity > 3:  # Much lower threshold to allow context-aware mutations earlier
            # Increase probability of context-aware mutations (was 0.3, now 0.7)
            if X is not None and y is not None and random.random() < 0.7:
                strategies.append(('context_aware', self.mutation_strategies.context_aware_mutation))
            
            # Lower threshold for semantic preserving (was 0.8, now 0.5 of max complexity)
            if complexity > self.max_complexity * 0.5 and random.random() < 0.5:
                strategies.append(('semantic_preserving', self.mutation_strategies.semantic_preserving_mutation))
        
        # Always include simplification
        strategies.append(('simplify', self.mutation_strategies.simplify_mutation))
        
        # Add random exploration with higher probability for diversity
        if random.random() < 0.4:  # 40% chance to include more random exploration
            strategies.insert(0, ('point', self.mutation_strategies.point_mutation))  # Add extra random at start
        
        # Remove duplicates while preserving order
        unique_strategies = []
        seen = set()
        for name, func in strategies:
            if name not in seen:
                unique_strategies.append((name, func))
                seen.add(name)
        
        return unique_strategies
    
    def _is_beneficial_mutation(self, original: Expression, mutated: Expression,
                              X: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None) -> bool:
        """Check if a mutation is beneficial - now allowing neutral mutations for exploration"""
        if X is None or y is None:
            return True  # Accept if we can't evaluate
        
        try:
            # Use full dataset for accurate fitness evaluation (restore from sampling)
            original_predictions = original.evaluate(X)
            mutated_predictions = mutated.evaluate(X)
            
            if original_predictions.ndim == 1:
                original_predictions = original_predictions.reshape(-1, 1)
            if mutated_predictions.ndim == 1:
                mutated_predictions = mutated_predictions.reshape(-1, 1)
            
            # Quick stability check - reject obviously bad mutations
            if (np.any(~np.isfinite(mutated_predictions)) or 
                np.max(np.abs(mutated_predictions)) > 1e10):  # Increased threshold for more tolerance
                return False
            
            # Allow neutral mutations - essential for evolutionary search
            # Use MSE for accurate comparison on full dataset
            original_mse = np.mean((y - original_predictions) ** 2)
            mutated_mse = np.mean((y - mutated_predictions) ** 2)
            
            # Allow significantly worse mutations for exploration (increased from 5% to 30%)
            improvement_threshold = 0.3  # Allow 30% worse MSE for exploration
            exploration_probability = 0.25  # 25% chance to accept any valid mutation
            
            return bool(mutated_mse <= original_mse * (1 + improvement_threshold) or 
                       random.random() < exploration_probability)
            
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
        Evaluate the fitness of a single expression using R² score with PySR-style parsimony.
        If X and y are not provided, returns a large negative value.
        """
        if X is None or y is None:
            return -10.0  # Large negative R² score when data unavailable

        try:
            from ..data_processing import r2_score
            from ..adaptive_parsimony import PySRStyleComplexity
            
            predictions = expr.evaluate(X)
            if predictions.ndim == 1:
                predictions = predictions.reshape(-1, 1)
            
            # Calculate R² score using our robust implementation
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
            
            # Apply PySR-style parsimony penalty
            parsimony_penalty = PySRStyleComplexity.get_parsimony_penalty(
                expr, parsimony_coefficient
            )
            
            # Apply stability penalties
            stability_penalty = 0.0
            max_abs_pred = np.max(np.abs(predictions))
            
            if max_abs_pred > 1e6:
                stability_penalty = 0.3  # Adjust to R² scale
            elif max_abs_pred > 1e4:
                stability_penalty = 0.1
            
            if np.any(~np.isfinite(predictions)):
                stability_penalty += 0.5  # Heavily penalize infinite/NaN
            
            fitness = r2 - parsimony_penalty - stability_penalty
            return float(fitness)
        except Exception:
            return -10.0  # Large negative R² score for invalid expressions
