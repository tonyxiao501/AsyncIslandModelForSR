"""
Genetic Operations for MIMO Symbolic Regression
This module consolidates all genetic operations including selection, crossover, and mutation.
"""
import random
import numpy as np
from typing import List, Tuple, Optional, Dict, Callable

from .expression_tree import Expression
from .generator import ExpressionGenerator
from .genetic_ops import GeneticOperations as GeneticOpsClass


def enhanced_selection(population: List[Expression], fitness_scores: List[float],
                       tournament_size: int) -> Expression:
    """Enhanced selection balancing fitness and diversity"""
    # Use tournament selection as the primary method
    return tournament_selection(population, fitness_scores, tournament_size)


def diversity_selection(population: List[Expression]) -> Expression:
    """Select based on diversity (less common expressions)"""
    # Simple diversity selection - prefer less common expressions
    strings = [expr.to_string() for expr in population]
    string_counts = {}
    for s in strings:
        string_counts[s] = string_counts.get(s, 0) + 1

    # Weight selection inversely by frequency
    weights = [1.0 / string_counts[expr.to_string()] for expr in population]
    total_weight = sum(weights)

    if total_weight > 0:
        weights = [w / total_weight for w in weights]
        chosen_index = np.random.choice(len(population), p=weights)
        return population[chosen_index]
    else:
        return random.choice(population)


def tournament_selection(population: List[Expression], fitness_scores: List[float],
                         tournament_size: int, stagnation_counter: int = 0) -> Expression:
    """Enhanced tournament selection with adaptive tournament size"""
    # Adaptive tournament size based on stagnation
    if stagnation_counter > 15:
        adaptive_tournament_size = max(2, tournament_size - 1)  # Smaller tournaments for more diversity
    elif stagnation_counter > 8:
        adaptive_tournament_size = tournament_size
    else:
        adaptive_tournament_size = min(len(population), tournament_size + 1)  # Larger tournaments for better selection

    tournament_indices = random.sample(range(len(population)),
                                       min(adaptive_tournament_size, len(population)))
    tournament_fitness = [fitness_scores[i] for i in tournament_indices]
    winner_idx = tournament_indices[np.argmax(tournament_fitness)]
    return population[winner_idx]


class GeneticOperations:
    """
    Consolidated genetic operations class that wraps the more complex genetic_ops module
    for simplified use in the evolution engine.
    """
    
    def __init__(self, n_inputs: int, max_complexity: int = 20):
        self.n_inputs = n_inputs
        self.max_complexity = max_complexity
        self.generator = ExpressionGenerator(n_inputs)
        
        # Use the advanced genetic operations from the genetic_ops module
        self.genetic_ops = GeneticOpsClass(n_inputs, max_complexity)
        
    def mutate(self, expression: Expression, mutation_rate: float = 0.1, 
               X: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None) -> Expression:
        """Apply mutation to an expression"""
        return self.genetic_ops.mutate(expression, mutation_rate, X, y)
    
    def crossover(self, parent1: Expression, parent2: Expression) -> Expression:
        """Apply crossover to two parent expressions"""
        # The genetic_ops crossover returns a tuple, we'll take the first offspring
        offspring1, offspring2 = self.genetic_ops.crossover(parent1, parent2)
        
        # Return the offspring with better complexity (simpler is better for equal fitness)
        if offspring1.complexity() <= offspring2.complexity():
            return offspring1
        else:
            return offspring2
    
    def adaptive_mutate_with_feedback(self, expression: Expression, mutation_rate: float = 0.1,
                                    X: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None,
                                    generation: int = 0, stagnation_count: int = 0) -> Expression:
        """Apply adaptive mutation with feedback"""
        return self.genetic_ops.adaptive_mutate_with_feedback(
            expression, mutation_rate, X, y, generation, stagnation_count
        )
    
    def generate_replacement(self, population: List[Expression], fitness_scores: List[float]) -> Expression:
        """Generate a replacement expression"""
        return self.genetic_ops.generate_replacement(population, fitness_scores)


# Note: Advanced genetic operations are available in the genetic_ops module
# This module provides simplified interfaces for the evolution engine


__all__ = [
    'enhanced_selection',
    'diversity_selection', 
    'tournament_selection',
    'GeneticOperations'
]
