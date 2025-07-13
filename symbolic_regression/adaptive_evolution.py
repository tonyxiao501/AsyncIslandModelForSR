# adaptive_evolution.py
import numpy as np
from typing import List
from .expression_tree import Expression
from .generator import ExpressionGenerator
from .genetic_ops import GeneticOperations
from .population import PopulationManager

def update_adaptive_parameters(self, generation: int, diversity_score: float, plateau_counter: int,
                               diversity_threshold: float, mutation_rate: float, crossover_rate: float,
                               current_mutation_rate: float, current_crossover_rate: float,
                               stagnation_counter: int):
  """Enhanced adaptive parameter updates"""
  # Base adaptation based on diversity and stagnation - more conservative
  if diversity_score < diversity_threshold:
    # Low diversity - increase exploration moderately
    mutation_multiplier = 1.0 + (diversity_threshold - diversity_score) * 1.0  # Reduced from 2.0
    crossover_multiplier = 0.95  # Less aggressive reduction
  else:
    # Good diversity - normal rates
    mutation_multiplier = 1.0
    crossover_multiplier = 1.0

  # Additional adaptation based on plateau - more conservative
  if plateau_counter > 20:  # Increased threshold
    mutation_multiplier *= 1.3  # Reduced from 1.5
    crossover_multiplier *= 0.85  # Less aggressive
  elif plateau_counter > 15:  # Increased threshold
    mutation_multiplier *= 1.1  # Reduced from 1.2

  # Apply multipliers with bounds
  new_mutation_rate = np.clip(mutation_rate * mutation_multiplier, 0.05, 0.4)  # Reduced max
  new_crossover_rate = np.clip(crossover_rate * crossover_multiplier, 0.6, 0.95)  # Increased min

  # More gradual return to original rates when performing well
  if stagnation_counter < 3 and plateau_counter < 3:  # Stricter condition
    new_mutation_rate = (current_mutation_rate * 0.98 + mutation_rate * 0.02)  # Slower adaptation
    new_crossover_rate = (current_crossover_rate * 0.98 + crossover_rate * 0.02)
  else:
    new_mutation_rate = current_mutation_rate
    new_crossover_rate = current_crossover_rate

  return new_mutation_rate, new_crossover_rate

def restart_population_enhanced(population: List[Expression], fitness_scores: List[float],
                                generator: ExpressionGenerator, population_size: int, n_inputs: int,
                                pop_manager: PopulationManager):
  """Enhanced population restart with better elite preservation"""
  # Keep top performers (more aggressive selection)
  elite_count = max(2, int(population_size * 0.05))  # Keep top 5%
  elite_indices = np.argsort(fitness_scores)[-elite_count:]
  elites = [population[i].copy() for i in elite_indices]

  new_population = elites.copy()

  # Create variants of elites with different mutation strengths
  if n_inputs is None:
    raise ValueError("n_inputs must be set before creating genetic operations")
  genetic_ops = GeneticOperations(n_inputs, max_complexity=25)
  for elite in elites:
    # High mutation variants
    for _ in range(2):
      mutated = genetic_ops.mutate(elite, 0.4)
      if pop_manager.is_expression_valid_cached(mutated):
        new_population.append(mutated)

    # Medium mutation variants
    for _ in range(2):
      mutated = genetic_ops.mutate(elite, 0.2)
      if pop_manager.is_expression_valid_cached(mutated):
        new_population.append(mutated)

  # Fill rest with completely new diverse individuals
  while len(new_population) < population_size:
    new_expr = Expression(generator.generate_random_expression())
    if pop_manager.is_expression_valid_cached(new_expr):
      new_population.append(new_expr)

  return new_population[:population_size]

def should_optimize_constants_enhanced(expr_index: int, population_size: int, steps_unchanged: int, generation: int) -> bool:
  """Enhanced logic for when to optimize constants"""
  if expr_index < population_size * 0.1:  # Top 10%
    return True

  if steps_unchanged > 5:
    base_prob = min(0.3, steps_unchanged / 20)
    rank_factor = 1.0 - (expr_index / population_size)
    prob = base_prob * (1 + rank_factor)
    return np.random.rand() < prob

  if generation % 20 == 0 and expr_index < population_size * 0.3:
    return True

  return False
