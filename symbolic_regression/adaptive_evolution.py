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
  # Base adaptation based on diversity and stagnation
  if diversity_score < diversity_threshold:
    # Low diversity - increase exploration
    mutation_multiplier = 1.0 + (diversity_threshold - diversity_score) * 2.0
    crossover_multiplier = 0.9
  else:
    # Good diversity - normal rates
    mutation_multiplier = 1.0
    crossover_multiplier = 1.0

  # Additional adaptation based on plateau
  if plateau_counter > 15:
    mutation_multiplier *= 1.5
    crossover_multiplier *= 0.8
  elif plateau_counter > 10:
    mutation_multiplier *= 1.2

  # Apply multipliers with bounds
  new_mutation_rate = np.clip(mutation_rate * mutation_multiplier, 0.05, 0.5)
  new_crossover_rate = np.clip(crossover_rate * crossover_multiplier, 0.5, 0.95)

  # Gradually return to original rates when performing well
  if stagnation_counter < 5 and plateau_counter < 5:
    new_mutation_rate = (current_mutation_rate * 0.95 + mutation_rate * 0.05)
    new_crossover_rate = (current_crossover_rate * 0.95 + crossover_rate * 0.05)
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
