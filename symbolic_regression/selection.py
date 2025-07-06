# selection.py
import random
import numpy as np
from typing import List
from .expression_tree import Expression

def enhanced_selection(population: List[Expression], fitness_scores: List[float],
                       diversity_score: float, diversity_threshold: float, tournament_size: int,
                       stagnation_counter: int) -> Expression:
  """Enhanced selection balancing fitness and diversity"""
  # Adaptive selection pressure
  if diversity_score > diversity_threshold:
    # Good diversity - focus more on fitness
    if random.random() < 0.85:
      return tournament_selection(population, fitness_scores, tournament_size, stagnation_counter)
    else:
      return diversity_selection(population)
  else:
    # Low diversity - balance fitness and diversity
    if random.random() < 0.6:
      return tournament_selection(population, fitness_scores, tournament_size, stagnation_counter)
    else:
      return diversity_selection(population)

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
                         tournament_size: int, stagnation_counter: int) -> Expression:
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
