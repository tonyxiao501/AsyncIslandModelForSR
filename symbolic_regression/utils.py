# Miscellaneous utility functions (e.g., string similarity, etc.)
# (Move all _string_similarity and similar helpers here)
from typing import List

import numpy as np

from symbolic_regression import Expression


def string_similarity(s1, s2):
  """Calculate string similarity using character-based Jaccard index"""
  if s1 == s2:
    return 1.0
  set1 = set(s1.replace(' ', '').replace('(', '').replace(')', ''))
  set2 = set(s2.replace(' ', '').replace('(', '').replace(')', ''))
  if not set1 and not set2:
    return 1.0
  intersection = len(set1 & set2)
  union = len(set1 | set2)
  return intersection / union if union > 0 else 0.0

# Add all other utility functions here.
def calculate_expression_uniqueness(expr: Expression, population: List[Expression]) -> float:
    """Calculate how unique an expression is compared to population"""
    expr_string = expr.to_string()
    unique_score = 0.0

    for other in population:
      if expr is not other:
        similarity = string_similarity(expr_string, other.to_string())
        unique_score += (1.0 - similarity)

    return unique_score / max(1, len(population) - 1)

def calculate_population_diversity(population: List[Expression]) -> float:
    """Calculate population diversity using multiple metrics"""
    if len(population) < 2:
      return 1.0

    # String-based diversity
    strings = [expr.to_string() for expr in population]
    unique_strings = len(set(strings))
    string_diversity = unique_strings / len(population)

    # Complexity diversity
    complexities = [expr.complexity() for expr in population]
    complexity_std = np.std(complexities) / (np.mean(complexities) + 1e-6)
    complexity_diversity = min(1.0, float(complexity_std))

    # Size diversity
    sizes = [expr.size() for expr in population]
    size_diversity = len(set(sizes)) / len(population)

    # Combined diversity score
    return (string_diversity * 0.5 + complexity_diversity * 0.3 + size_diversity * 0.2)