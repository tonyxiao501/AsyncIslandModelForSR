import warnings
from typing import List

from scipy.optimize import curve_fit, OptimizeWarning


def _optimize_constants(self, X, y, popolation: List[Expression], steps_unchanged):
  for expr in popolation:
    if not self._should_optimize_constants(steps_unchanged):
      continue
    expr_vec = expr.vector_lambdify()
    if expr_vec is not None:
      try:
        with warnings.catch_warnings():
          warnings.simplefilter("error", OptimizeWarning)
          popt, pcov = curve_fit(expr_vec, X, y, expr.get_constants())
          expr.set_constants(popt)
      except OptimizeWarning:
        pass  # failed to optimize


def _should_optimize_constants(self, steps_unchanged):
  p = 1 / (1 + self.population_size * np.exp(-5 * steps_unchanged / self.generations))
  if np.random.rand() < p:
    return True
  else:
    return False


def _should_optimize_constants_enhanced(self, expr_index: int, population_size: int,
                                        steps_unchanged: int, generation: int) -> bool:
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

