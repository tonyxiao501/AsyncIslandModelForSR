# expression_utils.py
import numpy as np
import warnings
from scipy.optimize import OptimizeWarning, curve_fit
from typing import List, Optional
import sympy as sp
from .expression_tree import Expression

def to_sympy_expression(expr_string: str, advanced_simplify: bool = False,
                        n_inputs: Optional[int] = None) -> Optional[str]:
  """Convert expression to SymPy and simplify"""
  try:
    if advanced_simplify and n_inputs is not None:
      # Advanced simplification would require the sympy_simplifier
      # For now, use basic simplification
      pass

    # Basic SymPy simplification
    sympy_expr = sp.sympify(expr_string.replace('X', 'x'))
    simplified = sp.simplify(sympy_expr)
    return str(simplified).replace('x', 'X')
  except Exception:
    return expr_string

def optimize_constants(X, y, population: List[Expression], steps_unchanged: int,
                       population_size: int, generations: int):
  """Optimize constants in expressions using curve fitting"""
  for expr in population:
    if not should_optimize_constants(steps_unchanged, population_size, generations):
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

def should_optimize_constants(steps_unchanged: int, population_size: int, generations: int) -> bool:
  """Determine if constants should be optimized based on stagnation"""
  p = 1 / (1 + population_size * np.exp(-5 * steps_unchanged / generations))
  return np.random.rand() < p
