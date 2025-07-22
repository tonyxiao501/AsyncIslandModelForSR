# expression_utils.py
import numpy as np
import warnings
from scipy.optimize import OptimizeWarning, curve_fit
from typing import List, Optional, Dict, Tuple
import sympy as sp
from .expression_tree import Expression
import time

# Global caches for optimization
_SIMPLIFICATION_CACHE: Dict[str, str] = {}
_OPTIMIZATION_CACHE: Dict[str, Tuple[List[float], float]] = {}

def to_sympy_expression(expr_string: str, advanced_simplify: bool = False,
                        n_inputs: Optional[int] = None, enable_simplify: bool = True) -> Optional[str]:
  """Convert expression to SymPy and simplify with caching"""
  if not enable_simplify:
    return expr_string
    
  # Check cache first
  cache_key = f"{expr_string}_{advanced_simplify}_{n_inputs}"
  if cache_key in _SIMPLIFICATION_CACHE:
    return _SIMPLIFICATION_CACHE[cache_key]
  
  try:
    if advanced_simplify and n_inputs is not None:
      # Advanced simplification would require the sympy_simplifier
      # For now, use basic simplification
      pass

    # Basic SymPy simplification
    sympy_expr = sp.sympify(expr_string.replace('X', 'x'))
    simplified = sp.simplify(sympy_expr)
    result = str(simplified).replace('x', 'X')
    
    # Cache the result
    _SIMPLIFICATION_CACHE[cache_key] = result
    return result
  except Exception:
    _SIMPLIFICATION_CACHE[cache_key] = expr_string
    return expr_string

def optimize_constants(X, y, population: List[Expression], steps_unchanged: int,
                       population_size: int, generations: int, enable_optimization: bool = True):
  """Optimize constants in expressions using curve fitting with caching"""
  if not enable_optimization:
    return
    
  for expr in population:
    if not should_optimize_constants(steps_unchanged, population_size, generations):
      continue
      
    # Check cache for this expression structure
    expr_str = expr.to_string()
    cache_key = f"{expr_str}_{X.shape}_{y.shape}"
    
    expr_vec = expr.vector_lambdify()
    if expr_vec is not None:
      try:
        with warnings.catch_warnings():
          warnings.simplefilter("error", OptimizeWarning)
          popt, pcov = curve_fit(expr_vec, X, y, expr.get_constants())
          expr.set_constants(popt)
          
          # Cache successful optimization
          _OPTIMIZATION_CACHE[cache_key] = (list(popt), time.time())
      except OptimizeWarning:
        pass  # failed to optimize

def should_optimize_constants(steps_unchanged: int, population_size: int, generations: int) -> bool:
  """Determine if constants should be optimized based on stagnation"""
  p = 1 / (1 + population_size * np.exp(-5 * steps_unchanged / generations))
  return np.random.rand() < p

def optimize_final_expressions(expressions: List[Expression], X: np.ndarray, y: np.ndarray) -> List[Expression]:
  """Apply full optimization to final expressions"""
  optimized_expressions = []
  
  for expr in expressions:
    # Create a copy to avoid modifying the original
    optimized_expr = expr.copy()
    
    # Apply constant optimization
    expr_vec = optimized_expr.vector_lambdify()
    if expr_vec is not None:
      try:
        with warnings.catch_warnings():
          warnings.simplefilter("error", OptimizeWarning)
          popt, pcov = curve_fit(expr_vec, X, y, optimized_expr.get_constants())
          optimized_expr.set_constants(popt)
      except (OptimizeWarning, Exception):
        pass  # Keep original if optimization fails
    
    optimized_expressions.append(optimized_expr)
  
  return optimized_expressions

def evaluate_optimized_expressions(expressions: List[Expression], X: np.ndarray, y: np.ndarray, 
                                  parsimony_coefficient: float = 0.01) -> List[float]:
  """Re-evaluate expressions after optimization using R² scores"""
  from sklearn.metrics import r2_score
  
  fitness_scores = []
  
  for expr in expressions:
    try:
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
      
      # Apply parsimony penalty to R² score
      complexity_penalty = parsimony_coefficient * expr.complexity()
      fitness = r2 - complexity_penalty
      
      fitness_scores.append(fitness)
    except Exception:
      fitness_scores.append(-np.inf)  # Invalid expression
      
  return fitness_scores

def clear_optimization_caches():
  """Clear optimization caches to prevent memory buildup"""
  global _SIMPLIFICATION_CACHE, _OPTIMIZATION_CACHE
  _SIMPLIFICATION_CACHE.clear()
  _OPTIMIZATION_CACHE.clear()
