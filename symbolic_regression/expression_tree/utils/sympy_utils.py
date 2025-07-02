import sympy as sp
from typing import Optional, Dict, Any
import numpy as np

class SymPySimplifier:
  """Advanced SymPy-based expression simplifier"""

  def __init__(self):
    self.simplification_strategies = [
      'simplify',
      'expand',
      'factor',
      'trigsimp',
      'logcombine'
    ]

  def simplify_expression(self, expr_string: str, n_inputs: int = 1) -> Dict[str, Any]:
    """
    Simplify expression using multiple SymPy strategies

    Returns:
        Dict with simplified expression and metadata
    """
    try:
      sympy_expr = self._convert_to_sympy(expr_string, n_inputs)
      original_complexity = self._calculate_complexity(sympy_expr)

      best_simplified = sympy_expr
      best_complexity = original_complexity
      best_strategy = 'none'

      # Try different simplification strategies
      for strategy in self.simplification_strategies:
        try:
          if strategy == 'simplify':
            simplified = sp.simplify(sympy_expr)
          elif strategy == 'expand':
            simplified = sp.expand(sympy_expr)
          elif strategy == 'factor':
            simplified = sp.factor(sympy_expr)
          elif strategy == 'trigsimp':
            simplified = sp.trigsimp(sympy_expr)
          elif strategy == 'logcombine':
            simplified = sp.logcombine(sympy_expr)
          else:
            continue

          complexity = self._calculate_complexity(simplified)
          if complexity < best_complexity:
            best_simplified = simplified
            best_complexity = complexity
            best_strategy = strategy

        except Exception:
          continue

      return {
        'simplified': self._convert_from_sympy(best_simplified, n_inputs),
        'strategy_used': best_strategy,
        'complexity_reduction': original_complexity - best_complexity,
        'original_complexity': original_complexity,
        'simplified_complexity': best_complexity
      }

    except Exception as e:
      return {
        'simplified': expr_string,
        'strategy_used': 'failed',
        'complexity_reduction': 0,
        'error': str(e)
      }

  def _convert_to_sympy(self, expr_string: str, n_inputs: int) -> sp.Expr:
    """Convert expression string to SymPy expression"""
    sympy_string = expr_string

    # Replace variables
    for i in range(n_inputs):
      sympy_string = sympy_string.replace(f'X{i}', f'x{i}')

    # Replace operators and functions
    replacements = {
      '^': '**',
    }

    for old, new in replacements.items():
      sympy_string = sympy_string.replace(old, new)

    return sp.sympify(sympy_string)

  def _convert_from_sympy(self, sympy_expr: sp.Expr, n_inputs: int) -> str:
    """Convert SymPy expression back to string"""
    expr_string = str(sympy_expr)

    # Replace variables back
    for i in range(n_inputs):
      expr_string = expr_string.replace(f'x{i}', f'X{i}')

    # Replace operators back
    expr_string = expr_string.replace('**', '^')

    return expr_string

  def _calculate_complexity(self, expr: sp.Expr) -> int:
    """Calculate expression complexity for SymPy expressions"""
    return len(expr.free_symbols) + len(expr.atoms(sp.Function)) + expr.count_ops()

  def latex_representation(self, expr_string: str, n_inputs: int = 1) -> str:
    """Get LaTeX representation of the expression"""
    try:
      sympy_expr = self._convert_to_sympy(expr_string, n_inputs)
      return sp.latex(sympy_expr)
    except Exception:
      return expr_string