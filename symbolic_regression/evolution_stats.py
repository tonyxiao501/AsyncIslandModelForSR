# evolution_stats.py
from typing import List, Dict, Any, Optional
import sympy as sp
from .expression_tree import Expression

def get_evolution_stats(fitness_history: List[float], best_fitness_history: List[float],
                        diversity_history: List[float], current_mutation_rate: float,
                        current_crossover_rate: float, stagnation_counter: int) -> Dict[str, Any]:
  """Get detailed evolution statistics"""
  return {
    'fitness_history': fitness_history.copy(),
    'best_fitness_history': best_fitness_history.copy(),
    'diversity_history': diversity_history.copy(),
    'final_mutation_rate': current_mutation_rate,
    'final_crossover_rate': current_crossover_rate,
    'total_stagnation': stagnation_counter,
    'total_generations': len(fitness_history)
  }

def get_detailed_expressions(expressions: List[Expression], sympy_simplify: bool = False) -> List[Dict]:
  """Get detailed information about expressions"""
  if not expressions:
    return []

  detailed = []
  for i, expr in enumerate(expressions):
    info = {
      'expression': expr.to_string(),
      'complexity': expr.complexity(),
      'size': expr.size(),
      'simplified': None,
      'output_index': i
    }

    if sympy_simplify:
      info['simplified'] = to_sympy_expression(expr.to_string())

    detailed.append(info)

  return detailed

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
