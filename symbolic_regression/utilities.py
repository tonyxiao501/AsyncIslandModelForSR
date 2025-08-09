"""
Utility Functions for MIMO Symbolic Regression
This module consolidates utility functions, expression utilities, evolution statistics, and quality assessment.
"""

import numpy as np
import warnings
import time
from typing import List, Dict, Any, Optional, Tuple, Union
from scipy.optimize import OptimizeWarning, curve_fit
import sympy as sp
from .expression_tree import Expression
from .expression_tree.core.node import Node, BinaryOpNode, UnaryOpNode
from .expression_tree.utils.tree_utils import get_all_nodes
from .data_processing import r2_score

# Global caches for optimization
_SIMPLIFICATION_CACHE: Dict[str, str] = {}
_OPTIMIZATION_CACHE: Dict[str, Tuple[List[float], float]] = {}


# String and Population Utilities
def string_similarity(s1: str, s2: str) -> float:
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


# Expression Utilities
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


def optimize_constants(X: np.ndarray, y: np.ndarray, population: List[Expression], steps_unchanged: int,
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
            consts = expr.get_constants()
            if len(consts) == 0:
                continue
            # Build a safe wrapper to ensure 1-D float output and bounded values
            def safe_expr_vec(Xin, *params):
                try:
                    if expr_vec is None:
                        raise RuntimeError("expr_vec is None")
                    out = expr_vec(Xin, *params)
                except Exception:
                    # Return large constant to signal bad fit but keep shape
                    n = np.asarray(Xin).shape[0]
                    return np.full(n, 1e6, dtype=float)
                out = np.asarray(out, dtype=float).reshape(-1)
                # Sanitize to avoid NaN/Inf and huge magnitudes
                out = np.nan_to_num(out, nan=0.0, posinf=1e6, neginf=-1e6)
                return np.clip(out, -1e6, 1e6)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error", OptimizeWarning)
                    y_flat = np.asarray(y, dtype=float).reshape(-1)
                    # Bounded optimization to keep constants in a reasonable range
                    bounds = (-100.0 * np.ones(len(consts)), 100.0 * np.ones(len(consts)))
                    popt, pcov = curve_fit(safe_expr_vec, X, y_flat, consts, bounds=bounds, maxfev=2000)
                    expr.set_constants(popt)
                    # Cache successful optimization
                    _OPTIMIZATION_CACHE[cache_key] = (list(popt), time.time())
            except (OptimizeWarning, Exception):
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
            consts = optimized_expr.get_constants()
            if len(consts) > 0:
                def safe_expr_vec(Xin, *params):
                    try:
                        if expr_vec is None:
                            raise RuntimeError("expr_vec is None")
                        out = expr_vec(Xin, *params)
                    except Exception:
                        n = np.asarray(Xin).shape[0]
                        return np.full(n, 1e6, dtype=float)
                    out = np.asarray(out, dtype=float).reshape(-1)
                    out = np.nan_to_num(out, nan=0.0, posinf=1e6, neginf=-1e6)
                    return np.clip(out, -1e6, 1e6)
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("error", OptimizeWarning)
                        y_flat = np.asarray(y, dtype=float).reshape(-1)
                        bounds = (-100.0 * np.ones(len(consts)), 100.0 * np.ones(len(consts)))
                        popt, pcov = curve_fit(safe_expr_vec, X, y_flat, consts, bounds=bounds, maxfev=2000)
                        optimized_expr.set_constants(popt)
                except (OptimizeWarning, Exception):
                    pass  # Keep original if optimization fails
        
        optimized_expressions.append(optimized_expr)
    
    return optimized_expressions


def evaluate_optimized_expressions(expressions: List[Expression], X: np.ndarray, y: np.ndarray, 
                                  parsimony_coefficient: float = 0.01) -> List[float]:
    """Re-evaluate expressions after optimization using R² scores with PySR-style parsimony"""
    from .adaptive_parsimony import PySRStyleComplexity
    fitness_scores = []
    
    for expr in expressions:
        try:
            predictions = expr.evaluate(X)
            predictions = np.asarray(predictions, dtype=float)
            if predictions.ndim == 1:
                predictions = predictions.reshape(-1, 1)
            # Sanitize predictions to avoid NaN/Inf during scoring
            predictions = np.nan_to_num(predictions, nan=0.0, posinf=1e6, neginf=-1e6)
            predictions = np.clip(predictions, -1e6, 1e6)
                
            # Calculate R² score using scikit-learn
            try:
                r2 = r2_score(np.asarray(y, dtype=float).flatten(), predictions.flatten())
            except Exception:
                # Fallback calculation for edge cases
                yf = np.asarray(y, dtype=float)
                ss_res = np.sum((yf - predictions) ** 2)
                ss_tot = np.sum((yf - np.mean(yf)) ** 2)
                if ss_tot == 0:
                    r2 = 1.0 if ss_res == 0 else 0.0
                else:
                    r2 = 1.0 - (ss_res / ss_tot)
            
            # Apply PySR-style parsimony penalty
            parsimony_penalty = PySRStyleComplexity.get_parsimony_penalty(
                expr, parsimony_coefficient
            )
            fitness = r2 - parsimony_penalty
            
            fitness_scores.append(fitness)
        except Exception:
            fitness_scores.append(-10.0)  # Large negative R² score for invalid expressions
            
    return fitness_scores


def clear_optimization_caches():
    """Clear optimization caches to prevent memory buildup"""
    global _SIMPLIFICATION_CACHE, _OPTIMIZATION_CACHE
    _SIMPLIFICATION_CACHE.clear()
    _OPTIMIZATION_CACHE.clear()


# Evolution Statistics
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


# Quality Assessment
def calculate_subtree_qualities(expression: Expression, X: np.ndarray, residuals: np.ndarray) -> Dict[Node, float]:
    """Calculate quality scores for subtrees based on correlation with residuals"""
    qualities = {}
    nodes = get_all_nodes(expression.root)

    for node in nodes:
        try:
            subtree_output = node.evaluate(X)
            # Sanitize to avoid overflow in std/corr computations
            subtree_output = np.asarray(subtree_output, dtype=float)
            subtree_output = np.nan_to_num(subtree_output, nan=0.0, posinf=1e6, neginf=-1e6)
            subtree_output = np.clip(subtree_output, -1e6, 1e6)

            if np.std(subtree_output) < 1e-6 or np.any(~np.isfinite(subtree_output)):
                qualities[node] = 0.0
                continue

            correlation = np.corrcoef(subtree_output, residuals)[0, 1]
           
            # Convert correlation to a quality score (0 to 1)
            # A perfect negative correlation (-1) gives a quality of 0, while no correlation 
            quality_score = (1.0 - correlation) / 2.0
            qualities[node] = quality_score
            
        except Exception:
            qualities[node] = 0.0  # Assign zero quality if any error occurs
            
    return qualities


def assess_expression_quality(expression: Expression, X: np.ndarray, y: np.ndarray,
                             return_detailed: bool = False) -> Union[float, Dict[str, Any]]:
    """Comprehensive quality assessment of an expression"""
    try:
        predictions = expression.evaluate(X)
        predictions = np.asarray(predictions, dtype=float)
        if predictions.ndim == 1:
            predictions = predictions.reshape(-1, 1)
        # Sanitize predictions to avoid NaN/Inf and huge magnitudes
        predictions = np.nan_to_num(predictions, nan=0.0, posinf=1e6, neginf=-1e6)
        predictions = np.clip(predictions, -1e6, 1e6)

        # Calculate basic metrics
        r2 = r2_score(y.flatten(), predictions.flatten())
        # Clip y as well to avoid overflow in subtraction/square
        y_clip = np.clip(y, -1e6, 1e6)
        rmse = np.sqrt(np.mean((y_clip - predictions) ** 2))
        complexity = expression.complexity()
        size = expression.size()

        # Calculate residuals for subtree analysis
        residuals = y_clip.flatten() - predictions.flatten()
        subtree_qualities = calculate_subtree_qualities(expression, X, residuals)

        # Overall quality score (weighted combination)
        parsimony_score = 1.0 / (1.0 + complexity * 0.01)
        quality_score = r2 * 0.7 + parsimony_score * 0.3

        if return_detailed:
            return {
                'quality_score': quality_score,
                'r2': r2,
                'rmse': rmse,
                'complexity': complexity,
                'size': size,
                'parsimony_score': parsimony_score,
                'subtree_qualities': subtree_qualities,
                'mean_subtree_quality': np.mean(list(subtree_qualities.values())) if subtree_qualities else 0.0
            }
        else:
            return quality_score

    except Exception:
        if return_detailed:
            return {
                'quality_score': 0.0,
                'r2': -10.0,
                'rmse': float('inf'),
                'complexity': expression.complexity(),
                'size': expression.size(),
                'parsimony_score': 0.0,
                'subtree_qualities': {},
                'mean_subtree_quality': 0.0
            }
        else:
            return 0.0


# Mathematical Utilities
def safe_division(a: np.ndarray, b: np.ndarray, default_value: float = 1.0) -> np.ndarray:
    """Safe division that handles division by zero"""
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.divide(a, b)
        result = np.where(np.isfinite(result), result, default_value)
    return result


def safe_log(x: np.ndarray, offset: float = 1e-12) -> np.ndarray:
    """Safe logarithm that handles non-positive values"""
    return np.log(np.maximum(x, offset))


def safe_sqrt(x: np.ndarray) -> np.ndarray:
    """Safe square root that handles negative values"""
    return np.sqrt(np.maximum(x, 0.0))


def safe_exp(x: np.ndarray, max_exp: float = 700.0) -> np.ndarray:
    """Safe exponential that prevents overflow"""
    return np.exp(np.clip(x, -max_exp, max_exp))


def safe_power(base: np.ndarray, exponent: np.ndarray, max_exp: float = 100.0) -> np.ndarray:
    """Safe power operation that prevents overflow/underflow"""
    # Clip exponent to prevent extreme values
    clipped_exp = np.clip(exponent, -max_exp, max_exp)
    
    # Handle negative bases with non-integer exponents
    with np.errstate(invalid='ignore'):
        result = np.power(np.abs(base), clipped_exp)
        # Preserve sign for odd integer exponents
        result = np.where((base < 0) & (clipped_exp % 2 == 1), -result, result)
        result = np.where(np.isfinite(result), result, 1.0)
    
    return result


# Validation Utilities
def validate_data(X: np.ndarray, y: np.ndarray) -> Tuple[bool, str]:
    """Validate input data for symbolic regression"""
    try:
        # Check if arrays are numeric
        if not np.issubdtype(X.dtype, np.number) or not np.issubdtype(y.dtype, np.number):
            return False, "Data must be numeric"
        
        # Check for NaN or infinite values
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            return False, "Input data contains NaN or infinite values"
        
        if np.any(np.isnan(y)) or np.any(np.isinf(y)):
            return False, "Target data contains NaN or infinite values"
        
        # Check dimensions
        if X.ndim != 2:
            return False, "Input data must be 2-dimensional"
        
        if y.ndim not in [1, 2]:
            return False, "Target data must be 1 or 2-dimensional"
        
        # Check sample size consistency
        if X.shape[0] != y.shape[0]:
            return False, "Number of samples in X and y must match"
        
        # Check for sufficient data
        if X.shape[0] < 2:
            return False, "At least 2 samples required"
        
        return True, "Data validation passed"
        
    except Exception as e:
        return False, f"Data validation error: {str(e)}"


def normalize_fitness_scores(scores: List[float]) -> List[float]:
    """Normalize fitness scores to [0, 1] range"""
    if not scores:
        return []
    
    scores_array = np.array(scores)
    min_score = np.min(scores_array)
    max_score = np.max(scores_array)
    
    if max_score == min_score:
        return [1.0] * len(scores)
    
    normalized = (scores_array - min_score) / (max_score - min_score)
    return normalized.tolist()


def calculate_expression_metrics(expressions: List[Expression]) -> Dict[str, Any]:
    """Calculate various metrics for a collection of expressions"""
    if not expressions:
        return {}
    
    complexities = [expr.complexity() for expr in expressions]
    sizes = [expr.size() for expr in expressions]
    strings = [expr.to_string() for expr in expressions]
    
    return {
        'count': len(expressions),
        'mean_complexity': np.mean(complexities),
        'std_complexity': np.std(complexities),
        'min_complexity': np.min(complexities),
        'max_complexity': np.max(complexities),
        'mean_size': np.mean(sizes),
        'std_size': np.std(sizes),
        'min_size': np.min(sizes),
        'max_size': np.max(sizes),
        'unique_expressions': len(set(strings)),
        'diversity_ratio': len(set(strings)) / len(expressions)
    }
