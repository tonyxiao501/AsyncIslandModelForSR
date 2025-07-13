# population.py: population/diversity management helpers
from typing import List, Dict, Set, Tuple, Optional, Callable, Any
from collections import defaultdict
import numpy as np
import random

from symbolic_regression import Expression
from symbolic_regression.expression_tree import Expression
from symbolic_regression.expression_tree.core.node import BinaryOpNode, UnaryOpNode, VariableNode, ConstantNode
from symbolic_regression.generator import ExpressionGenerator
from utils import calculate_expression_uniqueness


class PopulationManager:
    """Optimized population management with caching and efficient operations"""
    
    def __init__(self, n_inputs: int, max_depth: int = 6):
        self.n_inputs = n_inputs
        self.max_depth = max_depth
        
        # Reduced cache for minimal memory overhead in multiprocessing
        self._validation_cache: Dict[str, bool] = {}
        self._complexity_cache: Dict[int, float] = {}
        
        # Pre-computed test data for validation
        # Use a fixed seed to ensure reproducible validation across processes
        rng = np.random.RandomState(42)
        self._test_X = rng.randn(5, n_inputs)  # Reduced test size for speed
        
        # Cache size limits to prevent memory bloat
        self._max_cache_size = 1000
        
    def clear_caches(self):
        """Clear all caches - call when population changes significantly"""
        self._validation_cache.clear()
        self._complexity_cache.clear()
        
    def get_expression_complexity(self, expr: Expression) -> float:
        """Cached complexity calculation with size limit"""
        expr_id = id(expr)
        if expr_id not in self._complexity_cache:
            if len(self._complexity_cache) > self._max_cache_size:
                # Clear oldest half of cache when full
                items = list(self._complexity_cache.items())
                self._complexity_cache = dict(items[len(items)//2:])
            self._complexity_cache[expr_id] = expr.complexity()
        return self._complexity_cache[expr_id]
    
    def is_expression_valid_cached(self, expr: Expression) -> bool:
        """Optimized validation with improved stability checking"""
        # Quick complexity check first without caching string
        try:
            complexity = expr.complexity()
            if complexity > 25:  # Reduced from 30 to be more selective
                return False
        except:
            return False
        
        # Use minimal string representation for cache key
        expr_str = str(hash(expr.to_string()))
        
        if expr_str in self._validation_cache:
            return self._validation_cache[expr_str]
        
        # Limit cache size
        if len(self._validation_cache) > self._max_cache_size:
            self._validation_cache.clear()
            # Continue with validation since cache was cleared
        
        try:
            # Use pre-computed test data for validation
            result = expr.evaluate(self._test_X)
            
            # Check for numerical issues
            if result is None or len(result) == 0:
                self._validation_cache[expr_str] = False
                return False
                
            # Convert to numpy array if needed
            if not isinstance(result, np.ndarray):
                result = np.array(result)
            
            # More strict validation for finite values and reasonable magnitude
            is_finite = np.all(np.isfinite(result))
            is_reasonable = np.max(np.abs(result)) < 1e8  # Reduced from 1e10
            has_variation = np.std(result) > 1e-10  # Ensure some variation in output
            
            is_valid = bool(is_finite and is_reasonable and has_variation)
            self._validation_cache[expr_str] = is_valid
            return is_valid
            
        except Exception as e:
            # Less permissive validation - only fallback for simple cases
            try:
                # Try with multiple test points for better validation
                test_points = np.array([[1.0] * self.n_inputs, [0.0] * self.n_inputs, [-1.0] * self.n_inputs])
                simple_result = expr.evaluate(test_points)
                
                if (simple_result is not None and 
                    np.isfinite(simple_result).all() and 
                    np.max(np.abs(simple_result)) < 1e6):
                    self._validation_cache[expr_str] = True
                    return True
                else:
                    self._validation_cache[expr_str] = False
                    return False
                    
            except Exception:
                # If validation fails completely, mark as invalid
                self._validation_cache[expr_str] = False
                return False
    
    def calculate_population_diversity_optimized(self, population: List[Expression]) -> Dict[str, float]:
        """Optimized diversity calculation with caching"""
        if len(population) < 2:
            return {'string_diversity': 1.0, 'complexity_diversity': 1.0, 'overall': 1.0}
        
        # Get cached strings and complexities
        strings = [self.get_expression_string(expr) for expr in population]
        complexities = [self.get_expression_complexity(expr) for expr in population]
        
        # String diversity
        unique_strings = len(set(strings))
        string_diversity = unique_strings / len(population)
        
        # Complexity diversity
        complexity_std = np.std(complexities)
        complexity_mean = np.mean(complexities)
        complexity_diversity = min(1.0, float(complexity_std / (complexity_mean + 1e-6)))
        
        # Size diversity (using complexity as proxy)
        size_diversity = len(set(complexities)) / len(population)
        
        overall_diversity = (string_diversity * 0.5 + complexity_diversity * 0.3 + size_diversity * 0.2)
        
        return {
            'string_diversity': string_diversity,
            'complexity_diversity': complexity_diversity,
            'size_diversity': size_diversity,
            'overall': overall_diversity
        }

    def get_expression_string(self, expr: Expression) -> str:
        """Simple string conversion without caching to reduce overhead"""
        return expr.to_string()

def generate_diverse_population_optimized(generator: ExpressionGenerator, 
                                        n_inputs: int, 
                                        population_size: int, 
                                        max_depth: int, 
                                        pop_manager: PopulationManager) -> List[Expression]:
    """Optimized diverse population generation with batching"""
    population = []
    
    # Strategy distribution
    strategies = {
        'random_full': 0.4,
        'random_grow': 0.3,
        'simple_combinations': 0.2,
        'constants_varied': 0.1
    }
    
    # Generate in batches to reduce validation overhead
    batch_size = 20
    target_counts = {strategy: int(ratio * population_size) 
                    for strategy, ratio in strategies.items()}
    
    for strategy, target_count in target_counts.items():
        current_count = 0
        attempts = 0
        max_attempts = target_count * 3
        
        while current_count < target_count and attempts < max_attempts:
            batch = []
            
            # Generate batch
            for _ in range(min(batch_size, target_count - current_count)):
                try:
                    if strategy == 'random_full':
                        depth = random.randint(2, max_depth)
                        expr = Expression(generator.generate_random_expression(max_depth=depth))
                    elif strategy == 'random_grow':
                        depth = random.randint(1, max_depth - 1)
                        expr = Expression(generator.generate_random_expression(max_depth=depth))
                    elif strategy == 'simple_combinations':
                        expr = generate_simple_combination_optimized(generator, n_inputs)
                    else:  # constants_varied
                        expr = generate_constant_heavy_optimized(generator, n_inputs)
                    
                    batch.append(expr)
                except Exception:
                    continue
            
            # Validate batch
            valid_expressions = [expr for expr in batch if pop_manager.is_expression_valid_cached(expr)]
            population.extend(valid_expressions)
            current_count += len(valid_expressions)
            attempts += len(batch)
    
    # Fill remaining slots efficiently with safety counter
    remaining_attempts = 0
    max_attempts_per_slot = 50  # Reduced from 100 to 50 for faster failure
    max_total_attempts = min(population_size * max_attempts_per_slot, 5000)  # Cap total attempts
    
    initial_population_size = len(population)
    
    while len(population) < population_size and remaining_attempts < max_total_attempts:
        try:
            expr = Expression(generator.generate_random_expression())
            if pop_manager.is_expression_valid_cached(expr):
                population.append(expr)
            remaining_attempts += 1
            
            # Early exit if we're making very slow progress
            if remaining_attempts > 1000 and len(population) - initial_population_size < 5:
                print(f"Warning: Slow population generation, breaking early after {remaining_attempts} attempts")
                break
                
        except Exception:
            remaining_attempts += 1
            continue
    
    # Debug: print if we had trouble filling the population
    if len(population) < population_size:
        print(f"Warning: Could only generate {len(population)} valid expressions out of {population_size} requested after {remaining_attempts} attempts")
    
    # If we couldn't fill the population, pad with the best expressions we have
    fill_attempts = 0
    while len(population) < population_size and fill_attempts < 100:  # Prevent infinite loop here too
        if population:
            # Duplicate a random existing expression
            population.append(population[np.random.randint(0, len(population))].copy())
        else:
            # Create a very simple expression as fallback
            from symbolic_regression.expression_tree.core.node import VariableNode
            simple_expr = Expression(VariableNode(0))
            population.append(simple_expr)
        fill_attempts += 1
    
    return population[:population_size]

def inject_diversity_optimized(population: List[Expression], 
                             fitness_scores: List[float], 
                             generator: ExpressionGenerator,
                             injection_rate: float,
                             pop_manager: PopulationManager,
                             stagnation_counter: int,
                             console_log: bool = True) -> List[Expression]:
    """Optimized diversity injection with single sort and batch operations"""
    
    # Adaptive injection rate - more conservative
    if stagnation_counter > 20:
        injection_rate = min(0.4, injection_rate * 1.5)
    elif stagnation_counter > 12:
        injection_rate = min(0.3, injection_rate * 1.2)
    
    n_to_replace = max(1, int(len(population) * injection_rate))
    
    # Single sort operation
    sorted_indices = np.argsort(fitness_scores)
    
    # Pre-calculate index ranges - focus on worst performers only
    worst_count = max(1, min(n_to_replace, len(population) // 4))  # Only worst 25%
    bottom_quarter_count = len(population) // 4
    
    # Batch replacement strategy - more conservative
    replacement_indices = []
    
    # Prioritize worst performers
    replacement_indices.extend(sorted_indices[:worst_count])
    
    # Only add random from bottom quarter if still needed
    if len(replacement_indices) < n_to_replace:
        remaining_needed = n_to_replace - len(replacement_indices)
        bottom_indices = sorted_indices[:bottom_quarter_count]
        available_bottom = [idx for idx in bottom_indices if idx not in replacement_indices]
        
        if available_bottom:
            random_count = min(remaining_needed, len(available_bottom))
            replacement_indices.extend(np.random.choice(available_bottom, size=random_count, replace=False))
    
    # Batch generate replacements
    new_population = population.copy()
    replacements_made = 0
    
    # Generate diverse expressions in batches
    diversity_generators = [
        lambda: generate_high_diversity_expression_optimized(generator, n_inputs=pop_manager.n_inputs),
        lambda: generate_targeted_diverse_expression_optimized(generator, population, pop_manager),
        lambda: generate_complex_diverse_expression_optimized(generator, n_inputs=pop_manager.n_inputs)
    ]
    
    for i, idx in enumerate(replacement_indices):
        try:
            # Cycle through different diversity generators
            gen_func = diversity_generators[i % len(diversity_generators)]
            new_expr = gen_func()
            
            if pop_manager.is_expression_valid_cached(new_expr):
                new_population[idx] = new_expr
                replacements_made += 1
            else:
                # Fallback: create a simple expression if validation fails
                from symbolic_regression.expression_tree.core.node import VariableNode
                fallback_expr = Expression(VariableNode(np.random.randint(0, pop_manager.n_inputs)))
                new_population[idx] = fallback_expr
                replacements_made += 1
            
        except Exception as e:
            # Fallback: keep the original expression if generation fails
            continue
    
    if console_log:
        print(f"Diversity injection: replaced {replacements_made}/{n_to_replace} individuals "
              f"(rate: {injection_rate:.2f}, stagnation: {stagnation_counter})")
    
    return new_population

def evaluate_population_enhanced_optimized(population: List[Expression],
                                         X: np.ndarray, 
                                         y: np.ndarray, 
                                         parsimony_coefficient: float,
                                         pop_manager: PopulationManager) -> List[float]:
    """Optimized fitness evaluation with improved convergence"""
    fitness_scores = []
    
    # Pre-calculate population diversity metrics once - reduce diversity bonus impact
    diversity_metrics = pop_manager.calculate_population_diversity_optimized(population)
    base_diversity_bonus = diversity_metrics['overall'] * 0.0005  # Reduced from 0.001
    
    # Batch evaluation
    for i, expr in enumerate(population):
        try:
            predictions = expr.evaluate(X)
            if predictions.ndim == 1:
                predictions = predictions.reshape(-1, 1)
            
            # Core fitness calculation
            mse = np.mean((y - predictions) ** 2)
            
            # Cached complexity - reduced penalty for moderate complexity
            complexity = pop_manager.get_expression_complexity(expr)
            complexity_penalty = parsimony_coefficient * complexity
            
            # Stability penalty (vectorized) - more aggressive for unstable solutions
            stability_penalty = 0.0
            max_abs_pred = np.max(np.abs(predictions))
            
            if max_abs_pred > 1e8:  # More aggressive for very large values
                stability_penalty = 2.0
            elif max_abs_pred > 1e6:
                stability_penalty = 1.0
            elif max_abs_pred > 1e4:
                stability_penalty = 0.2
            
            if np.any(~np.isfinite(predictions)):
                stability_penalty += 2.0  # Heavily penalize infinite/NaN
            
            # Reduced diversity bonus to focus more on fitness
            fitness = -mse - complexity_penalty - stability_penalty + base_diversity_bonus
            fitness_scores.append(float(fitness))
            
        except Exception:
            fitness_scores.append(-1e8)
    
    return fitness_scores

# Optimized helper functions
def generate_simple_combination_optimized(generator: ExpressionGenerator, n_inputs: int) -> Expression:
    """Optimized simple combination generation"""
    if n_inputs <= 0:
        raise ValueError("n_inputs must be greater than 0")
    
    var_node = VariableNode(random.randint(0, n_inputs - 1))
    const_node = ConstantNode(random.uniform(-2, 2))
    
    # Pre-defined combinations for efficiency
    combinations = [
        lambda: BinaryOpNode('+', var_node, const_node),
        lambda: BinaryOpNode('*', var_node, const_node),
        lambda: UnaryOpNode('sin', var_node),
        lambda: UnaryOpNode('cos', var_node),
        lambda: BinaryOpNode('*', var_node, var_node),
    ]
    
    return Expression(random.choice(combinations)())

def generate_high_diversity_expression_optimized(generator: ExpressionGenerator, n_inputs: int) -> Expression:
    """Optimized high diversity expression generation"""
    var = VariableNode(random.randint(0, n_inputs - 1))
    const1 = ConstantNode(np.random.uniform(-2, 2))
    const2 = ConstantNode(np.random.uniform(-2, 2))
    
    # Build expression tree efficiently
    inner_expr = BinaryOpNode('+', BinaryOpNode('*', const1, var), const2)
    
    # Random function selection
    func_choices = ['sin', 'cos', 'exp']
    func = random.choice(func_choices)
    
    if func == 'exp':
        # Limit exponential growth
        limited_inner = BinaryOpNode('*', ConstantNode(0.5), var)
        return Expression(UnaryOpNode(func, limited_inner))
    else:
        return Expression(UnaryOpNode(func, inner_expr))

def generate_targeted_diverse_expression_optimized(generator: ExpressionGenerator, 
                                                 population: List[Expression],
                                                 pop_manager: PopulationManager) -> Expression:
    """Optimized targeted diversity with cached string operations"""
    sample_size = min(20, len(population))
    sample_population = population[:sample_size]
    
    # Use cached strings
    sample_strings = [pop_manager.get_expression_string(expr) for expr in sample_population]
    
    # Efficient pattern detection
    has_trig = any('sin' in s or 'cos' in s for s in sample_strings)
    has_exp = any('exp' in s for s in sample_strings)
    has_poly = any('*' in s and '+' in s for s in sample_strings)
    
    if not has_trig or not has_exp:
        return generate_high_diversity_expression_optimized(generator, pop_manager.n_inputs)
    elif not has_poly:
        return generate_polynomial_expression_optimized(generator)
    else:
        return generate_mixed_expression_optimized(generator)

def generate_complex_diverse_expression_optimized(generator: ExpressionGenerator, n_inputs: int) -> Expression:
    """Optimized complex diversity generation"""
    return generate_high_diversity_expression_optimized(generator, n_inputs)

# Additional optimized helper functions
def generate_polynomial_expression_optimized(generator: ExpressionGenerator) -> Expression:
    """Optimized polynomial generation"""
    var = VariableNode(0)
    coeffs = [ConstantNode(np.random.uniform(-3, 3)) for _ in range(3)]
    
    # Build polynomial efficiently
    x_squared = BinaryOpNode('*', var, var)
    term1 = BinaryOpNode('*', coeffs[0], x_squared)
    term2 = BinaryOpNode('*', coeffs[1], var)
    
    return Expression(BinaryOpNode('+', BinaryOpNode('+', term1, term2), coeffs[2]))

def generate_mixed_expression_optimized(generator: ExpressionGenerator) -> Expression:
    """Optimized mixed expression generation"""
    var = VariableNode(0)
    const = ConstantNode(np.random.uniform(-2, 2))
    
    trig_part = UnaryOpNode('sin', var)
    linear_part = BinaryOpNode('*', const, var)
    
    return Expression(BinaryOpNode('+', trig_part, linear_part))

def generate_constant_heavy_optimized(generator: ExpressionGenerator, n_inputs: int) -> Expression:
    """Optimized constant-heavy generation"""
    if n_inputs <= 0:
        raise ValueError("n_inputs must be greater than 0")
    
    var_node = VariableNode(random.randint(0, n_inputs - 1))
    const1 = ConstantNode(random.uniform(-3, 3))
    const2 = ConstantNode(random.uniform(-3, 3))
    
    op = random.choice(['+', '-', '*'])
    return Expression(BinaryOpNode(op, BinaryOpNode('*', const1, var_node), const2))