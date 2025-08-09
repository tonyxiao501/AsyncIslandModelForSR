"""
Population Management for MIMO Symbolic Regression
This module consolidates population generation, diversity management, and the Great Powers mechanism.
"""
from typing import List, Dict, Set, Tuple, Optional, Callable, Any
from collections import defaultdict
import numpy as np
import random

from .expression_tree import Expression
from .expression_tree.core.node import BinaryOpNode, UnaryOpNode, VariableNode, ConstantNode
from .generator import ExpressionGenerator
from .utilities import calculate_expression_uniqueness, string_similarity


class PopulationManager:
    """Optimized population management with caching and efficient operations - EXACT COPY from working version"""
    
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
            # Compute magnitude first to avoid overflow in std for huge values
            try:
                max_abs = float(np.max(np.abs(result))) if is_finite else float('inf')
            except Exception:
                max_abs = float('inf')
            is_reasonable = max_abs < 1e8  # Reduced from 1e10

            # Only compute std if values are finite and within a reasonable magnitude
            if is_finite and is_reasonable:
                # Clip to a safe range to avoid overflow in variance computation
                res_clip = np.clip(result, -1e6, 1e6)
                with np.errstate(over='ignore', invalid='ignore'):
                    has_variation = np.std(res_clip) > 1e-10  # Ensure some variation in output
            else:
                has_variation = False

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
            return {'string_diversity': 1.0, 'complexity_diversity': 1.0, 'structural_diversity': 1.0, 'overall': 1.0}
        
        # Get cached strings and complexities
        strings = [expr.to_string() for expr in population]
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
        
        # Structural diversity (simplified)
        structural_diversity = string_diversity  # Use string diversity as proxy for structural
        
        overall_diversity = (string_diversity * 0.5 + complexity_diversity * 0.3 + size_diversity * 0.2)
        
        return {
            'string_diversity': string_diversity,
            'complexity_diversity': complexity_diversity,
            'size_diversity': size_diversity,
            'structural_diversity': structural_diversity,
            'overall': overall_diversity
        }


def generate_diverse_population_optimized(generator: ExpressionGenerator, 
                                        population_size: int, 
                                        n_inputs: int,
                                        pop_manager: PopulationManager) -> List[Expression]:
    """Optimized diverse population generation with batching - EXACT COPY from working version"""
    population = []
    max_depth = pop_manager.max_depth
    
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
                             console_log: bool = True,
                             protected_indices: Optional[List[int]] = None) -> List[Expression]:
    """Optimized diversity injection with elite protection and single sort and batch operations"""
    
    # Adaptive injection rate - more conservative
    if stagnation_counter > 20:
        injection_rate = min(0.4, injection_rate * 1.5)
    elif stagnation_counter > 12:
        injection_rate = min(0.3, injection_rate * 1.2)
    
    n_to_replace = max(1, int(len(population) * injection_rate))
    
    # Single sort operation
    sorted_indices = np.argsort(fitness_scores)
    
    # Protect elites and Great Powers from replacement
    if protected_indices is None:
        # Default protection: top 10% of population
        elite_count = max(1, int(0.1 * len(population)))
        protected_indices = sorted_indices[-elite_count:].tolist()
    
    # Ensure protected_indices is not None for type checker
    assert protected_indices is not None
    protected_set = set(protected_indices)
    
    # Pre-calculate index ranges - focus on worst performers only, but exclude protected
    available_indices = [idx for idx in sorted_indices if idx not in protected_set]
    
    # Focus on bottom performers among available (non-protected) indices
    worst_count = max(1, min(n_to_replace, len(available_indices) // 2))  # Worst 50% of available
    bottom_half_count = len(available_indices) // 2
    
    # Batch replacement strategy - more conservative, exclude protected
    replacement_indices = []
    
    # Prioritize worst performers (excluding protected)
    replacement_indices.extend(available_indices[:worst_count])
    
    # Only add random from bottom half if still needed
    if len(replacement_indices) < n_to_replace:
        remaining_needed = n_to_replace - len(replacement_indices)
        bottom_indices = available_indices[:bottom_half_count]
        available_bottom = [idx for idx in bottom_indices if idx not in replacement_indices]
        
        if available_bottom:
            random_count = min(remaining_needed, len(available_bottom))
            replacement_indices.extend(np.random.choice(available_bottom, size=random_count, replace=False))
    
    # Batch generate replacements
    new_population = population.copy()
    replacements_made = 0
    
    # Generate diverse expressions in batches
    for idx in replacement_indices[:n_to_replace]:
        try:
            # Use various strategies for diverse replacement
            strategy = np.random.choice(['high_diversity', 'targeted_diverse', 'complex_diverse', 'polynomial'])
            
            if strategy == 'high_diversity':
                new_expr = generate_high_diversity_expression_optimized(generator, pop_manager.n_inputs)
            elif strategy == 'targeted_diverse':
                new_expr = generate_targeted_diverse_expression_optimized(generator, pop_manager.n_inputs)
            elif strategy == 'complex_diverse':
                new_expr = generate_complex_diverse_expression_optimized(generator, pop_manager.n_inputs)
            else:  # polynomial
                new_expr = generate_polynomial_expression_optimized(generator)
            
            if pop_manager.is_expression_valid_cached(new_expr):
                new_population[idx] = new_expr
                replacements_made += 1
            
        except Exception:
            # Fallback to simple random generation
            try:
                fallback_expr = Expression(generator.generate_random_expression())
                if pop_manager.is_expression_valid_cached(fallback_expr):
                    new_population[idx] = fallback_expr
                    replacements_made += 1
            except:
                continue
    
    if console_log and replacements_made > 0:
        print(f"    Diversity injection: replaced {replacements_made} expressions (rate: {injection_rate:.2f})")
    
    return new_population


# Helper functions for expression generation - EXACT COPIES from working version
def generate_simple_combination_optimized(generator: ExpressionGenerator, n_inputs: int) -> Expression:
    """Generate simple combination expressions optimized for performance"""
    strategy = np.random.choice(['linear', 'product', 'ratio'])
    
    if strategy == 'linear':
        # Simple linear combination: a*x0 + b*x1 + c
        var1 = VariableNode(np.random.randint(0, n_inputs))
        var2 = VariableNode(np.random.randint(0, n_inputs)) if n_inputs > 1 else var1
        const1 = ConstantNode(np.random.uniform(-2, 2))
        const2 = ConstantNode(np.random.uniform(-2, 2))
        const3 = ConstantNode(np.random.uniform(-1, 1))
        
        term1 = BinaryOpNode('*', const1, var1)
        term2 = BinaryOpNode('*', const2, var2)
        sum_terms = BinaryOpNode('+', term1, term2)
        result = BinaryOpNode('+', sum_terms, const3)
        
    elif strategy == 'product':
        # Simple product: a * x0 * x1
        var1 = VariableNode(np.random.randint(0, n_inputs))
        var2 = VariableNode(np.random.randint(0, n_inputs)) if n_inputs > 1 else var1
        const = ConstantNode(np.random.uniform(-2, 2))
        
        product = BinaryOpNode('*', var1, var2)
        result = BinaryOpNode('*', const, product)
        
    else:  # ratio
        # Simple ratio: x0 / (x1 + c)
        var1 = VariableNode(np.random.randint(0, n_inputs))
        var2 = VariableNode(np.random.randint(0, n_inputs)) if n_inputs > 1 else var1
        const = ConstantNode(np.random.uniform(0.5, 2.0))  # Avoid division by zero
        
        denominator = BinaryOpNode('+', var2, const)
        result = BinaryOpNode('/', var1, denominator)
    
    return Expression(result)


def generate_high_diversity_expression_optimized(generator: ExpressionGenerator, n_inputs: int) -> Expression:
    """Generate high diversity expressions optimized for performance"""
    ops = ['+', '-', '*', '/', 'sin', 'cos', 'exp', 'log']
    op = np.random.choice(ops)
    
    if op in ['sin', 'cos', 'exp', 'log']:
        # Unary operation
        var = VariableNode(np.random.randint(0, n_inputs))
        const = ConstantNode(np.random.uniform(-1, 1))
        inner = BinaryOpNode('*', const, var)
        result = UnaryOpNode(op, inner)
    else:
        # Binary operation
        var1 = VariableNode(np.random.randint(0, n_inputs))
        var2 = VariableNode(np.random.randint(0, n_inputs))
        result = BinaryOpNode(op, var1, var2)
    
    return Expression(result)


def generate_targeted_diverse_expression_optimized(generator: ExpressionGenerator, n_inputs: int) -> Expression:
    """Generate targeted diverse expressions optimized for performance"""
    # Create expressions with specific patterns
    pattern = np.random.choice(['polynomial', 'trigonometric', 'exponential'])
    
    if pattern == 'polynomial':
        # x^2 + a*x + b
        var = VariableNode(np.random.randint(0, n_inputs))
        const1 = ConstantNode(np.random.uniform(-2, 2))
        const2 = ConstantNode(np.random.uniform(-1, 1))
        
        square = BinaryOpNode('^', var, ConstantNode(2.0))
        linear = BinaryOpNode('*', const1, var)
        sum1 = BinaryOpNode('+', square, linear)
        result = BinaryOpNode('+', sum1, const2)
        
    elif pattern == 'trigonometric':
        # sin(a*x) + cos(b*x)
        var = VariableNode(np.random.randint(0, n_inputs))
        const1 = ConstantNode(np.random.uniform(-3, 3))
        const2 = ConstantNode(np.random.uniform(-3, 3))
        
        inner1 = BinaryOpNode('*', const1, var)
        inner2 = BinaryOpNode('*', const2, var)
        sin_term = UnaryOpNode('sin', inner1)
        cos_term = UnaryOpNode('cos', inner2)
        result = BinaryOpNode('+', sin_term, cos_term)
        
    else:  # exponential
        # exp(a*x) or log(|x| + 1)
        var = VariableNode(np.random.randint(0, n_inputs))
        const = ConstantNode(np.random.uniform(-1, 1))
        
        if np.random.random() < 0.5:
            inner = BinaryOpNode('*', const, var)
            result = UnaryOpNode('exp', inner)
        else:
            abs_x = UnaryOpNode('abs', var)
            inner = BinaryOpNode('+', abs_x, ConstantNode(1.0))
            result = UnaryOpNode('log', inner)
    
    return Expression(result)


def generate_complex_diverse_expression_optimized(generator: ExpressionGenerator, n_inputs: int) -> Expression:
    """Generate complex diverse expressions optimized for performance"""
    return Expression(generator.generate_random_expression(max_depth=4))


def generate_polynomial_expression_optimized(generator: ExpressionGenerator) -> Expression:
    """Generate polynomial expressions optimized for performance"""
    var = VariableNode(0)  # Use first variable
    const1 = ConstantNode(np.random.uniform(-2, 2))
    const2 = ConstantNode(np.random.uniform(-2, 2))
    const3 = ConstantNode(np.random.uniform(-1, 1))
    
    # a*x^2 + b*x + c
    square = BinaryOpNode('^', var, ConstantNode(2.0))
    term1 = BinaryOpNode('*', const1, square)
    term2 = BinaryOpNode('*', const2, var)
    sum1 = BinaryOpNode('+', term1, term2)
    result = BinaryOpNode('+', sum1, const3)
    
    return Expression(result)


def generate_mixed_expression_optimized(generator: ExpressionGenerator) -> Expression:
    """Generate mixed expressions optimized for performance"""
    return Expression(generator.generate_random_expression(max_depth=3))


def generate_constant_heavy_optimized(generator: ExpressionGenerator, n_inputs: int) -> Expression:
    """Generate constant-heavy expressions optimized for performance"""
    var = VariableNode(np.random.randint(0, n_inputs))
    const1 = ConstantNode(np.random.uniform(-5, 5))
    const2 = ConstantNode(np.random.uniform(-2, 2))
    
    # c1 * x + c2
    term = BinaryOpNode('*', const1, var)
    result = BinaryOpNode('+', term, const2)
    
    return Expression(result)


class GreatPowers:
    """
    Maintains the top 5 expressions (Great Powers) dynamically across generations.
    These expressions are preserved from diversity injection and population restarts.
    Includes advanced redundancy elimination to prevent duplicate expressions.
    EXACT COPY from working version
    """
    
    def __init__(self, max_powers: int = 5, similarity_threshold: float = 0.98):
        self.max_powers = max_powers
        self.powers: List[Dict] = []  # List of {'expression': Expression, 'fitness': float, 'generation': int}
        self.generation_updates = 0
        self.similarity_threshold = similarity_threshold  # Threshold for considering expressions similar
        self.redundancy_stats = {
            'rejected_duplicates': 0,
            'semantic_rejections': 0,
            'structural_rejections': 0,
            'total_rejections': 0
        }
        
    def __len__(self) -> int:
        """Return the number of Great Powers"""
        return len(self.powers)
        
    def _is_expression_redundant(self, candidate_expr: Expression, X: Optional[np.ndarray] = None) -> Tuple[bool, str]:
        """
        Check if candidate expression is redundant with existing Great Powers.
        
        Args:
            candidate_expr: Expression to check for redundancy
            X: Optional data for semantic comparison
            
        Returns:
            Tuple of (is_redundant: bool, reason: str)
        """
        candidate_str = candidate_expr.to_string()
        
        for i, power in enumerate(self.powers):
            existing_expr = power['expression']
            existing_str = existing_expr.to_string()
            
            # 1. Exact string match (fastest check)
            if candidate_str == existing_str:
                return True, f"exact_duplicate_with_power_{i+1}"
            
            # 2. String similarity check
            str_similarity = string_similarity(candidate_str, existing_str)
            if str_similarity >= self.similarity_threshold:
                return True, f"string_similarity_{str_similarity:.3f}_with_power_{i+1}"
            
            # 3. Structural similarity check
            structural_similarity = self._calculate_structural_similarity(candidate_expr, existing_expr)
            if structural_similarity >= self.similarity_threshold:
                return True, f"structural_similarity_{structural_similarity:.3f}_with_power_{i+1}"
        
        return False, "not_redundant"
    
    def _calculate_structural_similarity(self, expr1: Expression, expr2: Expression) -> float:
        """Calculate structural similarity between two expressions"""
        try:
            # Compare complexity
            comp1, comp2 = expr1.complexity(), expr2.complexity()
            if comp1 == 0 and comp2 == 0:
                return 1.0
            
            complexity_similarity = 1.0 - abs(comp1 - comp2) / max(comp1, comp2, 1)
            
            # Compare size (number of nodes)
            size1, size2 = expr1.size(), expr2.size()
            if size1 == 0 and size2 == 0:
                return 1.0
                
            size_similarity = 1.0 - abs(size1 - size2) / max(size1, size2, 1)
            
            # Get operator signatures
            ops1 = self._get_operator_signature(expr1)
            ops2 = self._get_operator_signature(expr2)
            
            # Calculate Jaccard similarity for operators
            all_ops = set(ops1.keys()) | set(ops2.keys())
            if not all_ops:
                operator_similarity = 1.0
            else:
                intersection = sum(min(ops1.get(op, 0), ops2.get(op, 0)) for op in all_ops)
                union = sum(max(ops1.get(op, 0), ops2.get(op, 0)) for op in all_ops)
                operator_similarity = intersection / union if union > 0 else 0.0
            
            # Weighted combination
            return (0.3 * complexity_similarity + 0.3 * size_similarity + 0.4 * operator_similarity)
            
        except Exception:
            return 0.0
    
    def _get_operator_signature(self, expr: Expression) -> Dict[str, int]:
        """Get operator signature (count of each operator type)"""
        # This is a simplified implementation
        # In a real implementation, you'd traverse the expression tree
        expr_str = expr.to_string()
        operators = {'+': 0, '-': 0, '*': 0, '/': 0, '^': 0, 'sin': 0, 'cos': 0, 'exp': 0, 'log': 0}
        
        for op in operators:
            operators[op] = expr_str.count(op)
        
        return operators
    
    def update_powers(self, population: List[Expression], fitness_scores: List[float], generation: int):
        """Update Great Powers with new candidates from current generation"""
        if not population or not fitness_scores:
            return
            
        # Find the best expression in current generation
        best_idx = np.argmax(fitness_scores)
        best_expr = population[best_idx]
        best_fitness = fitness_scores[best_idx]
        
        # Check if this expression should be added to Great Powers
        self._add_candidate(best_expr, best_fitness, generation)
        
        # Also check top 3 expressions for potential additions
        top_indices = np.argsort(fitness_scores)[-3:]
        for idx in top_indices:
            if idx != best_idx:  # Skip the best one we already processed
                self._add_candidate(population[idx], fitness_scores[idx], generation)
    
    def _add_candidate(self, candidate_expr: Expression, fitness: float, generation: int):
        """Add a candidate expression to Great Powers if it qualifies"""
        # Quick fitness check - only consider high-fitness expressions
        if len(self.powers) == self.max_powers:
            min_fitness = min(power['fitness'] for power in self.powers)
            if fitness <= min_fitness:
                return  # Not good enough
        
        # Check for redundancy
        is_redundant, reason = self._is_expression_redundant(candidate_expr)
        if is_redundant:
            self.redundancy_stats['total_rejections'] += 1
            if 'exact_duplicate' in reason:
                self.redundancy_stats['rejected_duplicates'] += 1
            elif 'semantic' in reason:
                self.redundancy_stats['semantic_rejections'] += 1
            elif 'structural' in reason:
                self.redundancy_stats['structural_rejections'] += 1
            return
        
        # Add to Great Powers
        new_power = {
            'expression': candidate_expr.copy(),
            'fitness': fitness,
            'generation': generation
        }
        
        self.powers.append(new_power)
        self.generation_updates += 1
        
        # Sort by fitness (descending) and keep only top max_powers
        self.powers.sort(key=lambda x: x['fitness'], reverse=True)
        if len(self.powers) > self.max_powers:
            self.powers = self.powers[:self.max_powers]
    
    def get_best_fitness(self) -> float:
        """Get the fitness of the best Great Power"""
        if not self.powers:
            return -float('inf')
        return self.powers[0]['fitness']
    
    def inject_powers_into_population(self, population: List[Expression], 
                                    fitness_scores: List[float], 
                                    injection_rate: float = 0.1) -> Tuple[List[Expression], List[float]]:
        """Inject Great Powers into population, replacing worst performers"""
        if not self.powers or injection_rate <= 0:
            return population, fitness_scores
            
        n_to_inject = min(len(self.powers), max(1, int(len(population) * injection_rate)))
        
        # Find worst performers to replace
        worst_indices = np.argsort(fitness_scores)[:n_to_inject]
        
        # Replace with Great Powers
        new_population = population.copy()
        new_fitness_scores = fitness_scores.copy()
        
        for i, idx in enumerate(worst_indices):
            if i < len(self.powers):
                new_population[idx] = self.powers[i]['expression'].copy()
                new_fitness_scores[idx] = self.powers[i]['fitness']
        
        return new_population, new_fitness_scores
    
    def protect_elites_from_injection(self, population: List[Expression], 
                                    fitness_scores: List[float], 
                                    elite_fraction: float = 0.1) -> List[int]:
        """Get indices of expressions that should be protected from diversity injection"""
        # Protect top performers
        elite_count = max(1, int(elite_fraction * len(population)))
        elite_indices = np.argsort(fitness_scores)[-elite_count:].tolist()
        
        # Also protect any Great Powers currently in population
        great_power_strings = {power['expression'].to_string() for power in self.powers}
        for i, expr in enumerate(population):
            if expr.to_string() in great_power_strings:
                if i not in elite_indices:
                    elite_indices.append(i)
        
        return elite_indices
    
    def diagnose_fitness_drop(self, current_best_fitness: float, generation: int) -> Dict[str, Any]:
        """Diagnose potential fitness drops compared to Great Powers"""
        if not self.powers:
            return {"status": "no_great_powers"}
        
        best_great_power_fitness = self.get_best_fitness()
        fitness_gap = best_great_power_fitness - current_best_fitness
        
        if fitness_gap > 0.01:  # Significant fitness drop
            latest_generation = max(power['generation'] for power in self.powers)
            return {
                "status": "fitness_drop_detected",
                "fitness_gap": fitness_gap,
                "current_best": current_best_fitness,
                "great_power_best": best_great_power_fitness,
                "latest_great_power_generation": latest_generation,
                "generations_since_update": generation - latest_generation
            }
        
        return {"status": "no_fitness_drop"}
    
    def get_redundancy_stats(self) -> Dict[str, int]:
        """Get redundancy rejection statistics"""
        return self.redundancy_stats.copy()
