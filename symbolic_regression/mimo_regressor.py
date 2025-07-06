import numpy as np
from scipy.constants import golden_ratio, electron_mass, neutron_mass
from scipy.optimize import OptimizeWarning, curve_fit
import warnings
import random
from typing import List, Optional, Dict, Any
import sympy as sp
from .expression_tree import Expression
from .expression_tree.core.node import Node, ConstantNode
from .generator import ExpressionGenerator, BiasedExpressionGenerator
from .genetic_ops import GeneticOperations
from .expression_tree.utils.sympy_utils import SymPySimplifier
import scipy.constants as constants
import multiprocessing
from .population import (
  generate_diverse_population, inject_diversity, is_expression_valid,
  generate_high_diversity_expression, generate_targeted_diverse_expression, generate_complex_diverse_expression, enhanced_reproduction_v2
)
from .utils import string_similarity
from .ensemble_worker import _fit_worker


class MIMOSymbolicRegressor:
  """Enhanced Multiple Input Multiple Output Symbolic Regression Model with improved evolution dynamics"""

  def __init__(self,
               population_size: int = 100,
               generations: int = 50,
               mutation_rate: float = 0.1,
               crossover_rate: float = 0.8,
               tournament_size: int = 3,
               max_depth: int = 6,
               parsimony_coefficient: float = 0.001,
               sympy_simplify: bool = True,
               advanced_simplify: bool = False,
               diversity_threshold: float = 0.7,
               adaptive_rates: bool = True,
               restart_threshold: int = 25,
               elite_fraction: float = 0.1,
               console_log=True
               ):

    self.population_size = population_size
    self.generations = generations
    self.mutation_rate = mutation_rate
    self.crossover_rate = crossover_rate
    self.tournament_size = tournament_size
    self.max_depth = max_depth
    self.parsimony_coefficient = parsimony_coefficient
    self.sympy_simplify = sympy_simplify
    self.advanced_simplify = advanced_simplify

    self.console_log = console_log

    # Enhanced evolution parameters
    self.diversity_threshold = diversity_threshold
    self.adaptive_rates = adaptive_rates
    self.restart_threshold = restart_threshold
    self.elite_fraction = elite_fraction

    # Evolution state tracking
    self.stagnation_counter = 0
    self.best_fitness_history = []
    self.diversity_history = []
    self.current_mutation_rate = mutation_rate
    self.current_crossover_rate = crossover_rate
    self.generation_diversity_scores = []

    self.n_inputs: Optional[int] = None
    self.n_outputs: Optional[int] = None
    self.best_expressions: List[Expression] = []
    self.fitness_history: List[float] = []

    if self.advanced_simplify:
      self.sympy_simplifier = SymPySimplifier()

  def fit(self, X: np.ndarray, y: np.ndarray, constant_optimize=False):
    """Enhanced fit with diversity preservation and adaptive evolution"""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    last_update = 0

    if X.ndim == 1:
      X = X.reshape(-1, 1)
    if y.ndim == 1:
      y = y.reshape(-1, 1)

    self.n_inputs = X.shape[1]
    self.n_outputs = y.shape[1]

    # Reset evolution state
    self.fitness_history = []
    self.best_fitness_history = []
    self.diversity_history = []
    self.generation_diversity_scores = []
    self.stagnation_counter = 0
    self.current_mutation_rate = self.mutation_rate
    self.current_crossover_rate = self.crossover_rate

    # Generate diverse initial population
    if self.n_inputs is None:
      raise ValueError("n_inputs must be set before generating population")
    generator = ExpressionGenerator(self.n_inputs, self.max_depth)
    population = generate_diverse_population(generator, self.n_inputs, self.population_size, self.max_depth, lambda expr: is_expression_valid(expr, self.n_inputs))

    genetic_ops = GeneticOperations(self.n_inputs, max_complexity=25)
    best_fitness = -np.inf
    plateau_counter = 0
    if self.console_log:
      print(f"Starting evolution with {self.population_size} individuals for {self.generations} generations")

    for generation in range(self.generations):
      # Evaluate fitness with enhanced scoring

      fitness_scores = self._evaluate_population_enhanced(population, X, y)

      # Calculate diversity metrics
      diversity_score = self._calculate_population_diversity(population)
      self.diversity_history.append(diversity_score)
      self.generation_diversity_scores.append(diversity_score)

      # Track best fitness and detect stagnation
      generation_best_fitness = max(fitness_scores)
      generation_avg_fitness = np.mean(fitness_scores)
      self.fitness_history.append(generation_best_fitness)

      # Update best solution
      if generation_best_fitness > best_fitness + 1e-8:
        best_fitness = generation_best_fitness
        best_idx = fitness_scores.index(generation_best_fitness)
        self.best_expressions = [population[best_idx].copy()]
        self.stagnation_counter = 0
        plateau_counter = 0
      else:
        self.stagnation_counter += 1
        plateau_counter += 1

      if generation > 0 and best_fitness - self.best_fitness_history[-1] > 1e-3:
        last_update = generation
      self.best_fitness_history.append(best_fitness)

      # Enhanced progress reporting
      if self.console_log:
        if generation % 10 == 0 or generation < 20:
          print(f"Gen {generation:3d}: Best={best_fitness:.6f} Avg={generation_avg_fitness:.6f} "
                f"Div={diversity_score:.3f} Stag={self.stagnation_counter} "
                f"MutRate={self.current_mutation_rate:.3f}")

      # Adaptive parameter adjustment
      if self.adaptive_rates:
        self._update_adaptive_parameters(generation, diversity_score, plateau_counter)

      # Handle long-term stagnation with population restart
      if self.stagnation_counter >= self.restart_threshold:
        if self.console_log:
          print(f"Population restart at generation {generation} (stagnation: {self.stagnation_counter})")
        population = self._restart_population_enhanced(population, fitness_scores, generator)
        self.stagnation_counter = 0
        plateau_counter = 0
        continue

      # Enhanced diversity injection for moderate stagnation
      if self.stagnation_counter > 10 and diversity_score < self.diversity_threshold:
        population = inject_diversity(
          population, fitness_scores, generator, 0.3,
          lambda expr: is_expression_valid(expr, self.n_inputs),
          generate_high_diversity_expression,
          generate_targeted_diverse_expression,
          generate_complex_diverse_expression,
          self.stagnation_counter, self.console_log
        )
        if self.console_log:
          print(f"Diversity injection at generation {generation}")

      # Enhanced reproduction with multiple strategies
      new_population = enhanced_reproduction_v2(
        population, fitness_scores, genetic_ops, diversity_score, generation,
        self.population_size, self.elite_fraction, self.current_crossover_rate, self.current_mutation_rate,
        self.n_inputs, self.max_depth, lambda expr: is_expression_valid(expr, self.n_inputs), generate_high_diversity_expression,
        X, y
      )

      population = new_population

      # Constant Optimize
      if constant_optimize:
        self._optimize_constants(X.squeeze(), y, population, generation - last_update)

    # Final reporting
    final_best = max(self.fitness_history) if self.fitness_history else -np.inf
    if self.console_log:
      print(f"\nEvolution completed:")
      print(f"Final best fitness: {final_best:.6f}")
      if self.best_expressions:
        print(f"Best expression: {self.best_expressions[0].to_string()}")
        print(f"Expression complexity: {self.best_expressions[0].complexity():.2f}")

  def _evaluate_population_enhanced(self, population: List[Expression],
                                    X: np.ndarray, y: np.ndarray) -> List[float]:
    """Enhanced fitness evaluation with multiple objectives"""
    fitness_scores = []

    for expr in population:
      try:
        predictions = expr.evaluate(X)
        if predictions.ndim == 1:
          predictions = predictions.reshape(-1, 1)

        # Multi-objective fitness
        mse = np.mean((y - predictions) ** 2)

        # Complexity penalty (adjustable)
        complexity_penalty = self.parsimony_coefficient * expr.complexity()

        # Stability penalty for extreme values
        stability_penalty = 0.0
        max_abs_pred = np.max(np.abs(predictions))
        if max_abs_pred > 1e6:
          stability_penalty = 0.5
        elif max_abs_pred > 1e4:
          stability_penalty = 0.1

        # Numerical stability penalty
        if np.any(~np.isfinite(predictions)):
          stability_penalty += 1.0

        # Diversity bonus (small)
        diversity_bonus = self._calculate_expression_uniqueness(expr, population) * 0.001

        fitness = -mse - complexity_penalty - stability_penalty + diversity_bonus
        fitness_scores.append(float(fitness))

      except Exception:
        fitness_scores.append(-1e8)  # Severe penalty for failed evaluation

    return fitness_scores

  def _calculate_expression_uniqueness(self, expr: Expression, population: List[Expression]) -> float:
    """Calculate how unique an expression is compared to population"""
    expr_string = expr.to_string()
    unique_score = 0.0

    for other in population:
      if expr is not other:
        similarity = string_similarity(expr_string, other.to_string())
        unique_score += (1.0 - similarity)

    return unique_score / max(1, len(population) - 1)

  def _calculate_population_diversity(self, population: List[Expression]) -> float:
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

  def _update_adaptive_parameters(self, generation: int, diversity_score: float, plateau_counter: int):
    """Enhanced adaptive parameter updates"""
    # Base adaptation based on diversity and stagnation
    if diversity_score < self.diversity_threshold:
      # Low diversity - increase exploration
      mutation_multiplier = 1.0 + (self.diversity_threshold - diversity_score) * 2.0
      crossover_multiplier = 0.9
    else:
      # Good diversity - normal rates
      mutation_multiplier = 1.0
      crossover_multiplier = 1.0

    # Additional adaptation based on plateau
    if plateau_counter > 15:
      mutation_multiplier *= 1.5
      crossover_multiplier *= 0.8
    elif plateau_counter > 10:
      mutation_multiplier *= 1.2

    # Apply multipliers with bounds
    self.current_mutation_rate = np.clip(
      self.mutation_rate * mutation_multiplier, 0.05, 0.5)
    self.current_crossover_rate = np.clip(
      self.crossover_rate * crossover_multiplier, 0.5, 0.95)

    # Gradually return to original rates when performing well
    if self.stagnation_counter < 5 and plateau_counter < 5:
      self.current_mutation_rate = (self.current_mutation_rate * 0.95 +
                                    self.mutation_rate * 0.05)
      self.current_crossover_rate = (self.current_crossover_rate * 0.95 +
                                     self.crossover_rate * 0.05)

  def _restart_population_enhanced(self, population: List[Expression],
                                   fitness_scores: List[float],
                                   generator: ExpressionGenerator) -> List[Expression]:
    """Enhanced population restart with better elite preservation"""
    # Keep top performers (more aggressive selection)
    elite_count = max(2, int(self.population_size * 0.05))  # Keep top 5%
    elite_indices = np.argsort(fitness_scores)[-elite_count:]
    elites = [population[i].copy() for i in elite_indices]

    new_population = elites.copy()

    # Create variants of elites with different mutation strengths
    if self.n_inputs is None:
      raise ValueError("n_inputs must be set before creating genetic operations")
    genetic_ops = GeneticOperations(self.n_inputs, max_complexity=25)
    for elite in elites:
      # High mutation variants
      for _ in range(2):
        mutated = genetic_ops.mutate(elite, 0.4)
        if self._is_expression_valid(mutated):
          new_population.append(mutated)

      # Medium mutation variants
      for _ in range(2):
        mutated = genetic_ops.mutate(elite, 0.2)
        if self._is_expression_valid(mutated):
          new_population.append(mutated)

    # Fill rest with completely new diverse individuals
    while len(new_population) < self.population_size:
      new_expr = Expression(generator.generate_random_expression())
      if self._is_expression_valid(new_expr):
        new_population.append(new_expr)

    return new_population[:self.population_size]

  def _enhanced_selection(self, population: List[Expression], fitness_scores: List[float],
                          diversity_score: float) -> Expression:
    """Enhanced selection balancing fitness and diversity"""

    # Adaptive selection pressure
    if diversity_score > self.diversity_threshold:
      # Good diversity - focus more on fitness
      if random.random() < 0.85:
        return self._tournament_selection(population, fitness_scores)
      else:
        return self._diversity_selection(population)
    else:
      # Low diversity - balance fitness and diversity
      if random.random() < 0.6:
        return self._tournament_selection(population, fitness_scores)
      else:
        return self._diversity_selection(population)

  def _diversity_selection(self, population: List[Expression]) -> Expression:
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

  def _tournament_selection(self, population: List[Expression], fitness_scores: List[float]) -> Expression:
    """Enhanced tournament selection with adaptive tournament size"""
    # Adaptive tournament size based on stagnation
    base_tournament_size = self.tournament_size
    if self.stagnation_counter > 15:
      tournament_size = max(2, base_tournament_size - 1)  # Smaller tournaments for more diversity
    elif self.stagnation_counter > 8:
      tournament_size = base_tournament_size
    else:
      tournament_size = min(len(population), base_tournament_size + 1)  # Larger tournaments for better selection

    tournament_indices = random.sample(range(len(population)),
                                       min(tournament_size, len(population)))
    tournament_fitness = [fitness_scores[i] for i in tournament_indices]
    winner_idx = tournament_indices[np.argmax(tournament_fitness)]
    return population[winner_idx]

  def predict(self, X: np.ndarray) -> np.ndarray:
    """Make predictions using the best expressions"""
    if not self.best_expressions:
      raise ValueError("Model has not been fitted yet. Call fit() first.")

    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
      X = X.reshape(-1, 1)

    predictions = []
    for expr in self.best_expressions:
      pred = expr.evaluate(X)
      if pred.ndim == 1:
        pred = pred.reshape(-1, 1)
      predictions.append(pred)

    if len(predictions) == 1:
      return predictions[0]
    else:
      return np.concatenate(predictions, axis=1)

  def score(self, X: np.ndarray, y: np.ndarray) -> float:
    """Calculate R² score for the model"""
    if not self.best_expressions:
      raise ValueError("Model has not been fitted yet")

    predictions = self.predict(X)

    if y.ndim == 1:
      y = y.reshape(-1, 1)

    ss_res = float(np.sum((y - predictions) ** 2))
    ss_tot = float(np.sum((y - np.mean(y, axis=0)) ** 2))

    if ss_tot == 0:
      return 1.0 if ss_res == 0 else 0.0

    return 1.0 - (ss_res / ss_tot)

  def get_expressions(self) -> List[str]:
    """Get the best expressions as strings"""
    if not self.best_expressions:
      return []

    expressions = []
    for expr in self.best_expressions:
      expr_str = expr.to_string()
      if self.sympy_simplify:
        simplified = self._to_sympy_expression(expr_str)
        expressions.append(simplified if simplified else expr_str)
      else:
        expressions.append(expr_str)

    return expressions

  def get_expr_obj(self) -> List[Expression]:
    return self.best_expressions

  def get_raw_expressions(self) -> List[str]:
    """Get raw expressions without simplification"""
    return [expr.to_string() for expr in self.best_expressions]

  def get_detailed_expressions(self) -> List[Dict]:
    """Get detailed information about expressions"""
    if not self.best_expressions:
      return []

    detailed = []
    for i, expr in enumerate(self.best_expressions):
      info = {
        'expression': expr.to_string(),
        'complexity': expr.complexity(),
        'size': expr.size(),
        'simplified': None,
        'output_index': i
      }

      if self.sympy_simplify:
        info['simplified'] = self._to_sympy_expression(expr.to_string())

      detailed.append(info)

    return detailed

  def _to_sympy_expression(self, expr_string: str) -> Optional[str]:
    """Convert expression to SymPy and simplify"""
    try:
      if self.advanced_simplify and hasattr(self, 'sympy_simplifier') and self.n_inputs is not None:
        result = self.sympy_simplifier.simplify_expression(expr_string, self.n_inputs)
        return result.get('simplified', expr_string)
      else:
        # Basic SymPy simplification
        sympy_expr = sp.sympify(expr_string.replace('X', 'x'))
        simplified = sp.simplify(sympy_expr)
        return str(simplified).replace('x', 'X')
    except Exception:
      return expr_string

  def get_evolution_stats(self) -> Dict[str, Any]:
    """Get detailed evolution statistics"""
    return {
      'fitness_history': self.fitness_history.copy(),
      'best_fitness_history': self.best_fitness_history.copy(),
      'diversity_history': self.diversity_history.copy(),
      'final_mutation_rate': self.current_mutation_rate,
      'final_crossover_rate': self.current_crossover_rate,
      'total_stagnation': self.stagnation_counter,
      'total_generations': len(self.fitness_history)
    }

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

  def _generate_diverse_population(self, generator):
    return generate_diverse_population(generator, self.n_inputs, self.population_size, self.max_depth, lambda expr: is_expression_valid(expr, self.n_inputs))

  def _inject_diversity(self, population, fitness_scores, generator, injection_rate=0.3):
    return inject_diversity(
      population, fitness_scores, generator, injection_rate,
      lambda expr: is_expression_valid(expr, self.n_inputs),
      generate_high_diversity_expression,
      generate_targeted_diverse_expression,
      generate_complex_diverse_expression,
      self.stagnation_counter, self.console_log
    )

  def _is_expression_valid(self, expr):
    return is_expression_valid(expr, self.n_inputs)

  def _string_similarity(self, s1, s2):
    return string_similarity(s1, s2)
