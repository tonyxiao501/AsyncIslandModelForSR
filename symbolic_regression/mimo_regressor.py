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
  generate_high_diversity_expression, generate_targeted_diverse_expression, generate_complex_diverse_expression, enhanced_reproduction_v2, evaluate_population_enhanced
)
from .utils import string_similarity, calculate_expression_uniqueness, calculate_population_diversity
from .ensemble_worker import _fit_worker
from .adaptive_evolution import (
    update_adaptive_parameters, restart_population_enhanced, should_optimize_constants_enhanced
)
from .selection import enhanced_selection, diversity_selection, tournament_selection
from .expression_utils import to_sympy_expression, optimize_constants, should_optimize_constants
from .evolution_stats import get_evolution_stats, get_detailed_expressions

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
      fitness_scores = evaluate_population_enhanced(population, X, y, self.parsimony_coefficient)

      # Calculate diversity metrics
      diversity_score = calculate_population_diversity(population)
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
        self.current_mutation_rate, self.current_crossover_rate = update_adaptive_parameters(
          self, generation, diversity_score, plateau_counter,
          self.diversity_threshold, self.mutation_rate, self.crossover_rate,
          self.current_mutation_rate, self.current_crossover_rate, self.stagnation_counter
        )

      # Handle long-term stagnation with population restart
      if self.stagnation_counter >= self.restart_threshold:
        if self.console_log:
          print(f"Population restart at generation {generation} (stagnation: {self.stagnation_counter})")
        population = restart_population_enhanced(population, fitness_scores, generator, self.population_size, self.n_inputs)
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
        optimize_constants(X.squeeze(), y, population, generation - last_update, self.population_size, self.generations)

    # Final reporting
    final_best = max(self.fitness_history) if self.fitness_history else -np.inf
    if self.console_log:
      print(f"\nEvolution completed:")
      print(f"Final best fitness: {final_best:.6f}")
      if self.best_expressions:
        print(f"Best expression: {self.best_expressions[0].to_string()}")
        print(f"Expression complexity: {self.best_expressions[0].complexity():.2f}")

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
        simplified = to_sympy_expression(expr_str, self.advanced_simplify, self.n_inputs)
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
    return get_detailed_expressions(self.best_expressions, self.sympy_simplify)

  def get_evolution_stats(self) -> Dict[str, Any]:
    """Get detailed evolution statistics"""
    return get_evolution_stats(
      self.fitness_history, self.best_fitness_history, self.diversity_history,
      self.current_mutation_rate, self.current_crossover_rate, self.stagnation_counter
    )
