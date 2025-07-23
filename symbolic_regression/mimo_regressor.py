import numpy as np
import os
import time
from typing import List, Optional, Dict, Any, Tuple
from .expression_tree import Expression
from .generator import ExpressionGenerator
from .genetic_ops import GeneticOperations
from .expression_tree.utils.sympy_utils import SymPySimplifier
from .population import (
  PopulationManager, generate_diverse_population_optimized, inject_diversity_optimized,
  evaluate_population_enhanced_optimized,
  generate_high_diversity_expression_optimized, generate_targeted_diverse_expression_optimized, 
  generate_complex_diverse_expression_optimized
)
from .utils import calculate_population_diversity
from .adaptive_evolution import (
    update_adaptive_parameters, restart_population_enhanced
)
from .expression_utils import to_sympy_expression, optimize_constants
from .evolution_stats import get_evolution_stats, get_detailed_expressions
from .data_scaling import DataScaler
from .multi_scale_fitness import MultiScaleFitnessEvaluator
from .great_powers import GreatPowers

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
               console_log=True,
               # New optimization control parameters
               evolution_sympy_simplify: bool = False,  # Disabled during evolution
               evolution_constant_optimize: bool = False,  # Disabled during evolution
               final_optimization_generations: int = 5,  # Apply optimization in final N generations
               # Data scaling parameters
               enable_data_scaling: bool = True,
               input_scaling: str = 'auto',
               output_scaling: str = 'auto',
               scaling_target_range: Tuple[float, float] = (-5.0, 5.0),  # Expanded range for extreme physics scales
               shared_data_scaler: Optional[DataScaler] = None,  # Pre-fitted scaler for ensemble consistency
               # Multi-scale fitness evaluation
               use_multi_scale_fitness: bool = True,
               extreme_value_threshold: float = 1e6,
               # Early termination and late extension parameters
               enable_early_termination: bool = True,
               early_termination_threshold: float = 0.99,
               early_termination_check_interval: int = 10,
               enable_late_extension: bool = True,
               late_extension_threshold: float = 0.95,
               late_extension_generations: int = 50
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

    # Optimization control parameters
    self.evolution_sympy_simplify = evolution_sympy_simplify
    self.evolution_constant_optimize = evolution_constant_optimize
    self.final_optimization_generations = final_optimization_generations

    # Enhanced evolution parameters
    self.diversity_threshold = diversity_threshold
    self.adaptive_rates = adaptive_rates
    self.restart_threshold = restart_threshold
    self.elite_fraction = elite_fraction

    # Data scaling parameters
    self.enable_data_scaling = enable_data_scaling
    self.input_scaling = input_scaling
    self.output_scaling = output_scaling
    self.scaling_target_range = scaling_target_range
    self.data_scaler: Optional[DataScaler] = shared_data_scaler  # Use shared scaler if provided

    # Multi-scale fitness evaluation
    self.use_multi_scale_fitness = use_multi_scale_fitness
    self.fitness_evaluator: Optional[MultiScaleFitnessEvaluator] = None
    if use_multi_scale_fitness:
      self.fitness_evaluator = MultiScaleFitnessEvaluator(
        use_log_space=True,
        use_relative_metrics=True,
        extreme_value_threshold=extreme_value_threshold
      )

    # Early termination and late extension parameters
    self.enable_early_termination = enable_early_termination
    self.early_termination_threshold = early_termination_threshold
    self.early_termination_check_interval = early_termination_check_interval
    self.enable_late_extension = enable_late_extension
    self.late_extension_threshold = late_extension_threshold
    self.late_extension_generations = late_extension_generations
    self.late_extension_triggered = False  # Track if extension was already triggered

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
    
    # Population manager (will be initialized when n_inputs is set)
    self.pop_manager: Optional[PopulationManager] = None

    # Great Powers mechanism - tracks best 5 expressions across all generations
    self.great_powers = GreatPowers(max_powers=5)

    # Inter-thread communication components
    self.shared_manager = None
    self.worker_id = None
    self.inter_thread_enabled = False

    # Debug CSV tracking
    self.debug_csv_path = None
    self.debug_worker_id = None

    if self.advanced_simplify:
      self.sympy_simplifier = SymPySimplifier()

  def enable_inter_thread_communication(self, shared_data, worker_id: int):
    """Enable inter-thread communication for this regressor instance"""
    self.shared_manager = shared_data
    self.worker_id = worker_id
    self.inter_thread_enabled = True
    if self.console_log:
      print(f"Worker {worker_id}: Inter-thread communication enabled")

  def set_debug_csv_path(self, debug_csv_path: str, worker_id: int):
    """Set the debug CSV path for tracking evolution progress"""
    self.debug_csv_path = debug_csv_path
    self.debug_worker_id = worker_id

  def _write_debug_csv(self, generation: int, population: List, fitness_scores: List[float],
                       diversity_score: float):
    """Write the top 10 expressions from this generation to the debug CSV file"""
    if not self.debug_csv_path or self.debug_worker_id is None:
      return

    try:
      import csv
      import datetime

      # Get top 10 expressions by fitness
      top_indices = sorted(range(len(fitness_scores)),
                          key=lambda i: fitness_scores[i], reverse=True)[:10]

      # Write to CSV file (append mode)
      with open(self.debug_csv_path, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        for rank, idx in enumerate(top_indices, 1):
          expr = population[idx]
          fitness = fitness_scores[idx]
          complexity = expr.complexity()
          expression_str = expr.to_string()

          writer.writerow([
            timestamp,
            self.debug_worker_id,
            generation,
            rank,
            f"{fitness:.6f}",
            f"{complexity:.3f}",
            expression_str,
            f"{diversity_score:.3f}",
            self.stagnation_counter
          ])

    except Exception as e:
      # Fail silently to not disrupt evolution
      if self.console_log:
        print(f"Debug CSV write failed: {e}")

  def _evaluate_population_enhanced_with_scaling(self, population: List[Expression], 
                                               X_scaled: np.ndarray, y_scaled: np.ndarray,
                                               X_original: np.ndarray, y_original: np.ndarray) -> List[float]:
    """
    Enhanced population evaluation that properly handles scaling.
    Evaluates fitness on original scale to ensure true performance measurement.
    """
    from sklearn.metrics import r2_score
    fitness_scores = []
    
    # Pre-calculate population diversity metrics once (if pop_manager available)
    base_diversity_bonus = 0.0
    if self.pop_manager is not None:
        diversity_metrics = self.pop_manager.calculate_population_diversity_optimized(population)
        base_diversity_bonus = diversity_metrics['overall'] * 0.0005
    
    for i, expr in enumerate(population):
        try:
            # Get predictions on scaled input
            predictions_scaled = expr.evaluate(X_scaled)
            if predictions_scaled.ndim == 1:
                predictions_scaled = predictions_scaled.reshape(-1, 1)
            
            # Transform predictions back to original scale for fitness evaluation
            if self.enable_data_scaling and self.data_scaler is not None:
                try:
                    predictions_original = self.data_scaler.inverse_transform_output(predictions_scaled)
                except Exception as e:
                    # If inverse transform fails, heavily penalize
                    fitness_scores.append(-10.0)
                    continue
            else:
                predictions_original = predictions_scaled
            
            # Core fitness calculation using R² score on ORIGINAL scale
            try:
                r2 = r2_score(y_original.flatten(), predictions_original.flatten())
            except Exception:
                # Fallback for edge cases
                ss_res = np.sum((y_original - predictions_original) ** 2)
                ss_tot = np.sum((y_original - np.mean(y_original)) ** 2)
                if ss_tot == 0:
                    r2 = 1.0 if ss_res == 0 else 0.0
                else:
                    r2 = 1.0 - (ss_res / ss_tot)
            
            # Cached complexity penalty
            if self.pop_manager is not None:
                complexity = self.pop_manager.get_expression_complexity(expr)
            else:
                complexity = expr.complexity()
            complexity_penalty = self.parsimony_coefficient * complexity
            
            # Stability penalty - check both scaled and original predictions
            stability_penalty = 0.0
            max_abs_pred_scaled = np.max(np.abs(predictions_scaled))
            max_abs_pred_original = np.max(np.abs(predictions_original))
            
            if (max_abs_pred_scaled > 1e8 or max_abs_pred_original > 1e8):
                stability_penalty = 0.5
            elif (max_abs_pred_scaled > 1e6 or max_abs_pred_original > 1e6):
                stability_penalty = 0.3
            elif (max_abs_pred_scaled > 1e4 or max_abs_pred_original > 1e4):
                stability_penalty = 0.1
            
            if (np.any(~np.isfinite(predictions_scaled)) or np.any(~np.isfinite(predictions_original))):
                stability_penalty += 0.5  # Heavily penalize infinite/NaN
            
            # Apply small diversity bonus (as R² adjustment)
            diversity_bonus = base_diversity_bonus * 0.01
            
            # Final R² based fitness score
            fitness = r2 - complexity_penalty - stability_penalty + diversity_bonus
            fitness_scores.append(float(fitness))
            
        except Exception:
            fitness_scores.append(-10.0)  # Large negative R² score for invalid expressions
    
    return fitness_scores

  def _evaluate_population_multi_scale(self, population: List[Expression], 
                                      X_scaled: np.ndarray, y_scaled: np.ndarray,
                                      X_original: np.ndarray, y_original: np.ndarray) -> List[float]:
    """
    Enhanced population evaluation using multi-scale fitness metrics.
    Evaluates fitness on original scale to ensure true performance measurement.
    All fitness values are now R² based for consistency.
    """
    from sklearn.metrics import r2_score
    fitness_scores = []
    
    for expr in population:
      try:
        # Get predictions on scaled input
        predictions_scaled = expr.evaluate(X_scaled)
        if predictions_scaled.ndim == 1:
          predictions_scaled = predictions_scaled.reshape(-1, 1)
        
        # Transform predictions back to original scale for fitness evaluation
        if self.enable_data_scaling and self.data_scaler is not None:
          try:
            predictions_original = self.data_scaler.inverse_transform_output(predictions_scaled)
          except Exception as e:
            # If inverse transform fails, heavily penalize
            fitness_scores.append(-10.0)
            if self.console_log and len(fitness_scores) <= 5:
              print(f"Inverse transform failed during fitness evaluation: {e}")
            continue
        else:
          predictions_original = predictions_scaled
        
        # Use multi-scale fitness evaluator on ORIGINAL scale data
        if self.fitness_evaluator:
          base_fitness = self.fitness_evaluator.evaluate_fitness(
            y_original.flatten(), predictions_original.flatten(), 0.0  # No parsimony penalty here
          )
        else:
          # Fallback to standard R² using scikit-learn on ORIGINAL scale
          try:
            base_fitness = r2_score(y_original.flatten(), predictions_original.flatten())
          except Exception:
            ss_res = np.sum((y_original - predictions_original) ** 2)
            ss_tot = np.sum((y_original - np.mean(y_original)) ** 2)
            base_fitness = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        # Apply complexity penalty (as R² adjustment)
        complexity = expr.complexity()
        complexity_penalty = self.parsimony_coefficient * complexity
        
        # Apply stability penalty for extreme predictions (as R² adjustment)
        # Check both scaled and original predictions for stability
        stability_penalty = 0.0
        max_abs_pred_scaled = np.max(np.abs(predictions_scaled))
        max_abs_pred_original = np.max(np.abs(predictions_original))
        
        if (max_abs_pred_scaled > 1e10 or max_abs_pred_original > 1e10 or
            np.any(np.isnan(predictions_scaled)) or np.any(np.isinf(predictions_scaled)) or
            np.any(np.isnan(predictions_original)) or np.any(np.isinf(predictions_original))):
          stability_penalty = 1.0  # Heavy penalty in R² scale
        elif max_abs_pred_scaled > 1e8 or max_abs_pred_original > 1e8:
          stability_penalty = 0.5
        elif max_abs_pred_scaled > 1e6 or max_abs_pred_original > 1e6:
          stability_penalty = 0.3
        
        final_fitness = base_fitness - complexity_penalty - stability_penalty
        fitness_scores.append(final_fitness)
        
      except Exception as e:
        # Invalid expression - use consistent R² penalty
        fitness_scores.append(-10.0)  # Large negative R² score for invalid expressions
        if self.console_log and len(fitness_scores) <= 5:  # Only log first few failures
          print(f"Expression evaluation failed: {e}")
    
    return fitness_scores

  def fit(self, X: np.ndarray, y: np.ndarray, constant_optimize=False):
    """Enhanced fit with diversity preservation and adaptive evolution"""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    last_update = 0

    if X.ndim == 1:
      X = X.reshape(-1, 1)
    if y.ndim == 1:
      y = y.reshape(-1, 1)

    # Apply data scaling if enabled
    X_scaled, y_scaled = X.copy(), y.copy()
    if self.enable_data_scaling:
      if self.data_scaler is None:
        # Create new data scaler if none provided
        self.data_scaler = DataScaler(
          input_scaling=self.input_scaling,
          output_scaling=self.output_scaling,
          target_range=self.scaling_target_range
        )
        X_scaled, y_scaled = self.data_scaler.fit_transform(X, y)
      else:
        # Use pre-fitted shared data scaler (for ensemble consistency)
        X_scaled = self.data_scaler.transform_input(X)
        y_scaled = self.data_scaler.transform_output(y) if hasattr(self.data_scaler, 'transform_output') else y.copy()
      
      if self.console_log:
        scaling_info = self.data_scaler.get_scaling_info()
        print(f"Data scaling applied:")
        print(f"  Input transforms: {scaling_info['input_transforms']}")
        print(f"  Output transform: {scaling_info['output_transform']}")
        print(f"  Target range: {scaling_info['target_range']}")

    self.n_inputs = X_scaled.shape[1]
    self.n_outputs = y_scaled.shape[1]
    
    # Initialize population manager
    if self.n_inputs is None:
      raise ValueError("n_inputs must be set before generating population")
    self.pop_manager = PopulationManager(self.n_inputs, self.max_depth)

    # Reset evolution state
    self.fitness_history = []
    self.best_fitness_history = []
    self.diversity_history = []
    self.generation_diversity_scores = []
    self.stagnation_counter = 0
    self.current_mutation_rate = self.mutation_rate
    self.current_crossover_rate = self.crossover_rate
    self.late_extension_triggered = False  # Reset late extension flag
    
    scaling_range = max(int(np.log(np.mean(np.abs(X)))), int(np.log(np.mean(np.abs(y)))))
    
    # Generate diverse initial population
    if self.n_inputs is None:
      raise ValueError("n_inputs must be set before generating population")
    generator = ExpressionGenerator(self.n_inputs, self.max_depth, scaling_range) # Added X, for scaling.
    population = generate_diverse_population_optimized(generator, self.n_inputs, self.population_size, self.max_depth, self.pop_manager)

    genetic_ops = GeneticOperations(self.n_inputs, max_complexity=25, scaling_range=scaling_range)
    best_fitness = -10.0  # Start with large negative R² score
    plateau_counter = 0
    original_generations = self.generations  # Store original generation count for extension
    if self.console_log:
      print(f"Starting evolution with {self.population_size} individuals for {self.generations} generations")

    generation = 0
    while generation < self.generations:
      # Evaluate fitness with enhanced scoring (use scaled data for expressions, original data for fitness)
      if self.use_multi_scale_fitness and self.fitness_evaluator:
        fitness_scores = self._evaluate_population_multi_scale(population, X_scaled, y_scaled, X, y)
      else:
        fitness_scores = self._evaluate_population_enhanced_with_scaling(population, X_scaled, y_scaled, X, y)

      # Calculate diversity metrics
      diversity_score = calculate_population_diversity(population)
      self.diversity_history.append(diversity_score)
      self.generation_diversity_scores.append(diversity_score)

      # Track best fitness and detect stagnation
      generation_best_fitness = max(fitness_scores)
      generation_avg_fitness = np.mean(fitness_scores)
      self.fitness_history.append(generation_best_fitness)

      # Update Great Powers mechanism (with redundancy elimination)
      great_powers_updated = self.great_powers.update_powers(population, fitness_scores, generation, X_scaled)
      
      # Update best solution (now compare against Great Powers)
      great_powers_best_fitness = self.great_powers.get_best_fitness()
      if great_powers_best_fitness > best_fitness + 1e-8:
        best_fitness = great_powers_best_fitness
        best_expr = self.great_powers.get_best_expression()
        if best_expr is not None:
          self.best_expressions = [best_expr]
        self.stagnation_counter = 0
        plateau_counter = 0
      elif generation_best_fitness > best_fitness + 1e-8:
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

      # Early termination check - test every N generations
      if (self.enable_early_termination and 
          generation > 0 and 
          generation % self.early_termination_check_interval == 0):
        if best_fitness >= self.early_termination_threshold:
          if self.console_log:
            print(f"\n*** EARLY TERMINATION ***")
            print(f"Generation {generation}: R² score ({best_fitness:.6f}) reached threshold ({self.early_termination_threshold})")
            print(f"Terminating evolution early after {generation + 1} generations")
          break

      # Late extension check - only trigger once at the end if threshold not met
      if (self.enable_late_extension and 
          not self.late_extension_triggered and 
          generation == original_generations - 1):  # Last generation of original run
        if best_fitness < self.late_extension_threshold:
          self.late_extension_triggered = True
          self.generations += self.late_extension_generations
          if self.console_log:
            print(f"\n*** LATE EXTENSION ***")
            print(f"Generation {generation}: R² score ({best_fitness:.6f}) below threshold ({self.late_extension_threshold})")
            print(f"Extending evolution by {self.late_extension_generations} generations (total: {self.generations})")

      # Enhanced progress reporting
      if self.console_log:
        if generation % 10 == 0 or generation < 20:
          great_powers_info = f"GP={len(self.great_powers)}"
          if great_powers_updated:
            great_powers_info += "*"
          else:
            # Check if update was rejected due to redundancy
            redundancy_stats = self.great_powers.get_redundancy_stats()
            if redundancy_stats['total_rejections'] > 0:
              great_powers_info += f"[R:{redundancy_stats['total_rejections']}]"
          
          # Diagnose potential fitness drop issues
          diagnosis = self.great_powers.diagnose_fitness_drop(generation_best_fitness, generation)
          if diagnosis["status"] == "fitness_drop_detected":
            great_powers_info += f" [WARN: Fitness drop detected! Gap={diagnosis['fitness_gap']:.6f}]"
          
          print(f"Gen {generation:3d}: Best={best_fitness:.6f} Avg={generation_avg_fitness:.6f} "
                f"Div={diversity_score:.3f} Stag={self.stagnation_counter} "
                f"MutRate={self.current_mutation_rate:.3f} {great_powers_info}")
          
          # Detailed fitness drop warning
          if diagnosis["status"] == "fitness_drop_detected" and generation % 10 == 0:
            print(f"  >>> FITNESS DROP WARNING: Current best ({diagnosis['current_best']:.6f}) vs "
                  f"Great Power best ({diagnosis['great_power_best']:.6f})")
            print(f"  >>> Last Great Power update: generation {diagnosis['latest_great_power_generation']} "
                  f"({diagnosis['generations_since_update']} generations ago)")
      
      # Emergency Great Powers injection if significant fitness drop detected
      if len(self.great_powers) > 0:
        diagnosis = self.great_powers.diagnose_fitness_drop(generation_best_fitness, generation)
        if diagnosis["status"] == "fitness_drop_detected" and diagnosis["fitness_gap"] > 0.05:
          if self.console_log:
            print(f"  >>> EMERGENCY: Injecting Great Powers due to significant fitness drop!")
          
          # Emergency injection of Great Powers
          population, fitness_scores = self.great_powers.inject_powers_into_population(
            population, fitness_scores, injection_rate=0.3
          )
          
          # Re-evaluate after injection
          if self.use_multi_scale_fitness and self.fitness_evaluator:
            fitness_scores = self._evaluate_population_multi_scale(population, X_scaled, y_scaled, X, y)
          else:
            fitness_scores = self._evaluate_population_enhanced_with_scaling(population, X_scaled, y_scaled, X, y)
          
          # Update best fitness after emergency injection
          emergency_best = max(fitness_scores)
          if emergency_best > best_fitness:
            best_fitness = emergency_best
            best_idx = fitness_scores.index(emergency_best)
            self.best_expressions = [population[best_idx].copy()]
            if self.console_log:
              print(f"  >>> Emergency injection improved fitness to {emergency_best:.6f}")
      
      # Periodic redundancy cleanup for Great Powers (every 50 generations)
      if generation > 0 and generation % 50 == 0 and len(self.great_powers) > 1:
        removed_count = self.great_powers.clean_redundant_powers(X_scaled, self.console_log)
        if removed_count > 0 and self.console_log:
          redundancy_stats = self.great_powers.get_redundancy_stats()
          print(f"  >>> Great Powers redundancy cleanup: removed {removed_count} duplicates")
          print(f"  >>> Total redundancy rejections: {redundancy_stats['total_rejections']} "
                f"(exact: {redundancy_stats['rejected_duplicates']}, "
                f"semantic: {redundancy_stats['semantic_rejections']}, "
                f"structural: {redundancy_stats['structural_rejections']})")

      # Adaptive parameter adjustment
      if self.adaptive_rates:
        self.current_mutation_rate, self.current_crossover_rate = update_adaptive_parameters(
          self, generation, diversity_score, plateau_counter,
          self.diversity_threshold, self.mutation_rate, self.crossover_rate,
          self.current_mutation_rate, self.current_crossover_rate, self.stagnation_counter
        )

      # Handle long-term stagnation with population restart - more conservative threshold
      if self.stagnation_counter >= 25:  # Restored to original 25 for later restart
        if self.console_log:
          print(f"Population restart at generation {generation} (stagnation: {self.stagnation_counter})")
          print(f"  Great Powers before restart: {len(self.great_powers)}")
          if len(self.great_powers) > 0:
            print(f"  Best Great Power fitness: {self.great_powers.get_best_fitness():.6f}")
        
        # Enhanced restart that preserves Great Powers
        population = self._restart_population_with_great_powers(population, fitness_scores, generator)
        self.stagnation_counter = 0
        plateau_counter = 0
        continue

      # Enhanced diversity injection for moderate stagnation - more conservative
      if diversity_score < 0.3:  # Only intervene when diversity is very low
        # Get protected indices from Great Powers
        protected_indices = self.great_powers.protect_elites_from_injection(
          population, fitness_scores, elite_fraction=0.15
        )
        population = inject_diversity_optimized(
          population, fitness_scores, generator, 0.25,  # Inject 25% new diverse expressions
          self.pop_manager,
          self.stagnation_counter, self.console_log,
          protected_indices=protected_indices
        )
        if self.console_log:
          print(f"Emergency diversity injection at generation {generation} (diversity={diversity_score:.3f}, protected={len(protected_indices)})")
      elif self.stagnation_counter > 8 and diversity_score < 0.5:  # Later intervention
        # Get protected indices from Great Powers
        protected_indices = self.great_powers.protect_elites_from_injection(
          population, fitness_scores, elite_fraction=0.1
        )
        population = inject_diversity_optimized(
          population, fitness_scores, generator, 0.15,  # Reduced injection rate
          self.pop_manager,
          self.stagnation_counter, self.console_log,
          protected_indices=protected_indices
        )
        if self.console_log:
          print(f"Diversity injection at generation {generation} (diversity={diversity_score:.3f})")

      # Enhanced reproduction with multiple strategies
      new_population = []
      
      # Elite preservation - increase elite fraction during stagnation
      base_elite_fraction = self.elite_fraction
      if self.stagnation_counter > 10:
        elite_fraction = min(0.25, base_elite_fraction * 1.5)  # Increase elites during stagnation
      else:
        elite_fraction = base_elite_fraction
        
      elite_count = max(1, int(elite_fraction * self.population_size))
      elite_indices = np.argsort(fitness_scores)[-elite_count:]
      for idx in elite_indices:
        new_population.append(population[idx].copy())
      
      # Generate rest of population through improved selection
      from .selection import enhanced_selection, tournament_selection
      while len(new_population) < self.population_size:
        if len(new_population) + 1 < self.population_size and np.random.random() < self.current_crossover_rate:
          # Crossover with better parent selection
          parent1 = enhanced_selection(population, fitness_scores, diversity_score, 
                                     self.diversity_threshold, self.tournament_size, self.stagnation_counter)
          parent2 = enhanced_selection(population, fitness_scores, diversity_score, 
                                     self.diversity_threshold, self.tournament_size, self.stagnation_counter)
          child1, child2 = genetic_ops.crossover(parent1, parent2)
          
          if self.pop_manager.is_expression_valid_cached(child1):
            new_population.append(child1)
          if len(new_population) < self.population_size and self.pop_manager.is_expression_valid_cached(child2):
            new_population.append(child2)
        else:
          # Mutation with better parent selection and context-aware mutations
          parent = tournament_selection(population, fitness_scores, self.tournament_size, self.stagnation_counter)
          child = genetic_ops.adaptive_mutate_with_feedback(
            parent, self.current_mutation_rate, X, y, generation, self.stagnation_counter
          )
          
          if self.pop_manager.is_expression_valid_cached(child):
            new_population.append(child)
      
      new_population = new_population[:self.population_size]

      # Anti-convergence measure: If diversity is too low, force mutation of duplicates
      if diversity_score < 0.4:
        new_population = self._eliminate_duplicates(new_population, generator, 0.3)
        if self.console_log:
          print(f"Duplicate elimination at generation {generation} (diversity={diversity_score:.3f})")

      # Periodic injection of Great Powers to maintain them in population
      if generation > 0 and generation % 15 == 0 and len(self.great_powers) > 0:
        # Re-evaluate population after reproduction changes
        if self.use_multi_scale_fitness and self.fitness_evaluator:
          current_fitness = self._evaluate_population_multi_scale(new_population, X_scaled, y_scaled, X, y)
        else:
          current_fitness = self._evaluate_population_enhanced_with_scaling(new_population, X_scaled, y_scaled, X, y)
        
        new_population, current_fitness = self.great_powers.inject_powers_into_population(
          new_population, current_fitness, injection_rate=0.1
        )
        fitness_scores = current_fitness  # Update fitness scores for next iteration
        
        if self.console_log:
          print(f"Great Powers injection at generation {generation} (injected up to {min(len(self.great_powers), max(1, int(0.1 * len(new_population))))} powers)")

      population = new_population

      # Inter-thread communication: Exchange expressions with other workers
      if self.inter_thread_enabled and self.shared_manager:
        try:
          # Check if we should exchange this generation
          exchange_interval = self.shared_manager.get('exchange_interval', 20)
          if generation > 0 and self.worker_id is not None and generation % exchange_interval == (self.worker_id % 5):
            from .shared_population_manager import ImprovedSharedData

            # Create temporary shared data handler
            temp_shared = ImprovedSharedData(
              n_workers=self.shared_manager['n_workers'],
              exchange_interval=self.shared_manager['exchange_interval'],
              purge_percentage=self.shared_manager['purge_percentage'],
              import_percentage=self.shared_manager['import_percentage']
            )
            temp_shared.temp_dir = self.shared_manager['temp_dir']
            temp_shared.lock_file = os.path.join(temp_shared.temp_dir, "exchange.lock")

            if self.worker_id is None:
              raise ValueError("worker_id must be set (not None) for population exchange.")
            population, fitness_scores = temp_shared.exchange_population_data(
              self.worker_id, population, fitness_scores, generation, self.n_inputs
            )
            if self.console_log:
              print(f"Worker {self.worker_id}: Population exchange completed at generation {generation}")
        except Exception as e:
          if self.console_log:
            print(f"Worker {self.worker_id}: Population exchange failed: {e}")

      # Constant Optimize - only in final generations or when explicitly enabled
      apply_constant_optimization = (
        constant_optimize and 
        (self.evolution_constant_optimize or generation >= (self.generations - self.final_optimization_generations))
      )
      
      if apply_constant_optimization:
        from .expression_utils import optimize_constants
        optimize_constants(X.squeeze(), y, population, generation - last_update, 
                         self.population_size, self.generations, enable_optimization=True)

      # Debug CSV logging - reduced frequency for better performance
      if generation % 10 == 0:  # Only log every 10 generations instead of every generation
        self._write_debug_csv(generation, population, fitness_scores, diversity_score)

      # Increment generation counter for while loop
      generation += 1

    # FINAL OPTIMIZATION PHASE: Apply expensive operations to top expressions only
    if self.console_log:
      print(f"\nApplying final optimizations to top expressions...")
    
    start_time = time.time()
    
    # Get all candidates from Great Powers and current population
    all_candidates = self.great_powers.get_all_candidates(population, fitness_scores)
    
    if self.console_log:
      print(f"Final candidate pool: {len(all_candidates)} expressions ({len(self.great_powers)} from Great Powers)")
      for i, candidate in enumerate(all_candidates[:5]):  # Show top 5
        print(f"  {i+1}. {candidate['source']}: fitness={candidate['fitness']:.6f}")
    
    # Re-evaluate all candidates with consistent scaling approach
    candidate_expressions = [candidate['expression'] for candidate in all_candidates]
    if self.use_multi_scale_fitness and self.fitness_evaluator:
      final_fitness_scores = self._evaluate_population_multi_scale(candidate_expressions, X_scaled, y_scaled, X, y)
    else:
      final_fitness_scores = self._evaluate_population_enhanced_with_scaling(candidate_expressions, X_scaled, y_scaled, X, y)
    
    # Select top candidates for final optimization
    top_count = min(10, len(candidate_expressions))
    top_indices = sorted(range(len(final_fitness_scores)), key=lambda i: final_fitness_scores[i], reverse=True)[:top_count]
    top_expressions = [candidate_expressions[i] for i in top_indices]
    
    # Apply final optimizations (optimization should be done on the same scale as training)
    from .expression_utils import optimize_final_expressions
    optimized_expressions = optimize_final_expressions(top_expressions, X_scaled, y_scaled)
    
    # Re-evaluate with optimized constants using consistent scaling approach (evaluate on original scale)
    if self.use_multi_scale_fitness and self.fitness_evaluator:
      optimized_fitness_scores = self._evaluate_population_multi_scale(optimized_expressions, X_scaled, y_scaled, X, y)
    else:
      optimized_fitness_scores = self._evaluate_population_enhanced_with_scaling(optimized_expressions, X_scaled, y_scaled, X, y)
    
    # Select best expressions based on optimized fitness
    best_indices = sorted(range(len(optimized_fitness_scores)), key=lambda i: optimized_fitness_scores[i], reverse=True)
    max_expressions = self.n_outputs if self.n_outputs is not None else 1
    self.best_expressions = [optimized_expressions[i] for i in best_indices[:min(max_expressions, len(optimized_expressions))]]
    
    optimization_time = time.time() - start_time
    if self.console_log:
      print(f"Final optimization completed in {optimization_time:.2f}s")
      improvement_count = sum(1 for i, opt_fitness in enumerate(optimized_fitness_scores) 
                            if i < len(final_fitness_scores) and opt_fitness > final_fitness_scores[top_indices[i]])
      print(f"Optimization improved {improvement_count}/{len(optimized_expressions)} expressions")

    # Final reporting
    final_best = max(self.fitness_history) if self.fitness_history else -10.0
    actual_generations = len(self.fitness_history)
    if self.console_log:
      print(f"\nEvolution completed:")
      print(f"Actual generations run: {actual_generations}")
      print(f"Final best fitness: {final_best:.6f}")
      
      # Report early termination or late extension
      if self.enable_early_termination and actual_generations < original_generations:
        print(f"*** Early termination triggered (saved {original_generations - actual_generations} generations)")
      elif self.late_extension_triggered:
        print(f"*** Late extension triggered (added {self.late_extension_generations} generations)")
      
      if self.best_expressions:
        print(f"Best expression: {self.best_expressions[0].to_string()}")
        print(f"Expression complexity: {self.best_expressions[0].complexity():.2f}")

  def predict(self, X: np.ndarray) -> np.ndarray:
    """Make predictions using the best expressions with proper scaling handling"""
    if not self.best_expressions:
      raise ValueError("Model has not been fitted yet. Call fit() first.")

    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
      X = X.reshape(-1, 1)

    # Apply input scaling if it was used during training
    X_for_prediction = X.copy()
    if self.enable_data_scaling and self.data_scaler is not None:
      try:
        X_for_prediction = self.data_scaler.transform_input(X)
      except Exception as e:
        if self.console_log:
          print(f"Warning: Could not apply input scaling during prediction: {e}")
        X_for_prediction = X

    predictions = []
    for expr in self.best_expressions:
      # Evaluate expression on scaled input
      pred = expr.evaluate(X_for_prediction)
      if pred.ndim == 1:
        pred = pred.reshape(-1, 1)
      predictions.append(pred)

    # Combine predictions
    if len(predictions) == 1:
      final_predictions = predictions[0]
    else:
      final_predictions = np.concatenate(predictions, axis=1)

    # Apply inverse output scaling if it was used during training
    if self.enable_data_scaling and self.data_scaler is not None:
      try:
        final_predictions = self.data_scaler.inverse_transform_output(final_predictions)
      except Exception as e:
        if self.console_log:
          print(f"Warning: Could not apply inverse output scaling during prediction: {e}")

    return final_predictions

  def score(self, X: np.ndarray, y: np.ndarray) -> float:
    """Calculate R² score for the model using scikit-learn's consistent implementation"""
    from sklearn.metrics import r2_score
    
    if not self.best_expressions:
      raise ValueError("Model has not been fitted yet")

    # Use predict method which handles scaling correctly
    predictions = self.predict(X)

    if y.ndim == 1:
      y = y.reshape(-1, 1)
    if predictions.ndim == 1:
      predictions = predictions.reshape(-1, 1)

    # Always use scikit-learn's R² implementation for consistency
    try:
      r2 = r2_score(y.flatten(), predictions.flatten())
      
      # For extreme values, clamp the result to prevent meaningless scores
      if abs(r2) > 100:
        if self.console_log:
          print(f"Warning: Extreme R² value ({r2:.2e}) detected, likely due to scaling mismatch")
        return 0.0  # Return neutral score for scaling mismatches
      
      return max(-10.0, min(1.0, r2))  # Clamp to consistent R² range (same as evaluation)
    
    except Exception as e:
      if self.console_log:
        print(f"Warning: R² calculation failed: {e}")
      return 0.0

  def get_expressions(self) -> List[str]:
    """Get the best expressions as strings (in original data scale if scaling was used)"""
    if not self.best_expressions:
      return []

    expressions = []
    for expr in self.best_expressions:
      expr_str = expr.to_string()
      
      # Apply simplification
      if self.sympy_simplify:
        from .expression_utils import to_sympy_expression
        simplified = to_sympy_expression(expr_str, self.advanced_simplify, self.n_inputs, enable_simplify=True)
        expressions.append(simplified if simplified else expr_str)
      else:
        expressions.append(expr_str)

    return expressions
  
  def get_scaled_expressions(self) -> List[str]:
    """Get expressions in scaled data space (useful for debugging)"""
    if not self.best_expressions:
      return []

    expressions = []
    for expr in self.best_expressions:
      expr_str = expr.to_string()
      if self.sympy_simplify:
        from .expression_utils import to_sympy_expression
        simplified = to_sympy_expression(expr_str, self.advanced_simplify, self.n_inputs, enable_simplify=True)
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

  def _eliminate_duplicates(self, population: List, generator: ExpressionGenerator,
                           replacement_fraction: float) -> List:
    """
    Eliminate duplicate expressions to prevent premature convergence.
    Replace duplicates with new diverse expressions.
    """
    # Count expression frequencies
    expr_strings = [expr.to_string() for expr in population]
    expr_counts = {}
    for i, expr_str in enumerate(expr_strings):
      if expr_str not in expr_counts:
        expr_counts[expr_str] = []
      expr_counts[expr_str].append(i)

    # Find duplicates (expressions that appear more than once)
    indices_to_replace = []
    for expr_str, indices in expr_counts.items():
      if len(indices) > 1:
        # Keep the first occurrence, mark others for replacement
        indices_to_replace.extend(indices[1:])

    # Also replace some random expressions to maintain diversity
    remaining_indices = [i for i in range(len(population)) if i not in indices_to_replace]
    additional_replacements = int(len(population) * replacement_fraction)
    if additional_replacements > 0 and remaining_indices:
      import random
      additional_indices = random.sample(
        remaining_indices,
        min(additional_replacements, len(remaining_indices))
      )
      indices_to_replace.extend(additional_indices)

    # Replace marked expressions with new diverse ones
    new_population = population.copy()
    for idx in indices_to_replace:
      try:
        # Generate a new diverse expression
        if self.n_inputs is not None and self.pop_manager is not None:
          new_expr = generate_high_diversity_expression_optimized(generator, self.n_inputs)
          if self.pop_manager.is_expression_valid_cached(new_expr):
            new_population[idx] = new_expr
          else:
            # Fallback to generator's method - FIX: use correct method name
            new_node = generator.generate_random_expression()
            new_population[idx] = Expression(new_node)
        else:
          new_node = generator.generate_random_expression()
          new_population[idx] = Expression(new_node)
      except:
        # If generation fails, use a simple fallback - FIX: use correct method name
        try:
          new_node = generator.generate_random_expression()
          new_population[idx] = Expression(new_node)
        except:
          # Ultimate fallback - create a simple variable node
          from .expression_tree.core.node import VariableNode
          new_population[idx] = Expression(VariableNode(0))

    return new_population

  def _restart_population_with_great_powers(self, population: List[Expression], 
                                          fitness_scores: List[float], 
                                          generator) -> List[Expression]:
    """
    Enhanced population restart that preserves Great Powers.
    """
    from .adaptive_evolution import restart_population_enhanced
    
    # Ensure we have valid values for required parameters
    if self.n_inputs is None:
      raise ValueError("n_inputs must be set before population restart")
    if self.pop_manager is None:
      raise ValueError("pop_manager must be initialized before population restart")
    
    # Get the regular restart population
    new_population = restart_population_enhanced(
      population, fitness_scores, generator, self.population_size, self.n_inputs, self.pop_manager
    )
    
    # Inject Great Powers into the restarted population if we have them
    if len(self.great_powers) > 0:
      # Create dummy fitness scores for the new population (will be re-evaluated anyway)
      new_fitness_scores = [-10.0] * len(new_population)  # Use consistent invalid fitness value
      
      # Inject Great Powers
      new_population, _ = self.great_powers.inject_powers_into_population(
        new_population, new_fitness_scores, injection_rate=0.2  # 20% for restart
      )
      
      if self.console_log:
        injected_count = min(len(self.great_powers), max(1, int(0.2 * len(new_population))))
        print(f"  Injected {injected_count} Great Powers into restart population")
    
    return new_population
