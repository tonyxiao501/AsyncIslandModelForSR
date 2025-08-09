"""
Evolution Engine for MIMO Symbolic Regression
This module consolidates all evolution-related functionality including adaptive parameters,
population restart mechanisms, and the main evolution loop.
"""
import numpy as np
import time
import warnings
from typing import List, Dict, Any, Optional, TYPE_CHECKING
from scipy.optimize import curve_fit, OptimizeWarning

from .expression_tree import Expression
from .generator import ExpressionGenerator
from .genetic_ops import GeneticOperations
from .population_management import inject_diversity_optimized
from .logging_system import get_logger, log_milestone, log_evolution_step, log_warning, log_debug

if TYPE_CHECKING:
    from .regressor import MIMOSymbolicRegressor
    from .population_management import PopulationManager, GreatPowers


def update_adaptive_parameters(regressor, generation: int, diversity_score: float, plateau_counter: int,
                               diversity_threshold: float, mutation_rate: float, crossover_rate: float,
                               current_mutation_rate: float, current_crossover_rate: float,
                               stagnation_counter: int):
    """Enhanced adaptive parameter updates - EXACT COPY from working version"""
    # Base adaptation based on diversity and stagnation - more conservative
    if diversity_score < diversity_threshold:
        # Low diversity - increase exploration moderately
        mutation_multiplier = 1.0 + (diversity_threshold - diversity_score) * 1.0  # Reduced from 2.0
        crossover_multiplier = 0.95  # Less aggressive reduction
    else:
        # Good diversity - normal rates
        mutation_multiplier = 1.0
        crossover_multiplier = 1.0

    # Additional adaptation based on plateau - more conservative
    if plateau_counter > 20:  # Increased threshold
        mutation_multiplier *= 1.3  # Reduced from 1.5
        crossover_multiplier *= 0.85  # Less aggressive
    elif plateau_counter > 15:  # Increased threshold
        mutation_multiplier *= 1.1  # Reduced from 1.2

    # Apply multipliers with bounds - KEEP WORKING VERSION BOUNDS
    new_mutation_rate = np.clip(mutation_rate * mutation_multiplier, 0.05, 0.4)  # Use working max of 0.4
    new_crossover_rate = np.clip(crossover_rate * crossover_multiplier, 0.6, 0.95)  # Keep same

    # More gradual return to original rates when performing well
    if stagnation_counter < 3 and plateau_counter < 3:  # Stricter condition
        new_mutation_rate = (current_mutation_rate * 0.98 + mutation_rate * 0.02)  # Slower adaptation
        new_crossover_rate = (current_crossover_rate * 0.98 + crossover_rate * 0.02)
    else:
        new_mutation_rate = current_mutation_rate
        new_crossover_rate = current_crossover_rate

    return new_mutation_rate, new_crossover_rate


def restart_population_enhanced(population: List[Expression], fitness_scores: List[float],
                                generator: ExpressionGenerator, population_size: int, n_inputs: int,
                                pop_manager: 'PopulationManager'):
    """Enhanced population restart with better elite preservation - EXACT COPY from working version"""
    # Keep top performers (more aggressive selection)
    elite_count = max(2, int(population_size * 0.05))  # Keep top 5%
    elite_indices = np.argsort(fitness_scores)[-elite_count:]
    elites = [population[i].copy() for i in elite_indices]

    new_population = elites.copy()

    # Create variants of elites with different mutation strengths
    if n_inputs is None:
        raise ValueError("n_inputs must be set before creating genetic operations")
    genetic_ops = GeneticOperations(n_inputs, max_complexity=25)
    for elite in elites:
        # High mutation variants
        for _ in range(2):
            mutated = genetic_ops.mutate(elite, 0.4)
            if pop_manager.is_expression_valid_cached(mutated):
                new_population.append(mutated)

        # Medium mutation variants
        for _ in range(2):
            mutated = genetic_ops.mutate(elite, 0.2)
            if pop_manager.is_expression_valid_cached(mutated):
                new_population.append(mutated)

    # Fill rest with completely new diverse individuals
    while len(new_population) < population_size:
        new_expr = Expression(generator.generate_random_expression())
        if pop_manager.is_expression_valid_cached(new_expr):
            new_population.append(new_expr)

    return new_population[:population_size]


def should_optimize_constants_enhanced(expr_index: int, population_size: int, steps_unchanged: int, generation: int) -> bool:
    """Enhanced logic for when to optimize constants - EXACT COPY from working version"""
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


def optimize_constants(expression: Expression, X: np.ndarray, y: np.ndarray) -> bool:
    """Optimize constants in an expression using curve fitting - EXACT COPY from working version"""
    try:
        expr_vec = expression.vector_lambdify()
        if expr_vec is not None:
            with warnings.catch_warnings():
                warnings.simplefilter("error", OptimizeWarning)
                popt, pcov = curve_fit(expr_vec, X, y, expression.get_constants())
                expression.set_constants(popt)
                return True
    except (OptimizeWarning, Exception):
        pass  # failed to optimize
    return False


class EvolutionEngine:
    """
    Main evolution engine that handles the complete evolutionary process
    for symbolic regression.
    """
    
    def __init__(self, regressor: 'MIMOSymbolicRegressor', pop_manager: 'PopulationManager', 
                 great_powers: 'GreatPowers', console_log: bool = True):
        self.regressor = regressor
        self.pop_manager = pop_manager
        self.great_powers = great_powers
        self.console_log = console_log
        self.logger = get_logger()
        
    def run_evolution(self, X_scaled: np.ndarray, y_scaled: np.ndarray, 
                     X_original: np.ndarray, y_original: np.ndarray, 
                     constant_optimize: bool = False) -> List[Expression]:
        """
        Run the complete evolution process and return the best expressions.
        EXACT COPY from working version evolution.py
        """
        # Initialize generator and genetic operations
        if self.regressor.n_inputs is None:
            raise ValueError("n_inputs must be set before running evolution")
            
        generator = ExpressionGenerator(self.regressor.n_inputs, max_depth=self.regressor.max_depth)
        genetic_ops = GeneticOperations(self.regressor.n_inputs, max_complexity=20)

        # Generate initial population using population management
        from .population_management import generate_diverse_population_optimized, inject_diversity_optimized
        
        population = generate_diverse_population_optimized(
            generator, self.regressor.population_size, self.regressor.n_inputs, self.pop_manager
        )

        # Evolution tracking variables
        plateau_counter = 0
        last_best_fitness = -float('inf')
        last_update = 0

        log_milestone("Starting evolution...")

        # Main evolution loop - EXACT COPY from working version
        for generation in range(self.regressor.generations):
            # Evaluate population
            fitness_scores = self._evaluate_population_enhanced_with_scaling(population, X_scaled, y_scaled, X_original, y_original)

            # Track best fitness
            generation_best_fitness = max(fitness_scores)
            generation_avg_fitness = np.mean(fitness_scores)
            
            self.regressor.fitness_history.append(float(generation_avg_fitness))
            self.regressor.best_fitness_history.append(float(generation_best_fitness))

            # Update best expressions if improved
            if generation_best_fitness > last_best_fitness:
                last_best_fitness = generation_best_fitness
                best_idx = fitness_scores.index(generation_best_fitness)
                self.regressor.best_expressions = [population[best_idx].copy()]
                self.regressor.stagnation_counter = 0
                plateau_counter = 0
                last_update = generation
            else:
                self.regressor.stagnation_counter += 1
                plateau_counter += 1

            # Calculate population diversity
            diversity_score = self.pop_manager.calculate_population_diversity_optimized(population)['structural_diversity']
            self.regressor.diversity_history.append(diversity_score)
            self.regressor.generation_diversity_scores.append(diversity_score)

            # Update Great Powers
            self.great_powers.update_powers(population, fitness_scores, generation)

            # Early termination check
            if (self.regressor.enable_early_termination and 
                generation % self.regressor.early_termination_check_interval == 0 and
                generation_best_fitness >= self.regressor.early_termination_threshold):
                log_milestone(f"Early termination at generation {generation}: fitness {generation_best_fitness:.6f} >= {self.regressor.early_termination_threshold}")
                break

            # Late extension check
            if (self.regressor.enable_late_extension and not self.regressor.late_extension_triggered and
                generation >= self.regressor.generations - 10 and
                generation_best_fitness >= self.regressor.late_extension_threshold):
                self.regressor.generations += self.regressor.late_extension_generations
                self.regressor.late_extension_triggered = True
                log_milestone(f"Late extension triggered at generation {generation}: extending by {self.regressor.late_extension_generations} generations")

            # Inter-thread communication
            if self.regressor.inter_thread_enabled and self.regressor.shared_manager:
                if self.regressor.shared_manager.should_exchange(self.regressor.worker_id, generation):
                    population, fitness_scores = self.regressor.shared_manager.exchange_population_data(
                        self.regressor.worker_id, population, fitness_scores, generation, self.regressor.n_inputs
                    )

            # Debug CSV logging
            if self.regressor.debug_csv_path:
                self._write_debug_csv(generation, population, fitness_scores, diversity_score)

            # Progress logging with better control
            additional_info = ""
            if len(self.great_powers) > 0:
                additional_info = f"GP: {len(self.great_powers)} (best: {self.great_powers.get_best_fitness():.6f})"
                # Diagnose potential fitness drop issues
                diagnosis = self.great_powers.diagnose_fitness_drop(generation_best_fitness, generation)
                if diagnosis["status"] == "fitness_drop_detected":
                    additional_info += f" [WARN: Fitness drop! Gap={diagnosis['fitness_gap']:.6f}]"
            
            additional_info += f" Stag={self.regressor.stagnation_counter} MutRate={self.regressor.current_mutation_rate:.3f}"
            
            # Use new logging system - only log every 20 generations or final for cleaner output
            if generation % 20 == 0 or generation == self.regressor.generations - 1:
                log_evolution_step(generation, float(generation_best_fitness), float(generation_avg_fitness), float(diversity_score), additional_info)

            # Emergency Great Powers injection if significant fitness drop detected
            if len(self.great_powers) > 0:
                diagnosis = self.great_powers.diagnose_fitness_drop(generation_best_fitness, generation)
                if diagnosis["status"] == "fitness_drop_detected" and diagnosis["fitness_gap"] > 0.05:
                    if self.console_log:
                        log_warning("EMERGENCY: Injecting Great Powers due to significant fitness drop!")
                    
                    # Emergency injection of Great Powers
                    population, fitness_scores = self.great_powers.inject_powers_into_population(
                        population, fitness_scores, injection_rate=0.3
                    )
                    
                    # Re-evaluate after injection
                    fitness_scores = self._evaluate_population_enhanced_with_scaling(population, X_scaled, y_scaled, X_original, y_original)

            # Adaptive parameter adjustment
            if self.regressor.adaptive_rates:
                self.regressor.current_mutation_rate, self.regressor.current_crossover_rate = update_adaptive_parameters(
                    self.regressor, generation, diversity_score, plateau_counter,
                    self.regressor.diversity_threshold, self.regressor.mutation_rate, self.regressor.crossover_rate,
                    self.regressor.current_mutation_rate, self.regressor.current_crossover_rate, self.regressor.stagnation_counter
                )

            # Handle long-term stagnation with population restart - WORKING VERSION: 25 threshold
            if self.regressor.stagnation_counter >= 25:  # KEEP ORIGINAL 25, not 40
                log_milestone(f"Population restart at generation {generation} (stagnation: {self.regressor.stagnation_counter})")
                
                population = self._restart_population_with_great_powers(population, fitness_scores, generator)
                self.regressor.stagnation_counter = 0
                plateau_counter = 0
                continue

            # Enhanced diversity injection for moderate stagnation
            if diversity_score < 0.25:  # More aggressive threshold
                # Get protected indices from Great Powers
                protected_indices = self.great_powers.protect_elites_from_injection(
                    population, fitness_scores, elite_fraction=0.15
                )
                population = inject_diversity_optimized(
                    population, fitness_scores, generator, 0.3,  # Increased injection rate
                    self.pop_manager, self.regressor.stagnation_counter, console_log=False,
                    protected_indices=protected_indices
                )
                log_debug(f"Emergency diversity injection at generation {generation} (diversity={diversity_score:.3f}, protected={len(protected_indices)})")
            elif self.regressor.stagnation_counter > 6 and diversity_score < 0.4:  # Lowered threshold
                # Get protected indices from Great Powers
                protected_indices = self.great_powers.protect_elites_from_injection(
                    population, fitness_scores, elite_fraction=0.1
                )
                population = inject_diversity_optimized(
                    population, fitness_scores, generator, 0.2,  # Increased injection rate
                    self.pop_manager, self.regressor.stagnation_counter, console_log=False,
                    protected_indices=protected_indices
                )

            # Genetic operations (selection, crossover, mutation)
            if generation < self.regressor.generations - 1:  # Don't evolve on the last generation
                population = self._apply_genetic_operations(
                    population, fitness_scores, genetic_ops, generator, diversity_score
                )

        # Final evaluation and return best expressions
        final_fitness_scores = self._evaluate_population_enhanced_with_scaling(population, X_scaled, y_scaled, X_original, y_original)

        # Find final best expression
        final_best_fitness = max(final_fitness_scores)
        if final_best_fitness > last_best_fitness:
            best_idx = final_fitness_scores.index(final_best_fitness)
            self.regressor.best_expressions = [population[best_idx].copy()]

        log_milestone(f"Evolution completed. Final best fitness: {max(final_fitness_scores):.6f}")

        return self.regressor.best_expressions

    def _evaluate_population_enhanced_with_scaling(self, population: List[Expression], 
                                                 X_scaled: np.ndarray, y_scaled: np.ndarray,
                                                 X_original: np.ndarray, y_original: np.ndarray) -> List[float]:
        """Evaluate population with enhanced fitness calculation and scaling support"""
        fitness_scores = []
        
        for expr in population:
            try:
                # Evaluate on scaled data
                predictions = expr.evaluate(X_scaled)
                # Sanitize predictions to reduce numerical warnings
                predictions = np.asarray(predictions, dtype=float)
                if predictions.ndim == 1:
                    predictions = predictions.reshape(-1, 1)
                predictions = np.nan_to_num(predictions, nan=0.0, posinf=1e6, neginf=-1e6)
                predictions = np.clip(predictions, -1e6, 1e6)

                # Calculate R² score
                from .data_processing import r2_score
                r2 = r2_score(y_scaled.flatten(), predictions.flatten())
                
                # Apply parsimony penalty
                complexity_penalty = self.regressor.parsimony_coefficient * expr.complexity()
                fitness = r2 - complexity_penalty
                
                fitness_scores.append(fitness)
                
            except Exception as e:
                fitness_scores.append(-10.0)  # Large negative R² score for invalid expressions
                log_debug(f"Expression evaluation failed: {e}")  # Only log in debug mode
        
        return fitness_scores

    def _apply_genetic_operations(self, population: List[Expression], fitness_scores: List[float],
                                genetic_ops: GeneticOperations, generator: ExpressionGenerator, diversity_score: float) -> List[Expression]:
        """Apply selection, crossover, and mutation operations"""
        from .selection import enhanced_selection, tournament_selection
        
        new_population = []
        
        # Elite preservation
        elite_count = max(1, int(self.regressor.population_size * self.regressor.elite_fraction))
        elite_indices = np.argsort(fitness_scores)[-elite_count:]
        new_population.extend([population[i].copy() for i in elite_indices])
        
        # Generate rest of population through genetic operations
        while len(new_population) < self.regressor.population_size:
            if np.random.rand() < self.regressor.current_crossover_rate:
                # Crossover
                parent1 = enhanced_selection(population, fitness_scores, diversity_score, 
                                             self.regressor.diversity_threshold, self.regressor.tournament_size, 
                                             self.regressor.stagnation_counter)
                parent2 = enhanced_selection(population, fitness_scores, diversity_score, 
                                             self.regressor.diversity_threshold, self.regressor.tournament_size, 
                                             self.regressor.stagnation_counter)
                
                # Get crossover offspring (returns tuple of two children)
                offspring1, offspring2 = genetic_ops.crossover(parent1, parent2)
                
                # Mutation
                if np.random.rand() < self.regressor.current_mutation_rate:
                    offspring1 = genetic_ops.mutate(offspring1, self.regressor.current_mutation_rate)
                if np.random.rand() < self.regressor.current_mutation_rate:
                    offspring2 = genetic_ops.mutate(offspring2, self.regressor.current_mutation_rate)
                
                # Add both offspring if valid
                if self.pop_manager.is_expression_valid_cached(offspring1):
                    new_population.append(offspring1)
                elif self.pop_manager.is_expression_valid_cached(parent1):
                    new_population.append(parent1.copy())
                
                if len(new_population) < self.regressor.population_size and self.pop_manager.is_expression_valid_cached(offspring2):
                    new_population.append(offspring2)
                elif len(new_population) < self.regressor.population_size:
                    new_population.append(parent2.copy())
            else:
                # Direct selection and mutation
                parent = tournament_selection(population, fitness_scores, self.regressor.tournament_size, 
                                              self.regressor.stagnation_counter)
                offspring = genetic_ops.mutate(parent, self.regressor.current_mutation_rate)
                
                if self.pop_manager.is_expression_valid_cached(offspring):
                    new_population.append(offspring)
                else:
                    new_population.append(parent.copy())
        
        return new_population[:self.regressor.population_size]

    def _restart_population_with_great_powers(self, population: List[Expression], 
                                            fitness_scores: List[float], 
                                            generator: ExpressionGenerator) -> List[Expression]:
        """
        Enhanced population restart that preserves Great Powers.
        """
        # Ensure we have valid values for required parameters
        if self.regressor.n_inputs is None:
            raise ValueError("n_inputs must be set before population restart")
        if self.pop_manager is None:
            raise ValueError("pop_manager must be initialized before population restart")
        
        # Get the regular restart population
        new_population = restart_population_enhanced(
            population, fitness_scores, generator, self.regressor.population_size, 
            self.regressor.n_inputs, self.pop_manager
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

    def _write_debug_csv(self, generation: int, population: List[Expression], 
                        fitness_scores: List[float], diversity_score: float):
        """Write the top 10 expressions from this generation to the debug CSV file"""
        if not self.regressor.debug_csv_path or self.regressor.debug_worker_id is None:
            return

        try:
            import csv
            import datetime

            # Get top 10 expressions by fitness
            top_indices = sorted(range(len(fitness_scores)),
                                key=lambda i: fitness_scores[i], reverse=True)[:10]

            # Write to CSV file (append mode)
            with open(self.regressor.debug_csv_path, 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)

                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                for rank, idx in enumerate(top_indices, 1):
                    expr = population[idx]
                    fitness = fitness_scores[idx]
                    complexity = expr.complexity()
                    expression_str = expr.to_string()

                    writer.writerow([
                        timestamp,
                        self.regressor.debug_worker_id,
                        generation,
                        rank,
                        f"{fitness:.6f}",
                        f"{complexity:.3f}",
                        expression_str,
                        f"{diversity_score:.3f}",
                        self.regressor.stagnation_counter
                    ])

        except Exception as e:
            # Fail silently to not disrupt evolution
            if self.console_log:
                print(f"Debug CSV write failed: {e}")
