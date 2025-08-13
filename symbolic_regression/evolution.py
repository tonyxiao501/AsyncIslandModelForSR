"""
Evolution Engine for MIMO Symbolic Regression
This module consolidates all evolution-related functionality including adaptive parameters,
population restart mechanisms, and the main evolution loop.
"""
import numpy as np
import time
import warnings
from typing import List, Dict, Any, Optional, TYPE_CHECKING, Tuple
from scipy.optimize import curve_fit, OptimizeWarning

from .expression_tree import Expression
from .generator import ExpressionGenerator
from .genetic_ops import GeneticOperations
from .population_management import inject_diversity_optimized
from .logging_system import get_logger, log_milestone, log_evolution_step, log_warning, log_debug
from .pareto import ParetoFront, ParetoItem
from .selection import enhanced_selection, tournament_selection, epsilon_lexicase_selection

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

        # Optional: Initialize Pareto tracking if requested on regressor
        pareto: Optional[ParetoFront] = None
        if getattr(self.regressor, 'enable_pareto_tracking', False):
            pareto = ParetoFront(capacity=getattr(self.regressor, 'pareto_capacity', 256))

        # Main evolution loop - EXACT COPY from working version
        for generation in range(self.regressor.generations):
            # Evaluate population and cache predictions/R² once per generation
            fitness_scores, pred_matrix, r2_list = self._evaluate_population_with_predictions(
                population, X_scaled, y_scaled, generation
            )
            # Update Pareto front (error = 1 - r2 approx from fitness + parsimony)
            if pareto is not None:
                yt = y_scaled.flatten()
                for i, expr in enumerate(population):
                    try:
                        r2 = float(r2_list[i])
                        err = float(max(0.0, 1.0 - r2))
                    except Exception:
                        err = 2.0
                    pareto.add(ParetoItem(
                        error=err,
                        complexity=float(expr.complexity()),
                        expression_str=expr.to_string(),
                        generation=generation,
                    ))

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

            # Asynchronous island cache migration (if configured)
            if getattr(self.regressor, '_async_cache_manager', None) is not None and getattr(self.regressor, '_async_island_id', None) is not None:
                try:
                    # Narrow types for type checker and access cache manager safely
                    from typing import cast
                    from .async_island_cache import AsynchronousIslandCacheManager, CachedExpression
                    assert self.regressor._async_cache_manager is not None
                    assert self.regressor._async_island_id is not None
                    cache_mgr = cast(AsynchronousIslandCacheManager, self.regressor._async_cache_manager)
                    island_id = int(self.regressor._async_island_id)
                    timer = getattr(self.regressor, '_async_migration_timer', None)

                    # Decide to send: export current top expressions to local island cache and file
                    can_send = True
                    if timer is not None:
                        try:
                            can_send = timer.should_send(generation, int(self.regressor._async_last_send_gen), self.regressor.best_fitness_history)
                        except Exception:
                            can_send = (generation - int(self.regressor._async_last_send_gen)) >= 15
                    if can_send:
                        # Share top-k expressions into island cache
                        k = max(2, len(population) // 20)  # send fewer to cut overhead
                        top_idx = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i], reverse=True)[:k]
                        for idx in top_idx:
                            expr = population[idx]
                            try:
                                ce = CachedExpression(
                                    expression_str=expr.to_string(),
                                    fitness=float(fitness_scores[idx]),
                                    complexity=float(expr.complexity()),
                                    from_island=island_id,
                                    timestamp=time.time(),
                                    generation=generation,
                                    is_initial_seed=False
                                )
                                cache_mgr.island_share_expression(island_id, ce)
                                try:
                                    self.regressor._async_stats['shared'] += 1
                                except Exception:
                                    pass
                            except Exception:
                                continue
                        # Export snapshot for other processes (throttled)
                        try:
                            # Only export 1 out of 3 sends to cut disk I/O
                            if (self.regressor._async_stats.get('send_events', 0) % 3) == 0:
                                cache_mgr.export_island_cache(island_id)
                                try:
                                    self.regressor._async_stats['exports'] += 1
                                except Exception:
                                    pass
                        except Exception:
                            pass
                        self.regressor._async_last_send_gen = generation
                        try:
                            self.regressor._async_stats['send_events'] += 1
                        except Exception:
                            pass

                    # Decide to receive: import candidates and merge
                    can_recv = True
                    if timer is not None:
                        try:
                            diversity_score = self.regressor.diversity_history[-1] if self.regressor.diversity_history else 1.0
                            can_recv = timer.should_receive(generation, int(self.regressor._async_last_recv_gen), float(diversity_score))
                        except Exception:
                            can_recv = (generation - int(self.regressor._async_last_recv_gen)) >= 15
                    if can_recv:
                        # Pull from in-memory topology caches first
                        candidates = cache_mgr.island_import_expressions(island_id, max_count=2)
                        try:
                            self.regressor._async_stats['received_candidates'] += len(candidates)
                        except Exception:
                            pass
                        # Also pull from file-backed caches if available
                        try:
                            file_candidates = cache_mgr.import_foreign_candidates(island_id, max_total=2)
                            candidates.extend(file_candidates)
                            try:
                                self.regressor._async_stats['file_candidates'] += len(file_candidates)
                            except Exception:
                                pass
                        except Exception:
                            pass

                        if candidates:
                            # Reconstruct expressions and inject by replacing worst individuals
                            from .expression_tree import Expression
                            rebuilt = []
                            for c in candidates:
                                try:
                                    rebuilt.append((Expression.from_string(c.expression_str, self.regressor.n_inputs), float(c.fitness)))
                                    try:
                                        self.regressor._async_stats['rebuilt'] += 1
                                    except Exception:
                                        pass
                                except Exception:
                                    continue
                            if rebuilt:
                                # Prepare indices to replace: worst by fitness
                                worst_idx = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i])[:len(rebuilt)]
                                for tgt, (expr_obj, fit_hint) in zip(worst_idx, rebuilt):
                                    if self.pop_manager.is_expression_valid_cached(expr_obj):
                                        population[tgt] = expr_obj
                                        # Defer exact recomputation to next loop; keep current fitness
                                        # to avoid immediate full re-eval overhead
                                        try:
                                            self.regressor._async_stats['injected'] += 1
                                        except Exception:
                                            pass
                                self.regressor._async_last_recv_gen = generation
                                try:
                                    self.regressor._async_stats['receive_events'] += 1
                                except Exception:
                                    pass
                except Exception:
                    # Best-effort; async path must not break evolution
                    pass

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
                    
                    # Re-evaluate after injection and refresh caches
                    fitness_scores, pred_matrix, r2_list = self._evaluate_population_with_predictions(
                        population, X_scaled, y_scaled, generation
                    )

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
                # After modifying population, refresh fitness and caches for consistent selection
                fitness_scores, pred_matrix, r2_list = self._evaluate_population_with_predictions(
                    population, X_scaled, y_scaled, generation
                )
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
                # After modifying population, refresh fitness and caches for consistent selection
                fitness_scores, pred_matrix, r2_list = self._evaluate_population_with_predictions(
                    population, X_scaled, y_scaled, generation
                )

            # Genetic operations (selection, crossover, mutation)
            if generation < self.regressor.generations - 1:  # Don't evolve on the last generation
                population = self._apply_genetic_operations(
                    population, fitness_scores, genetic_ops, generator, diversity_score,
                    pred_matrix if getattr(self.regressor, 'use_lexicase', False) else None,
                    y_scaled if getattr(self.regressor, 'use_lexicase', False) else None
                )

        # Final evaluation and return best expressions
        final_fitness_scores, _, _ = self._evaluate_population_with_predictions(population, X_scaled, y_scaled, generation)

        # Find final best expression
        final_best_fitness = max(final_fitness_scores)
        if final_best_fitness > last_best_fitness:
            best_idx = final_fitness_scores.index(final_best_fitness)
            self.regressor.best_expressions = [population[best_idx].copy()]

        log_milestone(f"Evolution completed. Final best fitness: {max(final_fitness_scores):.6f}")

        # Final Pareto dump if enabled
        if pareto is not None and getattr(self.regressor, 'pareto_csv_path', None):
            try:
                pareto.to_csv(getattr(self.regressor, 'pareto_csv_path'))
            except Exception:
                pass

        return self.regressor.best_expressions

    def _evaluate_population_with_predictions(self, population: List[Expression],
                                              X_scaled: np.ndarray,
                                              y_scaled: np.ndarray,
                                              generation: int) -> Tuple[List[float], np.ndarray, np.ndarray]:
        """Evaluate population and return fitness scores, predictions matrix, and per-individual R².

        This consolidates evaluation to avoid repeated predictions for lexicase selection
        and Pareto updates within the same generation.
        """
        fitness_scores: List[float] = []
        preds_list: List[np.ndarray] = []
        r2_list: List[float] = []

        n_samples = X_scaled.shape[0]

        from .data_processing import r2_score, mae_loss, huber_loss

        for expr in population:
            try:
                pred = expr.evaluate(X_scaled)
                pred = np.asarray(pred, dtype=float)
                if pred.ndim == 1:
                    pred = pred.reshape(-1, 1)
                pred = np.nan_to_num(pred, nan=0.0, posinf=1e6, neginf=-1e6)
                pred = np.clip(pred, -1e6, 1e6)

                # Compute per-output metric and aggregate across outputs
                if pred.shape[1] > 1 and y_scaled.ndim == 2 and y_scaled.shape[1] > 1:
                    # Align shapes
                    y_cols = y_scaled.shape[1]
                    p_cols = pred.shape[1]
                    m = min(y_cols, p_cols)
                    if self.regressor.loss == 'r2':
                        r2_vals = [r2_score(y_scaled[:, i], pred[:, i]) for i in range(m)]
                        score_agg = float(np.mean(r2_vals)) if m > 0 else -10.0
                    elif self.regressor.loss == 'mae':
                        maes = [mae_loss(y_scaled[:, i], pred[:, i]) for i in range(m)]
                        score_agg = -float(np.mean(maes)) if m > 0 else -10.0
                    else:  # huber
                        hub = [huber_loss(y_scaled[:, i], pred[:, i], delta=float(getattr(self.regressor, 'huber_delta', 1.0))) for i in range(m)]
                        score_agg = -float(np.mean(hub)) if m > 0 else -10.0
                else:
                    # Single-output or fallback flatten
                    if self.regressor.loss == 'r2':
                        score_agg = float(r2_score(y_scaled.flatten(), pred.flatten()))
                    elif self.regressor.loss == 'mae':
                        score_agg = -float(mae_loss(y_scaled.flatten(), pred.flatten()))
                    else:
                        score_agg = -float(huber_loss(y_scaled.flatten(), pred.flatten(), delta=float(getattr(self.regressor, 'huber_delta', 1.0))))

                # Adaptive parsimony coefficient (fallback to fixed if unavailable)
                _ps = getattr(self.regressor, '_parsimony_system', None)
                if _ps is not None:
                    diversity_hint = self.regressor.diversity_history[-1] if self.regressor.diversity_history else 1.0
                    parsimony_coeff = float(_ps.get_adaptive_coefficient(
                        generation, int(self.regressor.generations), float(diversity_hint)
                    ))
                else:
                    parsimony_coeff = float(self.regressor.parsimony_coefficient)

                complexity_penalty = parsimony_coeff * expr.complexity()
                fitness = float(score_agg - complexity_penalty)

                fitness_scores.append(fitness)
                # Store per-sample mean prediction for lexicase error construction
                preds_list.append(np.mean(pred, axis=1) if pred.ndim == 2 else pred.reshape(n_samples))
                r2_list.append(float(score_agg))
            except Exception:
                # Invalid expression
                fitness_scores.append(-10.0)
                preds_list.append(np.zeros(n_samples, dtype=float))
                r2_list.append(0.0)

        pred_matrix = np.stack(preds_list, axis=0) if preds_list else np.zeros((0, X_scaled.shape[0]))
        return fitness_scores, pred_matrix, np.asarray(r2_list, dtype=float)

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
                                  genetic_ops: GeneticOperations, generator: ExpressionGenerator, diversity_score: float,
                                  preds_for_lexicase: Optional[np.ndarray] = None,
                                  y_for_lexicase: Optional[np.ndarray] = None) -> List[Expression]:
        """Apply selection, crossover, and mutation operations"""
        
        
        new_population = []
        # Precompute errors matrix once if using lexicase with informative subsampling and sticky bags
        errors = None
        if preds_for_lexicase is not None and y_for_lexicase is not None and getattr(self.regressor, 'use_lexicase', False):
            try:
                # Build per-sample target as mean across outputs for lexicase
                if y_for_lexicase.ndim == 1:
                    Y = np.asarray(y_for_lexicase, dtype=float).flatten()
                else:
                    Y = np.asarray(np.mean(y_for_lexicase, axis=1), dtype=float).flatten()
                full_errors = np.abs(preds_for_lexicase - Y[None, :])  # (n_individuals, n_cases)
                n_cases = full_errors.shape[1]

                # Determine desired bag size k
                k = getattr(self.regressor, 'lexicase_case_subsample', None)
                frac = getattr(self.regressor, 'lexicase_case_fraction', None)
                # Support explicit full-case modes
                if isinstance(k, str) and k.lower() == 'all':
                    k = n_cases
                if k is None and frac is not None:
                    if isinstance(frac, (int, float)) and frac >= 1.0:
                        k = n_cases
                    else:
                        k = max(1, int(round(float(frac) * n_cases)))
                # Apply default safety: k = min(128, n_cases) if k invalid
                if not isinstance(k, int) or k <= 0 or k > n_cases:
                    k = min(128, n_cases)

                # Sticky bag: reuse indices for several generations
                sticky_T = int(getattr(self.regressor, 'lexicase_bag_sticky_generations', 0) or 0)
                if sticky_T > 0:
                    if not hasattr(self.regressor, '_lexicase_bag_state'):
                        setattr(self.regressor, '_lexicase_bag_state', {'indices': None, 'age': 0})
                    bag = getattr(self.regressor, '_lexicase_bag_state')
                else:
                    bag = {'indices': None, 'age': 0}

                need_new_bag = bag['indices'] is None or bag['age'] >= sticky_T
                if need_new_bag:
                    # Informative (disagreement-aware) sampling: rank cases by std of predictions across individuals
                    # Use preds_for_lexicase (n_individuals, n_cases)
                    try:
                        pred_std = np.std(preds_for_lexicase, axis=0)
                    except Exception:
                        pred_std = np.random.rand(n_cases)
                    informative_frac = float(getattr(self.regressor, 'lexicase_informative_fraction', 0.8) or 0.8)
                    informative_k = int(round(informative_frac * k))
                    informative_k = max(0, min(k, informative_k))

                    # Top informative_k by std
                    if informative_k > 0:
                        top_idx = np.argpartition(-pred_std, kth=min(informative_k - 1, n_cases - 1))[:informative_k]
                    else:
                        top_idx = np.array([], dtype=int)
                    # Fill remaining with random choices from the rest
                    remaining = k - informative_k
                    if remaining > 0:
                        mask = np.ones(n_cases, dtype=bool)
                        mask[top_idx] = False
                        pool = np.where(mask)[0]
                        if pool.size >= remaining:
                            rand_idx = np.random.choice(pool, size=remaining, replace=False)
                        else:
                            rand_idx = pool
                        indices = np.concatenate([top_idx, rand_idx]) if top_idx.size else rand_idx
                    else:
                        indices = top_idx

                    # Shuffle indices to avoid order bias in lexicase
                    if indices.size > 1:
                        np.random.shuffle(indices)

                    bag['indices'] = indices
                    bag['age'] = 0
                else:
                    indices = bag['indices']
                    bag['age'] += 1

                # Slice errors by bag indices
                if indices is not None and indices.size > 0:
                    errors = full_errors[:, indices]
                else:
                    errors = full_errors
                # Persist bag state back to regressor if sticky
                if getattr(self.regressor, 'lexicase_bag_sticky_generations', 0):
                    setattr(self.regressor, '_lexicase_bag_state', bag)
            except Exception:
                errors = None
        
        # Elite preservation
        elite_count = max(1, int(self.regressor.population_size * self.regressor.elite_fraction))
        elite_indices = np.argsort(fitness_scores)[-elite_count:]
        new_population.extend([population[i].copy() for i in elite_indices])
        
        # Generate rest of population through genetic operations
        while len(new_population) < self.regressor.population_size:
            if np.random.rand() < self.regressor.current_crossover_rate:
                # Crossover
                if errors is not None:
                    eps = getattr(self.regressor, 'lexicase_epsilon', None)
                    parent1 = epsilon_lexicase_selection(population, errors, epsilon=eps)
                    parent2 = epsilon_lexicase_selection(population, errors, epsilon=eps)
                else:
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
                if errors is not None:
                    eps = getattr(self.regressor, 'lexicase_epsilon', None)
                    parent = epsilon_lexicase_selection(population, errors, epsilon=eps)
                else:
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
