"""
Ensemble MIMO Symbolic Regression Implementation
This module consolidates ensemble functionality, including parallel processing,
worker coordination, and shared population management.
"""
import numpy as np
import os
import time
import random
import tempfile
import pickle
import multiprocessing
import warnings
from typing import List, Dict, Optional, Tuple, Any, TYPE_CHECKING

from .expression_tree import Expression
from .expression_tree.optimization.memory_pool import reset_global_pool

if TYPE_CHECKING:
    from .data_processing import DataScaler


# Worker function for multiprocessing
def _fit_worker(config: tuple):
    """
    Optimized worker function for multiprocessing with minimal synchronization overhead.
    Each process runs independently with unique parameters for diversity.
    """
    # Import here to avoid circular import
    from .regressor import MIMOSymbolicRegressor

    regressor_params, X, y, constant_optimize, shared_manager, worker_id, debug_csv_path = config

    # Reset the global pool for this process to avoid multiprocessing conflicts
    reset_global_pool()

    # Each process must have its own random seed to ensure diversity in runs
    # Use a combination of worker_id and process ID for uniqueness
    seed = hash((worker_id, os.getpid())) % 2**32
    random.seed(seed)
    np.random.seed(seed % 2**32)

    try:
        # Instantiate the regressor for this process
        reg = MIMOSymbolicRegressor(**regressor_params)

        # Only enable inter-thread communication if shared manager is provided
        # This avoids all synchronization overhead when not needed
        if shared_manager is not None:
            reg.enable_inter_thread_communication(shared_manager, worker_id)

        # Set debug CSV path if provided (minimal overhead)
        if debug_csv_path is not None:
            reg.set_debug_csv_path(debug_csv_path, worker_id)

        reg.fit(X, y, constant_optimize=constant_optimize)
        
        return reg
    except KeyboardInterrupt:
        return None
    except Exception as e:
        print(f"Worker process {worker_id} failed: {str(e)}")
        return None


class ImprovedSharedData:
    """
    Improved file-based sharing with better expression transfer and reduced overhead.
    """

    def __init__(self, n_workers: int, exchange_interval: int = 20,
                 purge_percentage: float = 0.10, import_percentage: float = 0.05):
        self.n_workers = n_workers
        self.exchange_interval = exchange_interval
        self.purge_percentage = purge_percentage
        self.import_percentage = import_percentage

        # Create temporary directory for sharing data
        self.temp_dir = tempfile.mkdtemp(prefix="symbolic_regression_comm_")
        self.lock_file = os.path.join(self.temp_dir, "exchange.lock")

        # Track exchanges to avoid too frequent operations
        self.last_exchange = {}

    def should_exchange(self, worker_id: int, generation: int) -> bool:
        """Check if it's time for this worker to exchange expressions"""
        if generation < self.exchange_interval:
            return False

        # Add randomization to prevent all workers exchanging simultaneously
        if generation % self.exchange_interval == (worker_id % 5):
            return True
        return False

    def exchange_population_data(self, worker_id: int, population: List,
                                 fitness_scores: List[float], generation: int,
                                 n_inputs: int) -> Tuple[List, List[float]]:
        """
        Exchange population data with other workers using improved strategy.
        """
        try:
            # Simple lock mechanism with timeout
            lock_acquired = self._acquire_lock(worker_id, timeout=2.0)
            if not lock_acquired:
                return population, fitness_scores

            try:
                # Read existing data from other workers
                foreign_expressions = self._read_foreign_expressions(worker_id, generation)

                # Save current worker's best expressions (more efficiently)
                self._save_worker_expressions(worker_id, population, fitness_scores, generation)

                # Perform exchange if we have good foreign expressions
                if foreign_expressions:
                    return self._perform_improved_exchange(
                        population, fitness_scores, foreign_expressions, n_inputs
                    )

            finally:
                self._release_lock()

        except Exception as e:
            # Fail silently to not disrupt evolution
            pass

        return population, fitness_scores

    def _acquire_lock(self, worker_id: int, timeout: float = 2.0) -> bool:
        """Acquire file lock with timeout"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                if not os.path.exists(self.lock_file):
                    with open(self.lock_file, 'w') as f:
                        f.write(str(worker_id))
                    time.sleep(0.01)  # Small delay to check if we got the lock

                    # Verify we got the lock
                    if os.path.exists(self.lock_file):
                        with open(self.lock_file, 'r') as f:
                            lock_owner = f.read().strip()
                        if lock_owner == str(worker_id):
                            return True

                time.sleep(0.05)
            except:
                continue
        return False

    def _release_lock(self):
        """Release file lock"""
        try:
            if os.path.exists(self.lock_file):
                os.remove(self.lock_file)
        except:
            pass

    def _read_foreign_expressions(self, worker_id: int, generation: int) -> List[Dict]:
        """Read expressions from other workers"""
        foreign_expressions = []

        # Look for recent expression files from other workers
        for other_worker_id in range(self.n_workers):
            if other_worker_id == worker_id:
                continue

            for gen_offset in range(0, min(5, generation)):
                check_gen = generation - gen_offset
                worker_file = os.path.join(
                    self.temp_dir,
                    f"worker_{other_worker_id}_gen_{check_gen}.pkl"
                )

                if os.path.exists(worker_file):
                    try:
                        with open(worker_file, 'rb') as f:
                            worker_data = pickle.load(f)
                            foreign_expressions.extend(worker_data)
                        break  # Found expressions for this worker
                    except:
                        continue

        return foreign_expressions

    def _save_worker_expressions(self, worker_id: int, population: List,
                                 fitness_scores: List[float], generation: int):
        """Save current worker's best expressions efficiently"""
        try:
            # Only save top 20% of expressions to reduce file size
            n_to_save = max(5, len(population) // 5)

            # Get indices of best expressions
            sorted_indices = sorted(range(len(fitness_scores)),
                                    key=lambda i: fitness_scores[i], reverse=True)
            best_indices = sorted_indices[:n_to_save]

            # Save expression data
            expressions_to_save = []
            for idx in best_indices:
                expr = population[idx]
                fitness = fitness_scores[idx]

                # Store the actual expression object and string
                expressions_to_save.append({
                    'expression_str': expr.to_string(),
                    'fitness': fitness,
                    'complexity': expr.complexity(),
                    'worker_id': worker_id,
                    'generation': generation,
                    'expression_obj': expr  # Store the actual object for reconstruction
                })

            # Save to file
            worker_file = os.path.join(self.temp_dir, f"worker_{worker_id}_gen_{generation}.pkl")
            with open(worker_file, 'wb') as f:
                pickle.dump(expressions_to_save, f)

            # Clean up old files to prevent disk space issues
            self._cleanup_old_files(worker_id, generation)

        except Exception:
            pass  # Fail silently

    def _cleanup_old_files(self, worker_id: int, current_generation: int):
        """Remove old expression files to save disk space"""
        try:
            cleanup_before = current_generation - 10
            for gen in range(max(0, cleanup_before - 5), cleanup_before):
                old_file = os.path.join(self.temp_dir, f"worker_{worker_id}_gen_{gen}.pkl")
                if os.path.exists(old_file):
                    os.remove(old_file)
        except:
            pass

    def _perform_improved_exchange(self, population: List, fitness_scores: List[float],
                                   foreign_expressions: List[Dict], n_inputs: int) -> Tuple[List, List[float]]:
        """Perform exchange with actual expression objects"""
        population_size = len(population)
        num_to_purge = max(1, int(population_size * self.purge_percentage))
        num_to_import = max(1, int(population_size * self.import_percentage))

        if not foreign_expressions:
            return population, fitness_scores

        # Sort foreign expressions by fitness (descending)
        foreign_expressions.sort(key=lambda x: x['fitness'], reverse=True)

        # Filter out expressions that are too similar to current population
        filtered_foreign = self._filter_diverse_expressions(
            foreign_expressions, population, num_to_import * 2
        )

        if not filtered_foreign:
            return population, fitness_scores

        # Take the best diverse expressions
        expressions_to_import = filtered_foreign[:num_to_import]

        # Remove worst expressions from current population
        sorted_indices = sorted(range(len(fitness_scores)),
                                key=lambda i: fitness_scores[i], reverse=True)
        indices_to_keep = sorted_indices[:-num_to_purge]

        new_population = [population[i] for i in indices_to_keep]
        new_fitness_scores = [fitness_scores[i] for i in indices_to_keep]

        # Import the actual expressions (or create similar ones)
        for expr_data in expressions_to_import:
            try:
                # Try to use the actual expression object if available
                if 'expression_obj' in expr_data and expr_data['expression_obj'] is not None:
                    imported_expr = expr_data['expression_obj'].copy()
                else:
                    # Use the new from_string method to properly reconstruct the expression
                    from .expression_tree import Expression
                    imported_expr = Expression.from_string(expr_data['expression_str'], n_inputs)

                new_population.append(imported_expr)
                # Use the original fitness (no penalty for good expressions!)
                new_fitness_scores.append(expr_data['fitness'])

            except Exception:
                # If import fails, skip this expression
                continue

        return new_population, new_fitness_scores

    def _filter_diverse_expressions(self, foreign_expressions: List[Dict],
                                    current_population: List, max_count: int) -> List[Dict]:
        """Filter foreign expressions to ensure diversity"""
        # Simple diversity filter based on expression strings
        current_strings = {expr.to_string() for expr in current_population}

        diverse_expressions = []
        seen_strings = set()

        for expr_data in foreign_expressions:
            expr_str = expr_data['expression_str']

            # Skip if too similar to current population or already seen
            if expr_str in current_strings or expr_str in seen_strings:
                continue

            # Skip if complexity is too different (avoid overly complex imports)
            if expr_data['complexity'] > 15:  # Reasonable complexity limit
                continue

            diverse_expressions.append(expr_data)
            seen_strings.add(expr_str)

            if len(diverse_expressions) >= max_count:
                break

        return diverse_expressions


def create_improved_shared_data(n_workers: int, exchange_interval: int = 15,
                               purge_percentage: float = 0.15, import_percentage: float = 0.08):
    """Create an improved shared data object that can be pickled with optimized exchange parameters"""
    return {
        'n_workers': n_workers,
        'exchange_interval': exchange_interval,
        'purge_percentage': purge_percentage,
        'import_percentage': import_percentage,
        'temp_dir': tempfile.mkdtemp(prefix="symbolic_regression_improved_"),
        'type': 'improved'
    }


class EnsembleMIMORegressor:
    """
    Runs multiple MIMOSymbolicRegressor fits concurrently and selects an
    ensemble of the best overall expressions.
    """
    def __init__(self, n_fits: int = 8, top_n_select: int = 5,
                 enable_inter_thread_communication: bool = True,
                 exchange_interval: int = 10, purge_percentage: float = 0.15,
                 import_percentage: float = 0.03, debug_csv_path: Optional[str] = None,
                 # Ensemble members work with raw data - no scaling 
                 **regressor_kwargs):
        """
        Initializes the Ensemble Regressor with optional inter-thread communication.

        Args:
            n_fits (int): The number of regressor fits to run concurrently.
                          Defaults to 8 as requested.
            top_n_select (int): The number of best expressions to select from all
                                runs. Defaults to 5 as requested.
            enable_inter_thread_communication (bool): Whether to enable communication
                                                      between worker threads.
            exchange_interval (int): How often (in generations) workers exchange expressions.
            purge_percentage (float): Percentage of worst expressions to remove (0.14-0.21).
            import_percentage (float): Percentage of best expressions to import from other workers.
            debug_csv_path (str): Path to a CSV file for debugging purposes, to log
                                  the progress and results of each fit. If None, logging
                                  to CSV is disabled.
            **regressor_kwargs: Keyword arguments to be passed to each
                                underlying MIMOSymbolicRegressor instance.
        """
        if not isinstance(n_fits, int) or n_fits <= 0:
            raise ValueError("n_fits must be a positive integer.")
        if not isinstance(top_n_select, int) or top_n_select <= 0:
            raise ValueError("top_n_select must be a positive integer.")
        if top_n_select > n_fits:
            raise ValueError("top_n_select cannot be greater than n_fits.")

        self.n_fits = n_fits
        self.top_n_select = top_n_select
        self.regressor_kwargs = regressor_kwargs
        self.best_expressions: List[Expression] = []
        self.best_fitnesses: List[float] = []
        self.all_results: List[Dict] = []

        # Inter-thread communication parameters
        self.enable_inter_thread_communication = enable_inter_thread_communication
        self.exchange_interval = exchange_interval
        self.purge_percentage = purge_percentage
        self.import_percentage = import_percentage

        # Debug CSV path
        self.debug_csv_path = debug_csv_path
        
        # REMOVED: All scaling functionality deprecated
        # Ensemble now works with raw data only

    def fit(self, X: np.ndarray, y: np.ndarray, constant_optimize: bool = False):
        """
        Fits 'n_fits' regressors concurrently using multiprocessing, then selects
        the top 'top_n_select' expressions based on their final fitness scores.

        IMPORTANT: Due to the use of 'multiprocessing' on some platforms (like
        Windows), this method should be called from within a
        `if __name__ == "__main__":` block in your script.
        """
        # DEPRECATED: Scaling functionality removed  
        # All data is now processed in raw form for better physical interpretation
        print(f"Starting ensemble fit with {self.n_fits} concurrent regressors...")
        print("Using raw data without scaling for better physical interpretability.")

        # Create shared data manager for inter-thread communication if enabled
        shared_manager = None
        if self.enable_inter_thread_communication:
            shared_manager = ImprovedSharedData(
                self.n_fits, self.exchange_interval, self.purge_percentage, self.import_percentage
            )
            print(f"Inter-thread communication enabled: exchange every {self.exchange_interval} generations")

        # Prepare configurations for each worker
        worker_configs = []
        for worker_id in range(self.n_fits):
            config = (
                self.regressor_kwargs.copy(),
                X, y, constant_optimize,
                shared_manager, worker_id, self.debug_csv_path
            )
            worker_configs.append(config)

        # Execute workers in parallel using multiprocessing
        start_time = time.time()
        with multiprocessing.Pool(processes=self.n_fits) as pool:
            fitted_regressors = pool.map(_fit_worker, worker_configs)

        # Filter out any failed workers
        fitted_regressors = [reg for reg in fitted_regressors if reg is not None]
        
        if not fitted_regressors:
            raise RuntimeError("All worker processes failed. Unable to fit ensemble.")

        print(f"Ensemble fitting completed in {time.time() - start_time:.2f}s")
        print(f"Successfully fitted {len(fitted_regressors)}/{self.n_fits} regressors")

        # Store fitted regressors for later use
        self.fitted_regressors = fitted_regressors

        # Collect all results from all runs
        all_run_results = []
        for run, regressor in enumerate(fitted_regressors):
            for expr in regressor.best_expressions:
                all_run_results.append({
                    'expression_obj': expr,
                    'expression_str': expr.to_string(),
                    'fitness': regressor.fitness_history[-1] if regressor.fitness_history else 0.0,
                    'complexity': expr.complexity(),
                    'run': run,
                    'regressor': regressor
                })

        # Sort by fitness (descending) and select top expressions
        all_run_results.sort(key=lambda x: x['fitness'], reverse=True)
        top_results = all_run_results[:self.top_n_select]

        # Apply final optimization if requested
        if constant_optimize:
            print("Applying final constant optimization to top expressions...")
            start_time = time.time()
            
            # Get the expressions to optimize
            candidate_expressions = [res['expression_obj'] for res in top_results]
            
            # Apply optimization using the utilities module
            from .utilities import optimize_final_expressions
            
            # Work with raw data (no scaling)
            optimized_expressions = optimize_final_expressions(candidate_expressions, X, y)
            from .utilities import evaluate_optimized_expressions
            parsimony_coeff = self.regressor_kwargs.get('parsimony_coefficient', 0.001)
            optimized_fitness_scores = evaluate_optimized_expressions(optimized_expressions, X, y, parsimony_coeff)
            
            # Update results with optimized fitness scores and re-rank
            for i, (optimized_expr, new_fitness) in enumerate(zip(optimized_expressions, optimized_fitness_scores)):
                if i < len(top_results):
                    top_results[i]['expression_obj'] = optimized_expr
                    top_results[i]['fitness'] = new_fitness
                    top_results[i]['expression_str'] = optimized_expr.to_string()
                    top_results[i]['complexity'] = optimized_expr.complexity()
            
            # Re-sort by new optimized fitness scores
            top_results.sort(key=lambda item: item['fitness'], reverse=True)
            
            optimization_time = time.time() - start_time
            improvement_count = sum(1 for i, new_fitness in enumerate(optimized_fitness_scores) 
                                  if i < len(all_run_results[:self.top_n_select]) and 
                                  new_fitness > all_run_results[i]['fitness'])
            
            print(f"Final optimization completed in {optimization_time:.2f}s")
            print(f"Optimization improved {improvement_count}/{len(optimized_expressions)} expressions")

        self.best_expressions = [res['expression_obj'] for res in top_results]
        self.best_fitnesses = [res['fitness'] for res in top_results]
        self.all_results = all_run_results

        # Print a summary of the best expressions found
        print(f"\nEnsemble fitting complete. Top {len(self.best_expressions)} of {len(all_run_results)} expressions selected:")
        for i, res in enumerate(top_results):
            print(f"  {i+1}. Fitness: {res['fitness']:.6f}, "
                  f"Complexity: {res['complexity']:.2f}, "
                  f"From Run: {res['run']}, "
                  f"Expression: {res['expression_str']}")

        if constant_optimize:
            improvement_count = sum(1 for i, new_fitness in enumerate([res['fitness'] for res in top_results]) 
                                  if i < len(all_run_results[:self.top_n_select]) and 
                                  new_fitness > all_run_results[i]['fitness'])
            print(f"Final optimization improved {improvement_count} expressions out of {len(top_results)} candidates.")

        if self.enable_inter_thread_communication:
            print(f"\nInter-thread communication summary:")
            print(f"Workers exchanged expressions every {self.exchange_interval} generations")
            print(f"This should have improved population diversity across threads")

    def predict(self, X: np.ndarray, strategy: str = 'mean') -> np.ndarray:
        """
        Makes predictions using the ensemble of best expressions.
        Handles data scaling if it was used during training.

        Args:
            X (np.ndarray): Input data for prediction.
            strategy (str): The strategy to combine predictions from the
                            expressions in the ensemble.
                            - 'mean': Average the predictions of all expressions (default).
                            - 'best_only': Use only the single best expression.

        Returns:
            np.ndarray: The predicted values.
        """
        if not self.best_expressions:
            raise ValueError("Model has not been fitted yet. Call fit() first.")

        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # Use raw data (no scaling)
        if strategy == 'best_only':
            # Use only the single best expression (the first in the sorted list)
            return self.best_expressions[0].evaluate(X)

        elif strategy == 'mean':
            # Collect predictions from all selected expressions
            all_predictions = []
            for expr in self.best_expressions:
                pred = expr.evaluate(X)
                if pred.ndim == 1:
                    pred = pred.reshape(-1, 1)
                all_predictions.append(pred)

            # Average the predictions column-wise
            return np.mean(np.array(all_predictions), axis=0)

        else:
            raise ValueError(f"Invalid prediction strategy '{strategy}'. Choose from 'mean' or 'best_only'.")

    def score(self, X: np.ndarray, y: np.ndarray, strategy: str = 'mean') -> float:
        """
        Calculates the R² (coefficient of determination) score using scikit-learn's implementation.

        Args:
            X (np.ndarray): Test samples.
            y (np.ndarray): True values for X.
            strategy (str): The prediction strategy to use for scoring ('mean' or 'best_only').

        Returns:
            float: The R² score.
        """
        from .data_processing import r2_score
        
        if not self.best_expressions:
            raise ValueError("Model has not been fitted yet. Call fit() first.")

        predictions = self.predict(X, strategy=strategy)

        if y.ndim == 1:
            y = y.reshape(-1, 1)

        # Use our robust R² implementation for better numerical stability
        try:
            return r2_score(y.flatten(), predictions.flatten())
        except Exception:
            # Fallback calculation for edge cases
            ss_res = float(np.sum((y - predictions) ** 2))
            ss_tot = float(np.sum((y - np.mean(y, axis=0)) ** 2))

            if ss_tot == 0.0:
                # Handle the case where the total sum of squares is zero
                return 1.0 if ss_res == 0.0 else 0.0

            return 1.0 - (ss_res / ss_tot)

    def get_expressions(self) -> List[str]:
        """Returns string representations of the top expressions (raw data, no scaling)."""
        if not self.best_expressions:
            return []
        
        # Return expressions in terms of raw variables (no scaling transformations)
        return [expr.to_string() for expr in self.best_expressions]
    def get_fitness_histories(self) -> List[List[float]]:
        """Returns the fitness histories for the top expressions in the ensemble."""
        if not hasattr(self, 'fitted_regressors') or not self.fitted_regressors:
            return []
        
        # Get the fitness histories for the top selected expressions
        fitness_histories = []
        for i, result in enumerate(self.all_results[:self.top_n_select]):
            regressor = result.get('regressor')
            if regressor and hasattr(regressor, 'fitness_history'):
                fitness_histories.append(regressor.fitness_history.copy())
            else:
                fitness_histories.append([])
        
        return fitness_histories
