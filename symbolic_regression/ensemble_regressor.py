import numpy as np
from typing import List, Dict, Optional
from .expression_tree import Expression
import multiprocessing
from .ensemble_worker import _fit_worker

class EnsembleMIMORegressor:
  """
  Runs multiple MIMOSymbolicRegressor fits concurrently and selects an
  ensemble of the best overall expressions.
  """
  def __init__(self, n_fits: int = 8, top_n_select: int = 5,
               enable_inter_thread_communication: bool = True,
               exchange_interval: int = 10, purge_percentage: float = 0.15,
               import_percentage: float = 0.03, debug_csv_path: Optional[str] = None, **regressor_kwargs):
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

  def fit(self, X: np.ndarray, y: np.ndarray, constant_optimize: bool = False):
    """
    Fits 'n_fits' regressors concurrently using multiprocessing, then selects
    the top 'top_n_select' expressions based on their final fitness scores.

    IMPORTANT: Due to the use of 'multiprocessing' on some platforms (like
    Windows), this method should be called from within a
    `if __name__ == "__main__":` block in your script.
    """
    print(f"Starting ensemble fit with {self.n_fits} concurrent regressors...")
    
    # Disable inter-thread communication by default to avoid synchronization overhead
    use_communication = (self.enable_inter_thread_communication and 
                        self.n_fits >= 4 and 
                        self.regressor_kwargs.get('generations', 50) >= 50)
    
    if use_communication:
      print(f"Inter-thread communication enabled: exchange every {self.exchange_interval} generations")
      print(f"Population exchange: purge {self.purge_percentage*100:.1f}%, import {self.import_percentage*100:.1f}%")
    else:
      print("Inter-thread communication disabled for optimal performance.")

    # Initialize shared population manager only if communication is actually used
    shared_manager = None
    if use_communication:
      from .shared_population_manager import create_improved_shared_data
      shared_manager = create_improved_shared_data(
        n_workers=self.n_fits,
        exchange_interval=self.exchange_interval,
        purge_percentage=self.purge_percentage,
        import_percentage=self.import_percentage
      )

    # Prepare configurations for each worker process.
    # We disable console logging for worker processes to keep the main output clean.
    worker_kwargs = self.regressor_kwargs.copy()
    worker_kwargs['console_log'] = False
    
    # Use different random seeds and parameters for diversity without communication overhead
    configs = []
    for i in range(self.n_fits):
      # Each worker gets slightly different parameters for natural diversity
      worker_specific_kwargs = worker_kwargs.copy()
      if not use_communication:
        # Add parameter diversity when not using communication
        base_mutation = worker_kwargs.get('mutation_rate', 0.1)
        base_parsimony = worker_kwargs.get('parsimony_coefficient', 0.001)
        
        worker_specific_kwargs['mutation_rate'] = base_mutation * (0.8 + 0.4 * i / max(1, self.n_fits - 1))
        worker_specific_kwargs['parsimony_coefficient'] = base_parsimony * (0.5 + 1.0 * i / max(1, self.n_fits - 1))
      
      configs.append((worker_specific_kwargs, X, y, constant_optimize, shared_manager, i, self.debug_csv_path))

    # Run fits in parallel using a process pool with optimized settings
    with multiprocessing.Pool(processes=self.n_fits, maxtasksperchild=1) as pool:
      results = pool.map(_fit_worker, configs)

    print("All concurrent fits completed. Aggregating and ranking results...")

    # Aggregate valid results from all runs
    all_run_results = []
    for i, reg in enumerate(results):
      if reg and reg.best_expressions and reg.best_fitness_history:
        fitness = reg.best_fitness_history[-1]
        expression = reg.best_expressions[0]
        all_run_results.append({
          "run": i,
          "fitness": fitness,
          "expression_obj": expression,
          "expression_str": expression.to_string(),
          "complexity": expression.complexity()
        })

    if not all_run_results:
      print("\nWarning: No valid expressions were found across all runs. The model is not fitted.")
      return

    # Sort all results by fitness in descending order (higher is better)
    all_run_results.sort(key=lambda item: item['fitness'], reverse=True)

    # Select the top N best expressions from the aggregated list
    top_results = all_run_results[:self.top_n_select]

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

    if self.enable_inter_thread_communication:
      print(f"\nInter-thread communication summary:")
      print(f"Workers exchanged expressions every {self.exchange_interval} generations")
      print(f"This should have improved population diversity across threads")

  def predict(self, X: np.ndarray, strategy: str = 'mean') -> np.ndarray:
    """
    Makes predictions using the ensemble of best expressions.

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

  def get_expressions(self) -> List[str]:
    """Returns the string representations of the top expressions in the ensemble."""
    if not self.best_expressions:
      return []
    return [expr.to_string() for expr in self.best_expressions]

  def score(self, X: np.ndarray, y: np.ndarray, strategy: str = 'mean') -> float:
    """
    Calculates the R² (coefficient of determination) score for the ensemble model.

    Args:
        X (np.ndarray): Test samples.
        y (np.ndarray): True values for X.
        strategy (str): The prediction strategy to use for scoring ('mean' or 'best_only').

    Returns:
        float: The R² score.
    """
    if not self.best_expressions:
      raise ValueError("Model has not been fitted yet. Call fit() first.")

    predictions = self.predict(X, strategy=strategy)

    if y.ndim == 1:
      y = y.reshape(-1, 1)

    ss_res = float(np.sum((y - predictions) ** 2))
    ss_tot = float(np.sum((y - np.mean(y, axis=0)) ** 2))

    if ss_tot == 0.0:
      # Handle the case where the total sum of squares is zero
      return 1.0 if ss_res == 0.0 else 0.0

    return 1.0 - (ss_res / ss_tot)