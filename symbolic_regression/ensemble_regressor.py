import numpy as np
from typing import List, Dict, Optional, TYPE_CHECKING
from .expression_tree import Expression
import multiprocessing
from .ensemble_worker import _fit_worker

if TYPE_CHECKING:
    from .data_scaling import DataScaler

class EnsembleMIMORegressor:
  """
  Runs multiple MIMOSymbolicRegressor fits concurrently and selects an
  ensemble of the best overall expressions.
  """
  def __init__(self, n_fits: int = 8, top_n_select: int = 5,
               enable_inter_thread_communication: bool = True,
               exchange_interval: int = 10, purge_percentage: float = 0.15,
               import_percentage: float = 0.03, debug_csv_path: Optional[str] = None,
               # Shared scaling parameters to ensure consistency across ensemble
               shared_data_scaler: Optional['DataScaler'] = None,
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
        shared_data_scaler (DataScaler): Pre-fitted data scaler to ensure consistent
                                        scaling across all ensemble members. If None,
                                        each worker will fit its own scaler.
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
    
    # Shared data scaler for consistent scaling across ensemble
    self.shared_data_scaler = shared_data_scaler

  def fit(self, X: np.ndarray, y: np.ndarray, constant_optimize: bool = False):
    """
    Fits 'n_fits' regressors concurrently using multiprocessing, then selects
    the top 'top_n_select' expressions based on their final fitness scores.

    IMPORTANT: Due to the use of 'multiprocessing' on some platforms (like
    Windows), this method should be called from within a
    `if __name__ == "__main__":` block in your script.
    """
    print(f"Starting ensemble fit with {self.n_fits} concurrent regressors...")
    
    # Pre-fit a shared data scaler if scaling is enabled and none provided
    if (self.shared_data_scaler is None and 
        self.regressor_kwargs.get('enable_data_scaling', False)):
      print("Fitting shared data scaler for consistent ensemble scaling...")
      from .data_scaling import DataScaler
      
      input_scaling = self.regressor_kwargs.get('input_scaling', 'auto')
      output_scaling = self.regressor_kwargs.get('output_scaling', 'auto')
      scaling_target_range = self.regressor_kwargs.get('scaling_target_range', (-5.0, 5.0))
      
      self.shared_data_scaler = DataScaler(
          input_scaling=input_scaling,
          output_scaling=output_scaling,
          target_range=scaling_target_range
      )
      
      # Fit the scaler on the training data
      X_scaled, y_scaled = self.shared_data_scaler.fit_transform(X, y)
      print(f"Data scaling applied: X {X.shape} -> {X_scaled.shape}, y {y.shape} -> {y_scaled.shape}")
    
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
    
    # If using shared scaling, pass the fitted scaler to workers
    if self.shared_data_scaler is not None:
      worker_kwargs['shared_data_scaler'] = self.shared_data_scaler
    
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
    self.fitted_regressors = []  # Store regressor objects for fitness history access
    for i, reg in enumerate(results):
      if reg and reg.best_expressions and reg.best_fitness_history:
        fitness = reg.best_fitness_history[-1]
        expression = reg.best_expressions[0]
        all_run_results.append({
          "run": i,
          "fitness": fitness,
          "expression_obj": expression,
          "expression_str": expression.to_string(),
          "complexity": expression.complexity(),
          "regressor": reg  # Store the regressor object
        })
        self.fitted_regressors.append(reg)

    if not all_run_results:
      print("\nWarning: No valid expressions were found across all runs. The model is not fitted.")
      return

    # Sort all results by fitness in descending order (higher is better)
    all_run_results.sort(key=lambda item: item['fitness'], reverse=True)

    # Select the top N best expressions from the aggregated list
    top_results = all_run_results[:self.top_n_select]

    # FINAL OPTIMIZATION PHASE: Apply optimizations to top candidate expressions
    print(f"\nApplying final optimizations to top {len(top_results)} candidate expressions...")
    
    import time
    start_time = time.time()
    
    # Extract expressions for optimization
    candidate_expressions = [res['expression_obj'] for res in top_results]
    
    # Apply final optimizations
    from .expression_utils import optimize_final_expressions, evaluate_optimized_expressions
    optimized_expressions = optimize_final_expressions(candidate_expressions, X, y)
    
    # Re-evaluate with optimized constants and get new fitness scores
    parsimony_coeff = self.regressor_kwargs.get('parsimony_coefficient', 0.01)
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

    # Check if we have scaling information
    data_scaler = self.shared_data_scaler
    if (data_scaler is None and hasattr(self, 'fitted_regressors') and 
        self.fitted_regressors and len(self.fitted_regressors) > 0 and 
        hasattr(self.fitted_regressors[0], 'data_scaler')):
      data_scaler = self.fitted_regressors[0].data_scaler

    if strategy == 'best_only':
      # Use only the single best expression (the first in the sorted list)
      if data_scaler is not None:
        X_scaled = data_scaler.transform_input(X)
        y_scaled = self.best_expressions[0].evaluate(X_scaled)
        return data_scaler.inverse_transform_output(y_scaled)
      else:
        return self.best_expressions[0].evaluate(X)

    elif strategy == 'mean':
      # Collect predictions from all selected expressions
      if data_scaler is not None:
        X_scaled = data_scaler.transform_input(X)
        all_predictions = []
        for expr in self.best_expressions:
          pred_scaled = expr.evaluate(X_scaled)
          pred_original = data_scaler.inverse_transform_output(pred_scaled)
          if pred_original.ndim == 1:
            pred_original = pred_original.reshape(-1, 1)
          all_predictions.append(pred_original)
      else:
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
    """Returns clean string representations of the top expressions with proper scaling information."""
    if not self.best_expressions:
      return []
    
    # Get basic expressions (these are in terms of scaled variables)
    basic_expressions = [expr.to_string() for expr in self.best_expressions]
    
    # Check if we have scaling information
    data_scaler = self.shared_data_scaler
    if (data_scaler is None and hasattr(self, 'fitted_regressors') and 
        self.fitted_regressors and len(self.fitted_regressors) > 0 and 
        hasattr(self.fitted_regressors[0], 'data_scaler')):
      data_scaler = self.fitted_regressors[0].data_scaler
    
    if data_scaler is not None:
      # Get scaling transformation expressions
      n_inputs = getattr(self.fitted_regressors[0], 'n_inputs', 1) if hasattr(self, 'fitted_regressors') and self.fitted_regressors else 1
      try:
        input_transforms, output_transform = data_scaler.get_scaling_transformation_expressions(n_inputs)
        
        # Check if any scaling was actually applied
        has_input_scaling = any(transform != f"x{i}" for i, transform in enumerate(input_transforms))
        has_output_scaling = output_transform != "y'"
        
        if has_input_scaling or has_output_scaling:
          # Add scaling information to expressions with new clean format
          expressions_with_scaling = []
          for expr_str in basic_expressions:
            # The expression is in terms of scaled variables, so we present it cleanly
            expr_with_scaling = expr_str
            
            # Add scaling information in the requested format
            scaling_lines = []
            
            # Add input scaling lines (x' = f(x))
            for i, transform in enumerate(input_transforms):
              if transform != f"x{i}":
                scaling_lines.append(f"x{i}' = {transform}")
            
            # Add output scaling line (y' = g(y))  
            if has_output_scaling and output_transform != "y'":
              # For output, we need the forward transformation, not inverse
              scaling_lines.append(f"y' = {self._get_forward_output_transform()}")
            
            # Combine everything
            if scaling_lines:
              expr_with_scaling += "\n with "
              expr_with_scaling += "\n      ".join(scaling_lines)
            
            expressions_with_scaling.append(expr_with_scaling)
          
          return expressions_with_scaling
        else:
          return basic_expressions
      except Exception as e:
        print(f"Warning: Failed to add scaling information: {e}")
        return basic_expressions
    else:
      return basic_expressions

  def _get_forward_output_transform(self) -> str:
    """Get the forward output transformation expression (y' = g(y))."""
    if not hasattr(self, 'fitted_regressors') or not self.fitted_regressors:
      return "y"
    
    data_scaler = self.shared_data_scaler
    if (data_scaler is None and hasattr(self.fitted_regressors[0], 'data_scaler')):
      data_scaler = self.fitted_regressors[0].data_scaler
    
    if data_scaler is None:
      return "y"
    
    transform = data_scaler.output_transform
    if transform == 'log':
      if data_scaler.output_log_offset > 0:
        if data_scaler.output_log_offset < 1e-6:
          return f"log(y + {data_scaler.output_log_offset:.2e})"
        else:
          return f"log(y + {data_scaler.output_log_offset:.3f})"
      else:
        return "log(y)"
    elif transform == 'standard':
      if data_scaler.output_scaler is not None:
        mean = data_scaler.output_scaler.mean_[0]  # type: ignore
        scale = data_scaler.output_scaler.scale_[0]  # type: ignore
        return f"(y - {mean:.3f}) / {scale:.3f}"
      else:
        return "y"
    elif transform == 'minmax':
      if data_scaler.output_scaler is not None:
        data_min = data_scaler.output_scaler.data_min_[0]  # type: ignore
        data_range = data_scaler.output_scaler.data_range_[0]  # type: ignore
        min_range, max_range = data_scaler.target_range
        range_width = max_range - min_range
        return f"{min_range:.1f} + {range_width:.1f} * (y - {data_min:.3f}) / {data_range:.3f}"
      else:
        return "y"
    elif transform == 'robust':
      if data_scaler.output_scaler is not None:
        center = data_scaler.output_scaler.center_[0]  # type: ignore
        scale = data_scaler.output_scaler.scale_[0]  # type: ignore
        return f"(y - {center:.3f}) / {scale:.3f}"
      else:
        return "y"
    else:  # 'none' or unknown
      return "y"

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
    from sklearn.metrics import r2_score
    
    if not self.best_expressions:
      raise ValueError("Model has not been fitted yet. Call fit() first.")

    predictions = self.predict(X, strategy=strategy)

    if y.ndim == 1:
      y = y.reshape(-1, 1)

    # Use scikit-learn's R² implementation for consistency
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