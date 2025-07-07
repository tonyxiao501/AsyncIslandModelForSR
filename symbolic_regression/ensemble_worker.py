import random
import numpy as np

def _fit_worker(config: tuple):
  """
  A top-level function to be executed by each process in a multiprocessing Pool.
  It initializes and fits a MIMOSymbolicRegressor instance.
  """
  # Move the import here to avoid circular import
  from .mimo_regressor import MIMOSymbolicRegressor
  from .expression_tree.optimization.memory_pool import reset_global_pool

  regressor_params, X, y, constant_optimize = config

  # Reset the global pool for this process to avoid multiprocessing conflicts
  reset_global_pool()

  # Each process must have its own random seed to ensure diversity in runs
  random.seed()
  np.random.seed()

  try:
    # Instantiate and fit the regressor for this process
    reg = MIMOSymbolicRegressor(**regressor_params)
    reg.fit(X, y, constant_optimize=constant_optimize)
    # Return the fully fitted object for result aggregation
    return reg
  except KeyboardInterrupt:
    # Handle keyboard interrupts gracefully
    return None
  except Exception as e:
    print(f"A worker process failed with an error: {e}")
    return None
