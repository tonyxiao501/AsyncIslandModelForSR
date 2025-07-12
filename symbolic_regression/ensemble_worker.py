import random
import numpy as np
import os

def _fit_worker(config: tuple):
  """
  Optimized worker function for multiprocessing with minimal synchronization overhead.
  Each process runs independently with unique parameters for diversity.
  """
  # Move the import here to avoid circular import
  from .mimo_regressor import MIMOSymbolicRegressor
  from .expression_tree.optimization.memory_pool import reset_global_pool

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
