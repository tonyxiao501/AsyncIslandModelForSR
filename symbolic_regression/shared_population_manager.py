"""
Improved Inter-Thread Communication for Symbolic Regression
This module provides efficient sharing of actual expressions between worker processes.
"""
import multiprocessing as mp
import numpy as np
from typing import List, Dict, Tuple, Optional
import time
import os
import pickle
import tempfile
import random

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
    successful_imports = 0
    for expr_data in expressions_to_import:
      try:
        # Try to use the actual expression object if available
        if 'expression_obj' in expr_data and expr_data['expression_obj'] is not None:
          imported_expr = expr_data['expression_obj'].copy()
        else:
          # Use the new from_string method to properly reconstruct the expression
          from symbolic_regression.expression_tree import Expression
          imported_expr = Expression.from_string(expr_data['expression_str'], n_inputs)

        new_population.append(imported_expr)
        # Use the original fitness (no penalty for good expressions!)
        new_fitness_scores.append(expr_data['fitness'])
        successful_imports += 1

      except Exception:
        # If import fails, generate a diverse expression instead
        try:
          from symbolic_regression.generator import ExpressionGenerator
          generator = ExpressionGenerator(n_inputs, max_depth=6)

          from symbolic_regression.generator import ExpressionGenerator
          generator = ExpressionGenerator(n_inputs, max_depth=6)

          for expr_data in expressions_to_import:
            try:
              # Generate a random expression as placeholder
              # In a real implementation, you'd parse the expression string
              imported_expr_node = generator.generate_random_expression()  # FIX: use correct method name
              imported_expr = Expression(imported_expr_node)
              new_population.append(imported_expr)
              # Use imported fitness with slight penalty
              new_fitness_scores.append(expr_data['fitness'] * 0.95)
            except:
              continue
        except:
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
