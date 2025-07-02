import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from symbolic_regression.mimo_regressor import MIMOSymbolicRegressor
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

diff_thresh = 1e-3
num_tests = 100
noise = 0

def generate_complex_function(X):
  """Generate a complex 1-input, 1-output function"""
  return 2 * np.sin(X) + 0.5 * X ** 2


def add_noise(y, noise_level=0.05):
  """Add Gaussian noise to the target function"""
  noise = np.random.normal(0, noise_level * np.std(y), size=y.shape)
  return y + noise

def get_last_rising_point(arr):
  diff_arr = np.diff(arr)
  rising_indices_diff = np.where(diff_arr > diff_thresh)[0] + 1
  if len(rising_indices_diff) > 0:
    return rising_indices_diff[-1]
  else:
    return -1

def run_test(X_train, y_train):
  model = MIMOSymbolicRegressor(
    population_size=200,
    generations=200,
    mutation_rate=0.15,
    crossover_rate=0.8,
    tournament_size=3,
    max_depth=5,
    parsimony_coefficient=0.005,
    diversity_threshold=0.65,
    adaptive_rates=True,
    restart_threshold=15,
    elite_fraction=0.12,
    console_log=False
  )
  model.fit(X_train, y_train, constant_optimize=True)
  return model

def run_and_score(args):
  _, X_train, y_train, X_test, y_true_test = args
  model = run_test(X_train, y_train)
  y_pred_train = model.predict(X_train)
  y_pred_test = model.predict(X_test)
  mse_test = np.mean((y_true_test - y_pred_test.flatten()) ** 2)
  rmse_test = np.sqrt(mse_test)
  last_rising = get_last_rising_point(model.best_fitness_history)

  return last_rising, rmse_test

if __name__ == '__main__':
  # Generate training data
  n_samples = 200

  X_train = np.linspace(-3, 3, n_samples).reshape(-1, 1)
  y_true_train = generate_complex_function(X_train.flatten())
  y_train = add_noise(y_true_train, noise_level=noise/100).reshape(-1, 1)

  # Generate test data (denser for smooth plotting)
  X_test = np.linspace(-3, 3, 300).reshape(-1, 1)
  y_true_test = generate_complex_function(X_test.flatten())

  print("Target function: f(x) = 2*sin(x) + 0.5*x²")
  print(f"Training data: {X_train.shape[0]} samples with {noise}% noise")

  results = []
  with ProcessPoolExecutor() as executor:
    futures = [executor.submit(run_and_score, (i, X_train, y_train, X_test, y_true_test)) for i in range(num_tests)]
    for f in tqdm(as_completed(futures), total=num_tests):
      results.append(f.result())

  best_evolutions, scores = zip(*results)

  fig, axes = plt.subplots(1, 2, figsize=(10, 5))
  ax1 = axes[0]
  ax2 = axes[1]
  ax1.hist(best_evolutions)
  ax2.hist(scores)
  plt.show()
