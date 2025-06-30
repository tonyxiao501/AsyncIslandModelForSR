import sys
import os
# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from symbolic_regression.mimo_regressor import MIMOSymbolicRegressor

def true_function(X):
  """Explicit vector-valued function: returns [sin(x0) + x1^2, x0 * x1 - x2, exp(x2)]"""
  y1 = np.sin(X[:, 0]) + X[:, 1] ** 2
  y2 = X[:, 0] * X[:, 1] - X[:, 2]
  y3 = np.exp(X[:, 2])
  return np.column_stack([y1, y2, y3])

def generate_data(n_samples=1000, noise_std=0.1):
  X = np.random.uniform(-2, 2, (n_samples, 3))
  y = true_function(X) + np.random.normal(0, noise_std, (n_samples, 3))
  return X, y

def main():
  # Generate data from explicit vector-valued function
  print("Generating data from explicit vector-valued function...")
  X_train, y_train = generate_data(800)
  X_test, y_test = generate_data(200, noise_std=0.0)  # No noise for test

  # Fit the model
  print("Training MIMO Symbolic Regression model...")
  model = MIMOSymbolicRegressor(
    population_size=100,
    generations=50,
    mutation_rate=0.1,
    crossover_rate=0.8,
    max_depth=5,
    parsimony_coefficient=0.001
  )
  model.fit(X_train, y_train)

  # Predict and evaluate
  print("Evaluating model...")
  y_pred = model.predict(X_test)
  train_score = model.score(X_train, y_train)
  test_score = model.score(X_test, y_test)

  print(f"Training R² score: {train_score:.4f}")
  print(f"Test R² score: {test_score:.4f}")

  # Show discovered expressions
  print("\nDiscovered expressions:")
  for i, expr in enumerate(model.get_expressions()):
    print(f"Output {i+1}: {expr}")

  plot_results(y_test, y_pred, model.fitness_history)

def plot_results(y_true, y_pred, fitness_history):
  fig, axes = plt.subplots(2, 2, figsize=(12, 10))
  n_outputs = y_true.shape[1]
  for i in range(min(3, n_outputs)):
    row = i // 2
    col = i % 2
    axes[row, col].scatter(y_true[:, i], y_pred[:, i], alpha=0.6)
    axes[row, col].plot([y_true[:, i].min(), y_true[:, i].max()],
                        [y_true[:, i].min(), y_true[:, i].max()], 'r--', lw=2)
    axes[row, col].set_xlabel(f'True Output {i+1}')
    axes[row, col].set_ylabel(f'Predicted Output {i+1}')
    axes[row, col].set_title(f'Output {i+1}: Predicted vs True')
    axes[row, col].grid(True, alpha=0.3)
  if n_outputs <= 2:
    axes[1, 1].plot(fitness_history)
    axes[1, 1].set_xlabel('Generation')
    axes[1, 1].set_ylabel('Average Fitness')
    axes[1, 1].set_title('Fitness Evolution')
    axes[1, 1].grid(True, alpha=0.3)
  plt.tight_layout()
  plt.show()

if __name__ == "__main__":
  main()