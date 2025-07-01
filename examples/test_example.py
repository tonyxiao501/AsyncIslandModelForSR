import sys
import os
import pdb

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from symbolic_regression.mimo_regressor import MIMOSymbolicRegressor


def generate_complex_function(X):
  """Generate a complex 1-input, 1-output function"""
  return 2 * np.sin(X) + 0.5 * X ** 2


def add_noise(y, noise_level=0.05):
  """Add Gaussian noise to the target function"""
  noise = np.random.normal(0, noise_level * np.std(y), size=y.shape)
  return y + noise


def main():
  # Set random seed for reproducibility
  np.random.seed(42)

  # Generate training data
  n_samples = 200
  X_train = np.linspace(-3, 3, n_samples).reshape(-1, 1)
  y_true_train = generate_complex_function(X_train.flatten())
  y_train = add_noise(y_true_train, noise_level=0.1).reshape(-1, 1)

  # Generate test data (denser for smooth plotting)
  X_test = np.linspace(-3, 3, 300).reshape(-1, 1)
  y_true_test = generate_complex_function(X_test.flatten())

  print("Target function: f(x) = 2*sin(x) + 0.5*x²")
  print(f"Training data: {X_train.shape[0]} samples with 5% noise")
  print(f"Test data: {X_test.shape[0]} samples (for plotting)")

  model = MIMOSymbolicRegressor(
    population_size=150,
    generations=500,
    mutation_rate=0.15,
    crossover_rate=0.8,
    tournament_size=3,
    max_depth=5,
    parsimony_coefficient=0.005,
    diversity_threshold=0.65,
    adaptive_rates=True,
    restart_threshold=15,
    elite_fraction=0.12
  )

  print("\nTraining symbolic regression model...")
  print("Target: 2*sin(x) + 0.5*x²")
  print("This should converge relatively quickly...")

  # Train the model
  model.fit(X_train, y_train)

  # Make predictions
  y_pred_train = model.predict(X_train)
  y_pred_test = model.predict(X_test)

  # Calculate R² scores
  r2_train = r2_score(y_train, y_pred_train)
  r2_test = r2_score(y_true_test, y_pred_test)

  # Calculate additional error metrics
  mse_test = np.mean((y_true_test - y_pred_test.flatten()) ** 2)
  rmse_test = np.sqrt(mse_test)
  mae_test = np.mean(np.abs(y_true_test - y_pred_test.flatten()))

  # Get discovered expression
  discovered_expressions = model.get_expressions()
  discovered_expr = discovered_expressions[0] if discovered_expressions else "No expression found"

  print(f"\n{'=' * 70}")
  print("SYMBOLIC REGRESSION RESULTS:")
  print(f"{'=' * 70}")
  print(f"Target function:      f(x) = 2*sin(x) + 0.5*x²")
  print(f"Discovered function:  f(x) = {discovered_expr}")
  print(f"Training R² score:    {r2_train:.6f}")
  print(f"Test R² score:        {r2_test:.6f}")
  print(f"Test RMSE:            {rmse_test:.6f}")
  print(f"Test MAE:             {mae_test:.6f}")
  if model.best_expressions:
    print(f"Expression complexity: {model.best_expressions[0].complexity():.2f}")
    print(f"Expression size:       {model.best_expressions[0].size()} nodes")
  print(f"Final fitness:        {model.fitness_history[-1]:.6f}")
  print(f"Generations run:      {len(model.fitness_history)}")

  # Create comprehensive visualization
  fig, axes = plt.subplots(2, 2, figsize=(15, 12))

  # Main function comparison plot
  ax1 = axes[0, 0]
  ax1.scatter(X_train.flatten(), y_train.flatten(),
              alpha=0.6, color='lightblue', s=25,
              label='Training Data (5% noise)', zorder=2)
  ax1.plot(X_test.flatten(), y_true_test,
           color='blue', linewidth=3,
           label='True: 2*sin(x) + 0.5*x²', zorder=3)
  ax1.plot(X_test.flatten(), y_pred_test.flatten(),
           color='red', linewidth=2, linestyle='--',
           label=f'Discovered: {discovered_expr}', zorder=4)
  ax1.set_xlabel('x', fontsize=11)
  ax1.set_ylabel('f(x)', fontsize=11)
  ax1.set_title('Function Comparison', fontsize=12, fontweight='bold')
  ax1.grid(True, alpha=0.3)
  ax1.legend(fontsize=9)

  # Add R² annotation
  textstr = f'Train R² = {r2_train:.4f}\nTest R² = {r2_test:.4f}\nRMSE = {rmse_test:.4f}'
  props = dict(boxstyle='round', facecolor='lightgreen', alpha=0.8)
  ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=10,
           verticalalignment='top', bbox=props)

  # Fitness evolution
  ax2 = axes[0, 1]
  ax2.plot(model.fitness_history, color='green', linewidth=2, label='Best Fitness')
  if hasattr(model, 'diversity_history') and model.diversity_history:
    ax2_twin = ax2.twinx()
    ax2_twin.plot(model.diversity_history, color='orange', linewidth=1, alpha=0.7, label='Diversity')
    ax2_twin.set_ylabel('Diversity Score', color='orange')
    ax2_twin.tick_params(axis='y', labelcolor='orange')
  ax2.set_xlabel('Generation')
  ax2.set_ylabel('Fitness', color='green')
  ax2.set_title('Evolution Progress')
  ax2.grid(True, alpha=0.3)
  ax2.tick_params(axis='y', labelcolor='green')

  # Residuals plot
  ax3 = axes[1, 0]
  residuals = y_true_test - y_pred_test.flatten()
  ax3.scatter(y_true_test, residuals, alpha=0.6, color='purple', s=20)
  ax3.axhline(y=0, color='red', linestyle='--', alpha=0.7)
  ax3.set_xlabel('True Values')
  ax3.set_ylabel('Residuals (True - Predicted)')
  ax3.set_title('Residual Analysis')
  ax3.grid(True, alpha=0.3)

  # Add residual statistics
  residual_std = np.std(residuals)
  ax3.axhline(y=residual_std, color='gray', linestyle=':', alpha=0.7, label=f'±1σ = ±{residual_std:.3f}')
  ax3.axhline(y=-residual_std, color='gray', linestyle=':', alpha=0.7)
  ax3.legend()

  # Prediction vs True scatter plot
  ax4 = axes[1, 1]
  ax4.scatter(y_true_test, y_pred_test.flatten(), alpha=0.6, color='darkblue', s=20)
  # Plot perfect prediction line
  min_val, max_val = min(y_true_test.min(), y_pred_test.min()), max(y_true_test.max(), y_pred_test.max())
  ax4.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='Perfect Prediction')
  ax4.set_xlabel('True Values')
  ax4.set_ylabel('Predicted Values')
  ax4.set_title('Prediction Accuracy')
  ax4.grid(True, alpha=0.3)
  ax4.legend()

  plt.tight_layout()
  plt.show()

  # Print detailed analysis
  print(f"\n{'=' * 70}")
  print("DETAILED ANALYSIS:")
  print(f"{'=' * 70}")

  if len(model.fitness_history) > 10:
    improvement = model.fitness_history[0] - model.fitness_history[-1]
    print(f"\nEvolution Statistics:")
    print(f"  Initial fitness:     {model.fitness_history[0]:.6f}")
    print(f"  Final fitness:       {model.fitness_history[-1]:.6f}")
    print(f"  Total improvement:   {improvement:.6f}")
    print(f"  Avg improvement/gen: {improvement/len(model.fitness_history):.6f}")


if __name__ == "__main__":
  main()