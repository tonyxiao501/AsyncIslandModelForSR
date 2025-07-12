import sys
import os
import pdb
import csv
import datetime
import cProfile
import pstats
import io
import time

# Add the project root to the Python path to resolve the import error
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from symbolic_regression.ensemble_regressor import EnsembleMIMORegressor  # <-- Use the ensemble version


def generate_complex_function(X):
  """Generate a complex but well-behaved function: 2*sin(x) + cos(2*x)"""
  return 2*np.sin(X) + np.cos(2*X)


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
  y_train = add_noise(y_true_train, noise_level=0.05).reshape(-1, 1)

  # Generate test data (denser for smooth plotting)
  X_test = np.linspace(-3, 3, 300).reshape(-1, 1)
  y_true_test = generate_complex_function(X_test.flatten())

  print("Target function: f(x) = 2*sin(x) + cos(2*x)")
  print(f"Training data: {X_train.shape[0]} samples with 5% noise")
  print(f"Test data: {X_test.shape[0]} samples (for plotting)")

  # Create debug CSV file for tracking thread progress
  timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
  debug_csv_path = f"thread_evolution_debug_{timestamp}.csv"

  # Initialize CSV file with headers
  with open(debug_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow([
      'timestamp', 'worker_id', 'generation', 'rank', 'fitness',
      'complexity', 'expression', 'diversity_score', 'stagnation_counter'
    ])

  print(f"Debug tracking enabled: {debug_csv_path}")

  # Instantiate the ensemble regressor with debug callback and optimized parameters
  model = EnsembleMIMORegressor(
    n_fits=8,                    
    top_n_select=5,             
    population_size=200,         
    generations=200,             
    mutation_rate=0.15,
    crossover_rate=0.8,
    tournament_size=3,
    max_depth=5,
    parsimony_coefficient=0.003,
    diversity_threshold=0.6,     
    adaptive_rates=True,
    restart_threshold=15,        
    elite_fraction=0.12,
    enable_inter_thread_communication=True,
    purge_percentage=0.15,       
    exchange_interval=15,       
    import_percentage=0.08,      
    debug_csv_path=debug_csv_path  # Pass debug file path
  )

  print("\nTraining symbolic regression ensemble model...")
  print("Target: 2*sin(x) + cos(2*x)")
  print("This should converge relatively quickly...")

  # Profile the model fitting section
  pr = cProfile.Profile()
  pr.enable()
  start = time.perf_counter()
  # Train the ensemble model (concurrent fitting)
  model.fit(X_train, y_train, constant_optimize=True)
  end = time.perf_counter()
  pr.disable()
  print(f"Model fitting took {end - start:.2f} seconds.")

  # Print top 20 cumulative time functions
  s = io.StringIO()
  ps = pstats.Stats(pr, stream=s).sort_stats('cumtime')
  ps.print_stats(20)
  print(s.getvalue())

  # Make predictions (using mean of ensemble)
  y_pred_train = model.predict(X_train, strategy='mean')
  y_pred_test = model.predict(X_test, strategy='mean')

  # Calculate R² scores
  r2_train = r2_score(y_train, y_pred_train)
  r2_test = r2_score(y_true_test, y_pred_test)

  # Calculate additional error metrics
  mse_test = np.mean((y_true_test - y_pred_test.flatten()) ** 2)
  rmse_test = np.sqrt(mse_test)
  mae_test = np.mean(np.abs(y_true_test - y_pred_test.flatten()))

  # Get discovered expressions
  discovered_expressions = model.get_expressions()
  discovered_expr = discovered_expressions[0] if discovered_expressions else "No expression found"

  print(f"\n{'=' * 70}")
  print("SYMBOLIC REGRESSION RESULTS (ENSEMBLE):")
  print(f"{'=' * 70}")
  print(f"Target function:      f(x) = 2*sin(x) + cos(2*x)")
  print(f"Discovered function:  f(x) = {discovered_expr}")
  print(f"Training R² score:    {r2_train:.6f}")
  print(f"Test R² score:        {r2_test:.6f}")
  print(f"Test RMSE:            {rmse_test:.6f}")
  print(f"Test MAE:             {mae_test:.6f}")
  print(f"Top ensemble expressions:")
  for i, expr in enumerate(discovered_expressions):
      print(f"  {i+1}. {expr}")

  # Create comprehensive visualization
  fig, axes = plt.subplots(2, 2, figsize=(10, 8))

  # Main function comparison plot
  ax1 = axes[0, 0]
  ax1.scatter(X_train.flatten(), y_train.flatten(),
              alpha=0.6, color='lightblue', s=25,
              label='Training Data (5% noise)', zorder=2)
  ax1.plot(X_test.flatten(), y_true_test,
           color='blue', linewidth=3,
           label='True: 2*sin(x) + cos(2*x)', zorder=3)
  ax1.plot(X_test.flatten(), y_pred_test.flatten(),
           color='red', linewidth=2, linestyle='--',
           label=f'Discovered (ensemble mean)', zorder=4)
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

  # Fitness evolution (not available for ensemble, so show placeholder)
  ax2 = axes[0, 1]
  ax2.set_title('Evolution Progress (see console for details)')
  ax2.axis('off')

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

  print(f"\n{'=' * 70}")
  print("DETAILED ANALYSIS (ENSEMBLE):")
  print(f"{'=' * 70}")

  # You can access more details from model.all_results if needed

if __name__ == "__main__":
  main()