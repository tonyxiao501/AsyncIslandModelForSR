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

  # Instantiate the ensemble regressor with optimized parameters
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
    import_percentage=0.08
  )

  print("\nTraining symbolic regression ensemble model...")
  print("Target: 2*sin(x) + cos(2*x)")
  print("This should converge relatively quickly...")

  # Train the ensemble model (concurrent fitting)
  model.fit(X_train, y_train, constant_optimize=True)

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

  # Get individual predictions for each of the top 5 expressions
  individual_predictions = []
  individual_r2_scores = []
  
  for expr in model.best_expressions[:5]:  # Get top 5 expressions
    pred = expr.evaluate(X_test)
    if pred.ndim == 1:
      pred = pred.reshape(-1, 1)
    individual_predictions.append(pred.flatten())
    
    # Calculate R² for this individual expression
    r2_individual = r2_score(y_true_test, pred.flatten())
    individual_r2_scores.append(r2_individual)

  # Create visualization with 5 separate plots for candidates
  fig, axes = plt.subplots(2, 3, figsize=(15, 10))
  
  # Plot each of the 5 candidate expressions
  colors = ['red', 'green', 'orange', 'purple', 'brown']
  for i in range(5):
    row = i // 3
    col = i % 3
    ax = axes[row, col]
    
    if i < len(individual_predictions):
      # Plot training data
      ax.scatter(X_train.flatten(), y_train.flatten(),
                alpha=0.6, color='lightblue', s=25,
                label='Training Data', zorder=2)
      
      # Plot true function
      ax.plot(X_test.flatten(), y_true_test,
             color='blue', linewidth=3,
             label='True Function', zorder=3)
      
      # Plot individual candidate prediction
      ax.plot(X_test.flatten(), individual_predictions[i],
             color=colors[i], linewidth=2, linestyle='--',
             label=f'Candidate {i+1}', zorder=4)
      
      ax.set_xlabel('x', fontsize=10)
      ax.set_ylabel('f(x)', fontsize=10)
      ax.set_title(f'Candidate {i+1} (R² = {individual_r2_scores[i]:.4f})', fontsize=11, fontweight='bold')
      ax.grid(True, alpha=0.3)
      ax.legend(fontsize=8)
      
      # Add expression as text annotation
      expr_text = discovered_expressions[i] if i < len(discovered_expressions) else "N/A"
      if len(expr_text) > 30:
        expr_text = expr_text[:27] + "..."
      ax.text(0.02, 0.98, f'f(x) = {expr_text}', transform=ax.transAxes, fontsize=8,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    else:
      ax.set_title(f'Candidate {i+1} - Not Available')
      ax.axis('off')

  # Use the 6th subplot for ensemble mean comparison
  ax_ensemble = axes[1, 2]
  ax_ensemble.scatter(X_train.flatten(), y_train.flatten(),
                     alpha=0.6, color='lightblue', s=25,
                     label='Training Data', zorder=2)
  ax_ensemble.plot(X_test.flatten(), y_true_test,
                  color='blue', linewidth=3,
                  label='True Function', zorder=3)
  ax_ensemble.plot(X_test.flatten(), y_pred_test.flatten(),
                  color='black', linewidth=2, linestyle='-',
                  label='Ensemble Mean', zorder=4)
  ax_ensemble.set_xlabel('x', fontsize=10)
  ax_ensemble.set_ylabel('f(x)', fontsize=10)
  ax_ensemble.set_title(f'Ensemble Mean (R² = {r2_test:.4f})', fontsize=11, fontweight='bold')
  ax_ensemble.grid(True, alpha=0.3)
  ax_ensemble.legend(fontsize=8)

  plt.tight_layout()
  plt.show()

  print(f"\n{'=' * 70}")
  print("DETAILED ANALYSIS (ENSEMBLE):")
  print(f"{'=' * 70}")
  print("Individual candidate R² scores:")
  for i, r2 in enumerate(individual_r2_scores):
      print(f"  Candidate {i+1}: {r2:.6f}")
  print(f"Ensemble mean R²: {r2_test:.6f}")

if __name__ == "__main__":
  main()