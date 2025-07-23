import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import os
from datetime import datetime
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
    import_percentage=0.08,
    enable_data_scaling=False,
    use_multi_scale_fitness=False,
    # Early termination and late extension parameters
    enable_early_termination=True,
    early_termination_threshold=0.99,     # Terminate if R² >= 0.99
    early_termination_check_interval=10,  # Check every 10 generations
    enable_late_extension=True,
    late_extension_threshold=0.95,        # Extend if R² < 0.95 at the end
    late_extension_generations=50         # Add 50 more generations if needed
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
    # Get the data scaler from the model
    data_scaler = model.shared_data_scaler
    if data_scaler is not None:
      # Transform test data and evaluate expression, then inverse transform
      X_test_scaled = data_scaler.transform_input(X_test)
      pred_scaled = expr.evaluate(X_test_scaled)
      pred = data_scaler.inverse_transform_output(pred_scaled)
    else:
      # No scaling was used
      pred = expr.evaluate(X_test)
    
    if pred.ndim == 1:
      pred = pred.reshape(-1, 1)
    individual_predictions.append(pred.flatten())
    
    # Calculate R² for this individual expression (on original scale)
    r2_individual = r2_score(y_true_test, pred.flatten())
    individual_r2_scores.append(r2_individual)

  # Get fitness histories for the top expressions
  fitness_histories = model.get_fitness_histories()

  # Create output directory with current date and time
  timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
  output_dir = os.path.join("..", f"{timestamp}_fit_result")
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)
  print(f"\nSaving plots to directory: {output_dir}")

  # Create 5 separate graphs for each candidate expression
  colors = ['red', 'green', 'orange', 'purple', 'brown']
  
  for i in range(5):
    if i < len(individual_predictions):
      # Create figure with 2 subplots: function plot and fitness plot
      fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
      
      # Left subplot: Function plot
      ax1.scatter(X_train.flatten(), y_train.flatten(),
                 alpha=0.6, color='lightblue', s=25,
                 label='Training Data', zorder=2)
      
      ax1.plot(X_test.flatten(), y_true_test,
              color='blue', linewidth=3,
              label='True Function', zorder=3)
      
      ax1.plot(X_test.flatten(), individual_predictions[i],
              color=colors[i], linewidth=2, linestyle='--',
              label=f'Candidate {i+1}', zorder=4)
      
      ax1.set_xlabel('x', fontsize=12)
      ax1.set_ylabel('f(x)', fontsize=12)
      ax1.set_title(f'Candidate {i+1} - Function Fit (R² = {individual_r2_scores[i]:.4f})', 
                   fontsize=14, fontweight='bold')
      ax1.grid(True, alpha=0.3)
      ax1.legend(fontsize=10)
      
      # Add expression as text annotation
      expr_text = discovered_expressions[i] if i < len(discovered_expressions) else "N/A"
      if len(expr_text) > 40:
        expr_text = expr_text[:37] + "..."
      ax1.text(0.02, 0.98, f'f(x) = {expr_text}', transform=ax1.transAxes, fontsize=10,
              verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
      
      # Right subplot: Fitness evolution plot
      if i < len(fitness_histories) and fitness_histories[i]:
        generations = range(1, len(fitness_histories[i]) + 1)
        ax2.plot(generations, fitness_histories[i], color=colors[i], linewidth=2, marker='o', markersize=2)
        ax2.set_xlabel('Generation', fontsize=12)
        ax2.set_ylabel('Fitness Score', fontsize=12)
        ax2.set_title(f'Candidate {i+1} - Fitness Evolution', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(1, len(fitness_histories[i]))
        
        # Add final fitness value as text
        final_fitness = fitness_histories[i][-1]
        ax2.text(0.98, 0.98, f'Final Fitness: {final_fitness:.6f}', 
                transform=ax2.transAxes, fontsize=10, ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
      else:
        ax2.text(0.5, 0.5, 'No fitness history available', 
                transform=ax2.transAxes, fontsize=12, ha='center', va='center')
        ax2.set_title(f'Candidate {i+1} - Fitness Evolution', fontsize=14, fontweight='bold')
      
      plt.tight_layout()
      
      # Save the plot
      filename = f"candidate_{i+1}_analysis.png"
      filepath = os.path.join(output_dir, filename)
      plt.savefig(filepath, dpi=300, bbox_inches='tight')
      print(f"Saved: {filename}")
      
      plt.show()
      plt.close()  # Close figure to free memory
    else:
      print(f"Candidate {i+1}: Not available")

  # Create ensemble summary plot
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
  
  # Left subplot: Ensemble mean comparison
  ax1.scatter(X_train.flatten(), y_train.flatten(),
             alpha=0.6, color='lightblue', s=25,
             label='Training Data', zorder=2)
  ax1.plot(X_test.flatten(), y_true_test,
          color='blue', linewidth=3,
          label='True Function', zorder=3)
  ax1.plot(X_test.flatten(), y_pred_test.flatten(),
          color='black', linewidth=2, linestyle='-',
          label='Ensemble Mean', zorder=4)
  ax1.set_xlabel('x', fontsize=12)
  ax1.set_ylabel('f(x)', fontsize=12)
  ax1.set_title(f'Ensemble Mean Prediction (R² = {r2_test:.4f})', fontsize=14, fontweight='bold')
  ax1.grid(True, alpha=0.3)
  ax1.legend(fontsize=10)
  
  # Right subplot: R² scores comparison
  candidate_names = [f'Candidate {i+1}' for i in range(len(individual_r2_scores))]
  candidate_names.append('Ensemble Mean')
  r2_values = individual_r2_scores + [r2_test]
  colors_extended = colors[:len(individual_r2_scores)] + ['black']
  
  bars = ax2.bar(candidate_names, r2_values, color=colors_extended, alpha=0.7)
  ax2.set_ylabel('R² Score', fontsize=12)
  ax2.set_title('R² Score Comparison', fontsize=14, fontweight='bold')
  ax2.grid(True, alpha=0.3, axis='y')
  ax2.set_ylim(0, 1.05)
  
  # Add value labels on bars
  for bar, value in zip(bars, r2_values):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
            f'{value:.4f}', ha='center', va='bottom', fontsize=10)
  
  plt.xticks(rotation=45, ha='right')
  plt.tight_layout()
  
  # Save ensemble summary
  ensemble_filename = "ensemble_summary.png"
  ensemble_filepath = os.path.join(output_dir, ensemble_filename)
  plt.savefig(ensemble_filepath, dpi=300, bbox_inches='tight')
  print(f"Saved: {ensemble_filename}")
  
  plt.show()
  plt.close()

  print(f"\n{'=' * 70}")
  print("DETAILED ANALYSIS (ENSEMBLE):")
  print(f"{'=' * 70}")
  print("Individual candidate R² scores:")
  for i, r2 in enumerate(individual_r2_scores):
      print(f"  Candidate {i+1}: {r2:.6f}")
  print(f"Ensemble mean R²: {r2_test:.6f}")
  print(f"\nAll plots saved to: {output_dir}")

if __name__ == "__main__":
  main()