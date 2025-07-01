import sys
import os
# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from symbolic_regression.mimo_regressor import MIMOSymbolicRegressor

def load_feynman_data():
  """Load and prepare Feynman dataset"""
  print("Loading Feynman dataset...")
  ds = load_dataset("yoshitomo-matsubara/srsd-feynman_easy")

  # Get the train split
  train_data = ds['train']

  # Inspect the structure of the first few items
  if len(train_data) > 0:
    first_item = train_data[0]
    print("First item type:", type(first_item))
    print("First item:", first_item)

    # Parse the text format
    sample_text = first_item['text']
    sample_values = sample_text.split()
    print(f"Sample text has {len(sample_values)} values")
    print("Sample values:", sample_values[:5], "..." if len(sample_values) > 5 else "")

  # Parse all data points and check for consistency
  all_data = []
  row_lengths = []

  for i, item in enumerate(train_data):
    text = item['text']
    values = [float(x) for x in text.split()]
    all_data.append(values)
    row_lengths.append(len(values))

    # Show first few rows for debugging
    if i < 5:
      print(f"Row {i}: {len(values)} values - {values}")

  # Check row length consistency
  unique_lengths = set(row_lengths)
  print(f"Unique row lengths: {unique_lengths}")
  print(f"Total rows: {len(all_data)}")

  if len(unique_lengths) > 1:
    # Find the most common length
    from collections import Counter
    length_counts = Counter(row_lengths)
    most_common_length = length_counts.most_common(1)[0][0]
    print(f"Most common row length: {most_common_length} (appears {length_counts[most_common_length]} times)")

    # Filter to only include rows with the most common length
    filtered_data = [row for row in all_data if len(row) == most_common_length]
    print(f"Filtered data: {len(filtered_data)} rows with {most_common_length} columns")
    all_data = filtered_data

  # Convert to numpy array
  try:
    data_array = np.array(all_data)
    print(f"Data shape: {data_array.shape}")
  except ValueError as e:
    print(f"Error creating numpy array: {e}")
    # Try to debug the issue
    print("First few row lengths:", row_lengths[:10])
    print("Last few row lengths:", row_lengths[-10:])
    return []

  # Check if we have enough columns (at least 2: features + target)
  if data_array.shape[1] < 2:
    print("Error: Dataset must have at least 2 columns (features + target)")
    return []

  # Assuming the last column is the target (y) and the rest are features (X)
  X = data_array[:, :-1]  # All columns except the last
  y = data_array[:, -1]   # Last column

  # Reshape y to be 2D
  y = y.reshape(-1, 1)

  print(f"Features shape: {X.shape}")
  print(f"Target shape: {y.shape}")

  # Create a single dataset
  datasets = [{
    'name': 'feynman_easy',
    'X': X,
    'y': y,
    'target_eq': 'unknown'
  }]

  print(f"Loaded {len(datasets)} dataset with {X.shape[0]} samples")
  return datasets

def test_single_equation(eq_data, model_params):
  """Test model on a single equation"""
  X, y = eq_data['X'], eq_data['y']
  eq_name = eq_data['name']
  target_eq = eq_data['target_eq']

  # Skip if not enough samples
  if len(X) < 100:
    return None

  # Split data
  X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
  )

  # Configure and train model
  model = MIMOSymbolicRegressor(**model_params)

  try:
    model.fit(X_train, y_train)

    # Get predictions and scores
    y_pred_test = model.predict(X_test)
    test_r2 = r2_score(y_test, y_pred_test)
    test_mse = mean_squared_error(y_test, y_pred_test)

    # Get discovered expression
    expressions = model.get_expressions()
    discovered_eq = expressions[0] if expressions else "No expression found"

    # Get final fitness
    final_fitness = model.fitness_history[-1] if model.fitness_history else 0.0

    return {
      'equation_name': eq_name,
      'target_equation': target_eq,
      'discovered_equation': discovered_eq,
      'fitness_score': final_fitness,
      'test_r2': test_r2,
      'test_mse': test_mse,
      'data_points': len(X)
    }

  except Exception as e:
    print(f"Error processing {eq_name}: {str(e)}")
    return None

def main():
  # Load Feynman dataset
  datasets = load_feynman_data()

  # Model parameters
  model_params = {
    'population_size': 50,
    'generations': 30,
    'mutation_rate': 0.1,
    'crossover_rate': 0.8,
    'max_depth': 6,
    'parsimony_coefficient': 0.001
  }

  print(f"\nTesting model on {len(datasets)} equations...")
  print("Model parameters:", model_params)

  # Test each equation
  results = []
  for i, eq_data in enumerate(datasets):
    print(f"\nProcessing equation {i+1}/{len(datasets)}: {eq_data['name']}")
    print(f"Target: {eq_data['target_eq']}")
    print(f"Data shape: {eq_data['X'].shape}")

    result = test_single_equation(eq_data, model_params)
    if result:
      results.append(result)
      print(f"✓ Completed - Fitness: {result['fitness_score']:.4f}, R²: {result['test_r2']:.4f}")
    else:
      print("✗ Skipped or failed")

  # Save results to CSV
  if results:
    df = pd.DataFrame(results)

    # Reorder columns for better readability
    column_order = [
      'equation_name',
      'target_equation',
      'discovered_equation',
      'fitness_score',
      'test_r2',
      'test_mse',
      'data_points'
    ]
    df = df[column_order]

    # Save to CSV
    output_file = 'feynman_results.csv'
    df.to_csv(output_file, index=False)
    print(f"\n{'='*60}")
    print(f"Results saved to: {output_file}")
    print(f"Total equations processed: {len(results)}")
    print(f"Average fitness score: {df['fitness_score'].mean():.4f}")
    print(f"Average R² score: {df['test_r2'].mean():.4f}")
    print(f"{'='*60}")

    # Show top 5 results
    print("\nTop 5 results by fitness score:")
    top_results = df.nlargest(5, 'fitness_score')[['equation_name', 'target_equation', 'discovered_equation', 'fitness_score']]
    for _, row in top_results.iterrows():
      print(f"\n{row['equation_name']}:")
      print(f"  Target:     {row['target_equation']}")
      print(f"  Discovered: {row['discovered_equation']}")
      print(f"  Fitness:    {row['fitness_score']:.4f}")

  else:
    print("No results to save.")

if __name__ == "__main__":
  main()