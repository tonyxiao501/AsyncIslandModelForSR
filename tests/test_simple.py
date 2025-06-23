import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from symbolic_regression.mimo_regressor import MIMOSymbolicRegressor

def test_simple_mimo():
    """Simple test of MIMO symbolic regression"""
    
    # Generate simple test data
    np.random.seed(42)
    X = np.random.uniform(-1, 1, (100, 2))
    
    # Simple known functions
    y1 = X[:, 0] + X[:, 1]  # Linear combination
    y2 = X[:, 0] * X[:, 1]  # Product
    
    y = np.column_stack([y1, y2])
    
    print("Test data shapes:")
    print(f"X: {X.shape}")
    print(f"y: {y.shape}")
    
    # Create and train model with small parameters for quick test
    model = MIMOSymbolicRegressor(
        population_size=20,
        generations=10,
        mutation_rate=0.1,
        crossover_rate=0.8,
        max_depth=3
    )
    
    print("\nTraining model...")
    model.fit(X, y)
    
    # Test predictions
    predictions = model.predict(X)
    print(f"\nPredictions shape: {predictions.shape}")
    
    # Show discovered expressions
    expressions = model.get_expressions()
    print(f"\nDiscovered expressions:")
    for i, expr in enumerate(expressions):
        print(f"Output {i+1}: {expr}")
    
    print("\nTest completed successfully!")

if __name__ == "__main__":
    test_simple_mimo()