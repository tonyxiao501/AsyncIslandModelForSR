"""
Simple test of the improved mutations without ensemble complexity
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from symbolic_regression.mimo_regressor import MIMOSymbolicRegressor

def test_simple_regression():
    """Test the improved mutation on a simple regression problem"""
    print("Testing simple regression with improved mutations...")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate simple test data: y = 2*sin(x)
    X = np.linspace(-2, 2, 50).reshape(-1, 1)
    y = 2 * np.sin(X.flatten()) + 0.1 * np.random.randn(50)
    y = y.reshape(-1, 1)
    
    print(f"Target function: y = 2*sin(x)")
    print(f"Training data: {X.shape[0]} samples")
    
    # Create regressor with improved parameters
    model = MIMOSymbolicRegressor(
        population_size=100,
        generations=50,
        mutation_rate=0.15,
        crossover_rate=0.8,
        tournament_size=3,
        max_depth=4,
        parsimony_coefficient=0.01,
        diversity_threshold=0.6,
        adaptive_rates=True,
        restart_threshold=15,
        elite_fraction=0.15,
        console_log=True
    )
    
    print("\nStarting evolution...")
    model.fit(X, y, constant_optimize=False)
    
    # Get results
    expressions = model.get_expressions()
    if expressions:
        print(f"\nBest expression: {expressions[0]}")
        
        # Test prediction
        y_pred = model.predict(X)
        mse = np.mean((y - y_pred) ** 2)
        print(f"MSE: {mse:.6f}")
        
        # Get mutation statistics if available
        try:
            genetic_ops = getattr(model, 'genetic_ops', None)
            if genetic_ops and hasattr(genetic_ops, 'get_mutation_statistics'):
                stats = genetic_ops.get_mutation_statistics()
                print("\nMutation statistics:")
                for key, value in stats.items():
                    if 'success_rate' in key:
                        print(f"  {key}: {value:.3f}")
        except Exception as e:
            print(f"Could not retrieve mutation statistics: {e}")
    else:
        print("No valid expressions found")

if __name__ == "__main__":
    test_simple_regression()
