#!/usr/bin/env python3
"""
Test script to verify that the architectural restoration fixes
allow the algorithm to solve functions like 2*sin(x) + cos(2x) again
"""

import numpy as np
import time
from symbolic_regression.mimo_regressor import MIMOSymbolicRegressor

def test_trigonometric_function():
    """Test the algorithm on 2*sin(x) + cos(2x) which it used to solve reliably"""
    print("Testing architectural restoration on: y = 2*sin(x) + cos(2x)")
    print("=" * 60)
    
    # Create test data
    np.random.seed(42)
    X = np.linspace(-np.pi, np.pi, 100).reshape(-1, 1)
    y_true = 2 * np.sin(X.flatten()) + np.cos(2 * X.flatten())
    y = y_true + 0.05 * np.random.normal(0, 1, len(y_true))  # Small amount of noise
    y = y.reshape(-1, 1)
    
    print(f"Target function: y = 2*sin(x) + cos(2x)")
    print(f"Training data: {X.shape[0]} samples")
    print(f"Data range: x ∈ [{X.min():.2f}, {X.max():.2f}]")
    print(f"Target range: y ∈ [{y_true.min():.2f}, {y_true.max():.2f}]")
    
    # Test with restored natural evolution parameters
    model = MIMOSymbolicRegressor(
        population_size=150,  # Larger population for better exploration
        generations=75,       # More generations for convergence
        mutation_rate=0.15,   # Standard mutation rate
        crossover_rate=0.8,   # High crossover rate
        tournament_size=3,
        max_depth=6,          # Allow deeper trees for trig functions
        parsimony_coefficient=0.001,  # Light parsimony penalty
        diversity_threshold=0.5,
        adaptive_rates=True,
        restart_threshold=20,
        elite_fraction=0.1,
        console_log=True
    )
    
    print("\nStarting evolution with restored natural selection...")
    start_time = time.time()
    
    model.fit(X, y, constant_optimize=True)
    
    evolution_time = time.time() - start_time
    
    # Analyze results
    expressions = model.get_expressions()
    if expressions:
        best_expr = expressions[0]  # This is a string
        y_pred = model.predict(X)
        
        # Calculate metrics
        mse = np.mean((y - y_pred) ** 2)
        r2 = 1 - (np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2))
        
        print(f"\n" + "=" * 60)
        print("RESULTS:")
        print(f"Evolution time: {evolution_time:.2f}s")
        print(f"Best expression: {best_expr}")
        
        # Handle complexity check safely - for string expressions, use length as approximation
        print(f"Expression length: {len(best_expr)} characters")
        
        print(f"MSE: {mse:.6f}")
        print(f"R² score: {r2:.6f}")
        
        # Check if it contains trigonometric functions
        expr_str = str(best_expr).lower()
        has_sin = 'sin' in expr_str
        has_cos = 'cos' in expr_str
        
        print(f"\nFunction analysis:")
        print(f"Contains 'sin': {has_sin}")
        print(f"Contains 'cos': {has_cos}")
        print(f"Contains trigonometric functions: {has_sin or has_cos}")
        
        # Success criteria: Good fit (R² > 0.9) and contains trig functions
        success = r2 > 0.9 and (has_sin or has_cos)
        
        print(f"\n{'SUCCESS' if success else 'NEEDS IMPROVEMENT'}: ", end="")
        if success:
            print("Algorithm successfully found trigonometric solution!")
        else:
            if r2 <= 0.9:
                print(f"Fitness too low (R² = {r2:.3f} <= 0.9)")
            if not (has_sin or has_cos):
                print(f"No trigonometric functions found in solution")
        
        # Get mutation statistics if available
        try:
            genetic_ops = getattr(model, 'genetic_ops', None)
            if genetic_ops and hasattr(genetic_ops, 'get_mutation_statistics'):
                stats = genetic_ops.get_mutation_statistics()
                print(f"\nMutation strategy statistics:")
                for key, value in stats.items():
                    if 'success_rate' in key:
                        strategy = key.replace('_success_rate', '')
                        attempts = stats.get(f"{strategy}_attempts", 0)
                        print(f"  {strategy:20} Success rate: {value:.3f} ({attempts} attempts)")
        except Exception as e:
            print(f"Could not retrieve mutation statistics: {e}")
        
        return success, r2, has_sin or has_cos
    else:
        print("No valid expressions found - algorithm failed!")
        return False, 0.0, False

def test_simple_function():
    """Test on a simpler function to verify basic functionality"""
    print("\n" + "=" * 60)
    print("Testing on simpler function: y = 2*x + 1")
    print("=" * 60)
    
    # Create simple test data
    np.random.seed(123)
    X = np.linspace(-2, 2, 50).reshape(-1, 1)
    y_true = 2 * X.flatten() + 1
    y = y_true + 0.1 * np.random.normal(0, 1, len(y_true))
    y = y.reshape(-1, 1)
    
    model = MIMOSymbolicRegressor(
        population_size=100,
        generations=50,
        mutation_rate=0.12,
        crossover_rate=0.8,
        console_log=True
    )
    
    start_time = time.time()
    model.fit(X, y, constant_optimize=True)
    evolution_time = time.time() - start_time
    
    expressions = model.get_expressions()
    if expressions:
        y_pred = model.predict(X)
        mse = np.mean((y - y_pred) ** 2)
        r2 = 1 - (np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2))
        
        print(f"Simple function result:")
        print(f"Expression: {expressions[0]}")
        print(f"R² score: {r2:.6f}")
        print(f"Evolution time: {evolution_time:.2f}s")
        
        return r2 > 0.95  # Should easily solve linear function
    return False

def main():
    """Run architectural restoration tests"""
    print("ARCHITECTURAL RESTORATION VERIFICATION")
    print("Testing whether natural evolution fixes restore algorithm capability")
    print("=" * 80)
    
    # Test 1: Simple function (should work)
    simple_success = test_simple_function()
    
    # Test 2: Trigonometric function (the main test)
    trig_success, r2, has_trig = test_trigonometric_function()
    
    print("\n" + "=" * 80)
    print("FINAL SUMMARY:")
    print("=" * 80)
    print(f"Simple function (2x + 1):      {'✓ PASS' if simple_success else '✗ FAIL'}")
    print(f"Trigonometric function:        {'✓ PASS' if trig_success else '✗ FAIL'}")
    
    if trig_success:
        print("\n🎉 ARCHITECTURAL RESTORATION SUCCESSFUL!")
        print("The algorithm can now solve trigonometric functions again.")
    elif has_trig and r2 > 0.7:
        print("\n⚠️  PARTIAL SUCCESS:")
        print("Found trigonometric functions but fit could be better.")
        print("May need parameter tuning or more generations.")
    else:
        print("\n❌ RESTORATION INCOMPLETE:")
        if not has_trig:
            print("Algorithm still not finding trigonometric solutions.")
        if r2 <= 0.7:
            print(f"Fitness too low (R² = {r2:.3f}).")
        print("May need additional architectural changes.")
    
    print(f"\nRecommendations:")
    if not trig_success:
        print("- Try increasing population size or generations")
        print("- Check if trigonometric operators are available in generator")
        print("- Verify mutation strategies are properly balanced")
    else:
        print("- Architecture restoration was successful!")
        print("- The natural evolution process is working correctly")

if __name__ == "__main__":
    main()
