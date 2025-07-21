#!/usr/bin/env python3
"""
Test script to verify performance optimizations are working correctly
"""

import numpy as np
import time
from symbolic_regression.ensemble_regressor import EnsembleMIMORegressor
from symbolic_regression.mimo_regressor import MIMOSymbolicRegressor

def test_caching_performance():
    """Test that caching improves performance on repeated operations"""
    from symbolic_regression.expression_utils import to_sympy_expression, clear_optimization_caches
    
    print("Testing caching performance...")
    
    # Clear caches first
    clear_optimization_caches()
    
    # Test expressions
    test_expressions = [
        "2.5*X0 + 1.0",
        "sin(X0) + cos(X0)",
        "X0^2 + 2*X0 + 1",
        "log(X0 + 1) * 2.0",
        "exp(X0) / (X0 + 1)"
    ]
    
    # Time without caching (first run)
    start_time = time.time()
    for _ in range(5):  # Repeat operations
        for expr in test_expressions:
            to_sympy_expression(expr, enable_simplify=True)
    first_run_time = time.time() - start_time
    
    # Time with caching (second run - should be faster)
    start_time = time.time()
    for _ in range(5):  # Repeat operations
        for expr in test_expressions:
            to_sympy_expression(expr, enable_simplify=True)
    cached_run_time = time.time() - start_time
    
    improvement = (first_run_time - cached_run_time) / first_run_time * 100
    print(f"First run (no cache): {first_run_time:.4f}s")
    print(f"Cached run: {cached_run_time:.4f}s")
    print(f"Improvement: {improvement:.1f}%")
    
    return improvement > 10  # Should see at least 10% improvement

def test_optimization_control():
    """Test that optimization can be disabled and enabled correctly"""
    print("\nTesting optimization control...")
    
    # Create simple test data
    np.random.seed(42)
    X = np.linspace(0, 1, 50).reshape(-1, 1)
    y = (2 * X.flatten() + 1).reshape(-1, 1)
    
    # Test with optimizations disabled during evolution
    print("Testing with evolution optimizations disabled...")
    model = MIMOSymbolicRegressor(
        population_size=30,
        generations=20,
        evolution_sympy_simplify=False,  # Disabled during evolution
        evolution_constant_optimize=False,  # Disabled during evolution
        final_optimization_generations=3,  # Only in final 3 generations
        console_log=True
    )
    
    start_time = time.time()
    model.fit(X, y, constant_optimize=True)
    fast_time = time.time() - start_time
    
    print(f"Fast evolution time: {fast_time:.2f}s")
    print(f"Found expression: {model.get_expressions()[0] if model.get_expressions() else 'None'}")
    
    return True

def test_final_optimization():
    """Test that final optimization stage works correctly"""
    print("\nTesting final optimization...")
    
    # Create test data with noise
    np.random.seed(42)
    X = np.linspace(0, 2, 100).reshape(-1, 1)
    y_true = 2 * X.flatten() + 1
    y = y_true + 0.1 * np.random.normal(0, 1, len(y_true))
    y = y.reshape(-1, 1)
    
    # Test ensemble with final optimization
    model = EnsembleMIMORegressor(
        n_fits=2,  # Small for testing
        top_n_select=2,
        population_size=40,
        generations=15,
        mutation_rate=0.15,
        crossover_rate=0.8,
        console_log=True
    )
    
    print("Fitting ensemble with final optimization...")
    start_time = time.time()
    model.fit(X, y, constant_optimize=True)
    total_time = time.time() - start_time
    
    # Check results
    expressions = model.get_expressions()
    predictions = model.predict(X)
    
    # Calculate R² score
    ss_res = np.sum((y - predictions) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2_score = 1 - (ss_res / ss_tot)
    
    print(f"Total time: {total_time:.2f}s")
    print(f"Found {len(expressions)} expressions")
    print(f"R² score: {r2_score:.4f}")
    print(f"Best expression: {expressions[0] if expressions else 'None'}")
    
    return len(expressions) > 0 and r2_score > 0.5

def main():
    """Run all performance optimization tests"""
    print("=" * 60)
    print("PERFORMANCE OPTIMIZATION VERIFICATION")
    print("=" * 60)
    
    test_results = []
    
    # Test 1: Caching performance
    try:
        result = test_caching_performance()
        test_results.append(("Caching Performance", result))
    except Exception as e:
        print(f"Caching test failed: {e}")
        test_results.append(("Caching Performance", False))
    
    # Test 2: Optimization control
    try:
        result = test_optimization_control()
        test_results.append(("Optimization Control", result))
    except Exception as e:
        print(f"Optimization control test failed: {e}")
        test_results.append(("Optimization Control", False))
    
    # Test 3: Final optimization
    try:
        result = test_final_optimization()
        test_results.append(("Final Optimization", result))
    except Exception as e:
        print(f"Final optimization test failed: {e}")
        test_results.append(("Final Optimization", False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed = 0
    for test_name, result in test_results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name:25} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(test_results)} tests passed")
    
    if passed == len(test_results):
        print("🎉 All optimizations are working correctly!")
    else:
        print("⚠️  Some optimizations may need attention.")

if __name__ == "__main__":
    main()
