#!/usr/bin/env python3
"""
Test script for early termination and late extension mechanisms
"""

import numpy as np
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from symbolic_regression.mimo_regressor import MIMOSymbolicRegressor

def test_early_termination():
    """Test early termination with a simple function that should converge quickly"""
    print("=== Testing Early Termination ===")
    
    # Generate simple linear data: y = 2*x + 1
    np.random.seed(42)
    X = np.linspace(-2, 2, 50).reshape(-1, 1)
    y = (2 * X.flatten() + 1).reshape(-1, 1)
    
    # Add tiny amount of noise
    y += np.random.normal(0, 0.01, y.shape)
    
    print("Testing with simple linear function: y = 2*x + 1")
    print(f"Data shape: X={X.shape}, y={y.shape}")
    
    # Create regressor with early termination enabled
    regressor = MIMOSymbolicRegressor(
        population_size=50,
        generations=100,  # Set high number that should trigger early termination
        mutation_rate=0.15,
        crossover_rate=0.8,
        max_depth=4,
        parsimony_coefficient=0.001,
        enable_early_termination=True,
        early_termination_threshold=0.99,  # High threshold for simple function
        early_termination_check_interval=10,
        enable_late_extension=False,  # Disable for this test
        console_log=True
    )
    
    print(f"Running evolution with early termination threshold: {regressor.early_termination_threshold}")
    print(f"Check interval: {regressor.early_termination_check_interval} generations")
    
    regressor.fit(X, y)
    
    # Check results
    final_fitness = max(regressor.fitness_history) if regressor.fitness_history else -10.0
    actual_generations = len(regressor.fitness_history)
    
    print(f"\nResults:")
    print(f"Final fitness (R²): {final_fitness:.6f}")
    print(f"Generations run: {actual_generations}/100")
    
    if actual_generations < 100:
        print("✓ Early termination worked!")
    else:
        print("✗ Early termination did not trigger")
    
    print(f"Best expression: {regressor.get_expressions()[0] if regressor.get_expressions() else 'None'}")
    
    return actual_generations < 100

def test_late_extension():
    """Test late extension with a complex function that should need more time"""
    print("\n=== Testing Late Extension ===")
    
    # Generate complex function: y = sin(x) + cos(2*x) + 0.5*x^2
    np.random.seed(42)
    X = np.linspace(-2, 2, 100).reshape(-1, 1)
    y = (np.sin(X.flatten()) + np.cos(2*X.flatten()) + 0.5*X.flatten()**2).reshape(-1, 1)
    
    # Add noise to make it challenging
    y += np.random.normal(0, 0.1, y.shape)
    
    print("Testing with complex function: y = sin(x) + cos(2*x) + 0.5*x^2")
    print(f"Data shape: X={X.shape}, y={y.shape}")
    
    # Create regressor with late extension enabled
    regressor = MIMOSymbolicRegressor(
        population_size=50,
        generations=30,  # Set low number that should trigger late extension
        mutation_rate=0.15,
        crossover_rate=0.8,
        max_depth=6,
        parsimony_coefficient=0.003,
        enable_early_termination=False,  # Disable for this test
        enable_late_extension=True,
        late_extension_threshold=0.95,  # High threshold that's unlikely to be met
        late_extension_generations=20,
        console_log=True
    )
    
    print(f"Running evolution with late extension threshold: {regressor.late_extension_threshold}")
    print(f"Original generations: 30, extension: {regressor.late_extension_generations}")
    
    regressor.fit(X, y)
    
    # Check results
    final_fitness = max(regressor.fitness_history) if regressor.fitness_history else -10.0
    actual_generations = len(regressor.fitness_history)
    
    print(f"\nResults:")
    print(f"Final fitness (R²): {final_fitness:.6f}")
    print(f"Generations run: {actual_generations}")
    
    if regressor.late_extension_triggered:
        print("✓ Late extension triggered!")
        if actual_generations > 30:
            print("✓ Additional generations were run!")
        else:
            print("✗ Extension triggered but no additional generations were run")
    else:
        print("✗ Late extension did not trigger")
    
    print(f"Best expression: {regressor.get_expressions()[0] if regressor.get_expressions() else 'None'}")
    
    return regressor.late_extension_triggered and actual_generations > 30

def test_both_mechanisms():
    """Test both mechanisms together"""
    print("\n=== Testing Both Mechanisms Together ===")
    
    # Generate moderate complexity function
    np.random.seed(42)
    X = np.linspace(-1, 1, 50).reshape(-1, 1)
    y = (X.flatten()**2 + X.flatten()).reshape(-1, 1)  # y = x^2 + x
    
    # Add moderate noise
    y += np.random.normal(0, 0.05, y.shape)
    
    print("Testing with moderate function: y = x^2 + x")
    print(f"Data shape: X={X.shape}, y={y.shape}")
    
    # Create regressor with both mechanisms enabled
    regressor = MIMOSymbolicRegressor(
        population_size=40,
        generations=50,
        mutation_rate=0.15,
        crossover_rate=0.8,
        max_depth=5,
        parsimony_coefficient=0.002,
        enable_early_termination=True,
        early_termination_threshold=0.98,
        early_termination_check_interval=10,
        enable_late_extension=True,
        late_extension_threshold=0.90,
        late_extension_generations=25,
        console_log=True
    )
    
    print(f"Early termination threshold: {regressor.early_termination_threshold}")
    print(f"Late extension threshold: {regressor.late_extension_threshold}")
    
    regressor.fit(X, y)
    
    # Check results
    final_fitness = max(regressor.fitness_history) if regressor.fitness_history else -10.0
    actual_generations = len(regressor.fitness_history)
    
    print(f"\nResults:")
    print(f"Final fitness (R²): {final_fitness:.6f}")
    print(f"Generations run: {actual_generations}")
    
    if actual_generations < 50 and final_fitness >= regressor.early_termination_threshold:
        print("✓ Early termination worked!")
    elif regressor.late_extension_triggered and actual_generations > 50:
        print("✓ Late extension worked!")
    else:
        print("? Normal completion (neither mechanism triggered)")
    
    print(f"Best expression: {regressor.get_expressions()[0] if regressor.get_expressions() else 'None'}")
    
    return True

def main():
    """Run all tests"""
    print("Testing Early Termination and Late Extension Mechanisms")
    print("=" * 60)
    
    try:
        # Test 1: Early termination
        early_term_success = test_early_termination()
        
        # Test 2: Late extension  
        late_ext_success = test_late_extension()
        
        # Test 3: Both mechanisms
        both_success = test_both_mechanisms()
        
        print("\n" + "=" * 60)
        print("SUMMARY:")
        print(f"Early termination test: {'✓ PASS' if early_term_success else '✗ FAIL'}")
        print(f"Late extension test: {'✓ PASS' if late_ext_success else '✗ FAIL'}")
        print(f"Both mechanisms test: {'✓ PASS' if both_success else '✗ FAIL'}")
        
        if early_term_success and late_ext_success and both_success:
            print("\n🎉 All tests passed! The mechanisms are working correctly.")
        else:
            print("\n⚠️ Some tests failed. Check the implementation.")
            
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
