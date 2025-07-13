"""
Test script to verify the Grammatical Evolution improvements work correctly.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from symbolic_regression.genetic_ops import GeneticOperations
from symbolic_regression.expression_tree import Expression, ConstantNode, VariableNode, BinaryOpNode

def test_context_aware_mutations():
    """Test the new context-aware mutation improvements"""
    print("Testing Grammatical Evolution improvements...")
    
    # Create test data
    X = np.linspace(-2, 2, 100).reshape(-1, 1)
    y = (2 * X + 1).flatten()  # Simple linear function
    
    # Create genetic operations with the improvements
    genetic_ops = GeneticOperations(n_inputs=1, max_complexity=20)
    
    # Create a simple expression: x + 1
    expr = Expression(
        BinaryOpNode('+', 
                    VariableNode(0), 
                    ConstantNode(1.0))
    )
    
    print(f"Original expression: {expr.to_string()}")
    print(f"Original complexity: {expr.complexity()}")
    
    # Test context-aware mutation
    mutated = genetic_ops.mutate(expr, mutation_rate=0.3, X=X, y=y.reshape(-1, 1))
    print(f"Context-aware mutated: {mutated.to_string()}")
    print(f"New complexity: {mutated.complexity()}")
    
    # Test adaptive mutation with feedback
    mutated_adaptive = genetic_ops.adaptive_mutate_with_feedback(
        expr, mutation_rate=0.2, X=X, y=y.reshape(-1, 1), 
        generation=10, stagnation_count=5
    )
    print(f"Adaptive mutated: {mutated_adaptive.to_string()}")
    
    # Test high stagnation mutation
    mutated_stagnant = genetic_ops.adaptive_mutate_with_feedback(
        expr, mutation_rate=0.2, X=X, y=y.reshape(-1, 1), 
        generation=50, stagnation_count=20
    )
    print(f"High stagnation mutated: {mutated_stagnant.to_string()}")
    
    # Get mutation statistics
    stats = genetic_ops.get_mutation_statistics()
    print("\nMutation statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value:.3f}")
    
    print("\nGrammatical Evolution improvements test completed successfully!")

if __name__ == "__main__":
    test_context_aware_mutations()
