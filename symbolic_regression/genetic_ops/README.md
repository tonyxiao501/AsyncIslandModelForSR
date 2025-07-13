# Genetic Operations Module

This module provides a comprehensive set of genetic operations for symbolic regression, organized into specialized submodules for better maintainability and extensibility.

## Module Structure

The genetic operations have been refactored into the following components:

### Core Modules

1. **`genetic_operations.py`** - Main interface that orchestrates all genetic operations
2. **`mutation_strategies.py`** - Collection of mutation strategies including context-aware and semantic-preserving mutations
3. **`crossover_operations.py`** - Various crossover operations including structural and quality-guided crossover
4. **`context_analysis.py`** - Advanced expression analysis for intelligent strategy selection
5. **`diversity_metrics.py`** - Population diversity measurement and maintenance utilities

## Usage

### Basic Usage (Backward Compatible)

```python
from symbolic_regression.genetic_ops import GeneticOperations

# Initialize genetic operations
genetic_ops = GeneticOperations(n_inputs=3, max_complexity=20)

# Perform mutations and crossover
mutated_expr = genetic_ops.mutate(expression, mutation_rate=0.1, X=X_train, y=y_train)
child1, child2 = genetic_ops.crossover(parent1, parent2)
```

### Advanced Usage (Using Individual Components)

```python
from symbolic_regression.genetic_ops import (
    GeneticOperations,
    MutationStrategies,
    CrossoverOperations,
    ExpressionContextAnalyzer,
    DiversityMetrics
)

# Use individual components for fine-grained control
mutation_strategies = MutationStrategies(n_inputs=3, max_complexity=20)
context_analyzer = ExpressionContextAnalyzer(n_inputs=3)

# Analyze expression context
context = context_analyzer.analyze_expression_context(expression)

# Apply specific mutation strategies
mutated = mutation_strategies.context_aware_mutation(expression, rate=0.1, X=X_train, y=y_train)
```

## Key Features

### Enhanced Mutation Strategies

- **Context-Aware Mutation**: Considers expression structure and node importance
- **Semantic-Preserving Mutation**: Applies algebraic transformations while preserving meaning
- **Adaptive Strategy Selection**: Dynamically selects best mutation strategy based on success rates
- **Feedback-Driven Adaptation**: Adjusts mutation aggressiveness based on evolution state

### Advanced Crossover Operations

- **Structural Crossover**: Exchanges subtrees between parent expressions
- **Quality-Guided Crossover**: Uses subtree quality assessment for intelligent crossover
- **Semantic Crossover**: Blends expression outputs for semantic combination
- **Uniform Crossover**: Node-by-node exchange for fine-grained recombination

### Intelligent Context Analysis

- **Structural Metrics**: Complexity, depth, balance, and operator diversity
- **Pattern Detection**: Redundancy and repeated subtree identification
- **Nonlinearity Assessment**: Detection of nonlinear operations and functions
- **Variable Usage Analysis**: Symmetry and distribution of variable usage

### Population Diversity Management

- **Multi-Metric Diversity**: Structural, operator, complexity, and semantic diversity
- **Distance Calculations**: Advanced structural and semantic distance measures
- **Diversity Maintenance**: Automatic replacement of similar individuals
- **Diverse Parent Selection**: Selection algorithms that promote genetic diversity

## Migration from Monolithic Structure

The original `genetic_ops.py` file has been refactored but maintains backward compatibility. Existing code will continue to work without changes, but new code should consider using the modular structure for better organization and extensibility.

### Benefits of Modular Structure

1. **Maintainability**: Each component has a clear responsibility
2. **Extensibility**: Easy to add new mutation or crossover strategies
3. **Testing**: Individual components can be tested in isolation
4. **Performance**: Selective imports reduce memory usage
5. **Documentation**: Better organized and component-specific documentation

## Performance Considerations

- **Adaptive Strategies**: Success rate tracking adds minimal overhead but provides significant performance gains
- **Context Analysis**: Comprehensive analysis provides better strategy selection
- **Quality Assessment**: Computational cost is offset by improved genetic operation effectiveness
- **Diversity Metrics**: Population-level analysis helps maintain exploration capability

## Future Enhancements

- **Multi-Objective Optimization**: Support for multiple fitness criteria
- **Island Model**: Population subdivision for enhanced diversity
- **Coevolution**: Competitive fitness evaluation
- **Neural-Guided Operations**: ML-based strategy selection and parameter tuning
