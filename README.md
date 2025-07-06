# Symbolic Regression Package

A comprehensive genetic programming approach to symbolic regression for Multiple Input Multiple Output (MIMO) systems with advanced evolution dynamics and modular architecture.

---

## Features

- **Multiple Input Multiple Output (MIMO) Symbolic Regression**
- **Modular Architecture** with separated components for easy extension and maintenance
- **Advanced Expression Tree System** with optimization and validation utilities
- **Genetic Programming** with sophisticated evolution dynamics
- **Diversity Preservation** and adaptive mutation/crossover rates
- **Population Management** with restart and elite preservation strategies
- **Ensemble Modeling**: Run multiple regressors in parallel and aggregate results
- **SymPy Integration** for expression simplification and validation
- **Constant Optimization** using curve fitting techniques
- **Detailed Evolution Statistics** and expression introspection
- **Memory Pool Optimization** for efficient expression evaluation

---

## Installation

Clone the repository and install dependencies:

```bash
git clone <your-repo-url>
cd PythonProject
pip install -r requirements.txt
```

---

## Quick Start

```python
import numpy as np
from symbolic_regression import MIMOSymbolicRegressor

# Generate synthetic data
X = np.random.rand(100, 3)
y = X[:, 0]**2 + np.sin(X[:, 1]) + X[:, 2]

# Create and train the regressor
regressor = MIMOSymbolicRegressor(
    population_size=100,
    generations=50,
    mutation_rate=0.1,
    crossover_rate=0.8,
    max_depth=6
)

# Fit the model
regressor.fit(X, y)

# Make predictions
predictions = regressor.predict(X)

# Get the discovered expressions
expressions = regressor.get_expressions()
print(f"Best expression: {expressions[0]}")

# Calculate R² score
score = regressor.score(X, y)
print(f"R² Score: {score:.4f}")
```

---

## Module Architecture

The package is organized into several modular components:

### Core Components

#### `expression_tree/`
- **`expression.py`**: Main Expression class for tree manipulation
- **`core/`**: Core node types and operators
  - `node.py`: Base node classes (VariableNode, ConstantNode, BinaryOpNode, UnaryOpNode)
  - `operators.py`: Operator definitions and evaluation functions
- **`optimization/`**: Performance optimization utilities
  - `memory_pool.py`: Memory pool for efficient expression evaluation
- **`utils/`**: Expression utilities
  - `simplifier.py`: Expression simplification algorithms
  - `sympy_utils.py`: SymPy integration for advanced simplification
  - `validator.py`: Expression validation utilities

#### `generator.py`
- `ExpressionGenerator`: Basic expression generation
- `BiasedExpressionGenerator`: Expression generation with operator bias control

#### `genetic_ops.py`
- `GeneticOperations`: Advanced genetic operations including:
  - Point mutation, subtree mutation, insert mutation
  - Structural and simple crossover
  - Diversity-preserving operations

#### `mimo_regressor.py`
- `MIMOSymbolicRegressor`: Main regressor class with adaptive evolution

### Population Management

#### `population.py`
- Diverse population generation strategies
- Diversity injection mechanisms
- Enhanced reproduction with simulated annealing
- Population validation and fitness evaluation

#### `selection.py`
- Enhanced selection balancing fitness and diversity
- Tournament selection with adaptive tournament size
- Diversity-based selection methods

### Evolution Control

#### `adaptive_evolution.py`
- Adaptive parameter adjustment
- Population restart strategies
- Stagnation detection and handling

#### `evolution_stats.py`
- Comprehensive evolution statistics tracking
- Expression analysis and reporting

### Utilities

#### `utils.py`
- String similarity calculations
- Population diversity metrics
- Expression uniqueness scoring

#### `expression_utils.py`
- SymPy expression conversion
- Constant optimization utilities

---

## Advanced Usage

### Custom Expression Generation

```python
from symbolic_regression import ExpressionGenerator, BiasedExpressionGenerator

# Basic generator
generator = ExpressionGenerator(n_inputs=3, max_depth=5)
expression = generator.generate_random_expression()

# Biased generator for specific operator distributions
biased_generator = BiasedExpressionGenerator(
    n_inputs=3,
    max_depth=5,
    operator_rates={'+': 0.3, '*': 0.3, 'sin': 0.2, 'exp': 0.2}
)
biased_expression = biased_generator.generate_biased_expression()
```

### Custom Genetic Operations

```python
from symbolic_regression import GeneticOperations

genetic_ops = GeneticOperations(n_inputs=3, max_complexity=20)

# Perform mutations
mutated_expr = genetic_ops.mutate(expression, mutation_rate=0.2)

# Perform crossover
child1, child2 = genetic_ops.crossover(parent1, parent2)
```

### Population Management

```python
from symbolic_regression import (
    generate_diverse_population, 
    inject_diversity, 
    enhanced_reproduction_v2
)

# Generate diverse initial population
population = generate_diverse_population(
    generator, n_inputs=3, population_size=100, 
    max_depth=6, is_expression_valid=lambda x: True
)

# Inject diversity during evolution
population = inject_diversity(
    population, fitness_scores, generator, 
    injection_rate=0.3, is_expression_valid=lambda x: True,
    generate_high_diversity_expression=lambda g: g.generate_random_expression(),
    # ... other parameters
)
```

### Expression Analysis

```python
from symbolic_regression import to_sympy_expression, get_evolution_stats

# Convert to SymPy for advanced analysis
sympy_expr = to_sympy_expression("X[0]**2 + sin(X[1])")

# Get detailed evolution statistics
stats = regressor.get_evolution_stats()
print(f"Final mutation rate: {stats['final_mutation_rate']}")
print(f"Total generations: {stats['total_generations']}")

# Get detailed expression information
detailed_exprs = regressor.get_detailed_expressions()
for expr_info in detailed_exprs:
    print(f"Expression: {expr_info['expression']}")
    print(f"Complexity: {expr_info['complexity']}")
    print(f"Size: {expr_info['size']}")
```

---

## Configuration Options

### MIMOSymbolicRegressor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `population_size` | int | 100 | Number of individuals in population |
| `generations` | int | 50 | Number of evolution generations |
| `mutation_rate` | float | 0.1 | Initial mutation rate |
| `crossover_rate` | float | 0.8 | Initial crossover rate |
| `tournament_size` | int | 3 | Tournament selection size |
| `max_depth` | int | 6 | Maximum expression tree depth |
| `parsimony_coefficient` | float | 0.001 | Penalty for expression complexity |
| `sympy_simplify` | bool | True | Enable SymPy simplification |
| `advanced_simplify` | bool | False | Enable advanced simplification |
| `diversity_threshold` | float | 0.7 | Diversity threshold for adaptive evolution |
| `adaptive_rates` | bool | True | Enable adaptive mutation/crossover rates |
| `restart_threshold` | int | 25 | Generations before population restart |
| `elite_fraction` | float | 0.1 | Fraction of elite individuals to preserve |
| `console_log` | bool | True | Enable console logging |

---

## Examples

### Basic Regression

```python
import numpy as np
from symbolic_regression import MIMOSymbolicRegressor

# Generate data for y = x^2 + 2*x + 1
X = np.linspace(-10, 10, 100).reshape(-1, 1)
y = X[:, 0]**2 + 2*X[:, 0] + 1

regressor = MIMOSymbolicRegressor(generations=100)
regressor.fit(X, y)

print(f"Discovered expression: {regressor.get_expressions()[0]}")
print(f"R² Score: {regressor.score(X, y):.4f}")
```

### Multi-Output Regression

```python
# Generate multi-output data
X = np.random.rand(100, 2)
y = np.column_stack([
    X[:, 0]**2 + X[:, 1],      # Output 1
    np.sin(X[:, 0]) + X[:, 1]   # Output 2
])

regressor = MIMOSymbolicRegressor(generations=100)
regressor.fit(X, y)

expressions = regressor.get_expressions()
print(f"Output 1 expression: {expressions[0]}")
print(f"Output 2 expression: {expressions[1]}")
```

### Custom Validation

```python
from symbolic_regression import ExpressionValidator

# Custom expression validation
validator = ExpressionValidator()

# Validate expression safety
is_safe = validator.is_safe_expression(expression)
print(f"Expression is safe: {is_safe}")

# Check for numerical stability
is_stable = validator.check_numerical_stability(expression, X)
print(f"Expression is numerically stable: {is_stable}")
```

---

## Performance Tips

1. **Population Size**: Start with 100-200 individuals for complex problems
2. **Generations**: Use 50-100 generations for simple problems, 200+ for complex ones
3. **Max Depth**: Keep between 4-8 to balance complexity and performance
4. **Diversity**: Enable adaptive rates and diversity injection for better exploration
5. **Simplification**: Use SymPy simplification for cleaner final expressions
6. **Memory Pool**: Automatically enabled for efficient expression evaluation

---

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes following the modular architecture
4. Add tests for new functionality
5. Submit a pull request

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## Citation

If you use this package in your research, please cite:

```bibtex
@software{symbolic_regression_mimo,
  title={Symbolic Regression Package for MIMO Systems},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/symbolic_regression}
}
```

---

## Changelog

### Version 0.1.0
- Initial modular architecture implementation
- Advanced expression tree system with optimization
- Enhanced genetic operations with diversity preservation
- Adaptive evolution parameters
- Comprehensive population management
- SymPy integration for expression simplification
- Memory pool optimization for performance
- Detailed evolution statistics and reporting
