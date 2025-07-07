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

### Basic Usage

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

# Get the best expression
best_expr = regressor.get_best_expression()
print(f"Best expression: {best_expr.to_string()}")
```

### Ensemble Modeling

```python
from symbolic_regression import EnsembleMIMORegressor

# Create ensemble regressor
ensemble = EnsembleMIMORegressor(
    n_fits=8,           # Number of concurrent fits
    top_n_select=5,     # Number of best expressions to select
    population_size=100,
    generations=50
)

# Fit the ensemble (use within if __name__ == "__main__": block)
if __name__ == "__main__":
    ensemble.fit(X, y)
    
    # Get ensemble predictions
    predictions = ensemble.predict(X)
    
    # Get best expressions from ensemble
    best_expressions = ensemble.get_best_expressions()
```

---

## Module Structure

The package is organized into several modular components:

### Core Components

- **`expression_tree/`**: Core expression tree functionality
  - `expression.py`: Expression class with caching and evaluation
  - `core/`: Node definitions and operators
  - `optimization/`: Memory pool for efficient node allocation
  - `utils/`: Simplification, validation, and utility functions

- **`mimo_regressor.py`**: Main MIMO symbolic regression implementation
- **`ensemble_regressor.py`**: Ensemble modeling with parallel processing
- **`genetic_ops.py`**: Genetic operations (mutation, crossover, selection)
- **`population.py`**: Population management and diversity functions
- **`adaptive_evolution.py`**: Adaptive parameters and restart mechanisms

### Utility Modules

- **`generator.py`**: Expression generation utilities
- **`selection.py`**: Selection strategies for genetic programming
- **`utils.py`**: General utility functions
- **`expression_utils.py`**: Expression manipulation utilities
- **`evolution_stats.py`**: Evolution statistics and monitoring
- **`constant_optimization.py`**: Constant optimization algorithms

---

## Advanced Features

### Expression Tree System

The expression tree system provides:

- **Efficient Node Types**: Variable, constant, binary operation, and unary operation nodes
- **Memory Pool**: High-performance memory allocation for nodes
- **Caching**: Intelligent caching of expression evaluations
- **Simplification**: Advanced simplification using SymPy integration

### Genetic Programming

Advanced genetic programming features include:

- **Adaptive Rates**: Mutation and crossover rates adapt based on population diversity
- **Population Restart**: Automatic restart when evolution stagnates
- **Elite Preservation**: Maintains best solutions across generations
- **Diversity Injection**: Ensures population diversity to prevent premature convergence

### Expression Validation

- **Syntax Validation**: Ensures expressions are syntactically correct
- **Semantic Validation**: Checks for mathematical validity
- **Complexity Control**: Manages expression complexity and depth

---

## API Reference

### MIMOSymbolicRegressor

```python
class MIMOSymbolicRegressor:
    def __init__(self,
                 population_size: int = 100,
                 generations: int = 50,
                 mutation_rate: float = 0.1,
                 crossover_rate: float = 0.8,
                 tournament_size: int = 3,
                 max_depth: int = 6,
                 parsimony_coefficient: float = 0.001,
                 sympy_simplify: bool = True,
                 advanced_simplify: bool = False,
                 diversity_threshold: float = 0.7,
                 adaptive_rates: bool = True,
                 restart_threshold: int = 25,
                 elite_fraction: float = 0.1,
                 console_log: bool = True)
    
    def fit(self, X: np.ndarray, y: np.ndarray, constant_optimize: bool = False)
    def predict(self, X: np.ndarray) -> np.ndarray
    def get_best_expression(self) -> Expression
    def get_evolution_stats(self) -> dict
```

### EnsembleMIMORegressor

```python
class EnsembleMIMORegressor:
    def __init__(self, n_fits: int = 8, top_n_select: int = 5, **regressor_kwargs)
    
    def fit(self, X: np.ndarray, y: np.ndarray, constant_optimize: bool = False)
    def predict(self, X: np.ndarray) -> np.ndarray
    def get_best_expressions(self) -> List[Expression]
    def get_ensemble_stats(self) -> dict
```

### Expression

```python
class Expression:
    def __init__(self, root: Node)
    
    def evaluate(self, X: np.ndarray) -> np.ndarray
    def to_string(self) -> str
    def to_sympy(self) -> sp.Expr
    def complexity(self) -> float
    def copy(self) -> Expression
```

---

## Examples

See the `examples/` directory for comprehensive usage examples:

- `example_usage.py`: Basic usage demonstration
- `test_example.py`: Testing with synthetic data
- `test_stability.py`: Stability and performance testing

---

## Performance Considerations

- **Memory Pool**: Uses optimized memory allocation for faster expression evaluation
- **Numba JIT**: Critical evaluation functions are JIT-compiled for speed
- **Parallel Processing**: Ensemble modeling leverages multiprocessing
- **Caching**: Intelligent caching reduces redundant computations

---

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## Citation

If you use this package in your research, please cite:

```bibtex
@software{symbolic_regression_package,
  title={Symbolic Regression Package: A Modular Genetic Programming Approach},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/symbolic-regression}
}
```
