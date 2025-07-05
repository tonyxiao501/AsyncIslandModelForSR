# MIMO Symbolic Regression

A Python library for advanced symbolic regression using evolutionary algorithms, supporting Multiple Input Multiple Output (MIMO) regression tasks. The project features diversity preservation, adaptive evolution, ensemble modeling, and advanced symbolic simplification.

---

## Features

- **Multiple Input Multiple Output (MIMO) Symbolic Regression**
- **Genetic Programming** with advanced evolution dynamics
- **Diversity Preservation** and adaptive mutation/crossover rates
- **Population Restart** and elite preservation strategies
- **Ensemble Modeling**: Run multiple regressors in parallel and aggregate the best results
- **SymPy-based Expression Simplification** (basic and advanced)
- **Constant Optimization** using curve fitting
- **Detailed Evolution Statistics** and expression introspection

---

## Installation

Clone the repository and install dependencies:

```bash
git clone <your-repo-url>
cd PythonProject
pip install -r requirements.txt
```

---

## Usage Example

```python
import numpy as np
from symbolic_regression.mimo_regressor import MIMOSymbolicRegressor, EnsembleMIMORegressor

# Generate synthetic data
X = np.random.rand(100, 3)
y = 2 * X[:, 0] + np.sin(X[:, 1]) - 0.5 * X[:, 2]**2

# Fit a single MIMO symbolic regressor
reg = MIMOSymbolicRegressor(population_size=200, generations=100)
reg.fit(X, y)
print("Best Expression:", reg.get_expressions())
print("R² Score:", reg.score(X, y))

# Fit an ensemble of regressors for more robust results
ensemble = EnsembleMIMORegressor(n_fits=4, top_n_select=2, population_size=100, generations=50)
ensemble.fit(X, y)
print("Ensemble Expressions:", ensemble.get_expressions())
print("Ensemble R² Score:", ensemble.score(X, y))
```

---

## Project Structure

```
symbolic_regression/
    mimo_regressor.py
    ensemble_worker.py
    generator.py
    genetic_ops.py
    population.py
    utils.py
    expression_tree/
        __init__.py
        expression.py
        core/
        optimization/
        utils/
examples/
    example_usage.py
    test_example.py
    test_stability.py
requirements.txt
pyproject.toml
```

---

## API Overview

### `MIMOSymbolicRegressor`

- `fit(X, y, constant_optimize=False)`: Fit the model to data.
- `predict(X)`: Predict outputs for new data.
- `score(X, y)`: R² score.
- `get_expressions()`: Get best expressions as strings.
- `get_detailed_expressions()`: Get detailed info (complexity, size, etc.).
- `get_evolution_stats()`: Get evolution statistics.

### `EnsembleMIMORegressor`

- `fit(X, y, constant_optimize=False)`: Fit multiple regressors in parallel.
- `predict(X, strategy='mean'|'best_only')`: Ensemble prediction.
- `score(X, y, strategy='mean'|'best_only')`: Ensemble R² score.
- `get_expressions()`: Get top expressions from the ensemble.

---

## Examples

See the `examples/` directory for scripts demonstrating usage on synthetic and real datasets.

---

## Notes

- For Windows users: When using `EnsembleMIMORegressor`, ensure you call `.fit()` inside a `if __name__ == "__main__":` block due to multiprocessing.
- Expressions are simplified using SymPy; advanced simplification requires additional configuration.
- For reproducibility, consider setting random seeds.

---

## License

MIT License

---

## Contributing

Contributions are welcome! Please open issues or submit pull requests.

---

## Acknowledgements

- [SymPy](https://www.sympy.org/)
- [NumPy](https://numpy.org/)
- [SciPy](https://scipy.org/)
