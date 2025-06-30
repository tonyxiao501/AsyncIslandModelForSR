import numpy as np
import random
from typing import List, Optional
from .expression_tree import Expression
from .generator import ExpressionGenerator
from .genetic_ops import GeneticOperations

class MIMOSymbolicRegressor:
    """Multiple Input Multiple Output Symbolic Regression Model"""

    def __init__(self,
                 population_size: int = 100,
                 generations: int = 50,
                 mutation_rate: float = 0.1,
                 crossover_rate: float = 0.8,
                 tournament_size: int = 3,
                 max_depth: int = 6,
                 parsimony_coefficient: float = 0.001):

        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        self.max_depth = max_depth
        self.parsimony_coefficient = parsimony_coefficient

        self.n_inputs: Optional[int] = None
        self.n_outputs: Optional[int] = None
        self.best_expressions: List[Expression] = []
        self.fitness_history: List[float] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'MIMOSymbolicRegressor':
        """Fit the symbolic regression model to MIMO data"""

        # Validate input shapes
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        self.n_inputs = X.shape[1]
        self.n_outputs = y.shape[1]

        # Type assertion for type checker - we know these are not None after assignment
        assert self.n_inputs is not None
        assert self.n_outputs is not None

        print(f"Training MIMO model: {self.n_inputs} inputs -> {self.n_outputs} outputs")

        # Initialize components - now we can safely pass n_inputs
        generator = ExpressionGenerator(self.n_inputs, self.max_depth)
        genetic_ops = GeneticOperations(self.n_inputs)

        # Create separate populations for each output
        populations: List[List[Expression]] = []
        for output_idx in range(self.n_outputs):
            population = generator.generate_population(self.population_size)
            populations.append(population)

        # Evolution loop
        for generation in range(self.generations):
            generation_fitness: List[float] = []

            # Evolve each output separately
            for output_idx in range(self.n_outputs):
                population = populations[output_idx]
                target = y[:, output_idx]

                # Evaluate fitness
                fitness_scores: List[float] = []
                for expr in population:
                    fitness = self._evaluate_fitness(expr, X, target)
                    fitness_scores.append(fitness)

                # Selection and reproduction
                new_population: List[Expression] = []

                # Keep best individuals (elitism)
                elite_size = self.population_size // 10
                elite_indices = np.argsort(fitness_scores)[:elite_size]
                for idx in elite_indices:
                    new_population.append(population[idx].copy())

                # Generate offspring
                while len(new_population) < self.population_size:
                    if random.random() < self.crossover_rate:
                        # Crossover
                        parent1 = self._tournament_selection(population, fitness_scores)
                        parent2 = self._tournament_selection(population, fitness_scores)
                        child1, child2 = genetic_ops.crossover(parent1, parent2)
                        new_population.extend([child1, child2])
                    else:
                        # Mutation
                        parent = self._tournament_selection(population, fitness_scores)
                        child = genetic_ops.mutate(parent, self.mutation_rate)
                        new_population.append(child)

                # Trim population to exact size
                populations[output_idx] = new_population[:self.population_size]
                generation_fitness.append(min(fitness_scores))

            # Track progress
            avg_fitness = float(np.mean(generation_fitness))
            self.fitness_history.append(avg_fitness)

            if generation % 10 == 0:
                print(f"Generation {generation}: Average fitness = {avg_fitness:.6f}")

        # Store best expressions for each output
        self.best_expressions = []
        for output_idx in range(self.n_outputs):
            population = populations[output_idx]
            target = y[:, output_idx]

            fitness_scores = [self._evaluate_fitness(expr, X, target)
                              for expr in population]
            best_idx = int(np.argmin(fitness_scores))
            self.best_expressions.append(population[best_idx])

            print(f"Output {output_idx} best expression: {population[best_idx].to_string()}")
            print(f"Output {output_idx} best fitness: {fitness_scores[best_idx]:.6f}")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the evolved expressions"""
        if not self.best_expressions:
            raise ValueError("Model has not been fitted yet")

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        predictions: List[np.ndarray] = []
        for expr in self.best_expressions:
            pred = expr.evaluate(X)
            predictions.append(pred)

        return np.column_stack(predictions)

    def _evaluate_fitness(self, expression: Expression, X: np.ndarray, y: np.ndarray) -> float:
        """Evaluate fitness of an expression (lower is better)"""
        try:
            predictions = expression.evaluate(X)

            # Mean squared error
            mse = float(np.mean((predictions - y) ** 2))

            # Add parsimony pressure (penalize complex expressions)
            complexity_penalty = float(self.parsimony_coefficient * expression.size())

            return mse + complexity_penalty

        except Exception:
            return float('inf')

    def _tournament_selection(self, population: List[Expression],
                              fitness_scores: List[float]) -> Expression:
        """Select individual using tournament selection"""
        tournament_indices = np.random.choice(len(population),
                                              size=self.tournament_size,
                                              replace=False)
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmin(tournament_fitness)]
        return population[winner_idx]

    def get_expressions(self) -> List[str]:
        """Get string representations of the best expressions"""
        if not self.best_expressions:
            raise ValueError("Model has not been fitted yet")
        return [expr.to_string() for expr in self.best_expressions]

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate R² score for the model"""
        if not self.best_expressions:
            raise ValueError("Model has not been fitted yet")

        predictions = self.predict(X)

        if y.ndim == 1:
            y = y.reshape(-1, 1)

        ss_res = float(np.sum((y - predictions) ** 2))
        ss_tot = float(np.sum((y - np.mean(y, axis=0)) ** 2))

        if ss_tot == 0:
            return 1.0

        return 1.0 - (ss_res / ss_tot)