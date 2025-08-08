"""
Core MIMO Symbolic Regression Implementation
This module contains the main MIMOSymbolicRegressor class with core functionality only.
Evolution logic has been moved to evolution.py for better organization.
"""
import numpy as np
import os
import time
import warnings
from typing import List, Optional, Dict, Any, Tuple

from .expression_tree import Expression
from .generator import ExpressionGenerator
from .expression_tree.utils.sympy_utils import SymPySimplifier

# Import from consolidated modules
from .evolution import EvolutionEngine
from .population_management import PopulationManager, GreatPowers
from .data_processing import DataScaler, standard_fitness_function, r2_score
from .utilities import get_evolution_stats, get_detailed_expressions


class MIMOSymbolicRegressor:
    """Enhanced Multiple Input Multiple Output Symbolic Regression Model with improved evolution dynamics"""

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
                 console_log=True,
                 # New optimization control parameters
                 evolution_sympy_simplify: bool = False,  # Disabled during evolution
                 evolution_constant_optimize: bool = False,  # Disabled during evolution
                 final_optimization_generations: int = 5,  # Apply optimization in final N generations
                 # Early termination and late extension parameters
                 enable_early_termination: bool = True,
                 early_termination_threshold: float = 0.99,
                 early_termination_check_interval: int = 10,
                 enable_late_extension: bool = True,
                 late_extension_threshold: float = 0.95,
                 late_extension_generations: int = 50
                 ):

        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        self.max_depth = max_depth
        self.parsimony_coefficient = parsimony_coefficient
        self.sympy_simplify = sympy_simplify
        self.advanced_simplify = advanced_simplify

        self.console_log = console_log

        # Optimization control parameters
        self.evolution_sympy_simplify = evolution_sympy_simplify
        self.evolution_constant_optimize = evolution_constant_optimize
        self.final_optimization_generations = final_optimization_generations

        # Enhanced evolution parameters
        self.diversity_threshold = diversity_threshold
        self.adaptive_rates = adaptive_rates
        self.restart_threshold = restart_threshold
        self.elite_fraction = elite_fraction

        # REMOVED: All data scaling functionality (deprecated)
        # Now works with raw data only for better physical interpretability

        # Early termination and late extension parameters
        self.enable_early_termination = enable_early_termination
        self.early_termination_threshold = early_termination_threshold
        self.early_termination_check_interval = early_termination_check_interval
        self.enable_late_extension = enable_late_extension
        self.late_extension_threshold = late_extension_threshold
        self.late_extension_generations = late_extension_generations
        self.late_extension_triggered = False  # Track if extension was already triggered

        # Evolution state tracking
        self.stagnation_counter = 0
        self.best_fitness_history = []
        self.diversity_history = []
        self.current_mutation_rate = mutation_rate
        self.current_crossover_rate = crossover_rate
        self.generation_diversity_scores = []

        self.n_inputs: Optional[int] = None
        self.n_outputs: Optional[int] = None
        self.best_expressions: List[Expression] = []
        self.fitness_history: List[float] = []
        
        # Population manager (will be initialized when n_inputs is set)
        self.pop_manager: Optional[PopulationManager] = None

        # Great Powers mechanism - tracks best 5 expressions across all generations
        self.great_powers = GreatPowers(max_powers=5)

        # Inter-thread communication components
        self.shared_manager = None
        self.worker_id = None
        self.inter_thread_enabled = False

        # Debug CSV tracking
        self.debug_csv_path = None
        self.debug_worker_id = None

        if self.advanced_simplify:
            self.sympy_simplifier = SymPySimplifier()

        # Evolution engine (will be initialized in fit())
        self.evolution_engine: Optional[EvolutionEngine] = None

    def enable_inter_thread_communication(self, shared_data, worker_id: int):
        """Enable inter-thread communication for this regressor instance"""
        self.shared_manager = shared_data
        self.worker_id = worker_id
        self.inter_thread_enabled = True
        if self.console_log:
            print(f"Worker {worker_id}: Inter-thread communication enabled")

    def set_debug_csv_path(self, debug_csv_path: str, worker_id: int):
        """Set the debug CSV path for tracking evolution progress"""
        self.debug_csv_path = debug_csv_path
        self.debug_worker_id = worker_id

    def fit(self, X: np.ndarray, y: np.ndarray, constant_optimize=False):
        """Enhanced fit with diversity preservation and adaptive evolution"""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        # Work with raw data (no scaling)
        X_scaled, y_scaled = X.copy(), y.copy()

        self.n_inputs = X_scaled.shape[1]
        self.n_outputs = y_scaled.shape[1]
        
        # Initialize population manager
        if self.n_inputs is None:
            raise ValueError("n_inputs must be set before generating population")
        self.pop_manager = PopulationManager(self.n_inputs, self.max_depth)

        # Reset evolution state
        self.fitness_history = []
        self.best_fitness_history = []
        self.diversity_history = []
        self.stagnation_counter = 0
        self.best_expressions = []
        self.current_mutation_rate = self.mutation_rate
        self.current_crossover_rate = self.crossover_rate

        # Initialize evolution engine
        self.evolution_engine = EvolutionEngine(
            regressor=self,
            pop_manager=self.pop_manager,
            great_powers=self.great_powers,
            console_log=self.console_log
        )

        if self.console_log:
            print(f"Starting evolution with {self.n_inputs} inputs and {self.n_outputs} outputs")
            print(f"Population size: {self.population_size}, Generations: {self.generations}")
            print("Using raw data without scaling for better physical interpretability")

        # Run evolution using the evolution engine
        self.best_expressions = self.evolution_engine.run_evolution(
            X_scaled, y_scaled, X, y, constant_optimize
        )

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the best expressions"""
        if not self.best_expressions:
            raise ValueError("Model must be fitted before making predictions")

        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # Use raw data (no scaling)
        X_scaled = X.copy()

        predictions = []
        for expr in self.best_expressions:
            try:
                pred = expr.evaluate(X_scaled)
                if pred.ndim == 1:
                    pred = pred.reshape(-1, 1)
                predictions.append(pred)
            except Exception as e:
                if self.console_log:
                    print(f"Warning: Prediction failed for expression: {e}")
                # Fallback to zeros
                predictions.append(np.zeros((X_scaled.shape[0], 1)))

        if not predictions:
            return np.zeros((X.shape[0], self.n_outputs or 1))

        # Combine predictions (use first expression for single output)
        if len(predictions) == 1:
            result = predictions[0]
        else:
            # For multiple expressions, concatenate along output dimension
            result = np.hstack(predictions)

        # Return raw predictions (no inverse scaling)
        return result

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate the R² score of the model on the given test data"""
        predictions = self.predict(X)
        y = np.asarray(y, dtype=np.float64)
        
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        
        try:
            # Handle multi-output case by computing mean R² across outputs
            if y.shape[1] > 1:
                r2_scores = []
                for i in range(y.shape[1]):
                    if i < predictions.shape[1]:
                        r2 = r2_score(y[:, i], predictions[:, i])
                        r2_scores.append(r2)
                return float(np.mean(r2_scores)) if r2_scores else 0.0
            else:
                return r2_score(y.flatten(), predictions.flatten())
        
        except Exception as e:
            if self.console_log:
                print(f"Warning: R² calculation failed: {e}")
            return 0.0

    def get_expressions(self) -> List[str]:
        """Get the best expressions as strings (in original data scale if scaling was used)"""
        if not self.best_expressions:
            return []

        expressions = []
        for expr in self.best_expressions:
            expr_str = expr.to_string()
            
            # Apply simplification
            if self.sympy_simplify:
                from .utilities import to_sympy_expression
                simplified = to_sympy_expression(expr_str, self.advanced_simplify, self.n_inputs, enable_simplify=True)
                expressions.append(simplified if simplified else expr_str)
            else:
                expressions.append(expr_str)

        return expressions
    
    def get_scaled_expressions(self) -> List[str]:
        """Get expressions in scaled data space (useful for debugging)"""
        if not self.best_expressions:
            return []

        expressions = []
        for expr in self.best_expressions:
            expr_str = expr.to_string()
            if self.sympy_simplify:
                from .utilities import to_sympy_expression
                simplified = to_sympy_expression(expr_str, self.advanced_simplify, self.n_inputs, enable_simplify=True)
                expressions.append(simplified if simplified else expr_str)
            else:
                expressions.append(expr_str)

        return expressions

    def get_expr_obj(self) -> List[Expression]:
        """Get the best expression objects"""
        return self.best_expressions

    def get_raw_expressions(self) -> List[str]:
        """Get raw expressions without simplification"""
        return [expr.to_string() for expr in self.best_expressions]

    def get_detailed_expressions(self) -> List[Dict]:
        """Get detailed information about expressions"""
        return get_detailed_expressions(self.best_expressions, self.sympy_simplify)

    def get_evolution_stats(self) -> Dict[str, Any]:
        """Get detailed evolution statistics"""
        return get_evolution_stats(
            self.fitness_history, self.best_fitness_history, self.diversity_history,
            self.current_mutation_rate, self.current_crossover_rate, self.stagnation_counter
        )
