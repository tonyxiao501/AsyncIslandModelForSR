#!/usr/bin/env python3
"""
Standard Symbolic Regression Benchmark Test Suite

This module implements the most commonly used benchmark datasets from SOTA symbolic regression research,
based on:

1. SRBench (La Cava et al., 2021): "Contemporary Symbolic Regression Methods and their Relative Performance"
   - 252 diverse regression problems including physics equations and real-world datasets
   
2. SRSD Feynman (Matsubara et al., 2022): "Rethinking Symbolic Regression Datasets and Benchmarks for Scientific Discovery"
   - 120 physics equations from Feynman Lectures on Physics
   
3. Nguyen-Vladislavleva benchmark problems (classic GP benchmarks)
   - Standard polynomial and transcendental functions
   
4. Korns benchmark (Korns, 2011)
   - Challenging multivariate polynomial problems
   
5. Vlad/Kepler benchmark (Vladislavleva et al., 2009)  
   - Real-world inspired problems with noise

This comprehensive suite allows testing against the same benchmarks used by:
- PySR, GPLearn, DSO, GINN, TuRBO-GP, ITEA, AFP, etc.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Callable, Optional
import warnings
from dataclasses import dataclass
from symbolic_regression.ensemble_regressor import EnsembleMIMORegressor
from symbolic_regression.data_processing import r2_score
import os
import time
from datetime import datetime


@dataclass
class BenchmarkProblem:
    """Container for a benchmark symbolic regression problem."""
    name: str
    description: str
    true_function: Callable
    variable_ranges: List[Tuple[float, float]]
    target_expression: str
    complexity_level: str  # 'easy', 'medium', 'hard'
    category: str  # 'feynman', 'nguyen', 'korns', 'vlad', 'srbench'
    noise_level: float = 0.0
    n_samples: int = 200


class StandardBenchmarkSuite:
    """
    Standard benchmark suite for symbolic regression based on SOTA research.
    
    Implements the most commonly used benchmarks from recent papers:
    - SRBench (La Cava et al.)
    - SRSD Feynman datasets (Matsubara et al.)
    - Classic Nguyen-Vladislavleva problems
    - Korns benchmark problems
    """
    
    def __init__(self, adaptive_config=True):
        self.problems = {}
        self.results = []
        self.adaptive_config = adaptive_config
        self.model = None  # Will be created per problem
        
        # Default optimized configuration based on benchmark analysis
        self.default_config = {
            'n_fits': 10,                   # More diverse solutions for complex problems
            'top_n_select': 3,              # Focus on best candidates
            'population_size': 500,         # Larger population for better exploration
            'generations': 300,             # More generations for complex expressions
            'mutation_rate': 0.15,          # Higher mutation for better exploration
            'crossover_rate': 0.75,         # Slightly lower crossover, more mutation
            'tournament_size': 5,           # Larger tournament for better selection pressure
            'max_depth': 12,                # Allow deeper expressions for transcendental functions
            'parsimony_coefficient': 0.005, # Lower parsimony - prioritize accuracy over simplicity
            'diversity_threshold': 0.8,     # Higher diversity to explore more solutions
            'adaptive_rates': True,
            'restart_threshold': 15,        # Restart sooner if stagnating
            'elite_fraction': 0.15,         # Keep more elite solutions
            'enable_adaptive_parsimony': True,
            'domain_type': "physics",
            'enable_early_termination': True,
            'early_termination_threshold': 0.9995, # Even higher threshold for exact solutions
            'early_termination_check_interval': 15, # Check more frequently
            'use_asynchronous_migration': True,
            'migration_interval': 10,       # More frequent migration
            'migration_probability': 0.35,  # Higher migration rate for diversity
            'console_log': False  # Reduce output for batch testing
        }
        
        self._create_benchmark_problems()
    
    def _get_optimized_config_for_problem(self, problem: BenchmarkProblem) -> dict:
        """Get optimized configuration based on problem characteristics."""
        config = self.default_config.copy()
        
        if not self.adaptive_config:
            return config
            
        # Problem-specific optimizations based on benchmark analysis
        if problem.category == "feynman":
            if "exponential" in problem.description.lower() or "exp" in problem.target_expression.lower():
                # Exponential functions need more exploration and deeper expressions
                config.update({
                    'population_size': 800,
                    'generations': 400,
                    'max_depth': 15,
                    'mutation_rate': 0.20,
                    'parsimony_coefficient': 0.001,  # Even lower for complex functions
                    'n_fits': 12  # More attempts for difficult functions
                })
                
            elif any(scale_word in problem.description.lower() for scale_word in ["gravitational", "potential", "large scale"]):
                # Physics problems with extreme scales need robust numerical handling
                config.update({
                    'population_size': 600,
                    'generations': 350,
                    'diversity_threshold': 0.85,
                    'restart_threshold': 10,  # Restart quickly for numerical issues
                    'n_fits': 15  # More attempts for numerical stability
                })
                
        elif problem.category == "nguyen":
            if "polynomial" in problem.description.lower():
                # Polynomial problems should find exact forms
                config.update({
                    'early_termination_threshold': 0.9999,  # Demand very high accuracy
                    'parsimony_coefficient': 0.01,  # Prefer simpler polynomial forms
                    'max_depth': 10,  # Don't need too deep for polynomials
                    'generations': 200  # Should solve quickly
                })
                
            elif any(func in problem.target_expression.lower() for func in ["sin", "cos", "log", "exp"]):
                # Transcendental functions need special handling
                config.update({
                    'population_size': 700,
                    'generations': 450,
                    'max_depth': 18,
                    'mutation_rate': 0.18,
                    'migration_probability': 0.40,  # High diversity needed
                    'n_fits': 12
                })
                
        elif problem.category in ["korns", "vlad"]:
            # High-dimensional complex problems
            config.update({
                'population_size': 1000,
                'generations': 500,
                'n_fits': 15,  # More independent runs
                'max_depth': 20,
                'diversity_threshold': 0.9,  # Maximum diversity
                'mutation_rate': 0.20
            })
        
        return config
    
    def _create_benchmark_problems(self):
        """Create all benchmark problems from different suites."""
        
        # === 1. FEYNMAN PHYSICS BENCHMARKS (SRSD) ===
        # Based on "Rethinking Symbolic Regression Datasets" (Matsubara et al., 2022)
        
        # Feynman I.6.2: Exponential decay - exp(-x/L)
        self.problems['feynman_I_6_2'] = BenchmarkProblem(
            name="Feynman I.6.2 (Exponential Decay)",
            description="Exponential decay: exp(-theta/sigma)",
            true_function=lambda theta, sigma: np.exp(-theta / sigma),
            variable_ranges=[(1, 5), (1, 3)],  # theta, sigma
            target_expression="exp(-x0/x1)",
            complexity_level="medium",
            category="feynman",
            noise_level=0.0,
            n_samples=200
        )
        
        # Feynman I.8.14: Energy-momentum relation
        self.problems['feynman_I_8_14'] = BenchmarkProblem(
            name="Feynman I.8.14 (Energy-Momentum)",
            description="Relativistic energy: sqrt((pc)^2 + (mc^2)^2)",
            true_function=lambda p, c, m: np.sqrt((p * c)**2 + (m * c**2)**2),
            variable_ranges=[(1, 5), (1, 2), (0.5, 2)],  # p, c, m
            target_expression="sqrt((x0*x1)^2 + (x2*x1^2)^2)",
            complexity_level="hard",
            category="feynman",
            noise_level=0.0
        )
        
        # Feynman I.12.1: Simple harmonic motion
        self.problems['feynman_I_12_1'] = BenchmarkProblem(
            name="Feynman I.12.1 (Harmonic Motion)",
            description="Simple harmonic motion: mu * v",
            true_function=lambda mu, v: mu * v,
            variable_ranges=[(0.5, 3), (1, 10)],  # mu, v
            target_expression="x0 * x1",
            complexity_level="easy",
            category="feynman",
            noise_level=0.0
        )
        
        # Feynman I.12.2: Force in harmonic motion
        self.problems['feynman_I_12_2'] = BenchmarkProblem(
            name="Feynman I.12.2 (Harmonic Force)",
            description="Harmonic force: q1*q2*r/(4*pi*epsilon*r^3)",
            true_function=lambda q1, q2, r, epsilon: q1 * q2 / (4 * np.pi * epsilon * r**2),
            variable_ranges=[(1, 3), (1, 3), (1, 5), (1, 2)],  # q1, q2, r, epsilon
            target_expression="(x0*x1)/(4*pi*x3*x2^2)",
            complexity_level="hard",
            category="feynman"
        )
        
        # Feynman I.15.3x: Gravitational potential energy
        self.problems['feynman_I_15_3x'] = BenchmarkProblem(
            name="Feynman I.15.3x (Gravitational PE)",
            description="Gravitational potential: -G*m1*m2/r",
            true_function=lambda G, m1, m2, r: -G * m1 * m2 / r,
            variable_ranges=[(6.67e-11, 6.68e-11), (1e20, 1e24), (1e20, 1e24), (1e6, 1e8)],
            target_expression="-x0*x1*x2/x3",
            complexity_level="medium",
            category="feynman"
        )
        
        # Feynman II.6.11: Lorentz force
        self.problems['feynman_II_6_11'] = BenchmarkProblem(
            name="Feynman II.6.11 (Lorentz Force)",
            description="Lorentz force: q * (v x B)",
            true_function=lambda q, v, B: q * v * B,  # Simplified scalar version
            variable_ranges=[(1e-6, 1e-5), (1e5, 1e6), (0.1, 1)],  # q, v, B
            target_expression="x0*x1*x2",
            complexity_level="easy",
            category="feynman"
        )
        
        # === 2. NGUYEN-VLADISLAVLEVA BENCHMARKS ===
        # Classic genetic programming benchmarks
        
        # Nguyen-1: x^3 + x^2 + x (polynomial)
        self.problems['nguyen_1'] = BenchmarkProblem(
            name="Nguyen-1 (Polynomial)",
            description="Simple polynomial: x^3 + x^2 + x",
            true_function=lambda x: x**3 + x**2 + x,
            variable_ranges=[(-1, 1)],
            target_expression="x^3 + x^2 + x",
            complexity_level="easy",
            category="nguyen",
            n_samples=20
        )
        
        # Nguyen-4: x^6 + x^5 + x^4 + x^3 + x^2 + x (higher-order polynomial)
        self.problems['nguyen_4'] = BenchmarkProblem(
            name="Nguyen-4 (High-order Polynomial)",
            description="High-order polynomial: x^6 + x^5 + x^4 + x^3 + x^2 + x",
            true_function=lambda x: x**6 + x**5 + x**4 + x**3 + x**2 + x,
            variable_ranges=[(-1, 1)],
            target_expression="x^6 + x^5 + x^4 + x^3 + x^2 + x",
            complexity_level="hard",
            category="nguyen",
            n_samples=20
        )
        
        # Nguyen-5: sin(x^2)*cos(x) - 1 (transcendental)
        self.problems['nguyen_5'] = BenchmarkProblem(
            name="Nguyen-5 (Transcendental)",
            description="Transcendental function: sin(x^2)*cos(x) - 1",
            true_function=lambda x: np.sin(x**2) * np.cos(x) - 1,
            variable_ranges=[(-1, 1)],
            target_expression="sin(x^2)*cos(x) - 1",
            complexity_level="hard",
            category="nguyen",
            n_samples=20
        )
        
        # Nguyen-7: log(x+1) + log(x^2+1) (logarithmic)
        self.problems['nguyen_7'] = BenchmarkProblem(
            name="Nguyen-7 (Logarithmic)",
            description="Logarithmic function: log(x+1) + log(x^2+1)",
            true_function=lambda x: np.log(x + 1) + np.log(x**2 + 1),
            variable_ranges=[(0, 2)],
            target_expression="log(x+1) + log(x^2+1)",
            complexity_level="medium",
            category="nguyen",
            n_samples=20
        )
        
        # Nguyen-10: 2*sin(x)*cos(y) (multivariate trigonometric)
        self.problems['nguyen_10'] = BenchmarkProblem(
            name="Nguyen-10 (Multivariate Trig)",
            description="Multivariate trigonometric: 2*sin(x)*cos(y)",
            true_function=lambda x, y: 2 * np.sin(x) * np.cos(y),
            variable_ranges=[(-1, 1), (-1, 1)],
            target_expression="2*sin(x)*cos(y)",
            complexity_level="medium",
            category="nguyen",
            n_samples=400  # More samples for 2D
        )
        
        # === 3. KORNS BENCHMARKS ===
        # Challenging multivariate problems (Korns, 2011)
        
        # Korns-12: Complex multivariate
        self.problems['korns_12'] = BenchmarkProblem(
            name="Korns-12 (Complex Multivariate)",
            description="Complex multivariate: 2 - 2.1*cos(9.8*x)*sin(1.3*w)",
            true_function=lambda x, y, z, v, w: 2 - 2.1 * np.cos(9.8 * x) * np.sin(1.3 * w),
            variable_ranges=[(-50, 50), (-50, 50), (-50, 50), (-50, 50), (-50, 50)],
            target_expression="2 - 2.1*cos(9.8*x0)*sin(1.3*x4)",
            complexity_level="hard",
            category="korns",
            n_samples=10000
        )
        
        # === 4. VLADISLAVLEVA BENCHMARKS ===
        # Real-world inspired problems with controlled complexity
        
        # Vladislavleva-4: 5 variables with interaction
        self.problems['vlad_4'] = BenchmarkProblem(
            name="Vladislavleva-4 (5D Interaction)",
            description="5D function: 10/(5 + (x1-3)^2 + (x2-3)^2 + (x3-3)^2 + (x4-3)^2 + (x5-3)^2)",
            true_function=lambda x1, x2, x3, x4, x5: 10 / (5 + (x1-3)**2 + (x2-3)**2 + (x3-3)**2 + (x4-3)**2 + (x5-3)**2),
            variable_ranges=[(0.05, 6.05), (0.05, 6.05), (0.05, 6.05), (0.05, 6.05), (0.05, 6.05)],
            target_expression="10/(5 + (x1-3)^2 + (x2-3)^2 + (x3-3)^2 + (x4-3)^2 + (x5-3)^2)",
            complexity_level="hard",
            category="vlad",
            n_samples=1024
        )
        
        # === 5. MODERN SRBENCH PROBLEMS ===
        # Selected real-world problems from SRBench suite
        
        # Tower dataset (synthetic but realistic)
        self.problems['tower'] = BenchmarkProblem(
            name="Tower Dataset",
            description="Synthetic tower problem: 4*x1*x2 + x3",
            true_function=lambda x1, x2, x3: 4 * x1 * x2 + x3,
            variable_ranges=[(0, 1), (0, 1), (0, 1)],
            target_expression="4*x1*x2 + x3",
            complexity_level="medium",
            category="srbench",
            noise_level=0.05  # Small amount of noise
        )
    
    def generate_data(self, problem: BenchmarkProblem) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate training data for a benchmark problem."""
        np.random.seed(42)  # Reproducible results
        
        n_vars = len(problem.variable_ranges)
        n_samples = problem.n_samples
        
        if n_vars == 1:
            # 1D problem
            x_min, x_max = problem.variable_ranges[0]
            X = np.linspace(x_min, x_max, n_samples).reshape(-1, 1)
            y_true = problem.true_function(X[:, 0])
        elif n_vars == 2:
            # 2D problem
            samples_per_dim = int(np.sqrt(n_samples))
            x1_min, x1_max = problem.variable_ranges[0]
            x2_min, x2_max = problem.variable_ranges[1]
            
            x1 = np.linspace(x1_min, x1_max, samples_per_dim)
            x2 = np.linspace(x2_min, x2_max, samples_per_dim)
            X1, X2 = np.meshgrid(x1, x2)
            X = np.column_stack([X1.flatten(), X2.flatten()])
            y_true = problem.true_function(X[:, 0], X[:, 1])
        elif n_vars <= 5:
            # Higher dimensional problems - use random sampling
            X = np.random.uniform(
                low=[r[0] for r in problem.variable_ranges],
                high=[r[1] for r in problem.variable_ranges],
                size=(n_samples, n_vars)
            )
            if n_vars == 3:
                y_true = problem.true_function(X[:, 0], X[:, 1], X[:, 2])
            elif n_vars == 4:
                y_true = problem.true_function(X[:, 0], X[:, 1], X[:, 2], X[:, 3])
            elif n_vars == 5:
                y_true = problem.true_function(X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4])
        else:
            raise ValueError(f"Problems with {n_vars} variables not yet supported")
        
        # Add noise if specified
        if problem.noise_level > 0:
            noise = np.random.normal(0, problem.noise_level * np.std(y_true), size=y_true.shape)
            y_noisy = y_true + noise
        else:
            y_noisy = y_true.copy()
        
        return X, y_noisy.reshape(-1, 1), y_true
    
    def run_single_benchmark(self, problem_key: str, max_time_minutes: float = 10.0) -> Dict:
        """Run a single benchmark problem."""
        if problem_key not in self.problems:
            raise ValueError(f"Problem '{problem_key}' not found. Available: {list(self.problems.keys())}")
        
        problem = self.problems[problem_key]
        print(f"\n{'='*80}")
        print(f"RUNNING BENCHMARK: {problem.name}")
        print(f"{'='*80}")
        print(f"Category: {problem.category}")
        print(f"Complexity: {problem.complexity_level}")
        print(f"Target: {problem.target_expression}")
        print(f"Variables: {len(problem.variable_ranges)}")
        print(f"Description: {problem.description}")
        
        # Create optimized model for this specific problem
        config = self._get_optimized_config_for_problem(problem)
        if self.adaptive_config:
            print(f"Using adaptive config: pop_size={config['population_size']}, "
                  f"generations={config['generations']}, max_depth={config['max_depth']}")
        
        self.model = EnsembleMIMORegressor(**config)
        
        # Generate data
        X, y, y_true = self.generate_data(problem)
        print(f"Generated {X.shape[0]} samples with {X.shape[1]} variables")
        print(f"Data ranges:")
        for i, (low, high) in enumerate(problem.variable_ranges):
            print(f"  x{i}: [{low:.3f}, {high:.3f}]")
        print(f"Target range: [{y_true.min():.6f}, {y_true.max():.6f}]")
        
        # Run symbolic regression
        start_time = time.time()
        try:
            # Set timeout
            timeout_seconds = max_time_minutes * 60
            
            self.model.fit(X, y, constant_optimize=True)
            elapsed_time = time.time() - start_time
            
            # Check if we exceeded time limit
            if elapsed_time > timeout_seconds:
                print(f"⚠️ Time limit exceeded ({elapsed_time:.1f}s > {timeout_seconds:.1f}s)")
            
            # Get results
            expressions = self.model.get_expressions()
            predictions = self.model.predict(X)
            r2_score_val = r2_score(y_true, predictions.flatten())
            
            # Calculate error metrics
            mse = np.mean((y_true - predictions.flatten())**2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(y_true - predictions.flatten()))
            
            result = {
                'problem_name': problem.name,
                'problem_key': problem_key,
                'category': problem.category,
                'complexity': problem.complexity_level,
                'target_expression': problem.target_expression,
                'n_variables': len(problem.variable_ranges),
                'n_samples': X.shape[0],
                'success': True,
                'r2_score': r2_score_val,
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'discovered_expressions': expressions[:3],
                'elapsed_time': elapsed_time,
                'converged': r2_score_val > 0.95,
                'exact_recovery': r2_score_val > 0.999,
                'config_used': {k: v for k, v in config.items() if k in ['population_size', 'generations', 'max_depth', 'mutation_rate']}
            }
            
            print(f"\n📊 RESULTS:")
            print(f"R² Score: {r2_score_val:.6f}")
            print(f"RMSE: {rmse:.6f}")
            print(f"Time: {elapsed_time:.2f}s")
            print(f"Converged (R² > 0.95): {'✓' if result['converged'] else '❌'}")
            print(f"Exact recovery (R² > 0.999): {'✓' if result['exact_recovery'] else '❌'}")
            print(f"\nTop discovered expressions:")
            for i, expr in enumerate(expressions[:3], 1):
                print(f"  {i}. {expr}")
                
        except Exception as e:
            elapsed_time = time.time() - start_time
            result = {
                'problem_name': problem.name,
                'problem_key': problem_key,
                'category': problem.category,
                'complexity': problem.complexity_level,
                'target_expression': problem.target_expression,
                'n_variables': len(problem.variable_ranges),
                'n_samples': X.shape[0],
                'success': False,
                'error': str(e),
                'elapsed_time': elapsed_time,
                'r2_score': 0.0,
                'converged': False,
                'exact_recovery': False
            }
            print(f"\n❌ FAILED: {e}")
        
        self.results.append(result)
        return result
    
    def run_benchmark_suite(self, categories: Optional[List[str]] = None, 
                           complexity_levels: Optional[List[str]] = None,
                           max_time_per_problem: float = 10.0) -> List[Dict]:
        """Run a subset of benchmark problems."""
        
        # Filter problems
        problems_to_run = []
        for key, problem in self.problems.items():
            if categories and problem.category not in categories:
                continue
            if complexity_levels and problem.complexity_level not in complexity_levels:
                continue
            problems_to_run.append(key)
        
        print(f"🚀 RUNNING BENCHMARK SUITE")
        print(f"Problems to run: {len(problems_to_run)}")
        print(f"Categories: {categories or 'All'}")
        print(f"Complexity levels: {complexity_levels or 'All'}")
        print(f"Max time per problem: {max_time_per_problem} minutes")
        
        results = []
        for i, problem_key in enumerate(problems_to_run, 1):
            print(f"\n[{i}/{len(problems_to_run)}] Running {problem_key}...")
            result = self.run_single_benchmark(problem_key, max_time_per_problem)
            results.append(result)
        
        return results
    
    def generate_benchmark_report(self, save_to_file: bool = True) -> str:
        """Generate a comprehensive benchmark report."""
        if not self.results:
            return "No benchmark results available. Run some benchmarks first."
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Calculate statistics
        total_problems = len(self.results)
        successful_runs = sum(1 for r in self.results if r.get('success', False))
        convergence_rate = sum(1 for r in self.results if r.get('converged', False)) / total_problems
        exact_recovery_rate = sum(1 for r in self.results if r.get('exact_recovery', False)) / total_problems
        
        # Category breakdown
        categories = {}
        for result in self.results:
            cat = result['category']
            if cat not in categories:
                categories[cat] = {'total': 0, 'converged': 0, 'exact': 0}
            categories[cat]['total'] += 1
            if result.get('converged', False):
                categories[cat]['converged'] += 1
            if result.get('exact_recovery', False):
                categories[cat]['exact'] += 1
        
        # Generate report
        report = f"""
SYMBOLIC REGRESSION BENCHMARK REPORT
{'='*80}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

OVERALL STATISTICS:
Total problems: {total_problems}
Successful runs: {successful_runs}/{total_problems} ({successful_runs/total_problems*100:.1f}%)
Convergence rate (R² > 0.95): {convergence_rate*100:.1f}%
Exact recovery rate (R² > 0.999): {exact_recovery_rate*100:.1f}%

CATEGORY BREAKDOWN:
"""
        
        for cat, stats in categories.items():
            conv_rate = stats['converged'] / stats['total'] * 100 if stats['total'] > 0 else 0
            exact_rate = stats['exact'] / stats['total'] * 100 if stats['total'] > 0 else 0
            report += f"  {cat.upper()}: {stats['total']} problems, {conv_rate:.1f}% converged, {exact_rate:.1f}% exact\\n"
        
        report += f"\\n{'='*80}\\nDETAILED RESULTS:\\n{'='*80}\\n"
        
        # Sort results by R² score (descending)
        sorted_results = sorted([r for r in self.results if r.get('success', False)], 
                               key=lambda x: x.get('r2_score', 0), reverse=True)
        
        for i, result in enumerate(sorted_results, 1):
            status = "✓" if result.get('exact_recovery', False) else ("~" if result.get('converged', False) else "❌")
            report += f"{i:2d}. {status} {result['problem_name']} [{result['category']}]\\n"
            report += f"    Target: {result['target_expression']}\\n"
            report += f"    R² Score: {result.get('r2_score', 0):.6f}, Time: {result.get('elapsed_time', 0):.1f}s\\n"
            if result.get('discovered_expressions'):
                report += f"    Best: {result['discovered_expressions'][0]}\\n"
            report += "\\n"
        
        if save_to_file:
            filename = f"benchmark_report_{timestamp}.txt"
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"\\n📄 Report saved to: {filename}")
        
        return report


def main():
    """Main function to run standard benchmarks."""
    print("🎯 STANDARD SYMBOLIC REGRESSION BENCHMARK SUITE")
    print("Based on SOTA research: SRBench, SRSD Feynman, Nguyen-Vladislavleva, Korns")
    print("="*80)
    
    # Create benchmark suite
    suite = StandardBenchmarkSuite()
    
    print(f"\\nAvailable benchmark problems: {len(suite.problems)}")
    for category in ['feynman', 'nguyen', 'korns', 'vlad', 'srbench']:
        count = sum(1 for p in suite.problems.values() if p.category == category)
        print(f"  {category.upper()}: {count} problems")
    
    # Quick test: Run easy problems only
    print("\\n🚀 Running EASY benchmark problems...")
    results = suite.run_benchmark_suite(
        complexity_levels=['easy'],
        max_time_per_problem=5.0  # 5 minutes max per problem
    )
    
    # Generate report
    report = suite.generate_benchmark_report()
    print("\\n" + report)
    
    # Run medium problems if easy ones went well
    easy_success_rate = sum(1 for r in results if r.get('converged', False)) / len(results) if results else 0
    
    if easy_success_rate > 0.5:  # If we solved >50% of easy problems
        print("\\n🚀 Easy problems went well! Running MEDIUM benchmark problems...")
        medium_results = suite.run_benchmark_suite(
            complexity_levels=['medium'],
            max_time_per_problem=8.0  # 8 minutes max per problem
        )
        
        # Final comprehensive report
        final_report = suite.generate_benchmark_report()
        print("\\n📊 FINAL BENCHMARK REPORT:")
        print("="*80)
        print(final_report)
    else:
        print(f"\\n⚠️ Easy problem success rate was {easy_success_rate*100:.1f}%")
        print("Consider tuning parameters before running harder problems.")


if __name__ == "__main__":
    main()
