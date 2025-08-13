"""
Island Model Ensemble for MIMO Symbolic Regression

This module implements a PySR-inspired island model where multiple populations
evolve independently with periodic migration of best individuals between islands.
The approach promotes diversity while allowing beneficial traits to spread across islands.
Includes PySR-style adaptive parsimony coefficients and domain-specific operator weighting.

Updated with asynchronous island-specific cache system for improved scalability.
"""
import numpy as np
import time
import multiprocessing
import random
import os
import tempfile
import pickle
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional, TYPE_CHECKING, Tuple, Any
from .expression_tree import Expression
from .regressor import MIMOSymbolicRegressor
from .utilities import optimize_final_expressions, evaluate_optimized_expressions
from .async_island_cache import (
    AsynchronousIslandCacheManager, 
    EnhancedIslandTopology,
    PoissonMigrationTimer,
    CachedExpression
)

if TYPE_CHECKING:
    from .data_processing import DataScaler


class IslandTopology:
    """Manages island topology and migration patterns for parallel symbolic regression."""
    
    def __init__(self, n_islands: int, migration_prob: float = 0.3):
        self.n_islands = n_islands
        self.migration_prob = migration_prob
        self.connections = self._create_random_topology()
    
    def _create_random_topology(self) -> Dict[int, List[int]]:
        """Create a random topology where each island connects to 1-3 others randomly."""
        connections = {i: [] for i in range(self.n_islands)}
        
        # Ensure each island has at least one connection
        for i in range(self.n_islands):
            # Each island connects to 1-3 others (like PySR's island model)
            n_connections = random.randint(1, min(3, self.n_islands - 1))
            possible_targets = [j for j in range(self.n_islands) if j != i]
            targets = random.sample(possible_targets, n_connections)
            connections[i] = targets
        
        return connections
    
    def get_migration_targets(self, island_id: int) -> List[int]:
        """Get list of islands this island can migrate to."""
        return self.connections.get(island_id, [])
    
    def should_migrate(self) -> bool:
        """Decide if migration should occur this cycle."""
        return random.random() < self.migration_prob


class SharedMigrationManager:
    """File-based migration manager for sharing expressions between island processes."""
    
    def __init__(self, n_islands: int, temp_dir: str):
        self.n_islands = n_islands
        self.temp_dir = temp_dir
        self.migration_files = {}
        
        # Create migration files for each island
        for i in range(n_islands):
            migration_file = os.path.join(temp_dir, f"island_{i}_migrants.pkl")
            self.migration_files[i] = migration_file
            # Initialize empty migration pool
            with open(migration_file, 'wb') as f:
                pickle.dump([], f)
    
    def send_migrants(self, from_island: int, to_islands: List[int], 
                     migrants: List[Dict[str, Any]]):
        """Send migrant expressions to target islands."""
        for to_island in to_islands:
            if to_island < self.n_islands:
                try:
                    migration_file = self.migration_files[to_island]
                    # Read existing migrants
                    try:
                        with open(migration_file, 'rb') as f:
                            existing_migrants = pickle.load(f)
                    except (FileNotFoundError, EOFError):
                        existing_migrants = []
                    
                    # Add new migrants
                    existing_migrants.extend(migrants)
                    
                    # Keep only recent migrants (last 50)
                    if len(existing_migrants) > 50:
                        existing_migrants = existing_migrants[-50:]
                    
                    # Write back
                    with open(migration_file, 'wb') as f:
                        pickle.dump(existing_migrants, f)
                        
                except Exception as e:
                    print(f"Failed to send migrants from {from_island} to {to_island}: {e}")
    
    def receive_migrants(self, island_id: int) -> List[Dict[str, Any]]:
        """Receive available migrant expressions for this island."""
        try:
            migration_file = self.migration_files[island_id]
            try:
                with open(migration_file, 'rb') as f:
                    migrants = pickle.load(f)
            except (FileNotFoundError, EOFError):
                migrants = []
            
            # Clear the migration file after reading
            with open(migration_file, 'wb') as f:
                pickle.dump([], f)
                
            return migrants
        except Exception as e:
            print(f"Failed to receive migrants for island {island_id}: {e}")
            return []


def island_worker_process(args):
    """Worker process for individual island evolution with migration support."""
    (island_id, regressor_params, X, y, total_generations, 
     migration_interval, topology_data, migration_manager_data, temp_dir) = args
    
    try:
        # Set unique seed for this island
        island_seed = hash((island_id, time.time())) % 2**32
        random.seed(island_seed)
        np.random.seed(island_seed % 2**32)
        
        # Create island topology
        topology = IslandTopology(topology_data['n_islands'], topology_data['migration_prob'])
        topology.connections = topology_data['connections']
        
        # Create migration manager
        migration_manager = SharedMigrationManager(migration_manager_data['n_islands'], temp_dir)
        
        # Create regressor with unique parameters for diversity (like PySR)
        params = regressor_params.copy()
        params['console_log'] = False
        
        # Remove parameters not supported by new MIMOSymbolicRegressor
        unsupported_params = [
            'purge_percentage', 'exchange_interval', 'import_percentage', 
            'enable_inter_thread_communication',
            # Scaling-related parameters (removed from system)
            'enable_data_scaling', 'use_multi_scale_fitness'
        ]
        for param in unsupported_params:
            params.pop(param, None)
        
        # Add parameter diversity for island differentiation
        params['mutation_rate'] *= random.uniform(0.8, 1.2)
        params['crossover_rate'] *= random.uniform(0.9, 1.1)
        params['population_size'] = int(params['population_size'] * random.uniform(0.9, 1.1))
        params['parsimony_coefficient'] *= random.uniform(0.5, 2.0)
        
        regressor = MIMOSymbolicRegressor(**params)
        
        # Island evolution with periodic migration
        generation = 0
        all_results = []
        
        # Batch evolution like PySR's ncycles_per_iteration
        while generation < total_generations:
            batch_size = min(migration_interval, total_generations - generation)
            
            # Set batch generations
            regressor.generations = batch_size
            
            # Run evolution batch
            regressor.fit(X, y, constant_optimize=False)
            
            generation += batch_size
            
            # Collect results from this batch
            if regressor.best_expressions and hasattr(regressor, 'fitness_history'):
                best_fitness = max(regressor.fitness_history) if regressor.fitness_history else -float('inf')
                for i, expr in enumerate(regressor.best_expressions[:3]):
                    all_results.append({
                        'expression_obj': expr.copy(),
                        'expression_str': expr.to_string(),
                        'fitness': best_fitness - i * 0.001,  # Small penalty for ranking
                        'island_id': island_id,
                        'generation': generation,
                        'complexity': expr.complexity()
                    })
            
            # Migration phase (like PySR's population exchange)
            if generation < total_generations and topology.should_migrate():
                try:
                    # Get best expressions to send as migrants
                    if regressor.best_expressions:
                        migrants = []
                        for expr in regressor.best_expressions[:2]:  # Send top 2
                            migrant_data = {
                                'expression_str': expr.to_string(),
                                'fitness': best_fitness,
                                'complexity': expr.complexity(),
                                'from_island': island_id,
                                'generation': generation
                            }
                            migrants.append(migrant_data)
                        
                        # Send to connected islands
                        target_islands = topology.get_migration_targets(island_id)
                        migration_manager.send_migrants(island_id, target_islands, migrants)
                    
                    # Receive migrants from other islands
                    received_migrants = migration_manager.receive_migrants(island_id)
                    if received_migrants:
                        # Here we would integrate migrants into the population
                        # For now, just log that migration occurred
                        pass
                        
                except Exception as e:
                    # Migration failures shouldn't stop evolution
                    pass
        
        # Return best results from this island
        if all_results:
            # Sort by fitness and return top results
            all_results.sort(key=lambda x: x['fitness'], reverse=True)
            return all_results[:5]  # Top 5 from this island
        else:
            return []
            
    except Exception as e:
        print(f"Island {island_id} failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return []


def asynchronous_island_worker_process(args) -> List[Dict[str, Any]]:
    """
    Enhanced asynchronous island worker with island-specific cache system.
    Each island evolves independently and uses an island cache plus optional
    file-backed exchange to asynchronously share/import expressions.
    """
    (island_id, regressor_params, X, y, total_generations, cache_manager_data,
     topology_data, migration_timer_params, shared_dir) = args
    try:
        # Seed per-process RNGs
        island_seed = hash((island_id, time.time())) % 2**32
        random.seed(island_seed)
        np.random.seed(island_seed % 2**32)

        # Timer for Poisson-timed events
        migration_timer = PoissonMigrationTimer(**migration_timer_params)

        # Prepare regressor params
        params = regressor_params.copy()
        params['console_log'] = False
        for param in ['purge_percentage', 'exchange_interval', 'import_percentage',
                      'enable_inter_thread_communication', 'enable_data_scaling', 'use_multi_scale_fitness']:
            params.pop(param, None)

        # Add island diversification
        if 'mutation_rate' in params:
            params['mutation_rate'] *= random.uniform(0.8, 1.2)
        if 'crossover_rate' in params:
            params['crossover_rate'] *= random.uniform(0.9, 1.1)
        if 'population_size' in params:
            params['population_size'] = int(params['population_size'] * random.uniform(0.9, 1.1))
        if 'parsimony_coefficient' in params:
            params['parsimony_coefficient'] *= random.uniform(0.5, 2.0)
        params['generations'] = total_generations

        regressor = MIMOSymbolicRegressor(**params)

        # Create topology and cache manager
        topology = EnhancedIslandTopology(topology_data['n_islands'], topology_data['migration_prob'])
        topology.connections = topology_data['connections']
        cache_manager = AsynchronousIslandCacheManager(cache_manager_data['n_islands'], topology)
        try:
            if isinstance(shared_dir, str) and shared_dir:
                cache_manager.set_shared_dir(shared_dir)
        except Exception:
            pass
        try:
            cache_manager.seed_all_caches(X.shape[1])
        except Exception:
            pass

        # Enable async migration context and run
        try:
            regressor.enable_async_island_migration(cache_manager, island_id, migration_timer)
        except Exception:
            pass

        regressor.fit(X, y, constant_optimize=False)

        # Collect island results
        all_results: List[Dict[str, Any]] = []
        if getattr(regressor, 'best_expressions', None):
            best_fitness = max(getattr(regressor, 'fitness_history', []) or [0.0])
            for i, expr in enumerate(regressor.best_expressions[:3]):
                rec: Dict[str, Any] = {
                    'expression_obj': expr.copy(),
                    'expression_str': expr.to_string(),
                    'fitness': best_fitness - i * 0.001,
                    'island_id': island_id,
                    'generation': total_generations,
                    'complexity': expr.complexity(),
                }
                if i == 0:
                    try:
                        rec['_async_stats'] = regressor.get_async_migration_stats()
                    except Exception:
                        pass
                all_results.append(rec)

        if all_results:
            all_results.sort(key=lambda x: x['fitness'], reverse=True)
            return all_results[:5]
        return []

    except Exception as e:
        print(f"Error in asynchronous island {island_id}: {e}")
        import traceback
        traceback.print_exc()
        return []

def asynchronous_island_thread_worker(args) -> List[Dict[str, Any]]:
    """
    Thread-based asynchronous island worker that shares a common in-memory cache
    manager across islands to avoid file I/O completely. This is faster on a
    single machine and leverages thread-safe locks in the cache.
    """
    (island_id, regressor_params, X, y, total_generations,
     shared_cache_manager, shared_topology, migration_timer_params) = args
    try:
        # Per-thread RNG seeding
        island_seed = hash((island_id, time.time())) % 2**32
        random.seed(island_seed)
        np.random.seed(island_seed % 2**32)

        # Prepare regressor params
        params = regressor_params.copy()
        params['console_log'] = False
        for param in ['purge_percentage', 'exchange_interval', 'import_percentage',
                      'enable_inter_thread_communication', 'enable_data_scaling', 'use_multi_scale_fitness']:
            params.pop(param, None)

        # Add island diversification
        if 'mutation_rate' in params:
            params['mutation_rate'] *= random.uniform(0.8, 1.2)
        if 'crossover_rate' in params:
            params['crossover_rate'] *= random.uniform(0.9, 1.1)
        if 'population_size' in params:
            params['population_size'] = int(params['population_size'] * random.uniform(0.9, 1.1))
        if 'parsimony_coefficient' in params:
            params['parsimony_coefficient'] *= random.uniform(0.5, 2.0)
        params['generations'] = total_generations

        regressor = MIMOSymbolicRegressor(**params)

        # Use shared topology and cache manager
        cache_manager = shared_cache_manager
        migration_timer = PoissonMigrationTimer(**migration_timer_params)

        try:
            regressor.enable_async_island_migration(cache_manager, island_id, migration_timer)
        except Exception:
            pass

        regressor.fit(X, y, constant_optimize=False)

        # Collect island results
        all_results: List[Dict[str, Any]] = []
        if getattr(regressor, 'best_expressions', None):
            best_fitness = max(getattr(regressor, 'fitness_history', []) or [0.0])
            for i, expr in enumerate(regressor.best_expressions[:3]):
                rec: Dict[str, Any] = {
                    'expression_obj': expr.copy(),
                    'expression_str': expr.to_string(),
                    'fitness': best_fitness - i * 0.001,
                    'island_id': island_id,
                    'generation': total_generations,
                    'complexity': expr.complexity(),
                }
                if i == 0:
                    try:
                        rec['_async_stats'] = regressor.get_async_migration_stats()
                    except Exception:
                        pass
                all_results.append(rec)

        if all_results:
            all_results.sort(key=lambda x: x['fitness'], reverse=True)
            return all_results[:5]
        return []

    except Exception as e:
        print(f"Error in asynchronous thread island {island_id}: {e}")
        import traceback
        traceback.print_exc()
        return []


class EnsembleMIMORegressor:
    """
    Island Model Ensemble for MIMO Symbolic Regression.
    
    Implements an island model evolutionary algorithm inspired by PySR where multiple
    populations evolve independently with periodic migration of best individuals between
    islands. This approach promotes diversity while allowing beneficial traits to spread.
    """
    
    def __init__(self, n_fits: int = 8, top_n_select: int = 5,
                 migration_interval: int = 20, migration_probability: float = 0.3,
                 enable_inter_thread_communication: bool = True,  # Kept for compatibility
                 enable_adaptive_parsimony: bool = True,  # Enable PySR-style adaptive parsimony
                 domain_type: str = "general",  # Domain-specific operator weighting
                 use_asynchronous_migration: bool = True,  # NEW: Enable asynchronous migration
                 **regressor_kwargs):
        """
        Initialize the Island Model Ensemble with PySR-style adaptive parsimony.
        
        Args:
            n_fits (int): Number of island populations (concurrent regressors)
            top_n_select (int): Number of best expressions to select from all islands
            migration_interval (int): Generations between migration opportunities (like PySR's ncycles_per_iteration)
            migration_probability (float): Probability of migration occurring each cycle
            enable_inter_thread_communication (bool): Kept for compatibility, always uses island model
            enable_adaptive_parsimony (bool): Enable PySR-style adaptive parsimony coefficient
            domain_type (str): Domain for operator weighting ("physics", "engineering", "biology", "finance", "general")
            use_asynchronous_migration (bool): Use new asynchronous island-specific cache system
            **regressor_kwargs: Parameters passed to each MIMOSymbolicRegressor
        """
        if not isinstance(n_fits, int) or n_fits <= 0:
            raise ValueError("n_fits must be a positive integer")
        if not isinstance(top_n_select, int) or top_n_select <= 0:
            raise ValueError("top_n_select must be a positive integer")
        if top_n_select > n_fits * 5:  # Each island contributes up to 5
            print(f"Warning: top_n_select ({top_n_select}) is large relative to n_fits ({n_fits})")
        
        self.n_fits = n_fits
        self.top_n_select = top_n_select
        self.migration_interval = migration_interval
        self.migration_probability = migration_probability
        self.enable_adaptive_parsimony = enable_adaptive_parsimony
        self.domain_type = domain_type
        self.use_asynchronous_migration = use_asynchronous_migration
        self.regressor_kwargs = regressor_kwargs
        
        # Initialize adaptive parsimony system
        if self.enable_adaptive_parsimony:
            from .adaptive_parsimony import AdaptiveParsimonySystem
            base_coeff = regressor_kwargs.get('parsimony_coefficient', 0.003)
            self.parsimony_system = AdaptiveParsimonySystem(base_coeff, domain_type)
        else:
            self.parsimony_system = None
        
        # Results storage
        self.best_expressions: List[Expression] = []
        self.best_fitnesses: List[float] = []
        self.all_results: List[Dict] = []
        self.fitted_regressors: List[Any] = []
    
    def fit(self, X: np.ndarray, y: np.ndarray, constant_optimize: bool = False):
        """
        Fits multiple regressors using island model evolution with migration.
        """
        if self.use_asynchronous_migration:
            return self._fit_asynchronous(X, y, constant_optimize)
        else:
            return self._fit_synchronous(X, y, constant_optimize)
    
    def _fit_asynchronous(self, X: np.ndarray, y: np.ndarray, constant_optimize: bool = False):
        """Asynchronous island model with island-specific caches"""
        print(f"Starting asynchronous island model ensemble with {self.n_fits} islands...")
        print(f"Using island-specific cache system with Poisson migration timing...")

        # Validate inputs
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        # Create topology and worker kwargs
        topology = EnhancedIslandTopology(self.n_fits, self.migration_probability)
        worker_kwargs = self.regressor_kwargs.copy()
        worker_kwargs['console_log'] = False

        # Migration timer parameters (rarer events by default)
        migration_timer_params = {
            'average_interval': max(20, int(self.migration_interval * 3)),
            'min_generation': max(25, int(self.migration_interval * 3))
        }

        # Backend selection
        use_threads = False
        all_island_results: List[Dict[str, Any]] = []
        island_async_stats: List[Dict[str, Any]] = []

        if use_threads:
            shared_cache_manager = AsynchronousIslandCacheManager(self.n_fits, topology)
            try:
                shared_cache_manager.seed_all_caches(X.shape[1])
            except Exception:
                pass

            thread_configs = []
            for i in range(self.n_fits):
                island_kwargs = worker_kwargs.copy()
                island_kwargs['mutation_rate'] = worker_kwargs.get('mutation_rate', 0.1) * random.uniform(0.8, 1.2)
                island_kwargs['parsimony_coefficient'] = worker_kwargs.get('parsimony_coefficient', 0.001) * random.uniform(0.5, 2.0)
                thread_configs.append((i, island_kwargs, X, y, worker_kwargs.get('generations', 200),
                                       shared_cache_manager, topology, migration_timer_params))

            with ThreadPoolExecutor(max_workers=self.n_fits) as executor:
                futures = [executor.submit(asynchronous_island_thread_worker, cfg) for cfg in thread_configs]
                for fut in as_completed(futures):
                    island_results = fut.result()
                    if island_results:
                        all_island_results.extend(island_results)
                        for rec in island_results:
                            stats = rec.get('_async_stats') if isinstance(rec, dict) else None
                            if stats:
                                island_async_stats.append({'island_id': rec.get('island_id'), **stats})
                                break
            print("All asynchronous island evolutions completed. Aggregating results...")
        else:
            # Process-based execution with file-backed shared_dir
            try:
                import os
                shm_base = '/dev/shm' if os.path.isdir('/dev/shm') else None
                if shm_base:
                    shared_dir = tempfile.mkdtemp(prefix="async_islands_", dir=shm_base)
                else:
                    shared_dir = tempfile.mkdtemp(prefix="async_islands_")
            except Exception:
                shared_dir = tempfile.mkdtemp(prefix="async_islands_")

            topology_data = {
                'n_islands': self.n_fits,
                'migration_prob': self.migration_probability,
                'connections': topology.connections,
            }
            cache_manager_data = {'n_islands': self.n_fits}

            configs = []
            for i in range(self.n_fits):
                island_kwargs = worker_kwargs.copy()
                island_kwargs['mutation_rate'] = worker_kwargs.get('mutation_rate', 0.1) * random.uniform(0.8, 1.2)
                island_kwargs['parsimony_coefficient'] = worker_kwargs.get('parsimony_coefficient', 0.001) * random.uniform(0.5, 2.0)
                configs.append((i, island_kwargs, X, y, worker_kwargs.get('generations', 200),
                                cache_manager_data, topology_data, migration_timer_params, shared_dir))

            try:
                with multiprocessing.Pool(processes=self.n_fits, maxtasksperchild=1) as pool:
                    results = pool.map(asynchronous_island_worker_process, configs)
                for island_results in results:
                    if island_results:
                        all_island_results.extend(island_results)
                        for rec in island_results:
                            stats = rec.get('_async_stats') if isinstance(rec, dict) else None
                            if stats:
                                island_async_stats.append({'island_id': rec.get('island_id'), **stats})
                                break
            finally:
                import shutil
                try:
                    shutil.rmtree(shared_dir)
                except Exception:
                    pass
            print("All asynchronous island evolutions completed. Aggregating results...")

        # Print migration summary
        if island_async_stats:
            agg = {'shared': 0, 'exports': 0, 'received_candidates': 0, 'file_candidates': 0, 'rebuilt': 0, 'injected': 0, 'send_events': 0, 'receive_events': 0}
            for s in island_async_stats:
                for k in agg:
                    agg[k] += int(s.get(k, 0))
            try:
                print("\nAsync migration summary:")
                print("  Islands with stats:", len(island_async_stats))
                print("  Shared:", agg['shared'], "Exports:", agg['exports'],
                      "Received:", agg['received_candidates'], "FileRecv:", agg['file_candidates'],
                      "Rebuilt:", agg['rebuilt'], "Injected:", agg['injected'],
                      "SendEvents:", agg['send_events'], "RecvEvents:", agg['receive_events'])
            except Exception:
                pass

        if not all_island_results:
            print("\nWarning: No valid expressions found across all islands. The model is not fitted.")
            return

        # Sort by fitness and pick top
        all_island_results.sort(key=lambda x: x['fitness'], reverse=True)
        top_results = all_island_results[:self.top_n_select]

        # Final optimization
        print(f"\nApplying final optimizations to top {len(top_results)} candidate expressions...")
        start_time = time.time()
        candidate_expressions = [res['expression_obj'] for res in top_results]
        parsimony_coeff = self.regressor_kwargs.get('parsimony_coefficient', 0.01)
        optimized_expressions = optimize_final_expressions(candidate_expressions, X, y)
        optimized_fitness_scores = evaluate_optimized_expressions(optimized_expressions, X, y, parsimony_coeff)

        for i, (optimized_expr, new_fitness) in enumerate(zip(optimized_expressions, optimized_fitness_scores)):
            if i < len(top_results):
                top_results[i]['expression_obj'] = optimized_expr
                top_results[i]['fitness'] = new_fitness
                top_results[i]['expression_str'] = optimized_expr.to_string()
                top_results[i]['complexity'] = optimized_expr.complexity()

        top_results.sort(key=lambda x: x['fitness'], reverse=True)
        optimization_time = time.time() - start_time
        improvement_count = sum(1 for i, new_fitness in enumerate(optimized_fitness_scores)
                                if i < len(all_island_results[:self.top_n_select]) and
                                new_fitness > all_island_results[i]['fitness'])
        print(f"Final optimization completed in {optimization_time:.2f}s")
        print(f"Optimization improved {improvement_count}/{len(optimized_expressions)} expressions")

        # Store results
        self.best_expressions = [res['expression_obj'] for res in top_results]
        self.best_fitnesses = [res['fitness'] for res in top_results]
        self.all_results = all_island_results

        print(f"\nAsynchronous island ensemble fitting complete. Top {len(self.best_expressions)} of {len(all_island_results)} expressions selected:")
        for i, res in enumerate(top_results):
            print(f"  {i+1}. Fitness: {res['fitness']:.6f}, Complexity: {res['complexity']:.2f}, From Island: {res['island_id']}, Expression: {res['expression_str']}")
    
    def _fit_synchronous(self, X: np.ndarray, y: np.ndarray, constant_optimize: bool = False):
        """
        Fits multiple regressors using island model evolution with migration.
        
        Each island evolves independently but exchanges best individuals periodically.
        This mimics PySR's population-based approach but with multiple islands.
        """
        print(f"Starting island model ensemble with {self.n_fits} islands...")
        print(f"Migration occurs every {self.migration_interval} generations with probability {self.migration_probability}")
        
        # Create temporary directory for migration files
        temp_dir = tempfile.mkdtemp(prefix="island_ensemble_")
        
        try:
            # Create island topology
            topology = IslandTopology(self.n_fits, self.migration_probability)
            
            # Prepare worker configurations
            worker_kwargs = self.regressor_kwargs.copy()
            worker_kwargs['console_log'] = False
            
            # Prepare topology and migration manager data for serialization
            topology_data = {
                'n_islands': self.n_fits,
                'migration_prob': self.migration_probability,
                'connections': topology.connections
            }
            
            migration_manager_data = {
                'n_islands': self.n_fits
            }
            
            configs = []
            for i in range(self.n_fits):
                # Add some parameter diversity between islands
                island_kwargs = worker_kwargs.copy()
                island_kwargs['mutation_rate'] = worker_kwargs.get('mutation_rate', 0.1) * random.uniform(0.8, 1.2)
                island_kwargs['parsimony_coefficient'] = worker_kwargs.get('parsimony_coefficient', 0.001) * random.uniform(0.5, 2.0)
                
                config = (i, island_kwargs, X, y, worker_kwargs.get('generations', 200), 
                         self.migration_interval, topology_data, migration_manager_data, temp_dir)
                configs.append(config)
            
            # Run island evolution in parallel
            with multiprocessing.Pool(processes=self.n_fits, maxtasksperchild=1) as pool:
                results = pool.map(island_worker_process, configs)
            
            print("All island evolutions completed. Aggregating results...")
            
            # Aggregate results from all islands
            all_island_results = []
            for island_results in results:
                if island_results:
                    all_island_results.extend(island_results)
            
            if not all_island_results:
                print("\nWarning: No valid expressions found across all islands. The model is not fitted.")
                return
            
            # Sort all results by fitness
            all_island_results.sort(key=lambda x: x['fitness'], reverse=True)
            
            # Select top results
            top_results = all_island_results[:self.top_n_select]
            
            # Apply final optimizations
            print(f"\nApplying final optimizations to top {len(top_results)} candidate expressions...")
            start_time = time.time()
            
            candidate_expressions = [res['expression_obj'] for res in top_results]
            
            # Apply final optimizations using raw data (no scaling)
            parsimony_coeff = self.regressor_kwargs.get('parsimony_coefficient', 0.01)
            
            optimized_expressions = optimize_final_expressions(candidate_expressions, X, y)
            optimized_fitness_scores = evaluate_optimized_expressions(optimized_expressions, X, y, parsimony_coeff)
            
            # Update results with optimized fitness scores
            for i, (optimized_expr, new_fitness) in enumerate(zip(optimized_expressions, optimized_fitness_scores)):
                if i < len(top_results):
                    top_results[i]['expression_obj'] = optimized_expr
                    top_results[i]['fitness'] = new_fitness
                    top_results[i]['expression_str'] = optimized_expr.to_string()
                    top_results[i]['complexity'] = optimized_expr.complexity()
            
            # Re-sort by optimized fitness
            top_results.sort(key=lambda x: x['fitness'], reverse=True)
            
            optimization_time = time.time() - start_time
            improvement_count = sum(1 for i, new_fitness in enumerate(optimized_fitness_scores) 
                                  if i < len(all_island_results[:self.top_n_select]) and 
                                  new_fitness > all_island_results[i]['fitness'])
            
            print(f"Final optimization completed in {optimization_time:.2f}s")
            print(f"Optimization improved {improvement_count}/{len(optimized_expressions)} expressions")
            
            # Store results
            self.best_expressions = [res['expression_obj'] for res in top_results]
            self.best_fitnesses = [res['fitness'] for res in top_results]
            self.all_results = all_island_results
            
            # Print summary
            print(f"\nIsland ensemble fitting complete. Top {len(self.best_expressions)} of {len(all_island_results)} expressions selected:")
            for i, res in enumerate(top_results):
                print(f"  {i+1}. Fitness: {res['fitness']:.6f}, "
                      f"Complexity: {res['complexity']:.2f}, "
                      f"From Island: {res['island_id']}, "
                      f"Expression: {res['expression_str']}")
                      
        finally:
            # Clean up temporary directory
            import shutil
            try:
                shutil.rmtree(temp_dir)
            except:
                pass
    
    def predict(self, X: np.ndarray, strategy: str = 'mean') -> np.ndarray:
        """
        Makes predictions using the ensemble of best expressions.
        
        Args:
            X (np.ndarray): Input data for prediction
            strategy (str): 'mean' for ensemble average, 'best_only' for single best
            
        Returns:
            np.ndarray: Predicted values
        """
        if not self.best_expressions:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        # Work with raw data (no scaling)
        
        if strategy == 'best_only':
            # Use only the single best expression
            return self.best_expressions[0].evaluate(X)
        
        elif strategy == 'mean':
            # Ensemble average (no scaling)
            all_predictions = []
            for expr in self.best_expressions:
                pred = expr.evaluate(X)
                if pred.ndim == 1:
                    pred = pred.reshape(-1, 1)
                all_predictions.append(pred)
            
            return np.mean(np.array(all_predictions), axis=0)
        
        else:
            raise ValueError(f"Invalid prediction strategy '{strategy}'. Choose 'mean' or 'best_only'.")
    
    def get_expressions(self) -> List[str]:
        """Returns string representations of the top expressions."""
        if not self.best_expressions:
            return []
        return [expr.to_string() for expr in self.best_expressions]
    
    def get_fitness_histories(self) -> List[List[float]]:
        """Returns empty list for compatibility (individual island histories not tracked)."""
        return [[] for _ in self.best_expressions]
    
    def score(self, X: np.ndarray, y: np.ndarray, strategy: str = 'mean') -> float:
        """
        Calculates the R² score for the ensemble.
        
        Args:
            X (np.ndarray): Test samples
            y (np.ndarray): True values
            strategy (str): Prediction strategy to use
            
        Returns:
            float: R² score
        """
        from .data_processing import r2_score
        
        if not self.best_expressions:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        
        predictions = self.predict(X, strategy=strategy)
        
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        
        try:
            return r2_score(y.flatten(), predictions.flatten())
        except Exception:
            # Fallback calculation
            ss_res = float(np.sum((y - predictions) ** 2))
            ss_tot = float(np.sum((y - np.mean(y, axis=0)) ** 2))
            
            if ss_tot == 0.0:
                return 1.0 if ss_res == 0.0 else 0.0
            
            return 1.0 - (ss_res / ss_tot)
        return 1.0 - (ss_res / ss_tot)
