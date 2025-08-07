"""
Asynchronous Island-Specific Cache System for MIMO Symbolic Regression

This module implements an advanced asynchronous migration system where each island
maintains its own cache that other islands can read from. This eliminates topology
accessibility issues and provides better scalability.
"""
import numpy as np
import time
import threading
import random
import multiprocessing as mp
from dataclasses import dataclass
from typing import List, Dict, Optional, Any, Union
from .expression_tree import Expression
from .generator import ExpressionGenerator


@dataclass
class CachedExpression:
    """Represents a cached expression with metadata for migration"""
    expression_str: str
    fitness: float
    complexity: float
    from_island: int
    timestamp: float
    generation: int
    is_initial_seed: bool = False
    
    def __post_init__(self):
        """Validate fields after initialization"""
        if not isinstance(self.expression_str, str):
            raise TypeError("expression_str must be a string")
        if not isinstance(self.fitness, (int, float)):
            raise TypeError("fitness must be numeric")
        if not isinstance(self.complexity, (int, float)):
            raise TypeError("complexity must be numeric")
        if not isinstance(self.from_island, int):
            raise TypeError("from_island must be an integer")
        if not isinstance(self.timestamp, (int, float)):
            raise TypeError("timestamp must be numeric")
        if not isinstance(self.generation, int):
            raise TypeError("generation must be an integer")


class PoissonMigrationTimer:
    """Generates random migration events with equivalent frequency using Poisson process"""
    
    def __init__(self, average_interval: int = 20, min_generation: int = 10):
        if not isinstance(average_interval, int) or average_interval <= 0:
            raise ValueError("average_interval must be a positive integer")
        if not isinstance(min_generation, int) or min_generation < 0:
            raise ValueError("min_generation must be a non-negative integer")
            
        self.average_interval = average_interval
        self.min_generation = min_generation
        self.lambda_rate = 1.0 / average_interval
    
    def should_send(self, generation: int, last_send: int, fitness_history: List[float]) -> bool:
        """Enhanced send decision with fitness-based gating"""
        if not isinstance(generation, int) or not isinstance(last_send, int):
            raise TypeError("generation and last_send must be integers")
        if not isinstance(fitness_history, list):
            raise TypeError("fitness_history must be a list")
        
        # Don't send in very early generations
        if generation < self.min_generation:
            return False
        
        # Don't send if fitness is too low (random noise)
        if fitness_history and len(fitness_history) >= 5:
            recent_fitness = np.mean(fitness_history[-5:])
            if recent_fitness < 0.1:  # Threshold for meaningful fitness
                return False
        
        # Standard Poisson timing
        if generation <= last_send:
            return False
        
        interval = np.random.exponential(self.average_interval)
        return (generation - last_send) >= interval
    
    def should_receive(self, generation: int, last_receive: int, diversity_score: float) -> bool:
        """Enhanced receive decision based on diversity need"""
        if not isinstance(generation, int) or not isinstance(last_receive, int):
            raise TypeError("generation and last_receive must be integers")
        if not isinstance(diversity_score, (int, float)):
            raise TypeError("diversity_score must be numeric")
        
        if generation <= last_receive or generation < 5:
            return False
        
        # More frequent receiving when diversity is low
        base_interval = self.average_interval * 0.7
        if diversity_score < 0.3:
            base_interval *= 0.5  # Receive twice as often when diversity is low
        
        receive_interval = np.random.exponential(base_interval)
        return (generation - last_receive) >= receive_interval


class IslandSpecificCache:
    """Each island has its own cache that others can read from"""
    
    def __init__(self, island_id: int, cache_size: int = 50):
        if not isinstance(island_id, int) or island_id < 0:
            raise ValueError("island_id must be a non-negative integer")
        if not isinstance(cache_size, int) or cache_size <= 0:
            raise ValueError("cache_size must be a positive integer")
            
        self.island_id = island_id
        self.cache_size = cache_size
        self.expressions: List[CachedExpression] = []
        self.initial_seed_count = 0
        self.lock = threading.Lock()
        
    def add_expression(self, expression: CachedExpression, is_seed: bool = False) -> None:
        """Add expression to this island's cache"""
        if not isinstance(expression, CachedExpression):
            raise TypeError("expression must be a CachedExpression instance")
        if not isinstance(is_seed, bool):
            raise TypeError("is_seed must be a boolean")
            
        with self.lock:
            if is_seed:
                self.initial_seed_count += 1
                expression.is_initial_seed = True
            else:
                expression.is_initial_seed = False
                
            self.expressions.append(expression)
            
            # Maintain cache size with smart eviction
            if len(self.expressions) > self.cache_size:
                self._smart_eviction()
    
    def _smart_eviction(self) -> None:
        """Evict expressions while preserving diversity"""
        # Always keep at least 20% initial seeds if available
        seed_expressions = [e for e in self.expressions if e.is_initial_seed]
        non_seed_expressions = [e for e in self.expressions if not e.is_initial_seed]
        
        target_seed_count = max(1, int(self.cache_size * 0.2))
        target_non_seed_count = self.cache_size - target_seed_count
        
        # Keep best seeds and best non-seeds
        kept_seeds = sorted(seed_expressions, key=lambda x: x.fitness, reverse=True)[:target_seed_count]
        kept_non_seeds = sorted(non_seed_expressions, key=lambda x: x.fitness, reverse=True)[:target_non_seed_count]
        
        self.expressions = kept_seeds + kept_non_seeds
    
    def sample_expressions(self, count: int, prefer_seeds: bool = True) -> List[CachedExpression]:
        """Sample expressions with preference for initial seeds"""
        if not isinstance(count, int) or count <= 0:
            raise ValueError("count must be a positive integer")
        if not isinstance(prefer_seeds, bool):
            raise TypeError("prefer_seeds must be a boolean")
            
        with self.lock:
            if not self.expressions:
                return []
            
            available = self.expressions.copy()
            
            if prefer_seeds:
                # 70% chance to pick seeds if available
                seed_expressions = [e for e in available if e.is_initial_seed]
                non_seed_expressions = [e for e in available if not e.is_initial_seed]
                
                result = []
                remaining_count = count
                
                # First, try to get seeds
                if seed_expressions and remaining_count > 0:
                    seed_sample_count = min(int(count * 0.7), len(seed_expressions), remaining_count)
                    seed_sample = random.sample(seed_expressions, seed_sample_count)
                    result.extend(seed_sample)
                    remaining_count -= len(seed_sample)
                
                # Fill remaining with non-seeds
                if non_seed_expressions and remaining_count > 0:
                    non_seed_sample_count = min(remaining_count, len(non_seed_expressions))
                    non_seed_sample = random.sample(non_seed_expressions, non_seed_sample_count)
                    result.extend(non_seed_sample)
                
                return result
            else:
                # Random sampling without preference
                sample_count = min(count, len(available))
                return random.sample(available, sample_count)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about this cache"""
        with self.lock:
            total_count = len(self.expressions)
            seed_count = sum(1 for e in self.expressions if e.is_initial_seed)
            
            if total_count > 0:
                avg_fitness = np.mean([e.fitness for e in self.expressions])
                max_fitness = max([e.fitness for e in self.expressions])
                avg_complexity = np.mean([e.complexity for e in self.expressions])
            else:
                avg_fitness = max_fitness = avg_complexity = 0.0
            
            return {
                'total_expressions': total_count,
                'seed_expressions': seed_count,
                'non_seed_expressions': total_count - seed_count,
                'avg_fitness': float(avg_fitness),
                'max_fitness': float(max_fitness),
                'avg_complexity': float(avg_complexity)
            }


class DisconnectionDetector:
    """Detects and handles topology disconnection"""
    
    def __init__(self, n_islands: int):
        if not isinstance(n_islands, int) or n_islands <= 0:
            raise ValueError("n_islands must be a positive integer")
            
        self.n_islands = n_islands
        self.exchange_counts: Dict[int, int] = {i: 0 for i in range(n_islands)}
        self.last_check_generation = 0
        
    def record_exchange(self, island_id: int) -> None:
        """Record successful exchange"""
        if not isinstance(island_id, int) or island_id < 0:
            raise ValueError("island_id must be a non-negative integer")
            
        if island_id in self.exchange_counts:
            self.exchange_counts[island_id] += 1
    
    def check_disconnection(self, generation: int) -> bool:
        """Check if islands are disconnected"""
        if not isinstance(generation, int):
            raise TypeError("generation must be an integer")
            
        if generation - self.last_check_generation < 50:
            return False  # Check every 50 generations
        
        self.last_check_generation = generation
        
        # Count islands with very few exchanges
        low_exchange_islands = sum(1 for count in self.exchange_counts.values() if count < 3)
        
        # If more than 60% of islands have low exchanges, likely disconnected
        return low_exchange_islands > (self.n_islands * 0.6)


class EmergencyBroadcastManager:
    """Manages emergency broadcast mode when islands are disconnected"""
    
    def __init__(self, n_islands: int):
        if not isinstance(n_islands, int) or n_islands <= 0:
            raise ValueError("n_islands must be a positive integer")
            
        self.n_islands = n_islands
        self.emergency_mode = False
        self.emergency_threshold = 100  # generations
        
    def should_activate_emergency(self, generation: int, exchange_count: int) -> bool:
        """Activate if very few exchanges happened"""
        if not isinstance(generation, int) or not isinstance(exchange_count, int):
            raise TypeError("generation and exchange_count must be integers")
            
        expected_exchanges = generation * self.n_islands * 0.1  # Expected rate
        return generation > self.emergency_threshold and exchange_count < expected_exchanges * 0.2
    
    def emergency_broadcast_topology(self) -> Dict[int, List[int]]:
        """Emergency: everyone broadcasts to everyone"""
        return {i: [j for j in range(self.n_islands) if j != i] for i in range(self.n_islands)}


class EnhancedIslandTopology:
    """Enhanced topology with connectivity guarantees and repair mechanisms"""
    
    def __init__(self, n_islands: int, migration_prob: float = 0.3):
        if not isinstance(n_islands, int) or n_islands <= 0:
            raise ValueError("n_islands must be a positive integer")
        if not isinstance(migration_prob, (int, float)) or not 0 <= migration_prob <= 1:
            raise ValueError("migration_prob must be a number between 0 and 1")
            
        self.n_islands = n_islands
        self.migration_prob = migration_prob
        self.connections = self._create_guaranteed_connected_topology()
        self.backup_connections = self._create_backup_topology()
        self.repair_attempts = 0
        self.max_repairs = 3
    
    def _create_guaranteed_connected_topology(self) -> Dict[int, List[int]]:
        """Ensure every island has at least one incoming connection"""
        connections = {i: [] for i in range(self.n_islands)}
        
        # Create ring topology as base (guarantees connectivity)
        for i in range(self.n_islands):
            next_island = (i + 1) % self.n_islands
            connections[i].append(next_island)
        
        # Add random additional connections
        for i in range(self.n_islands):
            n_additional = random.randint(0, 2)  # 0-2 additional connections
            possible_targets = [j for j in range(self.n_islands) 
                              if j != i and j not in connections[i]]
            
            if possible_targets:
                additional_targets = random.sample(
                    possible_targets, 
                    min(n_additional, len(possible_targets))
                )
                connections[i].extend(additional_targets)
        
        # Verify connectivity
        self._verify_connectivity(connections)
        return connections
    
    def _verify_connectivity(self, connections: Dict[int, List[int]]) -> None:
        """Ensure every island can receive from at least one other"""
        can_receive = set()
        
        for source, targets in connections.items():
            for target in targets:
                can_receive.add(target)
        
        # If any island can't receive, add emergency connections
        for island in range(self.n_islands):
            if island not in can_receive:
                # Find a random island to connect to this one
                source = random.choice([i for i in range(self.n_islands) if i != island])
                if island not in connections[source]:
                    connections[source].append(island)
    
    def _create_backup_topology(self) -> Dict[int, List[int]]:
        """Backup fully connected topology for emergencies"""
        backup = {}
        for i in range(self.n_islands):
            backup[i] = [j for j in range(self.n_islands) if j != i]
        return backup
    
    def get_migration_targets(self, island_id: int, use_backup: bool = False) -> List[int]:
        """Get targets with backup option"""
        if not isinstance(island_id, int) or island_id < 0:
            raise ValueError("island_id must be a non-negative integer")
        if not isinstance(use_backup, bool):
            raise TypeError("use_backup must be a boolean")
            
        if use_backup:
            return self.backup_connections.get(island_id, [])
        return self.connections.get(island_id, [])
    
    def should_migrate(self) -> bool:
        """Decide if migration should occur this cycle"""
        return random.random() < self.migration_prob


class AsynchronousIslandCacheManager:
    """Manages island-specific caches with topology awareness and edge case handling"""
    
    def __init__(self, n_islands: int, topology: EnhancedIslandTopology):
        if not isinstance(n_islands, int) or n_islands <= 0:
            raise ValueError("n_islands must be a positive integer")
        if not isinstance(topology, EnhancedIslandTopology):
            raise TypeError("topology must be an EnhancedIslandTopology instance")
            
        self.n_islands = n_islands
        self.topology = topology
        self.island_caches: Dict[int, IslandSpecificCache] = {}
        self.disconnection_detector = DisconnectionDetector(n_islands)
        self.emergency_manager = EmergencyBroadcastManager(n_islands)
        
        # Create cache for each island
        for i in range(n_islands):
            self.island_caches[i] = IslandSpecificCache(i)
    
    def seed_all_caches(self, n_inputs: int) -> None:
        """Seed all island caches with diverse initial expressions"""
        if not isinstance(n_inputs, int) or n_inputs <= 0:
            raise ValueError("n_inputs must be a positive integer")
            
        generator = ExpressionGenerator(n_inputs, max_depth=3)
        
        for island_id in range(self.n_islands):
            self._seed_island_cache(island_id, generator)
    
    def _seed_island_cache(self, island_id: int, generator: ExpressionGenerator) -> None:
        """Seed specific island cache"""
        cache = self.island_caches[island_id]
        
        # Generate diverse seed expressions
        for _ in range(10):  # 10 seed expressions per island
            expr = generator.generate_random_expression()
            seed_expr = CachedExpression(
                expression_str=expr.to_string(),
                fitness=random.uniform(0.05, 0.15),  # Low but varied baseline
                complexity=expr.complexity(),
                from_island=-1,  # Marker for seed
                timestamp=time.time(),
                generation=0,
                is_initial_seed=True
            )
            cache.add_expression(seed_expr, is_seed=True)
    
    def island_share_expression(self, island_id: int, expression: CachedExpression) -> None:
        """Island shares expression to its own cache"""
        if not isinstance(island_id, int) or island_id < 0:
            raise ValueError("island_id must be a non-negative integer")
        if not isinstance(expression, CachedExpression):
            raise TypeError("expression must be a CachedExpression instance")
            
        if island_id in self.island_caches:
            self.island_caches[island_id].add_expression(expression, is_seed=False)
            self.disconnection_detector.record_exchange(island_id)
    
    def island_import_expressions(self, requesting_island: int, max_count: int = 3) -> List[CachedExpression]:
        """Import expressions from accessible islands"""
        if not isinstance(requesting_island, int) or requesting_island < 0:
            raise ValueError("requesting_island must be a non-negative integer")
        if not isinstance(max_count, int) or max_count <= 0:
            raise ValueError("max_count must be a positive integer")
        
        # Get accessible islands from topology
        accessible_islands = self.topology.get_migration_targets(requesting_island)
        
        # Emergency mode: access all if disconnected
        if self.emergency_manager.emergency_mode:
            accessible_islands = [i for i in range(self.n_islands) if i != requesting_island]
        
        if not accessible_islands:
            return []  # Completely isolated
        
        # Sample from accessible islands
        all_candidates: List[CachedExpression] = []
        for island_id in accessible_islands:
            if island_id in self.island_caches:
                # Prefer initial seeds as requested
                candidates = self.island_caches[island_id].sample_expressions(
                    max_count, prefer_seeds=True
                )
                all_candidates.extend(candidates)
        
        if not all_candidates:
            return []
        
        # Return random sample from all candidates
        final_count = min(max_count, len(all_candidates))
        return random.sample(all_candidates, final_count)
    
    def check_and_handle_disconnection(self, generation: int) -> bool:
        """Check for disconnection and handle if necessary"""
        if not isinstance(generation, int):
            raise TypeError("generation must be an integer")
            
        total_exchanges = sum(self.disconnection_detector.exchange_counts.values())
        
        # Check if emergency mode should be activated
        if self.emergency_manager.should_activate_emergency(generation, total_exchanges):
            self.emergency_manager.emergency_mode = True
            return True
        
        # Check for topology disconnection
        if self.disconnection_detector.check_disconnection(generation):
            # Try topology repair
            if self.topology.repair_attempts < self.topology.max_repairs:
                self.topology.connections = self.topology._create_guaranteed_connected_topology()
                self.topology.repair_attempts += 1
                return True
        
        return False
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get statistics for all island caches"""
        stats = {}
        for island_id, cache in self.island_caches.items():
            stats[f'island_{island_id}'] = cache.get_cache_stats()
        
        # Global statistics
        total_expressions = sum(stats[key]['total_expressions'] for key in stats)
        total_seed_expressions = sum(stats[key]['seed_expressions'] for key in stats)
        
        stats['global'] = {
            'total_expressions': total_expressions,
            'total_seed_expressions': total_seed_expressions,
            'exchange_counts': dict(self.disconnection_detector.exchange_counts),
            'emergency_mode': self.emergency_manager.emergency_mode
        }
        
        return stats
