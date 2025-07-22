"""
Great Powers Mechanism - Elite Expression Repository
Maintains the best 5 expressions across all generations
"""
import numpy as np
from typing import List, Dict, Optional, Tuple
from .expression_tree import Expression


class GreatPowers:
    """
    Maintains the top 5 expressions (Great Powers) dynamically across generations.
    These expressions are preserved from diversity injection and population restarts.
    """
    
    def __init__(self, max_powers: int = 5):
        self.max_powers = max_powers
        self.powers: List[Dict] = []  # List of {'expression': Expression, 'fitness': float, 'generation': int}
        self.generation_updates = 0
        
    def update_powers(self, population: List[Expression], fitness_scores: List[float], generation: int) -> bool:
        """
        Update the Great Powers with the best expression from current generation.
        Includes complexity bias to favor simpler expressions.
        
        Returns:
            bool: True if a new Great Power was added or updated
        """
        # Find the best expression in current generation
        best_idx = np.argmax(fitness_scores)
        best_fitness = fitness_scores[best_idx]
        best_expr = population[best_idx].copy()
        
        # Apply complexity bias: prefer simpler expressions at similar fitness levels
        complexity = best_expr.complexity()
        complexity_penalty = min(0.1, complexity * 0.005)  # Max 0.1 penalty
        adjusted_fitness = best_fitness - complexity_penalty
        
        # Check if this expression should become a Great Power
        updated = False
        
        if len(self.powers) < self.max_powers:
            # Still have slots available
            self.powers.append({
                'expression': best_expr,
                'fitness': best_fitness,
                'adjusted_fitness': adjusted_fitness,
                'complexity': complexity,
                'generation': generation
            })
            updated = True
        else:
            # Find the weakest Great Power based on adjusted fitness
            weakest_idx = min(range(len(self.powers)), 
                            key=lambda i: self.powers[i].get('adjusted_fitness', self.powers[i]['fitness']))
            weakest_adjusted_fitness = self.powers[weakest_idx].get('adjusted_fitness', self.powers[weakest_idx]['fitness'])
            
            # Replace if current best is better than weakest Great Power
            if adjusted_fitness > weakest_adjusted_fitness + 1e-8:  # Small threshold to avoid numerical issues
                self.powers[weakest_idx] = {
                    'expression': best_expr,
                    'fitness': best_fitness,
                    'adjusted_fitness': adjusted_fitness,
                    'complexity': complexity,
                    'generation': generation
                }
                updated = True
        
        if updated:
            # Sort powers by adjusted fitness (best first)
            self.powers.sort(key=lambda p: p.get('adjusted_fitness', p['fitness']), reverse=True)
            self.generation_updates += 1
            
        return updated
    
    def get_best_expression(self) -> Optional[Expression]:
        """Get the best Great Power expression"""
        if self.powers:
            return self.powers[0]['expression'].copy()
        return None
    
    def get_best_fitness(self) -> float:
        """Get the fitness of the best Great Power (raw fitness, not adjusted)"""
        if self.powers:
            return self.powers[0]['fitness']
        return -10.0  # Large negative R² score when no powers exist
    
    def get_all_powers(self) -> List[Expression]:
        """Get all Great Power expressions"""
        return [power['expression'].copy() for power in self.powers]
    
    def get_powers_info(self) -> List[Dict]:
        """Get detailed information about all Great Powers"""
        return [
            {
                'expression': power['expression'].to_string(),
                'fitness': power['fitness'],
                'adjusted_fitness': power.get('adjusted_fitness', power['fitness']),
                'complexity': power.get('complexity', power['expression'].complexity()),
                'generation': power['generation'],
                'rank': i + 1
            }
            for i, power in enumerate(self.powers)
        ]
    
    def inject_powers_into_population(self, population: List[Expression], 
                                    fitness_scores: List[float], 
                                    injection_rate: float = 0.1) -> Tuple[List[Expression], List[float]]:
        """
        Inject Great Powers into population, replacing worst performers.
        
        Args:
            population: Current population
            fitness_scores: Current fitness scores
            injection_rate: Fraction of population to replace with Great Powers
            
        Returns:
            Tuple of updated population and fitness scores
        """
        if not self.powers:
            return population, fitness_scores
            
        n_to_inject = min(len(self.powers), max(1, int(len(population) * injection_rate)))
        
        # Find worst performers to replace, but avoid replacing Great Powers already in population
        power_strings = {power['expression'].to_string() for power in self.powers}
        population_strings = [expr.to_string() for expr in population]
        
        # Create list of indices that don't contain Great Powers already
        non_power_indices = [i for i, expr_str in enumerate(population_strings) 
                           if expr_str not in power_strings]
        
        if not non_power_indices:
            return population, fitness_scores
            
        # Sort non-power indices by fitness to find worst performers
        non_power_fitness = [(i, fitness_scores[i]) for i in non_power_indices]
        non_power_fitness.sort(key=lambda x: x[1])  # Sort by fitness ascending
        
        worst_indices = [idx for idx, _ in non_power_fitness[:n_to_inject]]
        
        new_population = population.copy()
        new_fitness_scores = fitness_scores.copy()
        
        # Inject Great Powers
        for i, worst_idx in enumerate(worst_indices):
            if i < len(self.powers):
                new_population[worst_idx] = self.powers[i]['expression'].copy()
                new_fitness_scores[worst_idx] = self.powers[i]['fitness']
        
        return new_population, new_fitness_scores
    
    def protect_elites_from_injection(self, population: List[Expression], 
                                    fitness_scores: List[float],
                                    elite_fraction: float = 0.1) -> List[int]:
        """
        Get indices of population members that should be protected from diversity injection.
        Includes both current elites and any Great Powers present in the population.
        
        Returns:
            List of indices to protect
        """
        protected_indices = set()
        
        # Protect current elites
        elite_count = max(1, int(elite_fraction * len(population)))
        elite_indices = np.argsort(fitness_scores)[-elite_count:]
        protected_indices.update(elite_indices)
        
        # Protect Great Powers if they're in the population
        power_strings = [power['expression'].to_string() for power in self.powers]
        for i, expr in enumerate(population):
            if expr.to_string() in power_strings:
                protected_indices.add(i)
        
        return list(protected_indices)
    
    def get_final_candidate(self, current_population: List[Expression], 
                          current_fitness_scores: List[float]) -> Expression:
        """
        Get the final candidate by comparing Great Powers with current generation's best.
        
        Args:
            current_population: Current generation population
            current_fitness_scores: Current generation fitness scores
            
        Returns:
            The best expression overall (from Great Powers or current generation)
        """
        if not self.powers:
            # No Great Powers, return best from current generation
            best_idx = np.argmax(current_fitness_scores)
            return current_population[best_idx].copy()
        
        # Get best from current generation
        current_best_idx = np.argmax(current_fitness_scores)
        current_best_fitness = current_fitness_scores[current_best_idx]
        current_best_expr = current_population[current_best_idx]
        
        # Get best Great Power
        great_power_best_fitness = self.get_best_fitness()
        great_power_best_expr = self.get_best_expression()
        
        # Return the better one
        if great_power_best_expr is not None and great_power_best_fitness > current_best_fitness:
            return great_power_best_expr
        else:
            return current_best_expr.copy()
            
    def get_all_candidates(self, current_population: List[Expression], 
                         current_fitness_scores: List[float]) -> List[Dict]:
        """
        Get all candidate expressions with their fitness scores for final selection.
        
        Returns:
            List of dictionaries with 'expression', 'fitness', and 'source' keys
        """
        candidates = []
        
        # Add all Great Powers
        for i, power in enumerate(self.powers):
            candidates.append({
                'expression': power['expression'].copy(),
                'fitness': power['fitness'],
                'generation': power['generation'],
                'source': f'GreatPower_{i+1}'
            })
        
        # Add best from current generation
        if current_fitness_scores:
            best_idx = np.argmax(current_fitness_scores)
            candidates.append({
                'expression': current_population[best_idx].copy(),
                'fitness': current_fitness_scores[best_idx],
                'generation': 'current',
                'source': 'CurrentBest'
            })
        
        # Sort by fitness (best first)
        candidates.sort(key=lambda x: x['fitness'], reverse=True)
        return candidates
    
    def clear_powers(self):
        """Clear all Great Powers (for restart scenarios)"""
        self.powers.clear()
        self.generation_updates = 0
        
    def get_update_history(self) -> List[Dict]:
        """Get history of Great Powers updates for debugging"""
        return [{
            'generation': power['generation'],
            'fitness': power['fitness'],
            'expression': power['expression'].to_string()[:50] + "..." if len(power['expression'].to_string()) > 50 else power['expression'].to_string()
        } for power in self.powers]
    
    def diagnose_fitness_drop(self, current_best_fitness: float, generation: int) -> Dict:
        """
        Diagnose potential fitness drop issues by comparing with Great Powers.
        
        Returns:
            Dictionary with diagnostic information
        """
        if not self.powers:
            return {"status": "no_great_powers", "message": "No Great Powers available for comparison"}
        
        best_gp_fitness = self.get_best_fitness()
        fitness_gap = best_gp_fitness - current_best_fitness
        
        diagnosis = {
            "status": "normal",
            "current_best": current_best_fitness,
            "great_power_best": best_gp_fitness,
            "fitness_gap": fitness_gap,
            "generation": generation,
            "num_great_powers": len(self.powers)
        }
        
        if fitness_gap > 0.01:  # Significant fitness drop
            diagnosis["status"] = "fitness_drop_detected"
            diagnosis["message"] = f"Current best ({current_best_fitness:.6f}) is significantly worse than Great Power best ({best_gp_fitness:.6f})"
            
            # Find which Great Power was last updated
            latest_generation = max(power['generation'] for power in self.powers)
            diagnosis["latest_great_power_generation"] = latest_generation
            diagnosis["generations_since_update"] = generation - latest_generation
            
        return diagnosis
    
    def __len__(self):
        return len(self.powers)
    
    def __bool__(self):
        return len(self.powers) > 0
