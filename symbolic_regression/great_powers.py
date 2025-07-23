"""
Great Powers Mechanism - Elite Expression Repository
Maintains the best 5 expressions across all generations with redundancy elimination
"""
import numpy as np
from typing import List, Dict, Optional, Tuple
from .expression_tree import Expression
from .utils import string_similarity, calculate_expression_uniqueness


class GreatPowers:
    """
    Maintains the top 5 expressions (Great Powers) dynamically across generations.
    These expressions are preserved from diversity injection and population restarts.
    Includes advanced redundancy elimination to prevent duplicate expressions.
    """
    
    def __init__(self, max_powers: int = 5, similarity_threshold: float = 0.98):
        self.max_powers = max_powers
        self.powers: List[Dict] = []  # List of {'expression': Expression, 'fitness': float, 'generation': int}
        self.generation_updates = 0
        self.similarity_threshold = similarity_threshold  # Threshold for considering expressions similar (increased to 0.98 to be much less aggressive)
        self.redundancy_stats = {
            'rejected_duplicates': 0,
            'semantic_rejections': 0,
            'structural_rejections': 0
        }
        
    def _is_expression_redundant(self, candidate_expr: Expression, X: Optional[np.ndarray] = None) -> Tuple[bool, str]:
        """
        Check if candidate expression is redundant with existing Great Powers.
        
        Args:
            candidate_expr: Expression to check for redundancy
            X: Optional data for semantic comparison
            
        Returns:
            Tuple of (is_redundant: bool, reason: str)
        """
        candidate_str = candidate_expr.to_string()
        
        for i, power in enumerate(self.powers):
            existing_expr = power['expression']
            existing_str = existing_expr.to_string()
            
            # 1. Exact string match (fastest check)
            if candidate_str == existing_str:
                return True, f"exact_duplicate_with_power_{i+1}"
            
            # 2. String similarity check
            str_similarity = string_similarity(candidate_str, existing_str)
            if str_similarity >= self.similarity_threshold:
                return True, f"string_similarity_{str_similarity:.3f}_with_power_{i+1}"
            
            # 3. Structural similarity check
            structural_similarity = self._calculate_structural_similarity(candidate_expr, existing_expr)
            if structural_similarity >= self.similarity_threshold:
                return True, f"structural_similarity_{structural_similarity:.3f}_with_power_{i+1}"
            
            # 4. Semantic similarity check (if data available) - DISABLED for better performance
            # Semantic similarity was being too aggressive and never triggered with high thresholds anyway
            # if X is not None:
            #     semantic_similarity = self._calculate_semantic_similarity(candidate_expr, existing_expr, X)
            #     if semantic_similarity >= self.similarity_threshold + 0.05:  # Slightly higher threshold for semantic
            #         return True, f"semantic_similarity_{semantic_similarity:.3f}_with_power_{i+1}"
        
        return False, "not_redundant"
    
    def _calculate_structural_similarity(self, expr1: Expression, expr2: Expression) -> float:
        """Calculate structural similarity between two expressions"""
        try:
            # Compare complexity
            comp1, comp2 = expr1.complexity(), expr2.complexity()
            if comp1 == 0 and comp2 == 0:
                return 1.0
            
            complexity_similarity = 1.0 - abs(comp1 - comp2) / max(comp1, comp2, 1)
            
            # Compare size (number of nodes)
            size1, size2 = expr1.size(), expr2.size()
            if size1 == 0 and size2 == 0:
                return 1.0
                
            size_similarity = 1.0 - abs(size1 - size2) / max(size1, size2, 1)
            
            # Get operator signatures
            ops1 = self._get_operator_signature(expr1)
            ops2 = self._get_operator_signature(expr2)
            
            # Calculate Jaccard similarity for operators
            all_ops = set(ops1.keys()) | set(ops2.keys())
            if not all_ops:
                operator_similarity = 1.0
            else:
                intersection = sum(min(ops1.get(op, 0), ops2.get(op, 0)) for op in all_ops)
                union = sum(max(ops1.get(op, 0), ops2.get(op, 0)) for op in all_ops)
                operator_similarity = intersection / union if union > 0 else 0.0
            
            # Weighted combination
            return (0.3 * complexity_similarity + 0.3 * size_similarity + 0.4 * operator_similarity)
            
        except Exception:
            # Fallback to simple comparison
            return 1.0 if expr1.to_string() == expr2.to_string() else 0.0
    
    def _calculate_semantic_similarity(self, expr1: Expression, expr2: Expression, X: np.ndarray) -> float:
        """Calculate semantic similarity based on expression outputs"""
        try:
            # Use a sample for performance
            sample_size = min(100, len(X))
            sample_indices = np.random.choice(len(X), sample_size, replace=False) if len(X) > sample_size else slice(None)
            X_sample = X[sample_indices]
            
            vals1 = expr1.evaluate(X_sample)
            vals2 = expr2.evaluate(X_sample)
            
            if vals1 is None or vals2 is None:
                return 0.0
            
            # Flatten for correlation calculation
            vals1_flat = vals1.flatten()
            vals2_flat = vals2.flatten()
            
            # Check for constant outputs
            if np.std(vals1_flat) < 1e-10 and np.std(vals2_flat) < 1e-10:
                # Both constant, check if same value
                return 1.0 if np.allclose(vals1_flat, vals2_flat, rtol=1e-5) else 0.0
            
            # Calculate correlation
            correlation = np.corrcoef(vals1_flat, vals2_flat)[0, 1]
            
            if np.isnan(correlation):
                return 0.0
            
            # Return absolute correlation as similarity
            return abs(correlation)
            
        except Exception:
            return 0.0
    
    def _get_operator_signature(self, expr: Expression) -> Dict[str, int]:
        """Get operator frequency signature for an expression"""
        signature = {}
        
        def collect_operators(node):
            from .expression_tree.core.node import BinaryOpNode, UnaryOpNode, ConstantNode, VariableNode
            
            if hasattr(node, 'operator'):
                if hasattr(node, 'left') and hasattr(node, 'right'):  # BinaryOpNode
                    signature[f"binary_{node.operator}"] = signature.get(f"binary_{node.operator}", 0) + 1
                    collect_operators(node.left)
                    collect_operators(node.right)
                elif hasattr(node, 'operand'):  # UnaryOpNode
                    signature[f"unary_{node.operator}"] = signature.get(f"unary_{node.operator}", 0) + 1
                    collect_operators(node.operand)
            elif hasattr(node, 'value'):  # ConstantNode
                signature["constant"] = signature.get("constant", 0) + 1
            elif hasattr(node, 'index'):  # VariableNode
                signature[f"var_{node.index}"] = signature.get(f"var_{node.index}", 0) + 1
        
        try:
            collect_operators(expr.root)
        except Exception:
            pass
            
        return signature
        
    def update_powers(self, population: List[Expression], fitness_scores: List[float], generation: int, X: Optional[np.ndarray] = None) -> bool:
        """
        Update the Great Powers with the best expression from current generation.
        Includes redundancy elimination and complexity bias to favor simpler expressions.
        
        Args:
            population: Current generation population
            fitness_scores: Fitness scores for current generation
            generation: Current generation number
            X: Optional training data for semantic redundancy checking
        
        Returns:
            bool: True if a new Great Power was added or updated
        """
        # Find the best expression in current generation
        best_idx = np.argmax(fitness_scores)
        best_fitness = fitness_scores[best_idx]
        best_expr = population[best_idx].copy()
        
        # CHECK FOR REDUNDANCY - but allow significantly better expressions even if similar
        is_redundant, redundancy_reason = self._is_expression_redundant(best_expr, X)
        if is_redundant:
            # Extract similar power index from redundancy reason
            similar_power_idx = None
            if "_with_power_" in redundancy_reason:
                try:
                    similar_power_idx = int(redundancy_reason.split("_with_power_")[1]) - 1  # Convert to 0-based index
                except:
                    pass
            
            # If we found which power it's similar to, check if candidate is significantly better
            if similar_power_idx is not None and 0 <= similar_power_idx < len(self.powers):
                existing_fitness = self.powers[similar_power_idx]['fitness']
                fitness_improvement = best_fitness - existing_fitness
                
                # VERY RELAXED: Allow any improvement, even tiny ones (0.001 = 0.1% R² improvement)
                if fitness_improvement > 0.001:
                    # This is an improvement, allow it despite similarity
                    pass  # Continue with normal processing
                else:
                    # Only reject if truly not better and exact duplicate
                    if "exact_duplicate" in redundancy_reason:
                        self.redundancy_stats['rejected_duplicates'] += 1
                        return False  # Only reject exact duplicates that aren't better
                    else:
                        # For all other similarities, allow if fitness is decent
                        if best_fitness < 0.85:  # Only reject if fitness is below 85%
                            if "semantic_similarity" in redundancy_reason:
                                self.redundancy_stats['semantic_rejections'] += 1
                            elif "structural_similarity" in redundancy_reason or "string_similarity" in redundancy_reason:
                                self.redundancy_stats['structural_rejections'] += 1
                            return False
            else:
                # Couldn't determine which power it's similar to, be very lenient
                if "exact_duplicate" in redundancy_reason:
                    self.redundancy_stats['rejected_duplicates'] += 1
                    return False  # Always reject exact duplicates
                else:
                    # For other similarities, only reject if fitness is very poor
                    if best_fitness < 0.70:  # Only reject if fitness is below 70%
                        if "semantic_similarity" in redundancy_reason:
                            self.redundancy_stats['semantic_rejections'] += 1
                        elif "structural_similarity" in redundancy_reason or "string_similarity" in redundancy_reason:
                            self.redundancy_stats['structural_rejections'] += 1
                        return False
        
        # Apply complexity bias: prefer simpler expressions at similar fitness levels
        complexity = best_expr.complexity()
        complexity_penalty = min(0.1, complexity * 0.005)  # Max 0.1 penalty
        adjusted_fitness = best_fitness - complexity_penalty
        
        # Check if this expression should become a Great Power
        updated = False
        
        if len(self.powers) < self.max_powers:
            # Still have slots available - add the new Great Power
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
    
    def get_redundancy_stats(self) -> Dict:
        """Get statistics about rejected redundant expressions"""
        total_rejections = sum(self.redundancy_stats.values())
        return {
            **self.redundancy_stats,
            'total_rejections': total_rejections
        }
    
    def clean_redundant_powers(self, X: Optional[np.ndarray] = None, console_log: bool = False) -> int:
        """
        Clean up existing Great Powers to remove any redundant expressions.
        This is useful if Great Powers were added before redundancy checking was implemented.
        
        Args:
            X: Optional training data for semantic similarity checking
            console_log: Whether to log cleanup details
            
        Returns:
            Number of redundant powers removed
        """
        if len(self.powers) <= 1:
            return 0
        
        original_count = len(self.powers)
        cleaned_powers = []
        removed_count = 0
        
        # Keep track of expressions we've already added to avoid duplicates
        for i, power in enumerate(self.powers):
            expr = power['expression']
            is_redundant = False
            
            # Check if this expression is redundant with any previously added expression
            for existing_power in cleaned_powers:
                existing_expr = existing_power['expression']
                
                # Use the same redundancy checks as in _is_expression_redundant
                candidate_str = expr.to_string()
                existing_str = existing_expr.to_string()
                
                # 1. Exact string match
                if candidate_str == existing_str:
                    is_redundant = True
                    if console_log:
                        print(f"  Removing Great Power {i+1}: exact duplicate")
                    break
                
                # 2. String similarity check
                str_similarity = string_similarity(candidate_str, existing_str)
                if str_similarity >= self.similarity_threshold:
                    is_redundant = True
                    if console_log:
                        print(f"  Removing Great Power {i+1}: string similarity {str_similarity:.3f}")
                    break
                
                # 3. Structural similarity check
                structural_similarity = self._calculate_structural_similarity(expr, existing_expr)
                if structural_similarity >= self.similarity_threshold:
                    is_redundant = True
                    if console_log:
                        print(f"  Removing Great Power {i+1}: structural similarity {structural_similarity:.3f}")
                    break
                
                # 4. Semantic similarity check (if data available)
                if X is not None:
                    semantic_similarity = self._calculate_semantic_similarity(expr, existing_expr, X)
                    if semantic_similarity >= self.similarity_threshold + 0.05:
                        is_redundant = True
                        if console_log:
                            print(f"  Removing Great Power {i+1}: semantic similarity {semantic_similarity:.3f}")
                        break
            
            if not is_redundant:
                cleaned_powers.append(power)
            else:
                removed_count += 1
        
        # Update the powers list
        self.powers = cleaned_powers
        
        # Re-sort by adjusted fitness
        if self.powers:
            self.powers.sort(key=lambda p: p.get('adjusted_fitness', p['fitness']), reverse=True)
        
        if console_log and removed_count > 0:
            print(f"Great Powers cleanup: removed {removed_count} redundant expressions, {len(self.powers)} remaining")
        
        return removed_count
    
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
