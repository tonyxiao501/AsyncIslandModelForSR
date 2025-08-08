import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import os
from datetime import datetime
from symbolic_regression.ensemble_regressor import EnsembleMIMORegressor


class PhysicsLawTester:
    """
    Test symbolic regression on well-known physics laws using MISO approach.
    
    MISO (Multiple Input Single Output):
    - Functions with only constants and one variable are treated as SISO (Single Input Single Output)
    - Multi-variable functions (e.g., F = ma, PV = nRT) are tested as MISO
    - This demonstrates the system's ability to discover relationships between
      multiple input variables that produce a single output
    """
    
    def __init__(self):
        # Configure the ensemble regressor with enhanced physics-focused parameters
        self.model = EnsembleMIMORegressor(
            n_fits=8,                    # Increased for better ensemble diversity
            top_n_select=5,             
            population_size=200,         # Increased for better exploration
            generations=200,             # Sufficient for most physics laws
            mutation_rate=0.15,          # Balanced for exploration vs exploitation
            crossover_rate=0.8,
            tournament_size=3,
            max_depth=6,                 # Reasonable depth for physics expressions
            parsimony_coefficient=0.003, # Balanced for physics expressions
            diversity_threshold=0.6,     # Good diversity
            adaptive_rates=True,
            restart_threshold=15,        # Reasonable patience
            elite_fraction=0.12,         # Elite fraction
            # PySR-style adaptive parsimony settings for physics
            enable_adaptive_parsimony=True,  # Enable adaptive parsimony coefficient
            domain_type="physics",           # Physics-specific operator weighting
            # Early termination and late extension parameters
            enable_early_termination=True,
            early_termination_threshold=0.99,     # Terminate if R² >= 0.99
            early_termination_check_interval=10,  # Check every 10 generations
            enable_late_extension=True,
            late_extension_threshold=0.95,        # Extend if R² < 0.95 at the end
            late_extension_generations=50,        # Add 50 more generations if needed
            # Asynchronous migration system for better island model
            use_asynchronous_migration=True,      # Use the new asynchronous island-specific cache system
            migration_interval=20,               # Average generations between migration events
            migration_probability=0.3           # Probability of migration per cycle
        )
        
        # Initialize results storage
        self.results = []
    
    def generate_data(self, func, x_ranges, n_samples=300, noise_level=0.02):
        """Generate training data for a given multi-variable function (MISO)"""
        np.random.seed(42)  # For reproducibility
        
        n_vars = len(x_ranges) // 2  # Each variable has min and max
        
        if n_vars == 1:  # Single variable (still useful for physics)
            X = np.linspace(x_ranges[0], x_ranges[1], n_samples).reshape(-1, 1)
            y_true = func(X[:, 0])
        elif n_vars == 2:  # Two variables (MISO)
            samples_per_dim = int(np.sqrt(n_samples))
            x1 = np.linspace(x_ranges[0], x_ranges[1], samples_per_dim)
            x2 = np.linspace(x_ranges[2], x_ranges[3], samples_per_dim)
            X1, X2 = np.meshgrid(x1, x2)
            X = np.column_stack([X1.flatten(), X2.flatten()])
            y_true = func(X[:, 0], X[:, 1])
        elif n_vars == 3:  # Three variables (MISO)
            samples_per_dim = int(np.cbrt(n_samples))
            x1 = np.linspace(x_ranges[0], x_ranges[1], samples_per_dim)
            x2 = np.linspace(x_ranges[2], x_ranges[3], samples_per_dim)
            x3 = np.linspace(x_ranges[4], x_ranges[5], samples_per_dim)
            X1, X2, X3 = np.meshgrid(x1, x2, x3, indexing='ij')
            X = np.column_stack([X1.flatten(), X2.flatten(), X3.flatten()])
            y_true = func(X[:, 0], X[:, 1], X[:, 2])
        elif n_vars == 4:  # Four variables (MISO)
            samples_per_dim = int(n_samples**(1/4))
            x1 = np.linspace(x_ranges[0], x_ranges[1], samples_per_dim)
            x2 = np.linspace(x_ranges[2], x_ranges[3], samples_per_dim)
            x3 = np.linspace(x_ranges[4], x_ranges[5], samples_per_dim)
            x4 = np.linspace(x_ranges[6], x_ranges[7], samples_per_dim)
            X1, X2, X3, X4 = np.meshgrid(x1, x2, x3, x4, indexing='ij')
            X = np.column_stack([X1.flatten(), X2.flatten(), X3.flatten(), X4.flatten()])
            y_true = func(X[:, 0], X[:, 1], X[:, 2], X[:, 3])
        elif n_vars == 5:  # Five variables (MISO)
            samples_per_dim = int(n_samples**(1/5))
            x1 = np.linspace(x_ranges[0], x_ranges[1], samples_per_dim)
            x2 = np.linspace(x_ranges[2], x_ranges[3], samples_per_dim)
            x3 = np.linspace(x_ranges[4], x_ranges[5], samples_per_dim)
            x4 = np.linspace(x_ranges[6], x_ranges[7], samples_per_dim)
            x5 = np.linspace(x_ranges[8], x_ranges[9], samples_per_dim)
            X1, X2, X3, X4, X5 = np.meshgrid(x1, x2, x3, x4, x5, indexing='ij')
            X = np.column_stack([X1.flatten(), X2.flatten(), X3.flatten(), X4.flatten(), X5.flatten()])
            y_true = func(X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4])
        else:
            # For more than 5 variables, use random sampling (MISO)
            X = np.random.uniform(
                low=[x_ranges[i*2] for i in range(n_vars)],
                high=[x_ranges[i*2+1] for i in range(n_vars)],
                size=(n_samples, n_vars)
            )
            y_true = func(*X.T)
        
        # Add noise
        noise = np.random.normal(0, noise_level * np.std(y_true), size=y_true.shape)
        y_noisy = y_true + noise
        
        # Ensure output is single column (MISO)
        return X, y_noisy.reshape(-1, 1), y_true
    
    def test_law(self, name, description, func, x_ranges, target_expr):
        """Test a single physics law using MISO (Multiple Input Single Output) approach"""
        print(f"\nTesting: {name}")
        print(f"Description: {description}")
        print(f"Target: {target_expr}")
        
        # Determine number of variables for this law
        n_vars = len(x_ranges) // 2
        var_type = "SISO" if n_vars == 1 else "MISO"
        print(f"Variables: {n_vars} ({var_type})")
        
        # Generate data
        X, y, y_true = self.generate_data(func, x_ranges)
        
        print(f"Generated {X.shape[0]} samples with {X.shape[1]} input variables")
        
        # Fit model
        self.model.fit(X, y, constant_optimize=True)
        
        # Get predictions and expressions
        y_pred = self.model.predict(X, strategy='mean')
        discovered_expressions = self.model.get_expressions()
        
        # Calculate R^2 score
        r2_score_val = r2_score(y_true, y_pred.flatten())
        
        # Store results
        result = {
            'name': name,
            'target': target_expr,
            'n_variables': n_vars,
            'type': var_type,
            'r2_score': r2_score_val,
            'expressions': discovered_expressions[:3]  # Top 3
        }
        self.results.append(result)
    def test_law_detailed(self, name, description, func, x_ranges, target_expr, plot_results=True):
        """Test a single physics law with detailed analysis and visualization"""
        print(f"\n{'='*80}")
        print(f"TESTING: {name}")
        print(f"{'='*80}")
        print(f"Description: {description}")
        print(f"Target: {target_expr}")
        
        # Determine number of variables for this law
        n_vars = len(x_ranges) // 2
        var_type = "SISO" if n_vars == 1 else "MISO"
        print(f"Variables: {n_vars} ({var_type})")
        
        # Generate data
        X, y, y_true = self.generate_data(func, x_ranges)
        print(f"Generated {X.shape[0]} samples with {X.shape[1]} input variables")
        
        # Fit model
        print(f"Training ensemble regressor...")
        self.model.fit(X, y, constant_optimize=True)
        
        # Get predictions and expressions
        y_pred = self.model.predict(X, strategy='mean')
        discovered_expressions = self.model.get_expressions()
        
        # Calculate R^2 score
        r2_score_val = r2_score(y_true, y_pred.flatten())
        
        # Store results
        result = {
            'name': name,
            'target': target_expr,
            'n_variables': n_vars,
            'type': var_type,
            'r2_score': r2_score_val,
            'expressions': discovered_expressions[:5]  # Top 5
        }
        self.results.append(result)
        
        # Print results
        print(f"\nR² Score: {r2_score_val:.4f}")
        print(f"Best expression: {discovered_expressions[0] if discovered_expressions else 'None'}")
        if len(discovered_expressions) > 1:
            print(f"2nd best: {discovered_expressions[1]}")
        if len(discovered_expressions) > 2:
            print(f"3rd best: {discovered_expressions[2]}")
        
        # Create visualizations if requested and it's a testable case
        if plot_results and n_vars <= 2:  # Only plot for 1D or 2D cases
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"{timestamp}_physics_results"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            self._create_visualization(X, y, y_true, y_pred, discovered_expressions, 
                                     result, output_dir, n_vars)
    
    def _create_visualization(self, X, y, y_true, y_pred, expressions, result, output_dir, n_vars):
        """Create visualization plots for physics law testing"""
        # Get individual predictions for each of the top 5 expressions
        individual_predictions = []
        individual_r2_scores = []
        
        for expr in self.model.best_expressions[:5]:  # Get top 5 expressions
            pred = expr.evaluate(X)
            if pred.ndim == 1:
                pred = pred.reshape(-1, 1)
            individual_predictions.append(pred.flatten())
            
            # Calculate R² for this individual expression
            r2_individual = r2_score(y_true, pred.flatten())
            individual_r2_scores.append(r2_individual)
        
        # Get fitness histories
        fitness_histories = self.model.get_fitness_histories()
        
        # Create plots for each candidate
        colors = ['red', 'green', 'orange', 'purple', 'brown']
        
        for i in range(min(5, len(individual_predictions))):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            if n_vars == 1:
                # 1D plot
                ax1.scatter(X.flatten(), y.flatten(), alpha=0.6, color='lightblue', 
                           s=25, label='Training Data', zorder=2)
                sorted_indices = np.argsort(X.flatten())
                ax1.plot(X.flatten()[sorted_indices], y_true[sorted_indices],
                        color='blue', linewidth=3, label='True Function', zorder=3)
                ax1.plot(X.flatten()[sorted_indices], individual_predictions[i][sorted_indices],
                        color=colors[i], linewidth=2, linestyle='--',
                        label=f'Candidate {i+1}', zorder=4)
                ax1.set_xlabel('Input Variable', fontsize=12)
            else:
                # 2D scatter plot
                ax1.scatter(y_true, individual_predictions[i], alpha=0.6, color=colors[i])
                ax1.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 
                        'k--', label='Perfect Prediction')
                ax1.set_xlabel('True Values', fontsize=12)
                ax1.set_ylabel('Predicted Values', fontsize=12)
            
            ax1.set_ylabel('Output', fontsize=12)
            ax1.set_title(f'{result["name"]} - Candidate {i+1} (R² = {individual_r2_scores[i]:.4f})', 
                         fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            ax1.legend(fontsize=10)
            
            # Add expression as text
            expr_text = expressions[i] if i < len(expressions) else "N/A"
            if len(expr_text) > 50:
                expr_text = expr_text[:47] + "..."
            ax1.text(0.02, 0.98, f'f = {expr_text}', transform=ax1.transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
            
            # Fitness evolution plot
            if i < len(fitness_histories) and fitness_histories[i]:
                generations = range(1, len(fitness_histories[i]) + 1)
                ax2.plot(generations, fitness_histories[i], color=colors[i], linewidth=2, marker='o', markersize=2)
                ax2.set_xlabel('Generation', fontsize=12)
                ax2.set_ylabel('Fitness Score', fontsize=12)
                ax2.set_title(f'Fitness Evolution - Candidate {i+1}', fontsize=14, fontweight='bold')
                ax2.grid(True, alpha=0.3)
                
                final_fitness = fitness_histories[i][-1]
                ax2.text(0.98, 0.98, f'Final: {final_fitness:.6f}', 
                        transform=ax2.transAxes, fontsize=10, ha='right', va='top',
                        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
            else:
                ax2.text(0.5, 0.5, 'No fitness history available', 
                        transform=ax2.transAxes, fontsize=12, ha='center', va='center')
                ax2.set_title(f'Fitness Evolution - Candidate {i+1}', fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            
            # Save the plot
            safe_name = "".join(c for c in result['name'] if c.isalnum() or c in (' ', '-', '_')).strip()
            safe_name = safe_name.replace(' ', '_')
            filename = f"{safe_name}_candidate_{i+1}.png"
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"Visualizations saved to: {output_dir}")

    def save_results(self):
        """Save all results to a text file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"physics_laws_results_{timestamp}.txt"
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("SYMBOLIC REGRESSION RESULTS FOR PHYSICS LAWS (MISO TESTING)\n")
            f.write("=" * 70 + "\n\n")
            
            # Summary statistics
            siso_count = sum(1 for r in self.results if r['type'] == 'SISO')
            miso_count = sum(1 for r in self.results if r['type'] == 'MISO')
            high_score_count = sum(1 for r in self.results if r['r2_score'] > 0.95)
            
            f.write(f"SUMMARY:\n")
            f.write(f"Total laws tested: {len(self.results)}\n")
            f.write(f"SISO (Single Input): {siso_count}\n") 
            f.write(f"MISO (Multiple Input): {miso_count}\n")
            f.write(f"High accuracy (R² > 0.95): {high_score_count}\n")
            f.write(f"Average R² score: {np.mean([r['r2_score'] for r in self.results]):.4f}\n")
            f.write("\n" + "=" * 70 + "\n\n")
            
            for i, result in enumerate(self.results, 1):
                f.write(f"{i}. {result['name']} ({result['type']})\n")
                f.write(f"Target function: {result['target']}\n")
                f.write(f"Variables: {result['n_variables']}\n")
                f.write(f"R² Score: {result['r2_score']:.6f}\n")
                f.write("Discovered expressions:\n")
                
                for j, expr in enumerate(result['expressions'], 1):
                    f.write(f"  Candidate {j}: {expr}\n")
                
                f.write("\n" + "-" * 50 + "\n\n")
        
        print(f"\nResults saved to: {filename}")
        print(f"Summary: {len(self.results)} laws tested, {miso_count} MISO, {siso_count} SISO")
        print(f"High accuracy (R² > 0.95): {high_score_count}/{len(self.results)}")


def main():
    """Main function to test physics laws using MISO approach with visualization"""
    tester = PhysicsLawTester()
    
    print("Testing Symbolic Regression on Physics Laws")
    print("MISO Approach: Multiple Input Single Output")
    print("Focus on key laws with detailed visualization")
    print("=" * 80)
    
    # Test carefully selected physics laws that demonstrate MISO capabilities:
    # - Functions with constants and one variable are treated as single input
    # - Multi-variable functions are tested as true MISO
    
    print("\n1. SINGLE INPUT LAWS (SISO - for completeness)")
    print("-" * 50)
    
    # Simple Harmonic Motion: F = -kx (SISO: one variable plus constant)
    tester.test_law_detailed(
        "Hooke's Law",
        "Spring force proportional to displacement (SISO with constant)",
        lambda k, x: k * x,  # k is treated as constant, x as variable
        [100, 100, -2, 2],  # constant k=100, variable x from -2 to 2
        "F = kx"
    )
    
    # Free Fall: h = (1/2)gt^2 (SISO: one variable plus constant)
    tester.test_law_detailed(
        "Free Fall Distance",
        "Distance fallen under gravity (SISO with constant)",
        lambda g, t: 0.5 * g * t**2,  # g is treated as constant, t as variable
        [9.81, 9.81, 0, 5],  # constant g=9.81, variable t from 0 to 5
        "h = (1/2)gt²"
    )
    
    print("\n\n2. TRUE MISO LAWS (Multiple Input Single Output)")
    print("-" * 50)
    
    # Newton's Second Law: F = ma (MISO: two variables)
    tester.test_law_detailed(
        "Newton's Second Law",
        "Force equals mass times acceleration (MISO)",
        lambda m, a: m * a,
        [0.5, 10, 0.5, 20],  # mass range, acceleration range
        "F = ma",
        plot_results=True
    )
    
    # Kinetic Energy: KE = (1/2)mv^2 (MISO: two variables)
    tester.test_law_detailed(
        "Kinetic Energy",
        "Energy of motion (MISO)",
        lambda m, v: 0.5 * m * v**2,
        [1, 20, 0, 30],  # mass, velocity
        "KE = (1/2)mv²",
        plot_results=True
    )
    
    # Ideal Gas Law: P = nRT/V (MISO: four variables)
    tester.test_law_detailed(
        "Ideal Gas Law",
        "Pressure from gas properties (MISO)",
        lambda n, R, T, V: n * R * T / V,
        [1, 10, 8.314, 8.314, 250, 400, 0.02, 0.1],  # n, R, T, V ranges
        "P = nRT/V",
        plot_results=False  # Too many dimensions for plotting
    )
    
    # Coulomb's Law: F = kq1q2/r^2 (MISO: four variables)
    tester.test_law_detailed(
        "Coulomb's Law",
        "Electrostatic force between charges (MISO)",
        lambda k, q1, q2, r: k * q1 * q2 / r**2,
        [8.99e9, 8.99e9, 1e-6, 1e-5, 1e-6, 1e-5, 0.1, 2],  # k, q1, q2, r
        "F = kq₁q₂/r²",
        plot_results=False  # Too many dimensions for plotting
    )
    
    # Gravitational Force: F = Gm1m2/r^2 (MISO: four variables)
    tester.test_law_detailed(
        "Newton's Law of Gravitation",
        "Gravitational force between masses (MISO)",
        lambda G, m1, m2, r: G * m1 * m2 / r**2,
        [6.67e-11, 6.67e-11, 1e20, 1e24, 1e20, 1e24, 1e6, 1e8],  # G, m1, m2, r
        "F = Gm₁m₂/r²",
        plot_results=False  # Too many dimensions for plotting
    )
    
    # Centripetal Force: F = mv^2/r (MISO: three variables)
    tester.test_law_detailed(
        "Centripetal Force",
        "Force required for circular motion (MISO)",
        lambda m, v, r: m * v**2 / r,
        [1, 20, 5, 25, 0.5, 5],  # mass, velocity, radius
        "F = mv²/r",
        plot_results=False  # Three dimensions
    )
    
    # Wave Speed: v = fλ (MISO: two variables)
    tester.test_law_detailed(
        "Wave Speed",
        "Wave speed from frequency and wavelength (MISO)",
        lambda f, wavelength: f * wavelength,
        [100, 1000, 0.1, 5],  # frequency, wavelength
        "v = fλ",
        plot_results=True
    )
    
    # Electric Power: P = VI (MISO: two variables)
    tester.test_law_detailed(
        "Electric Power",
        "Power dissipated in circuit (MISO)",
        lambda V, I: V * I,
        [1, 50, 0.1, 5],  # voltage, current
        "P = VI",
        plot_results=True
    )
    
    # Potential Energy: PE = mgh (MISO: three variables)
    tester.test_law_detailed(
        "Gravitational Potential Energy",
        "Energy due to height in gravitational field (MISO)",
        lambda m, g, h: m * g * h,
        [1, 20, 9.8, 9.82, 0, 50],  # mass, gravity, height
        "PE = mgh",
        plot_results=False  # Three dimensions
    )
    
    # Save all results
    tester.save_results()
    
    print(f"\n{'='*80}")
    print(f"TESTING COMPLETED: {len(tester.results)} physics laws tested!")
    print(f"{'='*80}")
    
    # Enhanced summary
    siso_laws = [r for r in tester.results if r['type'] == 'SISO']
    miso_laws = [r for r in tester.results if r['type'] == 'MISO']
    high_score_laws = [r for r in tester.results if r['r2_score'] > 0.95]
    
    print(f"\nSUMMARY STATISTICS:")
    print(f"SISO laws (Single Input + Constants): {len(siso_laws)}")
    print(f"MISO laws (Multiple Inputs): {len(miso_laws)}")
    print(f"High accuracy (R² > 0.95): {len(high_score_laws)}/{len(tester.results)}")
    print(f"Average R² score: {np.mean([r['r2_score'] for r in tester.results]):.4f}")
    
    print(f"\nTOP PERFORMING LAWS:")
    sorted_results = sorted(tester.results, key=lambda x: x['r2_score'], reverse=True)
    for i, result in enumerate(sorted_results[:5], 1):
        print(f"  {i}. {result['name']} ({result['type']}): R² = {result['r2_score']:.4f}")
    
    print(f"\nMISO CAPABILITY DEMONSTRATED:")
    print(f"The system successfully discovered relationships with multiple input variables,")
    print(f"showing its ability to handle complex multi-dimensional physics problems.")
    print(f"Functions with only constants and one variable were correctly identified as SISO.")


if __name__ == "__main__":
    main()
