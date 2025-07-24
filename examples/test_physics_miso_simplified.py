import numpy as np
from sklearn.metrics import r2_score
import os
from datetime import datetime
from symbolic_regression.ensemble_regressor import EnsembleMIMORegressor


class SimplifiedPhysicsLawTester:
    """Test symbolic regression on well-known physics laws using MISO approach"""
    
    def __init__(self):
        # Configure the ensemble regressor with enhanced physics-focused parameters
        self.model = EnsembleMIMORegressor(
            n_fits=6,                    # Reduced for faster testing
            top_n_select=3,             
            population_size=150,         # Reduced for faster testing
            generations=200,             # Sufficient for most physics laws
            mutation_rate=0.3,           # Higher for more exploration
            crossover_rate=0.8,
            tournament_size=3,
            max_depth=10,                # Increased for complex expressions
            parsimony_coefficient=0.001, # Further reduced to allow more complex expressions
            diversity_threshold=0.7,     # Increased for better diversity
            adaptive_rates=True,
            restart_threshold=25,        # Increased patience
            elite_fraction=0.15,         # Higher elite fraction
            enable_inter_thread_communication=True,
            purge_percentage=0.15,       
            exchange_interval=15,       
            import_percentage=0.08,
            # MINIMAL scaling for physics - preserve mathematical structure
            enable_data_scaling=False,   # DISABLE scaling to preserve physics relationships
            use_multi_scale_fitness=True,
            # When scaling is disabled, these are ignored but kept for compatibility
            input_scaling='none',        
            output_scaling='none',       
            scaling_target_range=(-5.0, 5.0),
            extreme_value_threshold=1e12
        )
        
        # Initialize results storage
        self.results = []
    
    def generate_data(self, func, x_ranges, n_samples=200, noise_level=0.02):
        """Generate training data for a given multi-variable function"""
        np.random.seed(42)  # For reproducibility
        
        n_vars = len(x_ranges) // 2  # Each variable has min and max
        
        if n_vars == 1:  # Single variable
            X = np.linspace(x_ranges[0], x_ranges[1], n_samples).reshape(-1, 1)
            y_true = func(X[:, 0])
        elif n_vars == 2:  # Two variables
            samples_per_dim = int(np.sqrt(n_samples))
            x1 = np.linspace(x_ranges[0], x_ranges[1], samples_per_dim)
            x2 = np.linspace(x_ranges[2], x_ranges[3], samples_per_dim)
            X1, X2 = np.meshgrid(x1, x2)
            X = np.column_stack([X1.flatten(), X2.flatten()])
            y_true = func(X[:, 0], X[:, 1])
        elif n_vars == 3:  # Three variables
            samples_per_dim = int(np.cbrt(n_samples))
            x1 = np.linspace(x_ranges[0], x_ranges[1], samples_per_dim)
            x2 = np.linspace(x_ranges[2], x_ranges[3], samples_per_dim)
            x3 = np.linspace(x_ranges[4], x_ranges[5], samples_per_dim)
            X1, X2, X3 = np.meshgrid(x1, x2, x3, indexing='ij')
            X = np.column_stack([X1.flatten(), X2.flatten(), X3.flatten()])
            y_true = func(X[:, 0], X[:, 1], X[:, 2])
        elif n_vars == 4:  # Four variables
            samples_per_dim = int(n_samples**(1/4))
            x1 = np.linspace(x_ranges[0], x_ranges[1], samples_per_dim)
            x2 = np.linspace(x_ranges[2], x_ranges[3], samples_per_dim)
            x3 = np.linspace(x_ranges[4], x_ranges[5], samples_per_dim)
            x4 = np.linspace(x_ranges[6], x_ranges[7], samples_per_dim)
            X1, X2, X3, X4 = np.meshgrid(x1, x2, x3, x4, indexing='ij')
            X = np.column_stack([X1.flatten(), X2.flatten(), X3.flatten(), X4.flatten()])
            y_true = func(X[:, 0], X[:, 1], X[:, 2], X[:, 3])
        else:
            # For more than 4 variables, use random sampling
            X = np.random.uniform(
                low=[x_ranges[i*2] for i in range(n_vars)],
                high=[x_ranges[i*2+1] for i in range(n_vars)],
                size=(n_samples, n_vars)
            )
            y_true = func(*X.T)
        
        # Add noise
        noise = np.random.normal(0, noise_level * np.std(y_true), size=y_true.shape)
        y_noisy = y_true + noise
        
        return X, y_noisy.reshape(-1, 1), y_true
    
    def test_law(self, name, description, func, x_ranges, target_expr):
        """Test a single physics law"""
        print(f"\nTesting: {name}")
        print(f"Description: {description}")
        print(f"Target: {target_expr}")
        
        # Generate data
        X, y, y_true = self.generate_data(func, x_ranges)
        
        # Fit model
        self.model.fit(X, y, constant_optimize=True)
        
        # Get predictions and expressions
        y_pred = self.model.predict(X, strategy='mean')
        discovered_expressions = self.model.get_expressions()
        
        # Calculate R^2 score
        r2_score_val = r2_score(y_true, y_pred)
        
        # Store results
        result = {
            'name': name,
            'target': target_expr,
            'r2_score': r2_score_val,
            'expressions': discovered_expressions[:5]  # Top 5
        }
        self.results.append(result)
        
        print(f"R^2 Score: {r2_score_val:.4f}")
        print(f"Best expression: {discovered_expressions[0] if discovered_expressions else 'None'}")
    
    def save_results(self):
        """Save all results to a text file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"physics_laws_miso_results_{timestamp}.txt"
        filepath = os.path.join("..", filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("SYMBOLIC REGRESSION RESULTS FOR PHYSICS LAWS (MISO)\n")
            f.write("=" * 60 + "\n\n")
            
            for i, result in enumerate(self.results, 1):
                f.write(f"{i}. {result['name']}\n")
                f.write(f"Target function: {result['target']}\n")
                f.write(f"R^2 Score: {result['r2_score']:.6f}\n")
                f.write("Discovered expressions:\n")
                
                for j, expr in enumerate(result['expressions'], 1):
                    f.write(f"  Candidate {j}: {expr}\n")
                
                f.write("\n" + "-" * 50 + "\n\n")
        
        print(f"\nResults saved to: {filepath}")


def main():
    """Main function to test selected physics laws using MISO approach"""
    tester = SimplifiedPhysicsLawTester()
    
    print("Testing Symbolic Regression on Physics Laws (MISO Approach)")
    print("=" * 60)
    print("This version tests physics laws with their actual multiple variables")
    print("instead of simplified single-variable versions.")
    
    # 1. CLASSICAL MECHANICS
    
    # Newton's Second Law: F = ma
    tester.test_law(
        "Newton's Second Law",
        "Force equals mass times acceleration",
        lambda m, a: m * a,
        [0.1, 10, 0.1, 20],  # mass range, acceleration range
        "F = ma"
    )
    
    # Kinematic equation: s = ut + (1/2)at^2
    tester.test_law(
        "Kinematic Equation",
        "Position with constant acceleration",
        lambda u, t, a: u*t + 0.5*a*t**2,
        [0, 10, 0.1, 5, 1, 20],  # initial velocity, time, acceleration
        "s = ut + (1/2)at^2"
    )
    
    # Gravitational Force: F = Gm1m2/r^2 (normalized for better fitting)
    tester.test_law(
        "Newton's Law of Gravitation (normalized)",
        "Gravitational force between two masses",
        lambda m1, m2, r: (m1 * m2) / r**2,  # G factored out for simplicity
        [1, 100, 1, 100, 1, 10],  # m1, m2, r ranges (normalized)
        "F ∝ m1m2/r^2"
    )
    
    # Hooke's Law: F = -kx
    tester.test_law(
        "Hooke's Law",
        "Spring force proportional to displacement",
        lambda k, x: -k * x,
        [10, 100, -2, 2],  # spring constant, displacement
        "F = -kx"
    )
    
    # Centripetal Force: F = mv^2/r
    tester.test_law(
        "Centripetal Force",
        "Force required for circular motion",
        lambda m, v, r: m * v**2 / r,
        [1, 20, 1, 30, 0.5, 10],  # mass, velocity, radius
        "F = mv^2/r"
    )
    
    # Kinetic Energy: KE = (1/2)mv^2
    tester.test_law(
        "Kinetic Energy",
        "Energy of motion",
        lambda m, v: 0.5 * m * v**2,
        [1, 20, 0, 30],  # mass, velocity
        "KE = (1/2)mv^2"
    )
    
    # Potential Energy: PE = mgh
    tester.test_law(
        "Gravitational Potential Energy",
        "Energy due to height in gravitational field",
        lambda m, g, h: m * g * h,
        [1, 50, 9.8, 9.82, 0, 100],  # mass, gravity, height
        "PE = mgh"
    )
    
    # Period of Simple Pendulum: T = 2π√(L/g)
    tester.test_law(
        "Simple Pendulum Period",
        "Period of small oscillations",
        lambda L, g: 2 * np.pi * np.sqrt(L / g),
        [0.1, 5, 9.7, 9.9],  # length, gravity
        "T = 2π√(L/g)"
    )
    
    # Elastic Potential Energy: PE = (1/2)kx^2
    tester.test_law(
        "Elastic Potential Energy",
        "Energy stored in compressed/stretched spring",
        lambda k, x: 0.5 * k * x**2,
        [10, 200, -3, 3],  # spring constant, displacement
        "PE = (1/2)kx^2"
    )
    
    # Work-Energy Theorem: W = F*d*cos(θ)
    tester.test_law(
        "Work Done",
        "Work equals force times distance times cosine of angle",
        lambda F, d, theta: F * d * np.cos(theta),
        [10, 100, 0, 20, 0, np.pi/3],  # force, distance, angle
        "W = F*d*cos(θ)"
    )
    
    # 2. ELECTROMAGNETISM
    
    # Ohm's Law: V = IR
    tester.test_law(
        "Ohm's Law",
        "Voltage equals current times resistance",
        lambda I, R: I * R,
        [0.01, 1, 10, 1000],  # current, resistance
        "V = IR"
    )
    
    # Electric Power: P = VI
    tester.test_law(
        "Electric Power",
        "Power dissipated in resistor",
        lambda V, I: V * I,
        [1, 100, 0.001, 10],  # voltage, current
        "P = VI"
    )
    
    # Capacitor Energy: U = (1/2)CV^2
    tester.test_law(
        "Capacitor Energy",
        "Energy stored in capacitor",
        lambda C, V: 0.5 * C * V**2,
        [1e-6, 1e-3, 1, 1000],  # capacitance, voltage (scaled for numerical stability)
        "U = (1/2)CV^2"
    )
    
    # Magnetic Force on Current: F = BIL sin(θ)
    tester.test_law(
        "Magnetic Force on Current",
        "Force on current-carrying conductor in magnetic field",
        lambda B, I, L, theta: B * I * L * np.sin(theta),
        [0.1, 2, 0.1, 10, 0.01, 1, 0, np.pi],  # B, I, L, theta
        "F = BIL sin(θ)"
    )
    
    # RC Time Constant: τ = RC
    tester.test_law(
        "RC Time Constant",
        "Characteristic time for RC circuit",
        lambda R, C: R * C,
        [100, 1e4, 1e-6, 1e-3],  # resistance, capacitance (scaled)
        "τ = RC"
    )
    
    # 3. THERMODYNAMICS
    
    # Ideal Gas Law: P = nRT/V (simplified)
    tester.test_law(
        "Ideal Gas Law (simplified)",
        "Pressure proportional to nT/V",
        lambda n, T, V: n * T / V,  # R = 1 for simplicity
        [1, 10, 200, 500, 0.01, 1],  # n, T, V ranges
        "P ∝ nT/V"
    )
    
    # Heat Conduction: q = -kA(dT/dx)
    tester.test_law(
        "Fourier's Heat Conduction",
        "Heat flux through material",
        lambda k, A, dT_dx: -k * A * dT_dx,
        [1, 500, 0.001, 1, -100, 100],  # thermal conductivity, area, temp gradient
        "q = -kA(dT/dx)"
    )
    
    # Heat Capacity: Q = mcΔT
    tester.test_law(
        "Heat Capacity",
        "Heat required to change temperature",
        lambda m, c, dT: m * c * dT,
        [0.1, 10, 100, 5000, 1, 100],  # mass, specific heat, temp change
        "Q = mcΔT"
    )
    
    # 4. BASIC QUANTUM PHYSICS (with reasonable scaling)
    
    # Energy-frequency relation: E ∝ f (Planck's law, h factored out)
    tester.test_law(
        "Energy-Frequency Relation",
        "Energy proportional to frequency (Planck-like)",
        lambda f: f,  # h = 1 for simplicity
        [1e14, 1e16],  # frequency range
        "E ∝ f"
    )
    
    # Hydrogen Energy Levels: E = -13.6Z^2/n^2
    tester.test_law(
        "Hydrogen-like Energy Levels",
        "Energy of electron in hydrogen-like atom",
        lambda Z, n: -13.6 * Z**2 / n**2,
        [1, 5, 1, 10],  # atomic number, principal quantum number
        "E = -13.6Z^2/n^2"
    )
    
    # 5. ASTROPHYSICS
    
    # Escape Velocity: v = √(2GM/r) (simplified)
    tester.test_law(
        "Escape Velocity (simplified)",
        "Escape velocity from gravitational field",
        lambda M, r: np.sqrt(M / r),  # 2G = 1 for simplicity
        [1e24, 1e25, 1e6, 1e8],  # mass, radius (normalized)
        "v ∝ √(M/r)"
    )
    
    # Hubble's Law: v = H₀d
    tester.test_law(
        "Hubble's Law",
        "Recession velocity proportional to distance",
        lambda H0, d: H0 * d,
        [50, 100, 1, 1000],  # Hubble constant, distance
        "v = H0d"
    )
    
    # Save all results
    tester.save_results()
    
    print(f"\nCompleted testing {len(tester.results)} physics laws using MISO approach!")
    print("Results summary:")
    for result in tester.results:
        print(f"  {result['name']}: R^2 = {result['r2_score']:.4f}")
    
    print("\nKey improvements with MISO approach:")
    print("1. Laws tested with their actual multiple variables")
    print("2. Better reflects how scientific laws are discovered")
    print("3. Tests the model's ability to handle multi-dimensional relationships")
    print("4. More realistic representation of physical phenomena")


if __name__ == "__main__":
    main()
