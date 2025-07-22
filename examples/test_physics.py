import numpy as np
from sklearn.metrics import r2_score
import os
from datetime import datetime
from symbolic_regression.ensemble_regressor import EnsembleMIMORegressor


class PhysicsLawTester:
    """Test symbolic regression on well-known physics laws"""
    
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
    
    def generate_data(self, func, x_range, n_samples=200, noise_level=0.02):
        """Generate training data for a given function"""
        np.random.seed(42)  # For reproducibility
        
        if len(x_range) == 2:  # Single variable
            X = np.linspace(x_range[0], x_range[1], n_samples).reshape(-1, 1)
            y_true = func(X.flatten())
        elif len(x_range) == 4:  # Two variables
            x1 = np.linspace(x_range[0], x_range[1], int(np.sqrt(n_samples)))
            x2 = np.linspace(x_range[2], x_range[3], int(np.sqrt(n_samples)))
            X1, X2 = np.meshgrid(x1, x2)
            X = np.column_stack([X1.flatten(), X2.flatten()])
            y_true = func(X[:, 0], X[:, 1])
        else:
            raise ValueError("Only 1D and 2D functions supported")
        
        # Add noise
        noise = np.random.normal(0, noise_level * np.std(y_true), size=y_true.shape)
        y_noisy = y_true + noise
        
        return X, y_noisy.reshape(-1, 1), y_true
    
    def test_law(self, name, description, func, x_range, target_expr):
        """Test a single physics law"""
        print(f"\nTesting: {name}")
        print(f"Description: {description}")
        print(f"Target: {target_expr}")
        
        # Generate data
        X, y, y_true = self.generate_data(func, x_range)
        
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
        filename = f"physics_laws_results_{timestamp}.txt"
        filepath = os.path.join("..", filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("SYMBOLIC REGRESSION RESULTS FOR PHYSICS LAWS\n")
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
    """Main function to test all physics laws"""
    tester = PhysicsLawTester()
    
    print("Testing Symbolic Regression on Physics Laws")
    print("=" * 50)
    
    # 1. CLASSICAL MECHANICS (10 laws)
    
    # Newton's Second Law: F = ma (simplified as F = a, assuming m=1)
    tester.test_law(
        "Newton's Second Law",
        "Force equals mass times acceleration",
        lambda a: a,  # F = ma, with m=1
        [0.1, 10],
        "F = a"
    )
    
    # Kinematic equation: s = ut + (1/2)at^2
    tester.test_law(
        "Kinematic Equation",
        "Position with constant acceleration",
        lambda t: 2*t + 0.5*9.81*t**2,  # u=2, a=9.81
        [0, 5],
        "s = ut + (1/2)at^2"
    )
    
    # Gravitational Force: F = Gm1m2/r^2 (normalized)
    tester.test_law(
        "Newton's Law of Gravitation",
        "Gravitational force between two masses",
        lambda r: 1.334 / r**2,  # Normalized: 6.67e-11 * 100 * 200 / 1e-11 = 1.334
        [1, 100],
        "F = k/r^2"
    )
    
    # Hooke's Law: F = -kx
    tester.test_law(
        "Hooke's Law",
        "Spring force proportional to displacement",
        lambda x: -50 * x,  # k = 50
        [-2, 2],
        "F = -kx"
    )
    
    # Centripetal Force: F = mv^2/r
    tester.test_law(
        "Centripetal Force",
        "Force required for circular motion",
        lambda v: 10 * v**2 / 5,  # m=10, r=5
        [1, 20],
        "F = mv^2/r"
    )
    
    # Kinetic Energy: KE = (1/2)mv^2
    tester.test_law(
        "Kinetic Energy",
        "Energy of motion",
        lambda v: 0.5 * 5 * v**2,  # m = 5
        [0, 30],
        "KE = (1/2)mv^2"
    )
    
    # Potential Energy: PE = mgh
    tester.test_law(
        "Gravitational Potential Energy",
        "Energy due to height in gravitational field",
        lambda h: 10 * 9.81 * h,  # m=10, g=9.81
        [0, 100],
        "PE = mgh"
    )
    
    # Period of Simple Pendulum: T = 2π√(L/g)
    tester.test_law(
        "Simple Pendulum Period",
        "Period of small oscillations",
        lambda L: 2 * np.pi * np.sqrt(L / 9.81),
        [0.1, 5],
        "T = 2π√(L/g)"
    )
    
    # Elastic Potential Energy: PE = (1/2)kx^2
    tester.test_law(
        "Elastic Potential Energy",
        "Energy stored in compressed/stretched spring",
        lambda x: 0.5 * 100 * x**2,  # k = 100
        [-3, 3],
        "PE = (1/2)kx^2"
    )
    
    # Work-Energy Theorem: W = ΔKE = F*d
    tester.test_law(
        "Work Done",
        "Work equals force times distance",
        lambda d: 50 * d,  # F = 50
        [0, 20],
        "W = F*d"
    )
    
    # 2. THERMODYNAMICS (5 laws)
    
    # Ideal Gas Law: P = nRT/V (simplified)
    tester.test_law(
        "Ideal Gas Law (P vs T)",
        "Pressure proportional to temperature at constant volume",
        lambda T: 8.314 * T / 0.1,  # nR/V = 8.314/0.1
        [200, 500],
        "P = nRT/V"
    )
    
    # Stefan-Boltzmann Law: j = σT^4 (normalized)
    tester.test_law(
        "Stefan-Boltzmann Law",
        "Radiated power proportional to T^4",
        lambda T: (T/1000)**4,  # Normalized to avoid tiny constants
        [300, 1000],
        "j ∝ T^4"
    )
    
    # Heat Conduction: q = -kA(dT/dx)
    tester.test_law(
        "Fourier's Heat Conduction",
        "Heat flux proportional to temperature gradient",
        lambda dT: -100 * dT,  # kA = 100
        [-50, 50],
        "q = -kA(dT/dx)"
    )
    
    # Charles's Law: V/T = constant
    tester.test_law(
        "Charles's Law",
        "Volume proportional to temperature",
        lambda T: 0.02 * T,  # V/T = 0.02
        [200, 600],
        "V = cT"
    )
    
    # Carnot Efficiency: η = 1 - Tc/Th
    tester.test_law(
        "Carnot Efficiency",
        "Maximum efficiency of heat engine",
        lambda Th: 1 - 300/Th,  # Tc = 300
        [400, 1000],
        "η = 1 - Tc/Th"
    )
    
    # 3. ELECTROMAGNETISM (8 laws)
    
    # Coulomb's Law: F = kq1q2/r^2 (normalized)
    tester.test_law(
        "Coulomb's Law",
        "Electrostatic force between charges",
        lambda r: 1.798 / r**2,  # Normalized: 8.99e9 * 1e-6 * 2e-6 = 1.798e-2
        [0.1, 5],
        "F = kq1q2/r^2"
    )
    
    # Ohm's Law: V = IR
    tester.test_law(
        "Ohm's Law",
        "Voltage equals current times resistance",
        lambda I: 100 * I,  # R = 100
        [0.01, 1],
        "V = IR"
    )
    
    # Electric Power: P = V^2/R
    tester.test_law(
        "Electric Power (V^2/R)",
        "Power dissipated in resistor",
        lambda V: V**2 / 50,  # R = 50
        [1, 100],
        "P = V^2/R"
    )
    
    # Capacitor Energy: U = (1/2)CV^2
    tester.test_law(
        "Capacitor Energy",
        "Energy stored in capacitor",
        lambda V: 0.5 * 1e-6 * V**2,  # C = 1uF
        [0, 1000],
        "U = (1/2)CV^2"
    )
    
    # Magnetic Force: F = qvB (normalized)
    tester.test_law(
        "Magnetic Force on Moving Charge",
        "Force on charge in magnetic field",
        lambda v: 0.5 * v / 1000,  # Normalized to reasonable scale
        [1000, 1e6],
        "F ∝ v"
    )
    
    # Faraday's Law: ε = -dΦ/dt
    tester.test_law(
        "Faraday's Law (EMF)",
        "Induced EMF proportional to flux change",
        lambda dPhi_dt: -dPhi_dt,
        [-10, 10],
        "ε = -dΦ/dt"
    )
    
    # LC Circuit Frequency: f = 1/(2π√(LC))
    tester.test_law(
        "LC Circuit Resonance",
        "Natural frequency of LC circuit",
        lambda L: 1 / (2 * np.pi * np.sqrt(L * 1e-6)),  # C = 1uF
        [1e-3, 1],
        "f = 1/(2π√(LC))"
    )
    
    # Electromagnetic Wave: c = λf (rearranged as λ = c/f)
    tester.test_law(
        "Electromagnetic Wave Relation",
        "Wavelength inversely proportional to frequency",
        lambda f: 3e8 / f,
        [1e6, 1e12],
        "λ = c/f"
    )
    
    # 4. QUANTUM PHYSICS (8 laws - maximum allowed)
    
    # Planck's Energy: E = hf (normalized)
    tester.test_law(
        "Planck's Energy Equation",
        "Energy of photon proportional to frequency",
        lambda f: 6.626e-20 * f,  # Scaled Planck constant for better fitting
        [1e14, 1e16],
        "E = hf"
    )
    
    # de Broglie Wavelength: λ = h/p (normalized)
    tester.test_law(
        "de Broglie Wavelength",
        "Matter wave wavelength",
        lambda p: 6.626e-10 / p,  # Scaled for better fitting
        [1e-24, 1e-20],
        "λ = h/p"
    )
    
    # Photoelectric Effect: KE = hf - φ (normalized)
    tester.test_law(
        "Photoelectric Effect",
        "Kinetic energy of photoelectrons",
        lambda f: 6.626e-20 * f - 3e-5,  # Scaled constants for better fitting
        [1e15, 5e15],
        "KE = hf - φ"
    )
    
    # Uncertainty Principle: Δx·Δp ≥ ℏ/2 (simplified as Δp = ℏ/(2Δx)) (normalized)
    tester.test_law(
        "Heisenberg Uncertainty Principle",
        "Position-momentum uncertainty relation",
        lambda dx: 5.275e-21 / dx,  # Scaled ℏ/2 for better fitting
        [1e-12, 1e-9],
        "Δp = ℏ/(2Δx)"
    )
    
    # Hydrogen Energy Levels: E = -13.6/n^2
    tester.test_law(
        "Hydrogen Energy Levels",
        "Energy of electron in hydrogen atom",
        lambda n: -13.6 / n**2,
        [1, 10],
        "E = -13.6/n^2"
    )
    
    # Compton Scattering: Δλ = h/(mₑc)(1-cosθ)
    tester.test_law(
        "Compton Scattering (θ dependence)",
        "Wavelength shift in photon scattering",
        lambda theta: 2.426e-12 * (1 - np.cos(theta)),
        [0, np.pi],
        "Δλ = λc(1-cosθ)"
    )
    
    # Blackbody Radiation (Wien's Law): λmax = b/T
    tester.test_law(
        "Wien's Displacement Law",
        "Peak wavelength inversely proportional to temperature",
        lambda T: 2.898e-3 / T,
        [1000, 6000],
        "λmax = b/T"
    )
    
    # Rydberg Formula: 1/λ = R(1/n1^2 - 1/n2^2)
    tester.test_law(
        "Rydberg Formula (simplified)",
        "Hydrogen spectral lines",
        lambda n: 1.097e7 * (1/4 - 1/n**2),  # n1=2, n2=n
        [3, 10],
        "1/λ = R(1/n1^2 - 1/n2^2)"
    )
    
    # 5. ASTROPHYSICS AND COSMOLOGY (7 laws)
    
    # Kepler's Third Law: T^2 ∝ a^3
    tester.test_law(
        "Kepler's Third Law",
        "Orbital period squared proportional to semi-major axis cubed",
        lambda a: np.sqrt(4 * np.pi**2 * a**3 / (6.67e-11 * 2e30)),  # GM_sun
        [1e11, 1e13],
        "T = 2π√(a^3/GM)"
    )
    
    # Schwarzschild Radius: rs = 2GM/c^2
    tester.test_law(
        "Schwarzschild Radius",
        "Black hole event horizon radius",
        lambda M: 2 * 6.67e-11 * M / (3e8)**2,
        [1e30, 1e32],
        "rs = 2GM/c^2"
    )
    
    # Hubble's Law: v = H₀d
    tester.test_law(
        "Hubble's Law",
        "Recession velocity proportional to distance",
        lambda d: 70 * d,  # H₀ = 70 km/s/Mpc
        [1, 1000],
        "v = H0d"
    )
    
    # Luminosity Distance: dL = √(L/(4πF))
    tester.test_law(
        "Luminosity Distance",
        "Distance from luminosity and flux",
        lambda F: np.sqrt(3.8e26 / (4 * np.pi * F)),  # L_sun
        [1e-12, 1e-8],
        "dL = √(L/(4πF))"
    )
    
    # Escape Velocity: v = √(2GM/r)
    tester.test_law(
        "Escape Velocity",
        "Minimum velocity to escape gravitational field",
        lambda r: np.sqrt(2 * 6.67e-11 * 6e24 / r),  # M_earth
        [6.4e6, 1e8],
        "v = √(2GM/r)"
    )
    
    # Tidal Force: F ∝ GMm/r³
    tester.test_law(
        "Tidal Force",
        "Differential gravitational force",
        lambda r: 6.67e-11 * 7.3e22 * 1000 / r**3,  # M_moon, m=1000kg
        [3.8e8, 1e9],
        "F ∝ GMm/r^3"
    )
    
    # Solar Luminosity (Stefan-Boltzmann): L = 4πR^2σT^4
    tester.test_law(
        "Stellar Luminosity",
        "Total power radiated by star",
        lambda T: 4 * np.pi * (7e8)**2 * 5.67e-8 * T**4,  # R_sun
        [3000, 10000],
        "L = 4πR^2σT^4"
    )
    
    # Save all results
    tester.save_results()
    
    print(f"\nCompleted testing {len(tester.results)} physics laws!")
    print("Results summary:")
    for result in tester.results:
        print(f"  {result['name']}: R^2 = {result['r2_score']:.4f}")


if __name__ == "__main__":
    main()
