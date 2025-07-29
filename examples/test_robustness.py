import numpy as np
import matplotlib.pyplot as plt
from time import time
from symbolic_regression import EnsembleMIMORegressor
from tqdm import tqdm
import sys
import io

def run_test(x_min, x_max, n_train, lambda_func, noise_level):
    # Generate two input features
    X1 = np.linspace(x_min, x_max, n_train)
    X2 = np.linspace(x_min, x_max, n_train)
    X_train = np.column_stack([X1, X2])
    y_true_train = lambda_func(X1, X2)

    # Add noise to training data
    noise = np.random.normal(0, noise_level * np.std(y_true_train), 
                            size=y_true_train.shape)
    y_train = (y_true_train + noise).reshape(-1, 1)

    # Generate test data (denser for evaluation)
    X1_test = np.linspace(x_min, x_max, 200)
    X2_test = np.linspace(x_min, x_max, 200)
    X_test = np.column_stack([X1_test, X2_test])
    y_true_test = lambda_func(X1_test, X2_test)
    
    # Create RL-enhanced ensemble model
    model = EnsembleMIMORegressor(
        n_fits=8,                    
        top_n_select=5,             
        population_size=200,         
        generations=200,              
        mutation_rate=0.15,
        crossover_rate=0.8,
        tournament_size=3,
        max_depth=4,
        parsimony_coefficient=0.005,
        diversity_threshold=0.6,     
        adaptive_rates=True,
        restart_threshold=10,        
        elite_fraction=0.1,
        enable_inter_thread_communication=True,
        purge_percentage=0.1,       
        exchange_interval=10,       
        import_percentage=0.05,
        enable_data_scaling=False,
        use_multi_scale_fitness=False,
        # Early termination parameters
        enable_early_termination=True,
        early_termination_threshold=0.99,
        early_termination_check_interval=8,
        enable_late_extension=True,
        late_extension_threshold=0.85,
        late_extension_generations=20
        # Note: RL features available through enhanced generation modules
    )
    
    # Train the model
    start_time = time()
    model.fit(X_train, y_train, constant_optimize=True)
    training_time = time() - start_time
    
    # Make predictions
    y_pred_train = model.predict(X_train, strategy='mean')
    y_pred_test = model.predict(X_test, strategy='mean')
    
    # Calculate metrics
    from sklearn.metrics import r2_score
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_true_test, y_pred_test)
    
    mse_test = np.mean((y_true_test - y_pred_test.flatten()) ** 2)
    rmse_test = np.sqrt(mse_test)
    
    # Get discovered expressions
    discovered_expressions = model.get_expressions()
    best_expression = discovered_expressions[0]
    
    return best_expression, r2_test
    
def main():
    noises = np.arange(start=0, stop=0.5, step=0.01)
    exprs = []
    r2s = []
    for noise in tqdm(noises):
        original = sys.stdout
        sys.stdout = io.StringIO()
        best_expr, r2 = run_test(-3, 3, 120, lambda x1, x2: x1*x2, noise)
        sys.stdout = original
        exprs.append(best_expr)
        r2s.append(r2)
    
    print()
    for (n, e, r) in zip(noises, exprs, r2s):
        print(f"{n:.2f} -> {r:.4f}: {e}")
    
    plt.figure(1)
    plt.plot(noises, r2s)
    plt.show()
        
if __name__ == '__main__':
    main()
