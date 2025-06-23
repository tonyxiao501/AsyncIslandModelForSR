import numpy as np
import matplotlib.pyplot as plt
from symbolic_regression.mimo_regressor import MIMOSymbolicRegressor

def generate_mimo_data(n_samples: int = 1000) -> tuple:
    """Generate synthetic MIMO data for testing"""
    
    # Create input data
    X = np.random.uniform(-2, 2, (n_samples, 3))
    
    # Define true functions for each output
    y1 = X[:, 0] ** 2 + np.sin(X[:, 1]) + 0.1 * np.random.normal(0, 1, n_samples)
    y2 = X[:, 0] * X[:, 1] + np.cos(X[:, 2]) + 0.1 * np.random.normal(0, 1, n_samples)
    y3 = np.exp(X[:, 0] * 0.5) + X[:, 1] * X[:, 2] + 0.1 * np.random.normal(0, 1, n_samples)
    
    y = np.column_stack([y1, y2, y3])
    
    return X, y

def main():
    # Generate synthetic MIMO data
    print("Generating synthetic MIMO data...")
    X_train, y_train = generate_mimo_data(800)
    X_test, y_test = generate_mimo_data(200)
    
    print(f"Training data shape: X={X_train.shape}, y={y_train.shape}")
    print(f"Test data shape: X={X_test.shape}, y={y_test.shape}")
    
    # Create and train the model
    print("\nTraining MIMO Symbolic Regression model...")
    model = MIMOSymbolicRegressor(
        population_size=100,
        generations=50,
        mutation_rate=0.1,
        crossover_rate=0.8,
        max_depth=5,
        parsimony_coefficient=0.001
    )
    
    model.fit(X_train, y_train)
    
    # Make predictions
    print("\nMaking predictions...")
    y_pred = model.predict(X_test)
    
    # Evaluate performance
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    print(f"\nModel Performance:")
    print(f"Training R² score: {train_score:.4f}")
    print(f"Test R² score: {test_score:.4f}")
    
    # Display discovered expressions
    print(f"\nDiscovered expressions:")
    expressions = model.get_expressions()
    for i, expr in enumerate(expressions):
        print(f"Output {i+1}: {expr}")
    
    # Plot results
    plot_results(y_test, y_pred, model.fitness_history)

def plot_results(y_true, y_pred, fitness_history):
    """Plot the results"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot predictions vs true values for each output
    n_outputs = y_true.shape[1]
    for i in range(min(3, n_outputs)):
        row = i // 2
        col = i % 2
        
        axes[row, col].scatter(y_true[:, i], y_pred[:, i], alpha=0.6)
        axes[row, col].plot([y_true[:, i].min(), y_true[:, i].max()], 
                           [y_true[:, i].min(), y_true[:, i].max()], 'r--', lw=2)
        axes[row, col].set_xlabel(f'True Output {i+1}')
        axes[row, col].set_ylabel(f'Predicted Output {i+1}')
        axes[row, col].set_title(f'Output {i+1}: Predicted vs True')
        axes[row, col].grid(True, alpha=0.3)
    
    # Plot fitness evolution
    if n_outputs <= 2:
        axes[1, 1].plot(fitness_history)
        axes[1, 1].set_xlabel('Generation')
        axes[1, 1].set_ylabel('Average Fitness')
        axes[1, 1].set_title('Fitness Evolution')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()