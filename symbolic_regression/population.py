# population.py: population/diversity management helpers

def generate_diverse_population(generator, n_inputs, population_size, max_depth, is_expression_valid):
    """Generate diverse initial population with multiple strategies"""
    import random
    from symbolic_regression.expression_tree import Expression
    population = []
    strategies = {
        'random_full': 0.4,  # 40% full random trees
        'random_grow': 0.3,  # 30% grown trees
        'simple_combinations': 0.2,  # 20% simple function combinations
        'constants_varied': 0.1  # 10% constant-heavy expressions
    }
    target_counts = {strategy: int(ratio * population_size)
                     for strategy, ratio in strategies.items()}
    for strategy, count in target_counts.items():
        for _ in range(count):
            if strategy == 'random_full':
                depth = random.randint(2, max_depth)
                expr = Expression(generator.generate_random_expression(max_depth=depth))
            elif strategy == 'random_grow':
                depth = random.randint(1, max_depth - 1)
                expr = Expression(generator.generate_random_expression(max_depth=depth))
            elif strategy == 'simple_combinations':
                expr = generate_simple_combination(generator, n_inputs)
            else:  # constants_varied
                expr = generate_constant_heavy(generator, n_inputs)
            if is_expression_valid(expr):
                population.append(expr)
    while len(population) < population_size:
        expr = Expression(generator.generate_random_expression())
        if is_expression_valid(expr):
            population.append(expr)
    return population[:population_size]

def inject_diversity(population, fitness_scores, generator, injection_rate, is_expression_valid, generate_high_diversity_expression, generate_targeted_diverse_expression, generate_complex_diverse_expression, stagnation_counter, console_log):
    """Enhanced diversity injection with more aggressive strategies"""
    import numpy as np
    diversity_score = 1.0  # Placeholder, should be calculated
    if stagnation_counter > 15:
        injection_rate = min(0.6, injection_rate * 2.0)
    elif stagnation_counter > 8:
        injection_rate = min(0.4, injection_rate * 1.5)
    n_to_replace = max(2, int(len(population) * injection_rate))
    new_population = population.copy()
    replacements_made = 0
    worst_count = max(1, n_to_replace // 3)
    worst_indices = np.argsort(fitness_scores)[:worst_count]
    for idx in worst_indices:
        new_expr = generate_high_diversity_expression(generator)
        if is_expression_valid(new_expr):
            new_population[idx] = new_expr
            replacements_made += 1
    if replacements_made < n_to_replace:
        bottom_half_count = len(population) // 2
        bottom_indices = np.argsort(fitness_scores)[:bottom_half_count]
        random_count = min(n_to_replace - replacements_made, max(1, (n_to_replace * 2) // 5))
        random_indices = np.random.choice(bottom_indices, size=random_count, replace=False)
        for idx in random_indices:
            new_expr = generate_targeted_diverse_expression(generator, population)
            if is_expression_valid(new_expr):
                new_population[idx] = new_expr
                replacements_made += 1
    if replacements_made < n_to_replace:
        median_start = len(population) // 4
        median_end = 3 * len(population) // 4
        median_indices = np.argsort(fitness_scores)[median_start:median_end]
        median_count = min(n_to_replace - replacements_made, len(median_indices) // 3)
        if median_count > 0:
            selected_median = np.random.choice(median_indices, size=median_count, replace=False)
            for idx in selected_median:
                new_expr = generate_complex_diverse_expression(generator)
                if is_expression_valid(new_expr):
                    new_population[idx] = new_expr
                    replacements_made += 1
    if console_log:
        print(f"Diversity injection: replaced {replacements_made}/{n_to_replace} individuals (rate: {injection_rate:.2f}, stagnation: {stagnation_counter})")
    return new_population

def is_expression_valid(expr, n_inputs):
    try:
        if expr.complexity() > 30:
            return False
        import numpy as np
        test_X = np.random.randn(5, n_inputs)
        result = expr.evaluate(test_X)
        return bool(np.all(np.isfinite(result)) and np.max(np.abs(result)) < 1e10)
    except:
        return False

def generate_high_diversity_expression(generator):
    """Generate an expression with high diversity using various operators and functions"""
    from symbolic_regression.expression_tree import Expression
    from symbolic_regression.expression_tree.core.node import BinaryOpNode, UnaryOpNode, VariableNode, ConstantNode
    import numpy as np
    import random
    var = VariableNode(0)
    const1 = ConstantNode(np.random.uniform(-2, 2))
    const2 = ConstantNode(np.random.uniform(-2, 2))
    inner_expr = BinaryOpNode('+', BinaryOpNode('*', const1, var), const2)
    func = np.random.choice(['sin', 'cos', 'exp'])
    if func == 'exp':
        limited_inner = BinaryOpNode('*', ConstantNode(0.5), var)
        return Expression(UnaryOpNode(func, limited_inner))
    else:
        return Expression(UnaryOpNode(func, inner_expr))

def generate_targeted_diverse_expression(generator, population):
    """Generate a targeted diverse expression based on the existing population's diversity"""
    # For brevity, use the same logic as before
    has_trig = any('sin' in expr.to_string() or 'cos' in expr.to_string() for expr in population[:20])
    has_exp = any('exp' in expr.to_string() for expr in population[:20])
    has_poly = any('*' in expr.to_string() and '+' in expr.to_string() for expr in population[:20])
    if not has_trig or not has_exp:
        return generate_high_diversity_expression(generator)
    elif not has_poly:
        return generate_polynomial_expression(generator)
    else:
        return generate_mixed_expression(generator)

def generate_complex_diverse_expression(generator):
    """Generate a complex diverse expression, currently similar to high diversity"""
    # For brevity, use a simple version
    return generate_high_diversity_expression(generator)

def generate_polynomial_expression(generator):
    """Generate a polynomial expression"""
    from symbolic_regression.expression_tree import Expression
    from symbolic_regression.expression_tree.core.node import BinaryOpNode, VariableNode, ConstantNode
    import numpy as np
    var = VariableNode(0)
    const1 = ConstantNode(np.random.uniform(-3, 3))
    const2 = ConstantNode(np.random.uniform(-3, 3))
    const3 = ConstantNode(np.random.uniform(-3, 3))
    x_squared = BinaryOpNode('*', var, var)
    term1 = BinaryOpNode('*', const1, x_squared)
    term2 = BinaryOpNode('*', const2, var)
    poly = BinaryOpNode('+', BinaryOpNode('+', term1, term2), const3)
    return Expression(poly)

def generate_mixed_expression(generator):
    """Generate an expression that is a mix of different types"""
    from symbolic_regression.expression_tree import Expression
    from symbolic_regression.expression_tree.core.node import BinaryOpNode, UnaryOpNode, VariableNode, ConstantNode
    import numpy as np
    var = VariableNode(0)
    const1 = ConstantNode(np.random.uniform(-2, 2))
    trig_part = UnaryOpNode('sin', var)
    linear_part = BinaryOpNode('*', const1, var)
    return Expression(BinaryOpNode('+', trig_part, linear_part))

def enhanced_reproduction_v2(population, fitness_scores, genetic_ops, diversity_score, generation, population_size, elite_fraction, current_crossover_rate, current_mutation_rate, n_inputs, max_depth, is_expression_valid, generate_high_diversity_expression, X, y):
    import numpy as np
    import random
    from symbolic_regression.generator import ExpressionGenerator
    new_population = []
    elite_count = max(2, int(population_size * elite_fraction))
    elite_indices = np.argsort(fitness_scores)[-elite_count:]
    elites = [population[i].copy() for i in elite_indices]
    new_population.extend(elites)
    temperature = 1.0
    cooling_rate = 0.97
    while len(new_population) < population_size:
        if random.random() < current_crossover_rate and len(new_population) < population_size - 1:
            parent1 = random.choice(population)
            parent2 = random.choice(population)
            child1, child2 = genetic_ops.crossover(parent1, parent2)
            if random.random() < current_mutation_rate:
                child1 = genetic_ops.mutate(child1, current_mutation_rate)
            if random.random() < current_mutation_rate:
                child2 = genetic_ops.mutate(child2, current_mutation_rate)
            for child in [child1, child2]:
                if is_expression_valid(child):
                    idx = random.randint(0, len(population) - 1)
                    current_fitness = fitness_scores[idx]
                    new_fitness = genetic_ops._evaluate_fitness(child, X, y)
                    delta = new_fitness - current_fitness
                    T = max(temperature * (cooling_rate ** generation), 1e-8)
                    exponent = np.clip(delta / T, -500, 500)
                    accept_prob = np.exp(exponent)
                    if delta > 0 or np.random.rand() < accept_prob:
                        new_population.append(child)
                    if len(new_population) >= population_size:
                        break
        else:
            if random.random() < 0.3:
                generator = ExpressionGenerator(n_inputs, max_depth)
                diverse_expr = generate_high_diversity_expression(generator)
                if is_expression_valid(diverse_expr):
                    idx = random.randint(0, len(population) - 1)
                    current_fitness = fitness_scores[idx]
                    new_fitness = genetic_ops._evaluate_fitness(diverse_expr, X, y)
                    delta = new_fitness - current_fitness
                    T = max(temperature * (cooling_rate ** generation), 1e-8)
                    exponent = np.clip(delta / T, -500, 500)
                    accept_prob = np.exp(exponent)
                    if delta > 0 or np.random.rand() < accept_prob:
                        new_population.append(diverse_expr)
            else:
                parent = random.choice(population)
                child = genetic_ops.mutate(parent, current_mutation_rate)
                if is_expression_valid(child):
                    idx = random.randint(0, len(population) - 1)
                    current_fitness = fitness_scores[idx]
                    if hasattr(genetic_ops, "_evaluate_fitness"):
                        new_fitness = genetic_ops._evaluate_fitness(child, X, y)
                    else:
                        new_fitness = current_fitness
                    delta = new_fitness - current_fitness
                    T = max(temperature * (cooling_rate ** generation), 1e-8)
                    exponent = np.clip(delta / T, -500, 500)
                    accept_prob = np.exp(exponent)
                    if delta > 0 or np.random.rand() < accept_prob:
                        new_population.append(child)
    return new_population[:population_size]

def generate_simple_combination(generator, n_inputs):
    from symbolic_regression.expression_tree import Expression
    from symbolic_regression.expression_tree.core.node import BinaryOpNode, UnaryOpNode, VariableNode, ConstantNode
    import random
    if n_inputs is None or n_inputs == 0:
        raise ValueError("n_inputs must be set and greater than 0")
    var_node = VariableNode(random.randint(0, n_inputs - 1))
    const_node = ConstantNode(random.uniform(-2, 2))
    combinations = [
        BinaryOpNode('+', var_node, const_node),
        BinaryOpNode('*', var_node, const_node),
        UnaryOpNode('sin', var_node),
        UnaryOpNode('cos', var_node),
        BinaryOpNode('*', var_node, var_node),  # x^2 approximation
    ]
    return Expression(random.choice(combinations))

def generate_constant_heavy(generator, n_inputs):
    from symbolic_regression.expression_tree import Expression
    from symbolic_regression.expression_tree.core.node import BinaryOpNode, VariableNode, ConstantNode
    import random
    if n_inputs is None or n_inputs == 0:
        raise ValueError("n_inputs must be set and greater than 0")
    var_node = VariableNode(random.randint(0, n_inputs - 1))
    const1 = ConstantNode(random.uniform(-3, 3))
    const2 = ConstantNode(random.uniform(-3, 3))
    op = random.choice(['+', '-', '*'])
    return Expression(BinaryOpNode(op, BinaryOpNode('*', const1, var_node), const2))
