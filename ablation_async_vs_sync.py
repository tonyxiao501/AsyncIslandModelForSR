#!/usr/bin/env python3
"""
Ablation: Async vs Sync vs No Migration

Runs quick comparisons on small benchmark problems with lightweight settings.
"""
import time
import argparse
import numpy as np
from benchmark_suite import StandardBenchmarkSuite
from symbolic_regression.ensemble_regressor import EnsembleMIMORegressor
from symbolic_regression.data_processing import r2_score


def run_problem(problem_key: str, mode: str, seed: int = 42) -> dict:
    np.random.seed(seed)
    suite = StandardBenchmarkSuite(adaptive_config=False)
    problem = suite.problems[problem_key]
    # Small dataset
    X, y, y_true = suite.generate_data(problem)

    # Lightweight config for speed
    base_cfg = dict(
        n_fits=3,
        top_n_select=2,
        population_size=200,
        generations=200,
        mutation_rate=0.15,
        crossover_rate=0.75,
        tournament_size=4,
        max_depth=8,
        parsimony_coefficient=0.005,
        diversity_threshold=0.7,
        adaptive_rates=True,
        restart_threshold=12,
        elite_fraction=0.12,
        migration_interval=8,
        migration_probability=0.35,
        console_log=False,
    )

    if mode == "async":
        base_cfg.update(use_asynchronous_migration=True)
    elif mode == "sync":
        base_cfg.update(use_asynchronous_migration=False, migration_probability=0.35)
    elif mode == "no_migration":
        base_cfg.update(use_asynchronous_migration=False, migration_probability=0.0)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    model = EnsembleMIMORegressor(**base_cfg)
    start = time.time()
    model.fit(X, y, constant_optimize=False)
    elapsed = time.time() - start

    preds = model.predict(X).flatten()
    r2 = r2_score(y_true, preds)
    return dict(mode=mode, r2=r2, time=elapsed, expressions=model.get_expressions()[:3])


def main():
    parser = argparse.ArgumentParser(description="Async vs Sync ablation")
    parser.add_argument("--problem", default="nguyen_1", help="Problem key (e.g., nguyen_1, feynman_I_12_1)")
    args = parser.parse_args()

    modes = ["no_migration", "sync", "async"]
    print(f"Running ablation on {args.problem} with modes: {modes}")

    results = []
    for mode in modes:
        print(f"\n=== Mode: {mode} ===")
        res = run_problem(args.problem, mode)
        print(f"R2: {res['r2']:.4f}, time: {res['time']:.2f}s")
        if res.get('expressions'):
            print("Top expr:", res['expressions'][0])
        results.append(res)

    print("\nSummary:")
    for r in results:
        print(f"  {r['mode']:<13} R2={r['r2']:.4f}  time={r['time']:.2f}s")


if __name__ == "__main__":
    main()
