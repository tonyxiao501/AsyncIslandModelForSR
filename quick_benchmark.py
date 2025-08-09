#!/usr/bin/env python3
"""
Quick Benchmark Runner

A simpler interface to run individual benchmark problems or small subsets.
Perfect for testing your fixes and comparing performance on standard problems.
"""

from benchmark_suite import StandardBenchmarkSuite
import sys


def run_quick_test():
    """Run a quick test on the easiest problems to verify everything works."""
    suite = StandardBenchmarkSuite()
    
    # Test the two simplest problems
    quick_problems = ['nguyen_1', 'feynman_I_12_1']
    
    print("🏃‍♂️ QUICK BENCHMARK TEST")
    print("Testing the two simplest problems to verify your fixes work...")
    print("="*60)
    
    for problem_key in quick_problems:
        result = suite.run_single_benchmark(problem_key, max_time_minutes=3.0)
        
        if result['success'] and result['r2_score'] > 0.9:
            print(f"✅ {problem_key}: PASSED (R² = {result['r2_score']:.4f})")
        else:
            print(f"❌ {problem_key}: FAILED (R² = {result.get('r2_score', 0):.4f})")
    
    # Generate quick report
    report = suite.generate_benchmark_report(save_to_file=False)
    print("\n" + "="*60)
    print("QUICK TEST SUMMARY:")
    print("="*60)
    
    success_count = sum(1 for r in suite.results if r.get('converged', False))
    total_count = len(suite.results)
    
    if success_count == total_count:
        print("🎉 ALL TESTS PASSED! Your fixes are working correctly.")
        print("You can now run the full benchmark suite with confidence.")
    elif success_count > 0:
        print(f"⚠️ PARTIAL SUCCESS: {success_count}/{total_count} tests passed.")
        print("Consider investigating the failing cases.")
    else:
        print("❌ ALL TESTS FAILED. Check your implementation.")
    
    return success_count == total_count


def run_feynman_physics():
    """Run all Feynman physics benchmarks."""
    suite = StandardBenchmarkSuite()
    
    print("🔬 FEYNMAN PHYSICS BENCHMARKS")
    print("Testing on real physics equations from Feynman Lectures")
    print("="*60)
    
    results = suite.run_benchmark_suite(
        categories=['feynman'],
        max_time_per_problem=8.0
    )
    
    # Show results
    physics_success = sum(1 for r in results if r.get('converged', False))
    print(f"\n🔬 PHYSICS BENCHMARK RESULTS:")
    print(f"Solved {physics_success}/{len(results)} physics equations")
    
    return suite.generate_benchmark_report()


def run_classic_gp():
    """Run classic GP benchmarks (Nguyen problems)."""
    suite = StandardBenchmarkSuite()
    
    print("🧬 CLASSIC GENETIC PROGRAMMING BENCHMARKS")
    print("Testing on Nguyen-Vladislavleva benchmark problems")
    print("="*60)
    
    results = suite.run_benchmark_suite(
        categories=['nguyen'],
        max_time_per_problem=6.0
    )
    
    # Show results
    gp_success = sum(1 for r in results if r.get('converged', False))
    print(f"\n🧬 GP BENCHMARK RESULTS:")
    print(f"Solved {gp_success}/{len(results)} classic GP problems")
    
    return suite.generate_benchmark_report()


def run_single_problem(problem_key: str):
    """Run a single specific problem."""
    suite = StandardBenchmarkSuite()
    
    if problem_key not in suite.problems:
        print(f"❌ Problem '{problem_key}' not found!")
        print(f"Available problems: {list(suite.problems.keys())}")
        return
    
    print(f"🎯 RUNNING SINGLE PROBLEM: {problem_key}")
    print("="*60)
    
    result = suite.run_single_benchmark(problem_key, max_time_minutes=10.0)
    
    # Show detailed results
    if result['success']:
        print(f"\n📊 DETAILED RESULTS:")
        print(f"Problem: {result['problem_name']}")
        print(f"Target: {result['target_expression']}")
        print(f"R² Score: {result['r2_score']:.6f}")
        print(f"RMSE: {result.get('rmse', 0):.6f}")
        print(f"Time: {result['elapsed_time']:.2f}s")
        print(f"Success: {'✅' if result['converged'] else '❌'}")
        
        if result.get('discovered_expressions'):
            print(f"\nDiscovered expressions:")
            for i, expr in enumerate(result['discovered_expressions'], 1):
                print(f"  {i}. {expr}")


def list_problems():
    """List all available benchmark problems."""
    suite = StandardBenchmarkSuite()
    
    print("📋 AVAILABLE BENCHMARK PROBLEMS")
    print("="*80)
    
    categories = {}
    for key, problem in suite.problems.items():
        if problem.category not in categories:
            categories[problem.category] = []
        categories[problem.category].append((key, problem))
    
    for category, problems in categories.items():
        print(f"\n{category.upper()} PROBLEMS:")
        print("-" * 40)
        for key, problem in problems:
            complexity_icon = {"easy": "🟢", "medium": "🟡", "hard": "🔴"}[problem.complexity_level]
            print(f"  {complexity_icon} {key:<20} - {problem.name}")
            print(f"     Target: {problem.target_expression}")
            print(f"     Variables: {len(problem.variable_ranges)}, Samples: {problem.n_samples}")


def main():
    """Main CLI interface."""
    if len(sys.argv) == 1:
        print("🎯 SYMBOLIC REGRESSION BENCHMARK RUNNER")
        print("="*50)
        print("Usage:")
        print("  python quick_benchmark.py quick        # Quick test (2 easy problems)")
        print("  python quick_benchmark.py physics      # All Feynman physics problems")
        print("  python quick_benchmark.py classic      # All Nguyen GP problems")
        print("  python quick_benchmark.py list         # List all available problems")
        print("  python quick_benchmark.py <problem>    # Run specific problem")
        print()
        print("Examples:")
        print("  python quick_benchmark.py nguyen_1")
        print("  python quick_benchmark.py feynman_I_6_2")
        print("  python quick_benchmark.py quick")
        return
    
    command = sys.argv[1].lower()
    
    if command == 'quick':
        run_quick_test()
    elif command == 'physics':
        run_feynman_physics()
    elif command == 'classic':
        run_classic_gp()
    elif command == 'list':
        list_problems()
    else:
        # Assume it's a specific problem name
        run_single_problem(command)


if __name__ == "__main__":
    main()
