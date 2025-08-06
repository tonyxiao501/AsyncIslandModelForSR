# Codebase Reorganization Plan

## Current Issues
- 19 Python files with inconsistent sizes (1KB to 48KB)
- Large files have too many responsibilities
- Small files could be consolidated
- Related functionality scattered across multiple files

## Proposed New Structure (9 files instead of 19)

### 1. `regressor.py` (from mimo_regressor.py)
**Functions to keep:**
- MIMOSymbolicRegressor.__init__()
- MIMOSymbolicRegressor.fit() (core logic only)
- MIMOSymbolicRegressor.predict()
- MIMOSymbolicRegressor.get_expressions()
- Basic configuration and initialization

**Functions to move out:**
- Evolution logic → `evolution.py`
- Data scaling → `data_processing.py`
- Population management → `population_management.py`

### 2. `ensemble.py` (merge ensemble_regressor.py + ensemble_worker.py + shared_population_manager.py)
**Functions to include:**
- EnsembleMIMORegressor class (from ensemble_regressor.py)
- _fit_worker() (from ensemble_worker.py)
- ImprovedSharedData class (from shared_population_manager.py)
- All ensemble coordination logic

### 3. `evolution.py` (merge adaptive_evolution.py + constant_optimization.py + evolution logic)
**Functions to include:**
- update_adaptive_parameters()
- restart_population_enhanced()
- _optimize_constants()
- _should_optimize_constants()
- _should_optimize_constants_enhanced()
- Evolution loop logic from mimo_regressor.py

### 4. `population_management.py` (merge population.py + great_powers.py + population utilities)
**Functions to include:**
- PopulationManager class
- GreatPowers class (all methods)
- generate_diverse_population_optimized()
- inject_diversity_optimized()
- evaluate_population_enhanced_optimized()
- All population generation functions
- calculate_population_diversity() (from utils.py)

### 5. `genetic_operations.py` (merge genetic_ops.py + selection.py + genetic_ops/)
**Functions to include:**
- enhanced_selection()
- diversity_selection()
- tournament_selection()
- All genetic operators from genetic_ops/ directory
- Crossover and mutation operations

### 6. `data_processing.py` (merge data_scaling.py + physics_data_preprocessor.py + multi_scale_fitness.py)
**Functions to include:**
- DataScaler class
- PhysicsDataPreprocessor class
- PhysicsAwareScaler class
- MultiScaleFitnessEvaluator class
- create_robust_fitness_function()
- All scaling and preprocessing utilities

### 7. `utilities.py` (merge utils.py + expression_utils.py + evolution_stats.py + quality_assessment.py)
**Functions to include:**
- string_similarity()
- calculate_expression_uniqueness()
- to_sympy_expression()
- optimize_constants()
- optimize_final_expressions()
- get_evolution_stats()
- get_detailed_expressions()
- calculate_subtree_qualities()
- All utility and helper functions

### 8. `generator.py` (keep as-is)
**Current size is appropriate (11KB)**
- ExpressionGenerator class
- BiasedExpressionGenerator class
- All expression generation logic

## Benefits of This Reorganization

1. **Reduced File Count**: 19 → 9 files (53% reduction)
2. **Better Cohesion**: Related functionality grouped together
3. **Balanced File Sizes**: Most files will be 5-15KB (manageable size)
4. **Clearer Dependencies**: Easier to understand module relationships
5. **Easier Maintenance**: Less file switching when working on related features

## Implementation Steps

1. Create new consolidated files
2. Move functions systematically
3. Update all imports across the codebase
4. Test that all functionality still works
5. Remove old fragmented files
6. Update documentation

## File Size Estimates After Reorganization

- `regressor.py`: ~15KB (core logic only)
- `ensemble.py`: ~25KB (3 files combined)
- `evolution.py`: ~8KB (evolution logic)
- `population_management.py`: ~35KB (largest, but cohesive)
- `genetic_operations.py`: ~8KB (genetic operators)
- `data_processing.py`: ~30KB (data processing pipeline)
- `utilities.py`: ~12KB (helper functions)
- `generator.py`: ~12KB (unchanged)

Total: 8 focused, well-sized files instead of 19 scattered ones.
