#!/usr/bin/env python3
"""
Quick test to verify cbrt complexity fix.
"""

import sys
sys.path.insert(0, '/home/tonyx/MIMOSymbolicRegression')

from symbolic_regression import *

# Test cbrt complexity now
x = VariableNode(0)
c1 = ConstantNode(2.0)
sin_x = UnaryOpNode('sin', x)
mult = BinaryOpNode('*', c1, sin_x)

# Direct expression: 2*sin(x)
direct_expr = Expression(mult)
print(f'Expression: 2*sin(x)')
print(f'Complexity: {direct_expr.complexity():.2f}')

# Cbrt wrapped expression: cbrt(2*sin(x))
cbrt_expr = UnaryOpNode('cbrt', mult)
cbrt_wrapped = Expression(cbrt_expr)
print(f'\nExpression: cbrt(2*sin(x))')
print(f'Complexity: {cbrt_wrapped.complexity():.2f}')

# Difference
diff = cbrt_wrapped.complexity() - direct_expr.complexity()
print(f'\nCbrt adds: {diff:.2f} complexity units')

# Test target function: 2*sin(x) + cos(2*x)
cos_2x = UnaryOpNode('cos', BinaryOpNode('*', c1, x))
target_expr = Expression(BinaryOpNode('+', mult, cos_2x))
print(f'\nTarget: 2*sin(x) + cos(2*x)')
print(f'Complexity: {target_expr.complexity():.2f}')

# Cbrt wrapped target
cbrt_target = UnaryOpNode('cbrt', BinaryOpNode('+', mult, cos_2x))
cbrt_target_expr = Expression(cbrt_target)
print(f'\nCbrt wrapped target: cbrt(2*sin(x) + cos(2*x))')
print(f'Complexity: {cbrt_target_expr.complexity():.2f}')

print(f'\nNow cbrt should add significant penalty: {cbrt_target_expr.complexity() - target_expr.complexity():.2f}')

if cbrt_wrapped.complexity() > direct_expr.complexity() + 2.0:
    print("✓ SUCCESS: cbrt now has proper penalty!")
else:
    print("✗ FAIL: cbrt still too cheap")
