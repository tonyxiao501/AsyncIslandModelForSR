import numpy as np
from typing import Optional
from .core.node import Node
import sympy as sp
from typing import Callable


class Expression:
  """Expression class with weighted complexity caching"""

  __slots__ = ('root', '_string_cache')

  def __init__(self, root: Node):
    self.root = root
    self._string_cache: Optional[str] = None

  def evaluate(self, X: np.ndarray) -> np.ndarray:
    return self.root.evaluate(X)

  def to_string(self) -> str:
    if self._string_cache is None:
      self._string_cache = self.root.to_string()
    return self._string_cache

  def copy(self) -> 'Expression':
    return Expression(self.root.copy())

  def size(self) -> int:
    """Node count (original complexity measure)"""
    return self.root.size()

  def complexity(self) -> float:
    """Weighted complexity score"""
    return self.root.complexity()

  def compress_constants(self) -> 'Expression':
    node = self.root.compress_constants()
    if node is None:
      node = self.root
    return Expression(node)

  def clear_cache(self):
    """Clear cached values"""
    self._string_cache = None
    
  def to_sympy(self) -> sp.Expr:   
    return self.root.to_sympy(sp.numbered_symbols(prefix='c'))

  def get_constants(self) -> tuple:
    const_list = []
    self.root.get_constants(const_list)
    return tuple(const_list)
  
  def set_constants(self, constants):
    if len(self.get_constants()) != len(constants):
        raise RuntimeError("Error for setting constants with different shape")
    if not isinstance(constants, list):
        constants = list(constants)
    self.root.set_constants(constants)   
    self.clear_cache() 

# Function: lambda X, *params -> Y
# Returns None if fails to lambdify or bad operation (x - x) -> 0
  def vector_lambdify(self) -> Optional[Callable]:
    constants = self.get_constants()
    sp_expr = self.to_sympy()
    c_dim = len(constants)
    x_dim = len(sp_expr.free_symbols) - c_dim
    if x_dim <= 0:
        return None
    symbols = sp.symbols(f'x0:{x_dim}')
    if c_dim > 0:
        symbols += sp.symbols(f'c0:{c_dim}')
    try:
        lambda_func = sp.lambdify(symbols, sp_expr, modules='numpy')
    except:
        return None
    if sp.simplify(sp_expr).free_symbols != symbols:
        return None
    # wrapper for unpack parameters
    def wrapper(X, *param):
        X_arr = np.array(X)
        if len(X_arr.shape) == 0:
            return lambda_func(X, *param)
        else:
            return lambda_func(*X, *param)
    return np.vectorize(wrapper)

  def __hash__(self) -> int:
    return hash(self.root)

  def __eq__(self, other) -> bool:
    if not isinstance(other, Expression):
      return False
    return hash(self) == hash(other)

  @classmethod
  def from_string(cls, expr_str: str, n_inputs: int = 1) -> 'Expression':
    try:
      # First check if this is a scaling operation format
      if expr_str.startswith('scale(') and expr_str.endswith(')'):
        return cls._parse_scaling_node(expr_str, n_inputs)
      
      # Check for nested scaling operations that SymPy can't handle
      if 'scale(' in expr_str:
        # If the expression contains scaling operations but isn't just a scaling operation,
        # we need to handle it specially since SymPy doesn't understand scale()
        return cls._parse_expression_with_scaling(expr_str, n_inputs)
      
      # Parse using sympy first to validate and normalize the expression
      import sympy as sp
      from .core.node import VariableNode, ConstantNode, BinaryOpNode, UnaryOpNode, ScalingOpNode
      from .core.operators import BINARY_OP_MAP, UNARY_OP_MAP

      # Replace variable notation for sympy compatibility
      normalized_str = expr_str
      for i in range(n_inputs):
        normalized_str = normalized_str.replace(f'X{i}', f'x{i}')

      # Parse with sympy
      try:
        sympy_expr = sp.sympify(normalized_str)
      except:
        # Fallback: try with different variable naming
        for i in range(n_inputs):
          normalized_str = normalized_str.replace(f'x{i}', f'x_{i}')
        sympy_expr = sp.sympify(normalized_str)

      # Convert sympy expression to our node structure
      root_node = cls._sympy_to_node(sympy_expr, n_inputs)
      return cls(root_node)

    except Exception as e:
      # If parsing fails, create a simple fallback expression
      from .core.node import VariableNode
      fallback_node = VariableNode(0)
      return cls(fallback_node)

  @classmethod
  def _parse_scaling_node(cls, expr_str: str, n_inputs: int) -> 'Expression':
    """Parse scaling node format: scale(operand, power)"""
    from .core.node import ScalingOpNode, VariableNode
    
    # Remove 'scale(' and final ')'
    content = expr_str[6:-1]
    
    # Find the last comma to separate operand from power
    # Need to be careful about nested parentheses
    paren_depth = 0
    last_comma_idx = -1
    
    for i, char in enumerate(content):
      if char == '(':
        paren_depth += 1
      elif char == ')':
        paren_depth -= 1
      elif char == ',' and paren_depth == 0:
        last_comma_idx = i
    
    if last_comma_idx == -1:
      raise ValueError(f"Invalid scaling format: {expr_str}")
    
    operand_str = content[:last_comma_idx].strip()
    power_str = content[last_comma_idx + 1:].strip()
    
    try:
      power = int(power_str)
    except ValueError:
      raise ValueError(f"Invalid power in scaling operation: {power_str}")
    
    # Protection against infinite recursion - if operand is also a scaling node,
    # and it's identical to the current expression, fall back to a simple node
    if operand_str.startswith('scale(') and operand_str == expr_str:
      # This would cause infinite recursion, so return a fallback
      return cls(VariableNode(0))
    
    # Recursively parse the operand with additional safety checks
    try:
      operand_expr = cls.from_string(operand_str, n_inputs)
      scaling_node = ScalingOpNode(power, operand_expr.root)
      return cls(scaling_node)
    except RecursionError:
      # If we hit recursion limit, return a fallback
      return cls(VariableNode(0))
    except Exception:
      # If parsing fails for any other reason, return a fallback
      return cls(VariableNode(0))

  @classmethod
  def _parse_expression_with_scaling(cls, expr_str: str, n_inputs: int) -> 'Expression':
    """Parse expressions that contain scaling operations mixed with other operations"""
    from .core.node import VariableNode
    
    # For now, this is a simplified approach that falls back to a variable node
    # In a full implementation, you would need to properly parse mixed expressions
    # containing scale() functions, but this prevents infinite loops
    try:
      # Try to replace scale() functions with simpler equivalents for SymPy
      simplified_str = expr_str
      
      # Find and replace scale() operations
      import re
      scale_pattern = r'scale\(([^,]+),\s*(-?\d+)\)'
      
      def replace_scale(match):
        operand = match.group(1).strip()
        power = int(match.group(2))
        # Convert scale(operand, power) to operand * 10^power
        if power >= 0:
          return f"({operand} * {10**power})"
        else:
          return f"({operand} / {10**abs(power)})"
      
      simplified_str = re.sub(scale_pattern, replace_scale, simplified_str)
      
      # Now try to parse with SymPy
      import sympy as sp
      normalized_str = simplified_str
      for i in range(n_inputs):
        normalized_str = normalized_str.replace(f'X{i}', f'x{i}')
      
      sympy_expr = sp.sympify(normalized_str)
      root_node = cls._sympy_to_node(sympy_expr, n_inputs)
      return cls(root_node)
      
    except Exception:
      # If all else fails, return a simple variable node
      fallback_node = VariableNode(0)
      return cls(fallback_node)

  @staticmethod
  def _sympy_to_node(sympy_expr, n_inputs: int):
    """Convert a sympy expression to our internal node structure"""
    import sympy as sp
    from .core.node import VariableNode, ConstantNode, BinaryOpNode, UnaryOpNode, ScalingOpNode
    from .core.operators import BINARY_OP_MAP, UNARY_OP_MAP

    # Handle atomic expressions
    if sympy_expr.is_symbol:
      # Extract variable index from symbol name (x0, x1, etc.)
      var_name = str(sympy_expr)
      if var_name.startswith('x'):
        try:
          var_index = int(var_name[1:].replace('_', ''))
          if 0 <= var_index < n_inputs:
            return VariableNode(var_index)
        except:
          pass
      # Default to first variable if parsing fails
      return VariableNode(0)

    if sympy_expr.is_number:
      return ConstantNode(float(sympy_expr))

    # Handle function calls (unary operations)
    if isinstance(sympy_expr, (sp.sin, sp.cos, sp.exp, sp.log)):
      func_name = str(type(sympy_expr).__name__).lower()
      if func_name in UNARY_OP_MAP:
        operand = Expression._sympy_to_node(sympy_expr.args[0], n_inputs)
        return UnaryOpNode(func_name, operand)
    
    if isinstance(sympy_expr, sp.Pow) and sympy_expr.args[1] == sp.Rational(1, 2):
      operand = Expression._sympy_to_node(sympy_expr.args[0], n_inputs)
      return UnaryOpNode('sqrt', operand)

    # Handle power operations
    if isinstance(sympy_expr, sp.Pow):
      base = Expression._sympy_to_node(sympy_expr.args[0], n_inputs)
      exponent = Expression._sympy_to_node(sympy_expr.args[1], n_inputs)
      return BinaryOpNode('^', base, exponent)

    # Handle binary operations
    if isinstance(sympy_expr, sp.Add):
      # Handle addition (potentially with multiple terms)
      if len(sympy_expr.args) == 2:
        left = Expression._sympy_to_node(sympy_expr.args[0], n_inputs)
        right = Expression._sympy_to_node(sympy_expr.args[1], n_inputs)
        return BinaryOpNode('+', left, right)
      else:
        # Multiple terms - build left-associative tree
        result = Expression._sympy_to_node(sympy_expr.args[0], n_inputs)
        for arg in sympy_expr.args[1:]:
          right = Expression._sympy_to_node(arg, n_inputs)
          result = BinaryOpNode('+', result, right)
        return result

    if isinstance(sympy_expr, sp.Mul):
      # Handle multiplication
      if len(sympy_expr.args) == 2:
        left = Expression._sympy_to_node(sympy_expr.args[0], n_inputs)
        right = Expression._sympy_to_node(sympy_expr.args[1], n_inputs)
        return BinaryOpNode('*', left, right)
      else:
        # Multiple factors - build left-associative tree
        result = Expression._sympy_to_node(sympy_expr.args[0], n_inputs)
        for arg in sympy_expr.args[1:]:
          right = Expression._sympy_to_node(arg, n_inputs)
          result = BinaryOpNode('*', result, right)
        return result

    # Handle subtraction and division (these are typically converted by sympy)
    if hasattr(sympy_expr, 'func'):
      func_name = str(sympy_expr.func)
      if 'Add' in func_name and len(sympy_expr.args) == 2:
        # Check if this is actually subtraction (second arg is negative)
        if sympy_expr.args[1].is_negative:
          left = Expression._sympy_to_node(sympy_expr.args[0], n_inputs)
          right = Expression._sympy_to_node(sp.Mul(-1, sympy_expr.args[1]), n_inputs)
          return BinaryOpNode('-', left, right)

      if 'Mul' in func_name and len(sympy_expr.args) == 2:
        # Check if this is actually division (second arg is a power with negative exponent)
        if isinstance(sympy_expr.args[1], sp.Pow) and sympy_expr.args[1].args[1] == -1:
          left = Expression._sympy_to_node(sympy_expr.args[0], n_inputs)
          right = Expression._sympy_to_node(sympy_expr.args[1].args[0], n_inputs)
          return BinaryOpNode('/', left, right)

    # Handle other expressions by trying to extract structure
    if hasattr(sympy_expr, 'args') and len(sympy_expr.args) >= 1:
      # Try to identify the operation from the expression type
      expr_type = type(sympy_expr).__name__.lower()

      if len(sympy_expr.args) == 1:
        # Unary operation
        if expr_type in ['sin', 'cos', 'exp', 'log', 'sqrt']:
          operand = Expression._sympy_to_node(sympy_expr.args[0], n_inputs)
          return UnaryOpNode(expr_type, operand)

      elif len(sympy_expr.args) == 2:
        # Binary operation
        left = Expression._sympy_to_node(sympy_expr.args[0], n_inputs)
        right = Expression._sympy_to_node(sympy_expr.args[1], n_inputs)

        # Try to infer operation type
        if 'add' in expr_type.lower():
          return BinaryOpNode('+', left, right)
        elif 'mul' in expr_type.lower():
          return BinaryOpNode('*', left, right)
        elif 'pow' in expr_type.lower():
          return BinaryOpNode('^', left, right)
        else:
          # Default to addition for unknown binary operations
          return BinaryOpNode('+', left, right)

    # Fallback: return a constant node
    try:
      value = float(sympy_expr)
      return ConstantNode(value)
    except:
      return ConstantNode(1.0)
