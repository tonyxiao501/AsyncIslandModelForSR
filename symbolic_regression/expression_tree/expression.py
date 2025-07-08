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
      # Parse using sympy first to validate and normalize the expression
      import sympy as sp
      from .core.node import VariableNode, ConstantNode, BinaryOpNode, UnaryOpNode
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

  @staticmethod
  def _sympy_to_node(sympy_expr, n_inputs: int):
    """Convert a sympy expression to our internal node structure"""
    import sympy as sp
    from .core.node import VariableNode, ConstantNode, BinaryOpNode, UnaryOpNode
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
    if isinstance(sympy_expr, (sp.sin, sp.cos, sp.exp, sp.log, sp.sqrt)):
      func_name = str(type(sympy_expr).__name__).lower()
      if func_name in UNARY_OP_MAP:
        operand = Expression._sympy_to_node(sympy_expr.args[0], n_inputs)
        return UnaryOpNode(func_name, operand)

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
          right = Expression._sympy_to_node(-sympy_expr.args[1], n_inputs)
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
