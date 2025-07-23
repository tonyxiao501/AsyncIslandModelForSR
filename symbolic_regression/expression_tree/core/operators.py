import numpy as np
import numba
from enum import IntEnum

class NodeType(IntEnum):
  VARIABLE = 0
  CONSTANT = 1
  BINARY_OP = 2
  UNARY_OP = 3
  SCALING_OP = 4

class OpType(IntEnum):
  # Binary ops
  ADD = 0
  SUB = 1
  MUL = 2
  DIV = 3
  POW = 4
  # Unary ops
  SIN = 5
  COS = 6
  EXP = 7
  LOG = 8
  SQRT = 9
  ABS = 10
  SQUARE = 11
  CUBE = 12
  RECIPROCAL = 13
  TAN = 14
  NEG = 15
  SQRT_ABS = 16
  LOG_ABS = 17
  INV_SQUARE = 18
  CBRT = 19
  FOURTH_ROOT = 20
  SINH = 21
  COSH = 22
  TANH = 23
  SCALE = 24  # Scaling operation

# Mapping dictionaries
BINARY_OP_MAP = {'+': OpType.ADD, '-': OpType.SUB, '*': OpType.MUL, '/': OpType.DIV, '^': OpType.POW}
UNARY_OP_MAP = {
    'sin': OpType.SIN, 'cos': OpType.COS, 'tan': OpType.TAN,
    'exp': OpType.EXP, 'log': OpType.LOG, 'sqrt': OpType.SQRT,
    'abs': OpType.ABS, 'square': OpType.SQUARE, 'cube': OpType.CUBE,
    'reciprocal': OpType.RECIPROCAL, 'neg': OpType.NEG,
    'sqrt_abs': OpType.SQRT_ABS, 'log_abs': OpType.LOG_ABS,
    'inv_square': OpType.INV_SQUARE, 'cbrt': OpType.CBRT,
    'fourth_root': OpType.FOURTH_ROOT, 'sinh': OpType.SINH,
    'cosh': OpType.COSH, 'tanh': OpType.TANH
}

# Scaling operation mapping
SCALING_OP_MAP = {'scale': OpType.SCALE}

@numba.njit(cache=True, inline='always')
def evaluate_variable(X, index):
  return X[:, index].astype(np.float64)

@numba.njit(cache=True, inline='always')
def evaluate_constant(n_samples, value):
  return np.full(n_samples, value, dtype=np.float64)

@numba.njit(cache=True, fastmath=True)
def evaluate_binary_op(left_val, right_val, operator):
  if operator == '+':
    return left_val + right_val
  elif operator == '-':
    return left_val - right_val
  elif operator == '*':
    return left_val * right_val
  elif operator == '/':
    out = np.ones_like(left_val)
    mask = right_val != 0
    out[mask] = left_val[mask] / right_val[mask]
    return out
  elif operator == '^':
    return np.power(left_val, np.clip(right_val, -10, 10))
  return np.zeros_like(left_val)

@numba.njit(cache=True, fastmath=True)
def evaluate_unary_op(operand_val, operator):
  if operator == 'sin':
    return np.sin(operand_val)
  elif operator == 'cos':
    return np.cos(operand_val)
  elif operator == 'tan':
    # Clip to avoid extreme values
    clipped = np.clip(operand_val, -1.5, 1.5)  # Avoid singularities near ±π/2
    return np.tan(clipped)
  elif operator == 'exp':
    return np.exp(np.clip(operand_val, -10, 10))
  elif operator == 'log':
    return np.log(np.abs(operand_val) + 1e-8)
  elif operator == 'sqrt':
    return np.sqrt(np.abs(operand_val))
  elif operator == 'abs':
    return np.abs(operand_val)
  elif operator == 'square':
    return operand_val * operand_val
  elif operator == 'cube':
    return operand_val * operand_val * operand_val
  elif operator == 'reciprocal':
    return 1.0 / (operand_val + 1e-12)
  elif operator == 'neg':
    return -operand_val
  elif operator == 'sqrt_abs':
    return np.sqrt(np.abs(operand_val))
  elif operator == 'log_abs':
    return np.log(np.abs(operand_val) + 1e-12)
  elif operator == 'inv_square':
    # Critical for inverse square laws (1/r^2)
    abs_val = np.abs(operand_val) + 1e-12
    return 1.0 / (abs_val * abs_val)
  elif operator == 'cbrt':
    # Cube root - preserve sign
    return np.sign(operand_val) * np.power(np.abs(operand_val), 1.0/3.0)
  elif operator == 'fourth_root':
    return np.power(np.abs(operand_val), 0.25)
  elif operator == 'sinh':
    clipped = np.clip(operand_val, -5, 5)
    return np.sinh(clipped)
  elif operator == 'cosh':
    clipped = np.clip(operand_val, -5, 5)
    return np.cosh(clipped)
  elif operator == 'tanh':
    return np.tanh(operand_val)  # tanh is bounded, no need to clip
  return np.zeros_like(operand_val)

@numba.njit(cache=True, fastmath=True)
def evaluate_binary_op_fast(left_val, right_val, op_type):
  if op_type == OpType.ADD:
    return left_val + right_val
  elif op_type == OpType.SUB:
    return left_val - right_val
  elif op_type == OpType.MUL:
    return left_val * right_val
  elif op_type == OpType.DIV:
    result = np.empty_like(left_val)
    mask = np.abs(right_val) > 1e-12
    result[mask] = left_val[mask] / right_val[mask]
    result[~mask] = np.sign(left_val[~mask]) * 1e6
    return result
  elif op_type == OpType.POW:
    clipped_right = np.clip(right_val, -10.0, 10.0)
    return np.power(np.abs(left_val) + 1e-12, clipped_right) * np.sign(left_val)
  return np.zeros_like(left_val)

@numba.njit(cache=True, fastmath=True)
def evaluate_unary_op_fast(operand_val, op_type):
  if op_type == OpType.SIN:
    return np.sin(operand_val)
  elif op_type == OpType.COS:
    return np.cos(operand_val)
  elif op_type == OpType.TAN:
    clipped = np.clip(operand_val, -1.5, 1.5)
    return np.tan(clipped)
  elif op_type == OpType.EXP:
    clipped = np.clip(operand_val, -10.0, 10.0)
    return np.exp(clipped)
  elif op_type == OpType.LOG:
    return np.log(np.abs(operand_val) + 1e-12)
  elif op_type == OpType.SQRT:
    return np.sqrt(np.abs(operand_val))
  elif op_type == OpType.ABS:
    return np.abs(operand_val)
  elif op_type == OpType.SQUARE:
    return operand_val * operand_val
  elif op_type == OpType.CUBE:
    return operand_val * operand_val * operand_val
  elif op_type == OpType.RECIPROCAL:
    return 1.0 / (operand_val + 1e-12)
  elif op_type == OpType.NEG:
    return -operand_val
  elif op_type == OpType.SQRT_ABS:
    return np.sqrt(np.abs(operand_val))
  elif op_type == OpType.LOG_ABS:
    return np.log(np.abs(operand_val) + 1e-12)
  elif op_type == OpType.INV_SQUARE:
    abs_val = np.abs(operand_val) + 1e-12
    return 1.0 / (abs_val * abs_val)
  elif op_type == OpType.CBRT:
    return np.sign(operand_val) * np.power(np.abs(operand_val), 1.0/3.0)
  elif op_type == OpType.FOURTH_ROOT:
    return np.power(np.abs(operand_val), 0.25)
  elif op_type == OpType.SINH:
    clipped = np.clip(operand_val, -5.0, 5.0)
    return np.sinh(clipped)
  elif op_type == OpType.COSH:
    clipped = np.clip(operand_val, -5.0, 5.0)
    return np.cosh(clipped)
  elif op_type == OpType.TANH:
    return np.tanh(operand_val)
  return np.zeros_like(operand_val)