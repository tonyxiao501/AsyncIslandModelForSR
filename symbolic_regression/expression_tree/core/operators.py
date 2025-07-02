import numpy as np
import numba
from enum import IntEnum

class NodeType(IntEnum):
  VARIABLE = 0
  CONSTANT = 1
  BINARY_OP = 2
  UNARY_OP = 3

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

# Mapping dictionaries
BINARY_OP_MAP = {'+': OpType.ADD, '-': OpType.SUB, '*': OpType.MUL, '/': OpType.DIV, '^': OpType.POW}
UNARY_OP_MAP = {'sin': OpType.SIN, 'cos': OpType.COS, 'exp': OpType.EXP, 'log': OpType.LOG, 'sqrt': OpType.SQRT}

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
  elif operator == 'exp':
    return np.exp(np.clip(operand_val, -10, 10))
  elif operator == 'log':
    return np.log(np.abs(operand_val) + 1e-8)
  elif operator == 'sqrt':
    return np.sqrt(np.abs(operand_val))
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
  elif op_type == OpType.EXP:
    clipped = np.clip(operand_val, -10.0, 10.0)
    return np.exp(clipped)
  elif op_type == OpType.LOG:
    return np.log(np.abs(operand_val) + 1e-12)
  elif op_type == OpType.SQRT:
    return np.sqrt(np.abs(operand_val))
  return np.zeros_like(operand_val)