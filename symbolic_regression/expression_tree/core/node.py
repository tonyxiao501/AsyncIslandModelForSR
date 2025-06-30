import numpy as np
from abc import ABC, abstractmethod
from typing import Optional
from .operators import (
    NodeType, OpType, BINARY_OP_MAP, UNARY_OP_MAP,
    evaluate_variable, evaluate_constant, evaluate_binary_op, evaluate_unary_op,
    evaluate_binary_op_fast, evaluate_unary_op_fast
)
from ..optimization.memory_pool import get_global_pool

class Node(ABC):
    """Base node class with caching"""
    
    __slots__ = ('_hash_cache', '_size_cache')
    
    def __init__(self):
        self._hash_cache: Optional[int] = None
        self._size_cache: Optional[int] = None
    
    def _clear_cache(self):
        self._hash_cache = None
        self._size_cache = None
    
    @abstractmethod
    def evaluate(self, X: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def to_string(self) -> str:
        pass

    @abstractmethod
    def copy(self) -> 'Node':
        pass

    def size(self) -> int:
        if self._size_cache is None:
            self._size_cache = self._compute_size()
        return self._size_cache
    
    @abstractmethod
    def _compute_size(self) -> int:
        pass

    @abstractmethod
    def compress_constants(self):
        pass
    
    def __hash__(self) -> int:
        if self._hash_cache is None:
            self._hash_cache = self._compute_hash()
        return self._hash_cache
    
    @abstractmethod
    def _compute_hash(self) -> int:
        pass

class VariableNode(Node):
    __slots__ = ('index',)
    
    def __init__(self, index: int):
        super().__init__()
        self.index = index

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        return evaluate_variable(X, self.index)

    def to_string(self) -> str:
        return f"X{self.index}"

    def copy(self) -> 'VariableNode':
        return get_global_pool().get_variable_node(self.index)
    
    def _compute_size(self) -> int:
        return 1
    
    def _compute_hash(self) -> int:
        return hash((NodeType.VARIABLE, self.index))

    def compress_constants(self):
        return self

class ConstantNode(Node):
    __slots__ = ('value',)
    
    def __init__(self, value: float):
        super().__init__()
        self.value = float(value)

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        return evaluate_constant(X.shape[0], self.value)

    def to_string(self) -> str:
        return f"{self.value:.3f}"

    def copy(self) -> 'ConstantNode':
        return get_global_pool().get_constant_node(self.value)
    
    def _compute_size(self) -> int:
        return 1
    
    def _compute_hash(self) -> int:
        return hash((NodeType.CONSTANT, self.value))

    def compress_constants(self):
        return self

class BinaryOpNode(Node):
    __slots__ = ('operator', 'left', 'right')
    
    def __init__(self, operator: str, left: Node, right: Node):
        super().__init__()
        self.operator = operator
        self.left = left
        self.right = right

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        left_val = self.left.evaluate(X)
        right_val = self.right.evaluate(X)
        try:
            result = evaluate_binary_op(left_val, right_val, self.operator)
            result = np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)
            return result.astype(np.float64)
        except Exception:
            return np.zeros(X.shape[0], dtype=np.float64)

    def to_string(self) -> str:
        return f"({self.left.to_string()} {self.operator} {self.right.to_string()})"

    def copy(self) -> 'BinaryOpNode':
        return get_global_pool().get_binary_node(self.operator, self.left.copy(), self.right.copy())
    
    def _compute_size(self) -> int:
        return 1 + self.left.size() + self.right.size()
    
    def _compute_hash(self) -> int:
        return hash((NodeType.BINARY_OP, self.operator, hash(self.left), hash(self.right)))

    def compress_constants(self):
        left_c = self.left.compress_constants()
        right_c = self.right.compress_constants()
        
        if isinstance(left_c, ConstantNode) and isinstance(right_c, ConstantNode):
            val = evaluate_binary_op(np.array([left_c.value]), np.array([right_c.value]), self.operator)[0]
            return get_global_pool().get_constant_node(val)
        
        if left_c is None:
            left_c = self.left
        if right_c is None:
            right_c = self.right
        
        return get_global_pool().get_binary_node(self.operator, left_c, right_c)

class UnaryOpNode(Node):
    __slots__ = ('operator', 'operand')
    
    def __init__(self, operator: str, operand: Node):
        super().__init__()
        self.operator = operator
        self.operand = operand

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        operand_val = self.operand.evaluate(X)
        try:
            result = evaluate_unary_op(operand_val, self.operator)
            result = np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)
            return result.astype(np.float64)
        except Exception:
            return np.zeros(X.shape[0], dtype=np.float64)

    def to_string(self) -> str:
        return f"{self.operator}({self.operand.to_string()})"

    def copy(self) -> 'UnaryOpNode':
        return get_global_pool().get_unary_node(self.operator, self.operand.copy())
    
    def _compute_size(self) -> int:
        return 1 + self.operand.size()
    
    def _compute_hash(self) -> int:
        return hash((NodeType.UNARY_OP, self.operator, hash(self.operand)))

    def compress_constants(self):
        operand_c = self.operand.compress_constants()
        if operand_c is None:
            operand_c = self.operand
        if isinstance(operand_c, ConstantNode):
            val = evaluate_unary_op(np.array([operand_c.value]), self.operator)[0]
            return get_global_pool().get_constant_node(val)
        return get_global_pool().get_unary_node(self.operator, operand_c)