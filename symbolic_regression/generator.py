import random
from typing import List
from .expression_tree import (
    Expression, Node, VariableNode, ConstantNode, 
    BinaryOpNode, UnaryOpNode
)

class ExpressionGenerator:
    """Generates random mathematical expressions"""

    def __init__(self, n_inputs: int, max_depth: int = 6):
        self.n_inputs = n_inputs
        self.max_depth = max_depth
        self.binary_ops = ['+', '-', '*', '/', '^']
        self.unary_ops = ['sin', 'cos', 'exp', 'log', 'sqrt']

    def generate_random_expression(self, depth: int = 0) -> Node:
        """Generate a random expression tree"""

        # Terminal nodes (variables or constants)
        if depth >= self.max_depth or (depth > 0 and random.random() < 0.3):
            if random.random() < 0.7:  # Variable
                return VariableNode(random.randint(0, self.n_inputs - 1))
            else:  # Constant
                return ConstantNode(random.uniform(-10, 10))

        # Function nodes
        if random.random() < 0.7:  # Binary operation
            op = random.choice(self.binary_ops)
            left = self.generate_random_expression(depth + 1)
            right = self.generate_random_expression(depth + 1)
            return BinaryOpNode(op, left, right)
        else:  # Unary operation
            op = random.choice(self.unary_ops)
            operand = self.generate_random_expression(depth + 1)
            return UnaryOpNode(op, operand)

    def generate_population(self, population_size: int) -> List[Expression]:
        """Generate a population of random expressions"""
        return [Expression(self.generate_random_expression())
                for _ in range(population_size)]