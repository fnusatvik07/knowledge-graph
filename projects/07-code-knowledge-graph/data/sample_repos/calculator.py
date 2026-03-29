"""A sample calculator module demonstrating classes, inheritance, and function calls.

This module is used as sample data for building a code knowledge graph.
It intentionally includes various Python constructs: imports, classes,
inheritance, decorators, static methods, and inter-function calls.
"""

import math
from decimal import Decimal, InvalidOperation
from utils import validate_input, format_result, logger


class CalculatorError(Exception):
    """Custom exception for calculator errors."""
    pass


class BaseCalculator:
    """Base calculator with fundamental arithmetic operations."""

    def __init__(self, precision: int = 10):
        self.precision = precision
        self.history: list[str] = []
        logger("BaseCalculator initialized")

    def add(self, a: float, b: float) -> float:
        """Add two numbers."""
        validate_input(a, b)
        result = a + b
        self._record_operation("add", a, b, result)
        return result

    def subtract(self, a: float, b: float) -> float:
        """Subtract b from a."""
        validate_input(a, b)
        result = a - b
        self._record_operation("subtract", a, b, result)
        return result

    def multiply(self, a: float, b: float) -> float:
        """Multiply two numbers."""
        validate_input(a, b)
        result = a * b
        self._record_operation("multiply", a, b, result)
        return result

    def divide(self, a: float, b: float) -> float:
        """Divide a by b with zero-division check."""
        validate_input(a, b)
        if b == 0:
            raise CalculatorError("Division by zero")
        result = a / b
        self._record_operation("divide", a, b, result)
        return result

    def _record_operation(self, op: str, a: float, b: float, result: float):
        """Record an operation in history."""
        entry = format_result(f"{op}({a}, {b}) = {result}")
        self.history.append(entry)

    def get_history(self) -> list[str]:
        """Return operation history."""
        return self.history.copy()

    def clear_history(self):
        """Clear all history."""
        self.history.clear()
        logger("History cleared")


class ScientificCalculator(BaseCalculator):
    """Extended calculator with scientific operations. Inherits from BaseCalculator."""

    def __init__(self, precision: int = 15):
        super().__init__(precision)
        self.angle_mode = "radians"

    def power(self, base: float, exponent: float) -> float:
        """Raise base to the power of exponent."""
        validate_input(base, exponent)
        result = math.pow(base, exponent)
        self._record_operation("power", base, exponent, result)
        return result

    def square_root(self, x: float) -> float:
        """Calculate the square root."""
        if x < 0:
            raise CalculatorError("Cannot take square root of negative number")
        result = math.sqrt(x)
        self._record_operation("sqrt", x, 0, result)
        return result

    def logarithm(self, x: float, base: float = math.e) -> float:
        """Calculate logarithm with given base."""
        validate_input(x, base)
        if x <= 0:
            raise CalculatorError("Logarithm undefined for non-positive numbers")
        result = math.log(x, base)
        self._record_operation("log", x, base, result)
        return result

    def sine(self, angle: float) -> float:
        """Calculate sine of an angle."""
        result = math.sin(angle)
        self._record_operation("sin", angle, 0, result)
        return result

    def cosine(self, angle: float) -> float:
        """Calculate cosine of an angle."""
        result = math.cos(angle)
        self._record_operation("cos", angle, 0, result)
        return result

    @staticmethod
    def degrees_to_radians(degrees: float) -> float:
        """Convert degrees to radians."""
        return math.radians(degrees)


class PrecisionCalculator(BaseCalculator):
    """Calculator using Decimal for arbitrary precision arithmetic."""

    def add_precise(self, a: str, b: str) -> str:
        """Add two decimal numbers with full precision."""
        try:
            result = Decimal(a) + Decimal(b)
            formatted = format_result(str(result))
            self._record_operation("add_precise", float(a), float(b), float(result))
            return formatted
        except InvalidOperation:
            raise CalculatorError("Invalid decimal input")


def calculate_expression(expression: str) -> float:
    """Parse and evaluate a simple arithmetic expression.

    Uses ScientificCalculator for the actual computation.
    """
    calc = ScientificCalculator()
    parts = expression.split()
    if len(parts) != 3:
        raise CalculatorError("Expression must be: number operator number")

    a, op, b = float(parts[0]), parts[1], float(parts[2])

    operations = {
        "+": calc.add,
        "-": calc.subtract,
        "*": calc.multiply,
        "/": calc.divide,
        "**": calc.power,
    }

    if op not in operations:
        raise CalculatorError(f"Unknown operator: {op}")

    return operations[op](a, b)


def batch_calculate(expressions: list[str]) -> list[float]:
    """Calculate multiple expressions and return results."""
    results = []
    for expr in expressions:
        try:
            result = calculate_expression(expr)
            results.append(result)
        except CalculatorError as e:
            logger(f"Error in expression '{expr}': {e}")
            results.append(float("nan"))
    return results
