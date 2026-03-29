"""Utility module providing helper functions for the calculator.

This module is imported by calculator.py and demonstrates cross-module
dependencies for the code knowledge graph.
"""

import logging
import sys
from typing import Any, Union

# Module-level logger
_logger = logging.getLogger("calculator")
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
_logger.addHandler(_handler)
_logger.setLevel(logging.INFO)


class ValidationError(Exception):
    """Raised when input validation fails."""
    pass


def validate_input(*args: Any) -> bool:
    """Validate that all arguments are numeric types.

    Args:
        *args: Values to validate.

    Returns:
        True if all inputs are valid numbers.

    Raises:
        ValidationError: If any input is not numeric.
    """
    for i, arg in enumerate(args):
        if not isinstance(arg, (int, float)):
            raise ValidationError(f"Argument {i} is not numeric: {arg!r} (type: {type(arg).__name__})")
    return True


def format_result(value: Union[str, float], precision: int = 6) -> str:
    """Format a numeric result for display.

    Args:
        value: The value to format (string or float).
        precision: Number of decimal places for float values.

    Returns:
        Formatted string representation.
    """
    if isinstance(value, float):
        return f"{value:.{precision}f}"
    return str(value).strip()


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp a value to a given range.

    Args:
        value: The input value.
        min_val: Minimum allowed value.
        max_val: Maximum allowed value.

    Returns:
        The clamped value.
    """
    validate_input(value, min_val, max_val)
    return max(min_val, min(value, max_val))


def percentage(part: float, whole: float) -> float:
    """Calculate what percentage 'part' is of 'whole'.

    Args:
        part: The partial value.
        whole: The total value.

    Returns:
        The percentage as a float.

    Raises:
        ValidationError: If whole is zero.
    """
    validate_input(part, whole)
    if whole == 0:
        raise ValidationError("Cannot calculate percentage with zero denominator")
    result = (part / whole) * 100
    return result


def logger(message: str, level: str = "info"):
    """Log a message at the specified level.

    Args:
        message: The message to log.
        level: Log level (debug, info, warning, error).
    """
    log_func = getattr(_logger, level.lower(), _logger.info)
    log_func(message)


def round_to_significant_figures(value: float, sig_figs: int = 4) -> float:
    """Round a number to a given number of significant figures.

    Args:
        value: The number to round.
        sig_figs: Number of significant figures.

    Returns:
        The rounded number.
    """
    validate_input(value)
    if value == 0:
        return 0.0
    import math
    magnitude = math.floor(math.log10(abs(value)))
    factor = 10 ** (sig_figs - 1 - magnitude)
    return round(value * factor) / factor
