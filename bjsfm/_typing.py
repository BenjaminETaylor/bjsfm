"""Shared type aliases for numpy arrays.

These aliases use :mod:`numpy.typing` (built into numpy) instead of the
unmaintained ``nptyping`` package. ``numpy.typing`` encodes the array *dtype*
but not its *shape*; array shapes (e.g. ``3x3``, ``Nx3``) are documented in the
relevant docstrings instead, since shapes are not statically checkable today.
"""
import numpy as np
from numpy.typing import NDArray, ArrayLike

__all__ = ['NDArray', 'ArrayLike', 'FloatArray', 'ComplexArray', 'IntArray', 'BoolArray']

#: Floating-point array (any shape).
FloatArray = NDArray[np.float64]
#: Complex array (any shape).
ComplexArray = NDArray[np.complexfloating]
#: Integer index array (any shape).
IntArray = NDArray[np.intp]
#: Boolean mask array (any shape).
BoolArray = NDArray[np.bool_]

