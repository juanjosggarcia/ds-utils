from ds_utils.preprocessing.preprocessing_config import (
    FEATURES_CONFIG,
    FeatureNamingConfig,
)

import numpy as np
import pandas as pd
from typing import Literal, Protocol
from numpy.typing import ArrayLike


# region Aux Class -------------------------------------------------------------

# endregion Aux Class ----------------------------------------------------------

# region Typing Class ----------------------------------------------------------

# endregion Typing Class -------------------------------------------------------

# region CONSTANTS -------------------------------------------------------------

# endregion CONSTANTS ----------------------------------------------------------

# region Aux Functions ---------------------------------------------------------

def _format_number(x: float, decimals: int = 2) -> str:
    """
    Format a numeric value for LaTeX matrix representation.

    Parameters
    ----------
    x : float
        Numeric value to format.

    decimals : int, default=2
        Maximum number of decimal places.

    Returns
    -------
    str
        Formatted numeric value without unnecessary trailing zeros.
    """

    x = round(float(x), decimals)

    if x.is_integer():
        return str(int(x))

    return str(x)

# endregion Aux Functions ------------------------------------------------------

# region Maths Functions -------------------------------------------------------



# endregion Maths --------------------------------------------------------------

def matrix_to_latex(A: ArrayLike, decimals: int = 2) -> str:
    """
    Convert a NumPy-compatible array into a LaTeX matrix.

    One-dimensional arrays are converted into column vectors before
    generating the LaTeX representation.

    Parameters
    ----------
    A : ArrayLike
        Matrix or vector to convert.

    decimals : int, default=2
        Maximum number of decimal places used to format the values.
        Trailing zeros are removed.

    Returns
    -------
    str
        LaTeX representation of the matrix using the ``pmatrix``
        environment.

    Examples
    --------
    A = np.array([[1, 2.5], [3.141592, 4]])

    matrix_to_latex(A)
    # '\\begin{pmatrix}1 & 2.5 \\\\ 3.14 & 4\\end{pmatrix}'
    """

    A = np.asarray(A)

    # Converts a vector into a column matrix
    if A.ndim == 1:
        A = A.reshape(-1, 1)

    rows = [
        " & ".join(_format_number(x, decimals) for x in row)
        for row in A
    ]

    return (
        r"\begin{pmatrix}"
        + r" \\ ".join(rows)
        + r"\end{pmatrix}"
    )


def frac_to_latex(numerator: str | int | float,
                  denominator: str | int | float) -> str:
    """
    Convert a numerator and denominator into a LaTeX fraction.

    Parameters
    ----------
    numerator : str, int or float
        Numerator of the fraction.
    denominator : str, int or float
        Denominator of the fraction.

    Returns
    -------
    str
        LaTeX representation of the fraction.

    Examples
    --------
    >>> frac_to_latex(3, 4)
    '\\\\frac{3}{4}'

    >>> frac_to_latex("x + 1", "x - 2")
    '\\\\frac{x + 1}{x - 2}'
    """
    return rf"\frac{{{numerator}}}{{{denominator}}}"


def sqrt_to_latex(x: str | int | float) -> str:
    """
    Convert an expression into a LaTeX square root.

    Parameters
    ----------
    x : str, int or float
        Expression inside the square root.

    Returns
    -------
    str
        LaTeX representation of the square root.

    Examples
    --------
    >>> sqrt_to_latex("x + 1")
    '\\\\sqrt{x + 1}'
    """
    return rf"\sqrt{{{x}}}"


def root_to_latex(x: str | int | float, index: int) -> str:
    """
    Convert an expression into a LaTeX nth root.

    Parameters
    ----------
    x : str, int or float
        Expression inside the root.
    index : int
        Index of the root.

    Returns
    -------
    str
        LaTeX representation of the nth root.

    Examples
    --------
    >>> root_to_latex("x + 1", 3)
    '\\\\sqrt[3]{x + 1}'
    """
    return rf"\sqrt[{index}]{{{x}}}"


def power_to_latex(base: str | int | float,
                   exponent: str | int | float) -> str:
    """
    Convert a base and exponent into a LaTeX power.

    Parameters
    ----------
    base : str, int or float
        Base of the power.
    exponent : str, int or float
        Exponent of the power.

    Returns
    -------
    str
        LaTeX representation of the power.

    Examples
    --------
    >>> power_to_latex("x", 2)
    'x^{2}'
    """
    return rf"{base}^{{{exponent}}}"


def subscript_to_latex(expression: str,
                        subscript: str | int) -> str:
    """
    Add a subscript to a LaTeX expression.

    Parameters
    ----------
    expression : str
        Base expression.
    subscript : str or int
        Subscript value.

    Returns
    -------
    str
        LaTeX representation with the specified subscript.

    Examples
    --------
    >>> subscript_to_latex("x", 1)
    'x_{1}'
    """
    return rf"{expression}_{{{subscript}}}"


def abs_to_latex(x: str | int | float) -> str:
    """
    Convert an expression into LaTeX absolute value notation.

    Parameters
    ----------
    x : str, int or float
        Expression whose absolute value is represented.

    Returns
    -------
    str
        LaTeX representation using scalable absolute-value delimiters.

    Examples
    --------
    >>> abs_to_latex("x - 3")
    '\\\\left|x - 3\\\\right|'
    """
    return rf"\left|{x}\right|"


def norm_to_latex(x: str) -> str:
    """
    Convert an expression into LaTeX norm notation.

    Parameters
    ----------
    x : str
        Expression whose norm is represented.

    Returns
    -------
    str
        LaTeX representation using scalable norm delimiters.

    Examples
    --------
    >>> norm_to_latex("x")
    '\\\\left\\\\|x\\\\right\\\\|'
    """
    return rf"\left\|{x}\right\|"


def transpose_to_latex(A: str) -> str:
    """
    Add transpose notation to a LaTeX expression.

    Parameters
    ----------
    A : str
        Matrix or expression.

    Returns
    -------
    str
        LaTeX representation of the transpose.

    Examples
    --------
    >>> transpose_to_latex("A")
    'A^{T}'
    """
    return rf"{A}^{{T}}"


def inverse_to_latex(A: str) -> str:
    """
    Add inverse notation to a LaTeX expression.

    Parameters
    ----------
    A : str
        Matrix or expression.

    Returns
    -------
    str
        LaTeX representation of the inverse.

    Examples
    --------
    >>> inverse_to_latex("A")
    'A^{-1}'
    """
    return rf"{A}^{{-1}}"


def determinant_to_latex(A: str) -> str:
    """
    Convert a matrix or expression into determinant notation.

    Parameters
    ----------
    A : str
        Matrix or expression.

    Returns
    -------
    str
        LaTeX representation of the determinant.

    Examples
    --------
    >>> determinant_to_latex("A")
    '\\\\det\\\\left(A\\\\right)'
    """
    return rf"\det\left({A}\right)"


# region Maths Functions with "lazy import" ------------------------------------

# endregion Maths Functions with "lazy import" ---------------------------------