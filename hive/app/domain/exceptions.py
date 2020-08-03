"""This module includes domain specific classes that inherit from Python's
builtin Exception class."""


class DistributionShapeError(Exception):
    """Raised when a vector shape is expected to match a matrix's column or row shape and doesn't."""
    def __init__(self, value=""):
        self.value = value

    def __str__(self):
        return self.value


class MatrixNotSquareError(Exception):
    """Raised when a function expects a square (N * N) matrix in any representation and some other dimensions are given"""
    def __init__(self, value=""):
        self.value = value

    def __str__(self):
        return self.value