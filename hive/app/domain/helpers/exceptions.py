"""Module with classes that inherit from Python's builtin Exception class
and represent domain specific errors."""


class MatlabEngineContainerError(Exception):
    """Raised an AttributeError or EngineError is found in matlab container
    modules."""
    def __init__(self, value=""):
        self.value = value

    def __str__(self):
        return self.value


class DistributionShapeError(Exception):
    """Raised when a vector shape is expected to match a matrix's column or
    row shape and does not."""
    def __init__(self, value=""):
        self.value = value

    def __str__(self):
        return self.value


class MatrixNotSquareError(Exception):
    """Raised when a function expects a square (N * N) matrix in any
    representation and some other dimensions are given."""
    def __init__(self, value=""):
        self.value = value

    def __str__(self):
        return self.value


class MatrixError(Exception):
    """Generic Matrix Error, can be used for example when a pandas DataFrame
    has shape (0, 0) or is one-dimensional."""
    def __init__(self, value=""):
        self.value = value

    def __str__(self):
        return self.value


class IllegalArgumentError(ValueError):
    """Generic error used to indicate a parameter is not valid or expected."""
    def __init__(self, value=""):
        super().__init__(value)
