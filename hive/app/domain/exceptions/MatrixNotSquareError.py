class MatrixNotSquareError(Exception):
    """
    Raised when a function expects a square (N * N) matrix in any representation and some other dimensions are given
    """
    def __init__(self, value=""):
        self.value = value

    def __str__(self):
        return self.value
