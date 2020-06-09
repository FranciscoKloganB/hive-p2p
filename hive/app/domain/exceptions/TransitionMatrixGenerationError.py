class TransitionMatrixGenerationError(Exception):
    """
    Raised when a function fails to generate a transition matrix with given inputs. Likely when optimization is not possible.
    """
    def __init__(self, value=""):
        self.value = value

    def __str__(self):
        return self.value
