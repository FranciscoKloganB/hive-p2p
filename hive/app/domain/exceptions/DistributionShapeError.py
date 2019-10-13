class DistributionShapeError(Exception):
    """
    Raised when a vector shape is expected to match a matrix's column or row shape and doesn't
    """
    def __init__(self, value=""):
        self.value = value

    def __str__(self):
        return self.value
