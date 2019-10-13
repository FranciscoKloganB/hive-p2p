class AlphaError(Exception):
    """
    Raised when an alpha value is expected to be within certain bounds and isn't
    """
    def __init__(self, value="(0, 1]"):
        self.value = value

    def __str__(self):
        return "Expected alpha value within bounds {}".format(self.value)
