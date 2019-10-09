class ConvergenceData:
    """
    Holds data that helps an domain.Hivemind keep track of converge in a simulation
    :cvar __DEVIATION_TOLERANCE: percentage in which a value on a distribution vector can deviate from another in eq cmp
    :type float
    :ivar cswc: indicates how many consecutive steps a file has in convergence
    :type int
    :ivar max_swc: indicates the biggest set of consecutive steps throughout the simulaton for this file
    :type int
    :ivar swc: indicates at which steps a file has seen convergence
    """

    __DEVIATION_TOLERANCE = 0.01

    def __init__(self, cswc=0, max_swc=0, swc=None):
        if swc is None:
            swc = []
        self.cswc = cswc
        self.max_swc = max_swc
        self.swc = swc

    @staticmethod
    def equal_distributions(one, another):
        row_count = len(one)
        if row_count != len(another):
            return False
        for i in range(0, row_count):
            deviation = another[i] * ConvergenceData.__DEVIATION_TOLERANCE
            lower_bound = another[i] - deviation
            upper_bound = another[i] + deviation
            if lower_bound < one[i] < upper_bound:
                continue
            else:
                return False
        return True
