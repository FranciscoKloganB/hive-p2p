class ConvergenceData:
    # region docstrings
    """
    Holds data that helps an domain.Hivemind keep track of converge in a simulation
    :cvar __DEVIATION_TOLERANCE: percentage in which a value on a distribution vector can deviate from another in eq cmp
    :type float
    :cvar MIN_CONVERGENCE_THRESHOLD: how many consecutive convergent stages must a file have to be considered converged
    :type int
    :ivar cswc: indicates how many consecutive steps a file has in convergence
    :type int
    :ivar largest_convergence_set: indicates the biggest set of consecutive steps throughout the simulaton for this file
    :type int
    :ivar convergence_set: list registering stages in which a file has seen convergence. Registers only when above min conv. threshold
    :type list<int>
    :ivar convergence_sets: list with all convergence sets found for this file during a simulation
    :type list<list<int>>
    """
    # endregion

    # region class variables, instance variables and constructors
    __DEVIATION_TOLERANCE = 0.01
    MIN_CONVERGENCE_THRESHOLD = 3

    def __init__(self):
        self.cswc = 0
        self.convergence_set = []
        self.convergence_sets = []
        self.largest_convergence_set = 0
    # endregion

    # region instance methods
    def cswc_increment_and_get(self, increment):
        self.cswc += increment
        return self.cswc

    def try_set_largest(self):
        if len(self.convergence_set) > self.largest_convergence_set:
            self.largest_convergence_set = self.cswc

    def try_update_convergence_set(self, stage):
        if self.cswc >= ConvergenceData.MIN_CONVERGENCE_THRESHOLD:
            self.convergence_set.append(stage)
            return True
        else:
            return False

    def save_sets_and_reset_data(self):
        self.cswc = 0
        if self.convergence_set:
            self.try_set_largest()
            self.convergence_sets.append(self.largest_convergence_set)
            self.convergence_set = []
    # endregion

    # region static methods
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
    # endregion
