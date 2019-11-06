import numpy as np


class ConvergenceData:
    # region docstrings
    """
    Holds data that helps an domain.Hivemind keep track of converge in a simulation
    :cvar MIN_CONVERGENCE_THRESHOLD: how many consecutive convergent stages must a file have to be considered converged
    :type int
    :ivar cswc: indicates how many consecutive steps a file has in convergence
    :type int
    :ivar largest_convergence_set: indicates the biggest set of consecutive steps throughout the simulaton for this file
    :type int
    :ivar convergence_set: Stages a file has seen convergence for in current set. Appends only when cswc > threshold
    :type list<int>
    :ivar convergence_sets: Set of all convergence sets found for this file during simulation
    :type list<list<int>>
    """
    # endregion

    # region class variables, instance variables and constructors
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
        set_len = len(self.convergence_set)
        if set_len > self.largest_convergence_set:
            self.largest_convergence_set = set_len

    def try_update_convergence_set(self, stage):
        if self.cswc >= ConvergenceData.MIN_CONVERGENCE_THRESHOLD:
            self.convergence_set.append(stage)
            return True
        else:
            return False

    def save_sets_and_reset(self):
        self.cswc = 0
        if self.convergence_set:
            self.try_set_largest()
            self.convergence_sets.append(self.convergence_set)
            self.convergence_set = []
    # endregion

    # region static methods
    @staticmethod
    def equal_distributions(one, another, parts_count):
        if len(one) != len(another):
            return False
        else:
            another /= parts_count
            return np.allclose(one, another, rtol=0.05, atol=1/parts_count)
    # endregion

    # region overrides
    def __str__(self):
        return "convergence_sets: {}\n, largest_convergence_set: {}\n".format(
            self.convergence_sets, self.largest_convergence_set
        )
    # endregion
