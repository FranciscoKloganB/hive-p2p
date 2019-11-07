import numpy as np

from globals.globals import MIN_CONVERGENCE_THRESHOLD

class ConvergenceData:
    # region docstrings
    """
    Holds data that helps an domain.Hivemind keep track of converge in a simulation
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
        if self.cswc >= MIN_CONVERGENCE_THRESHOLD:
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
            return np.allclose(one, another, rtol=0.3, atol=1e-2)
    # endregion

    # region overrides
    def __str__(self):
        return "time in convergence: " + str(ConvergenceData.recursive_len(self.convergence_sets)) + \
               "\nlargest_convergence_set: " + str(self.largest_convergence_set) + \
               "\nconvergence_sets:\n " + str(self.convergence_sets)

    def __repr__(self):
        return str(self.__dict__)
    # endregion

    # region helpers
    @staticmethod
    def recursive_len(item):
        if type(item) == list:
            return sum(ConvergenceData.recursive_len(subitem) for subitem in item)
        else:
            return 1
    # endregion
