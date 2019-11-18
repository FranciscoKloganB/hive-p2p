import numpy as np
import pandas as pd

from typing import Union, List
from globals.globals import MIN_CONVERGENCE_THRESHOLD, R_TOL, A_TOL


class ConvergenceData:
    # region docstrings
    """
    Holds data that helps an domain.Hivemind keep track of converge in a simulation
    :ivar int cswc: indicates how many consecutive steps a file has in convergence
    :ivar int largest_convergence_set: indicates the biggest set of consecutive steps throughout the simulation for a file
    :ivar List[int] convergence_set: current consecutive set of stages in which a file has seen convergence
    :ivar List[List[int]] convergence_sets: Set of all convergence sets found for this file during simulation
    """
    # endregion

    # region class variables, instance variables and constructors
    def __init__(self):
        self.cswc: int = 0
        self.convergence_set: List[int] = []
        self.convergence_sets: List[List[int]] = []
        self.largest_convergence_set: int = 0
    # endregion

    # region instance methods
    def cswc_increment(self, increment: int = 1) -> None:
        """
        Increases the counter that which registers how many consecutive stages had convergence
        :param increment: value to increment; default is 1
        """
        self.cswc += increment

    def try_set_largest_convergence_set(self) -> None:
        """
        Verifies if the current convergence set is the largest of all seen so far, if it is, updates the ConvergenceData
        instance field largest_convergence_set to be the length of the convergence_set
        """
        set_len = len(self.convergence_set)
        if set_len > self.largest_convergence_set:
            self.largest_convergence_set = set_len

    def try_append_to_convergence_set(self, stage: int) -> None:
        """
        Checks if the counter for consecutive stage convergence is bigger than the minimum threshold for verified
        convergence and if it is, appends the inputted stage to the current convergence set
        :param stage:
        """
        if self.cswc >= MIN_CONVERGENCE_THRESHOLD:
            self.convergence_set.append(stage)

    def save_sets_and_reset(self) -> None:
        """
        Appends the current convergence set to the list of convergence sets, verifies if the saved set is the largest
        seen so far and resets consecutive stages with convergence counter to zero.
        """
        self.cswc = 0
        if self.convergence_set:  # code below only executes if convergence_set isn't empty, i.e.: convergence_set != []
            self.try_set_largest_convergence_set()
            self.convergence_sets.append(self.convergence_set)
            self.convergence_set = []
    # endregion

    # region static methods
    @staticmethod
    def equal_distributions(one: Union[pd.DataFrame, pd.Series, np.array],
                            another: Union[pd.DataFrame, pd.Series, np.array]) -> bool:
        """
        :param pd.DataFrame one: labeled distribution
        :param pd.DataFrame another: another labeled distribution
        :returns bool: True if the elements at each index of both distributions are close enough, or False otherwise
        """
        if len(one) != len(another):
            return False
        else:
            return np.allclose(one, another, rtol=R_TOL, atol=A_TOL)
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
            return sum(ConvergenceData.recursive_len(sub_item) for sub_item in item)
        else:
            return 1
    # endregion
