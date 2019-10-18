import pandas as pd
import numpy as np


class MarkovMatrix:
    # region docstrings
    """
    Implements a matrix which adheres to markov chain theory and implements some basic markov chains' behaviour
    :ivar states: identifiers for the buckets existing on network passed in matching order to their transition arrays
    :type list<str>
    :ivar transition_matrix: concrete markov matrix data structure with named rows and columns according to passed states
    :type 2D pandas.DataFrame
    """
    # endregion

    # region class variables, instance variables and constructors
    def __init__(self, states, transition_arrays):
        """
        Initialize the Markov Chain instance.
        :param states: names of the buckets of the P2P network that will form the resilient Hive.
        :type list<str>
        :param transition_arrays: list containing lists, each defining jump probabilities of each state between stages
        :type list<list<float>>
        """
        self.current_state = None
        self.states = states
        self.transition_matrix = pd.DataFrame(
            np.array(transition_arrays).transpose(),
            columns=states,
            index=states
        )
    # endregion

    # region instance methods
    def next_state(self, current_state):
        """
        Choose list variable given probability of each variable
        :param current_state: state a file
        :type int or str
        :return next stage's state given the probabilities of each variable within transition_matrix for the current_state
        :type str
        """
        # https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.random.choice.html
        return np.random.choice(a=self.states, p=self.transition_matrix[current_state])
    # endregion
