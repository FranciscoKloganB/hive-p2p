import pandas as pd
import numpy as np


class MarkovMatrix:
    """
    Implements a matrix which adheres to markov chain theory and implements some basic markov chains' behaviour
    :ivar states: identifiers for the buckets existing on network passed in matching order to their transition arrays
    :type list<str>
    :ivar transition_matrix: concrete markov matrix data structure with named rows and columns according to passed states
    :type 2D pandas.DataFrame
    """

    def __init__(self, states, transition_arrays):
        """
        Initialize the Markov Chain instance.
        :param states: names of the buckets of the P2P network that will form the resillient Hive.
        :type list<str>
        :param transition_arrays: list containing lists, each defining jump probabilities of each state between stages
        :type list<list<float>>
        """

        self.states = states
        self.transition_matrix = pd.DataFrame(
            np.array(transition_arrays).transpose(),
            columns=states,
            index=states
        )

    def next_state(self, current_state):
        """
        Choose list variable given probability of each variable
        :param current_state: state a file
        :type int or str
        :return next stage's state given the probabilities of each variable within transition_matrix for the current_state
        :type str
        """
        # https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.random.choice.html
        return np.random.choice(self.states, p=self.transition_matrix[current_state])

    def generate_states(self, current_state, no=10):
        """
        Generates the next states of the system.
        :param current_state: The state of the current random variable.
        :type int or str
        :param no: The number of future states to generate.
        :type int
        """
        future_states = []
        for i in range(no):
            next_state = self.next_state(current_state)
            future_states.append(next_state)
            current_state = next_state
        return future_states


def main():
    mm = MarkovMatrix([[0.5, 0.5, 0], [0.4, 0.2, 0.4], [0.2, 0.2, 0.6]], ["A", "B", "C"])
    print(mm.transition_matrix.to_string())
    var = np.random.choice(mm.states, p=mm.transition_matrix["A"])
    print(var)


if __name__ == "__main__":
    main()