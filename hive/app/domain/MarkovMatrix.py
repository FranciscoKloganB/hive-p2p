import pandas as pd
import numpy as np


class MarkovMatrix:
    """
    Implements a matrix which adheres to markov chain theory and implements some basic markov chains' behaviour

    :param :ivar transition_line_vectors: line vector stating probability from 'this_worker' to 'another_worker'
    :type list<list<float>>
    :param :ivar worker_id_list: unique ID that identifies the worker bucket in the hive.
    :type list<str>  An array representing the states of the Markov Chain. It

    """

    def __init__(self, transition_line_vectors, worker_id_list):
        """
        Initialize the Markov Chain instance.
        """
        self.worker_id = worker_id_list
        self.transition_matrix = pd.DataFrame(
            np.array(transition_line_vectors).transpose(),
            columns=worker_id_list,
            index=worker_id_list
        )

    def next_state(self, current_state):
        """
        Returns the state of the random variable at the next time instance.
        :param current_state: the current state of the system.
        :type int or str
        """
        return np.random.choice(self.states, p=self.transition_matrix[self.index_dict[current_state], :])

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
