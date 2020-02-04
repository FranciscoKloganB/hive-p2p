import random
import numpy as np

from typing import List
from utils.randoms import random_index


def new_symmetric_adjency_matrix(size: int):
    """
    Generates a random symmetric matrix without transient state sets or absorbeent nodes
    :param int size: the size of the square matrix (size * size)
    :return List[List[int]] adj_matrix: the adjency matrix representing the connections between a group of peers
    """
    secure_random = random.SystemRandom()
    adj_matrix: List[List[int]] = [[0] * size for _ in range(size)]
    choices: List[int] = [0, 1]

    for i in range(size):
        for j in range(i, size):
            probability = secure_random.uniform(0.0, 1.0)
            edge_val = np.random.choice(a=choices, p=[probability, 1-probability]).item()  # converts numpy.int32 to int
            adj_matrix[i][j] = adj_matrix[j][i] = edge_val

    # Use guilty until proven innocent approach for both checks
    for i in range(size):
        is_absorbent_or_transient: bool = True
        for j in range(size):
            # Ensure state i can reach and be reached by some other state j, where i != j
            if adj_matrix[i][j] == 1 and i != j:
                is_absorbent_or_transient = False
                break
        if is_absorbent_or_transient:
            j = random_index(i, size)
            adj_matrix[i][j] = adj_matrix[j][i] = 1
    return adj_matrix
