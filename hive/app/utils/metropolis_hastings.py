import numpy as np

from typing import List, Tuple
from domain.exceptions.DistributionShapeError import DistributionShapeError
from domain.exceptions.MatrixNotSquareError import MatrixNotSquareError


# region module public functions
def metropolis_algorithm(
        a: List[List[int]], ddv: List[float], column_major_in: bool = False, column_major_out: bool = True) -> np.ndarray:
    """
    Constructs a transition matrix with desired distribution as steady state
    :param List[List[int]] a: any adjacency matrix
    :param List[float] ddv: a stochastic desired distribution vector
    :param bool column_major_in: indicates whether adj_matrix given in input is in row or column major form
    :param bool column_major_out: indicates whether to return transition_matrix output is in row or column major form
    :returns np.ndarray transition_matrix: unlabeled transition matrix
    """

    ddv: np.array = np.asarray(ddv)
    a: np.ndarray = np.asarray(a)

    # Input checking
    if ddv.shape[0] != a.shape[1]:
        raise DistributionShapeError("distribution shape: {}, proposal matrix shape: {}".format(ddv.shape, a.shape))
    if a.shape[0] != a.shape[1]:
        raise MatrixNotSquareError("rows: {}, columns: {}, expected square matrix".format(a.shape[0], a.shape[1]))

    if column_major_in:
        a = a.transpose()

    shape: Tuple[int, int] = a.shape
    size: int = a.shape[0]

    rw: np.ndarray = _construct_random_walk_matrix(a, shape, size)
    r: np.ndarray = _construct_rejection_matrix(ddv, rw, shape, size)

    transition_matrix: np.ndarray = np.zeros(shape=shape)

    for i in range(size):
        for j in range(size):
            if i != j:
                transition_matrix[i, j] = rw[i, j] * min(1, r[i, j])
        # after defining all p[i, j] we can safely defined p[i, i], i.e.: define p[i, j] when i = j
        transition_matrix[i, i] = _mh_summation(rw, r, i)

    return transition_matrix.transpose() if column_major_out else transition_matrix
    # endregion


# region module private functions
def _construct_random_walk_matrix(adj_matrix: np.ndarray, shape: Tuple[int, int], size: int) -> np.ndarray:
    """
    Constructs a random walk over the adjacency matrix
    :param np.ndarray adj_matrix: any adjacency matrix
    :param Tuple[int, int] shape: size of adj_matrix #(lines, columns)
    :param int size: the number of lines/columns matrix has. These should match the tuple values.
    :returns np.ndarray rw: a random_walk over the adjacency matrix with uniform distribution
    """
    rw: np.ndarray = np.zeros(shape=shape)
    for i in range(size):
        degree: np.int32 = np.sum(adj_matrix[i, :])  # all possible states reachable from state i, including self
        for j in range(size):
            rw[i, j] = adj_matrix[i, j] / degree
    return rw


def _construct_rejection_matrix(ddv: np.array, rw: np.ndarray, shape: Tuple[int, int], size: int) -> np.ndarray:
    """
    Constructs a rejection matrix for the random walk
    :param np.ndarray ddv: a stochastic desired distribution vector
    :param np.ndarray rw: a random_walk over an adjacency matrix
    :param Tuple[int, int] shape: size of adj_matrix #(lines, columns)
    :param int size: #lines or #columns, for effeciency
    :returns np.ndarray r: a matrix containing acceptance/rejectance probabilities for the random walk
    """
    r = np.zeros(shape=shape)
    with np.errstate(divide='ignore', invalid='ignore'):
        for i in range(size):
            for j in range(size):
                r[i, j] = (ddv[j] * rw[j, i]) / (ddv[i] * rw[i, j])
    return r


def _mh_summation(rw: np.ndarray, r: np.ndarray, i: int) -> np.int32:
    """
    Performs summation of the m-h branch when indices of m[i, j] are the same, i.e.: when i=j
    :param np.ndarray rw: a random_walk over an adjacency matrix
    :param np.ndarray r: a matrix containing acceptance/rejectance probabilities for the random walk
    :param int i: a fixed row index to simulate a simulation function
    :returns float: pii, the probability of going from state i to j, where i = j
    """
    size: int = rw.shape[0]
    pii: np.int32 = rw[i, i]
    print("pii: {}".format(rw[i, i]))
    for k in range(size):
        pii += rw[i, k] * (1 - min(1, r[i, k]))
    return pii
# endregion
