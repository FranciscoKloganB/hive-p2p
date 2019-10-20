import sys
import logging
import numpy as np

from domain.exceptions.AlphaError import AlphaError
from domain.exceptions.MatrixNotSquareError import MatrixNotSquareError
from domain.exceptions.DistributionShapeError import DistributionShapeError


# region module public functions
def metropolis_algorithm(adj_matrix, ddv, column_major_in=False, column_major_out=True):
    """
    :param adj_matrix: any adjacency matrix (list of lits) provided by the user in row major form
    :type list<list<float>>
    :param ddv: a stochastic desired distribution vector
    :type list<float>
    :param column_major_in: indicates wether adj_matrix given in input is in row or column major form
    :type bool
    :param column_major_out: indicates wether to return transition_matrix output is in row or column major form
    :type bool
    :return: transition matrix that converges to ddv in the long term
    :type: N-D numpy.array
    """

    ddv = np.asarray(ddv)
    adj_matrix = np.asarray(adj_matrix)

    if column_major_in:
        adj_matrix = adj_matrix.transpose()

    shape = adj_matrix.shape
    size = adj_matrix.shape[0]

    rw = _construct_random_walk_matrix(adj_matrix, shape, size)
    r = _construct_rejection_matrix(ddv, rw, shape, size)

    transition_matrix = np.zeros(shape=shape)

    for i in range(size):
        for j in range(size):
            if i != j:
                transition_matrix[i, j] = rw[i, j] * min(1, r[i, j])
        # after defining all p[i, j] we can safely defined p[i, i], i.e.: define p[i, j] when i = j
        transition_matrix[i, i] = _mh_summation(rw, r, i)

    return transition_matrix.transpose() if column_major_out else transition_matrix
    # endregion


# region module private functions
def _construct_random_walk_matrix(adj_matrix, shape, size):
    rw = np.zeros(shape=shape)
    for i in range(size):
        ext_degree = np.sum(adj_matrix[i, :])  # states reachable from node i, counting itself (hence ext) [0, 1, 1] = 2
        for j in range(size):
            rw[i, j] = adj_matrix[i, j] / ext_degree
    return rw


def _construct_rejection_matrix(ddv, rw, shape, size):
    r = np.zeros(shape=shape)
    with np.errstate(divide='ignore', invalid='ignore'):
        for i in range(size):
            for j in range(size):
                r[i, j] = (ddv[j] * rw[j, i]) / (ddv[i] * rw[i, j])
    return r


def _mh_summation(rw, r, i):
    """
    Performs summation of the m-h branch when indices of m[i, j] are the same, i.e.: when i=j
    :param rw: column stochastic, square, random walk matrix
    :type: N-D numpy.array
    :param r: column stochastic, square, rejection matrix
    :type: N-D numpy.array
    :return: pii, the jump probability from state i to state i in the metropolis hastings output (transition matrix)
    :type float
    """
    size = rw.shape[0]
    pii = rw[i, i]
    for k in range(size):
        pii += rw[i, k] * (1 - min(1, r[i, k]))
    return pii
# endregion



