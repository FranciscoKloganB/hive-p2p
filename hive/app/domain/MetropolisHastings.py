import sys
import logging
import numpy as np

from domain.exceptions.AlphaError import AlphaError
from domain.exceptions.MatrixNotSquareError import MatrixNotSquareError
from domain.exceptions.DistributionShapeError import DistributionShapeError


# region module public functions
def metropols_algorithm(q, r, major='r', f_alpha=0.8):
    """
    :param q: any square stochastic matrix, usually represented in row-major due to simfile inputs
    :type list<list<float>>
    :param r: python list given by user, doesn't matter if it's row or column major
    :type list<float>
    :param major: wether the proposal matrix is given as a row major ('r') or column ('c') major list of lists.
    :type str
    :param f_alpha: an arbitrary value in ]0, 1]
    :type float
    :return: column major N-D numpy.array matrix
    """

    r = np.asarray(r)
    q = np.asarray(q)

    if major == 'r':
        q = q.transpose()

    a = _construct_acceptance_matrix(q, r)
    m = _construct_transition_matrix(q, a)
    return m
# endregion


# region module private functions
def _construct_acceptance_matrix(k, v):
    """
    Constructs a acceptance matrix for a proposal matrix according to Behcet Acikmese & David S. Bayard in their paper:
    'A Markov Chain Approach to Probabilistic Swarm Guidance'
    :param a: an arbitrary value in ]0, 1]
    :type float
    :param k: column stochastic, column major, square, proposal matrix
    :type: N-D numpy.array
    :param v: distribution vector, irrelevant if it's column or row major
    :type: 1-D numpy.array
    :return:
    """
    size = k.shape[0]
    f = np.zeros(shape=k.shape)
    for i in range(size):
        for j in range(size):
            f[i, j] = min(1, (v[i] * k[i, i]) / (v[i]) * k[i, j])
    return f


def _construct_transition_matrix(q, a):
    """
    Constructs a transition matrix according to Behcet Acikmese & David S. Bayard in their paper:
    'A Markov Chain Approach to Probabilistic Swarm Guidance'
    :param q: column stochastic, column major, square, proposal matrix
    :type: N-D numpy.array
    :param a: column stochastic, column major, square, acceptance matrix
    :type: N-D numpy.array
    :return: p: transition matrix used by each node for probabilistic swarm guidance
    :type N-D numpy.array
    """
    size = q.shape[0]
    p = np.zeros(shape=q.shape)
    for i in range(size):
        for j in range(size):
            if i != j:
                p[i, j] = a[i, j] * q[i, j]
            else:
                p[i, j] = 1 - _mh_weighted_sum(q, a, i)
    return p


def _mh_weighted_sum(q, a, i):
    """
    Performs summation of the m-h branch when indices of m[i, j] are the same, i.e.: when i=j
    :param q: column stochastic, column major, square, proposal matrix
    :type: N-D numpy.array
    :param a: column stochastic, column major, square, acceptance matrix
    :type: N-D numpy.array
    :return: result
    :type float
    """
    size = q.shape[0]
    result = 0.0
    for k in range(size):
        if k != i:
            result += a[i, k] * q[i, k]
    return result
# endregion


# region lame unit testing
# noinspection DuplicatedCode
def test_matrix_pow():
    np.set_printoptions(threshold=sys.maxsize)
    m = [
        [0.1, 0.2, 0.3, 0, 0, 0.3, 0.05, 0.05],
        [0.2, 0, 0, 0.2, 0.4, 0.1, 0.1, 0],
        [0, 0.3, 0.3, 0.3, 0, 0, 0, 0.1],
        [0, 0, 0.05, 0.05, 0, 0.4, 0.3, 0.2],
        [0.5, 0.5, 0, 0, 0, 0, 0, 0],
        [0, 0, 0.3, 0.4, 0.2, 0.1, 0, 0],
        [0, 0.8, 0.05, 0, 0.05, 0, 0.05, 0.05],
        [0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.1]
    ]
    ma = np.asarray(m).transpose()
    print(ma)
    powma = np.linalg.matrix_power(ma, 100)
    print(powma[:, 0])


# noinspection DuplicatedCode
def test_mh_results():
    np.set_printoptions(threshold=sys.maxsize)

    k = [
        [0.1, 0.2, 0.3, 0, 0, 0.3, 0.05, 0.05],
        [0.2, 0, 0, 0.2, 0.4, 0.1, 0.1, 0],
        [0, 0.3, 0.3, 0.3, 0, 0, 0, 0.1],
        [0, 0, 0.05, 0.05, 0, 0.4, 0.3, 0.2],
        [0.5, 0.5, 0, 0, 0, 0, 0, 0],
        [0, 0, 0.3, 0.4, 0.2, 0.1, 0, 0],
        [0, 0.8, 0.05, 0, 0.05, 0, 0.05, 0.05],
        [0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.1]
    ]
    m = metropols_algorithm(q=k, r=[0.3, 0.3, 0.1, 0, 0.1, 0, 2.0, 0])
    print(m)
    powma = np.linalg.matrix_power(m, 100)
    print(powma[:, 0])
# endregion lame unit testing
