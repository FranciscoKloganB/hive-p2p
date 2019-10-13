import logging

import numpy as np

from domain.exceptions.AlphaError import AlphaError
from domain.exceptions.MatrixNotSquareError import MatrixNotSquareError
from domain.exceptions.DistributionShapeError import DistributionShapeError


# region module public functions
def metropols_algorithm(k, v, major='r', f_alpha=0.5):
    """
    :param k: any square stochastic matrix, usually represented in row-major due to simfile inputs
    :type list<list<float>>
    :param v: python list given by user, doesn't matter if it's row or column major
    :type list<float>
    :param major: wether the proposal matrix is given as a row major ('r') or column ('c') major list of lists.
    :type str
    :param f_alpha: an arbitrary value in ]0, 1]
    :type float
    :return: column major N-D numpy.array matrix
    """
    try:
        v = np.asarray(v, dtype=np.float64)
        k = np.asarray(k, dtype=np.float64)
    except ValueError as ve:
        logging.exception(str(ve))
        logging.debug("proposal matrix or distribution vector are missing elements or have incorrect sizes")

    # Input checking
    if f_alpha <= 0 or f_alpha > 1:
        raise AlphaError("(0, 1]")
    if v.shape[0] != k.shape[1]:
        raise DistributionShapeError("distribution shape: {}, proposal matrix shape: {}".format(v.shape, k.shape))
    if k.shape[0] != k.shape[1]:
        raise MatrixNotSquareError("rows: {}, columns: {}, expected square matrix".format(k.shape[0], k.shape[1]))

    # Ensure matrice is column major
    if major == 'r':
        k.transpose()

    acceptance_matrix = _construct_f_matrix(f_alpha, k, v)
    # Construct the transition_matrix that will be returned to the user
    transition_matrix = np.zeros(shape=k.shape, dtype=np.float64, order='F')

    return None
# endregion


# region module private functions
def _construct_f_matrix(a, k, v):
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
    r = _construct_r_matrix(k, v)
    f = np.zeros(shape=k.shape, dtype=np.float64, order='F')
    for i in range(size):
        for j in range(size):
            f[i, j] = a * min(1, r[i, j])
    return f


def _construct_r_matrix(k, v):
    """
    Constructs a acceptance probabilities matrix according to Behcet Acikmese & David S. Bayard in their paper:
    'A Markov Chain Approach to Probabilistic Swarm Guidance'
    :param k: column stochastic, column major, square, proposal matrix
    :type: N-D numpy.array
    :param v: distribution vector, irrelevant if it's column or row major
    :type: 1-D numpy.array
    :return: r: acceptance probabilities used by the acceptance matrix in metropols_algorithm
    :type N-D numpy.array
    """
    size = k.shape[0]
    r = np.zeros(shape=k.shape, dtype=np.float64, order='F')
    for i in range(size):
        for j in range(size):
            r[i, j] = v[i] * k[j, i] / v[j] * k[i, j]
    return r


def _construct_m_matrix(k, f):
    """
    Constructs a transition matrix according to Behcet Acikmese & David S. Bayard in their paper:
    'A Markov Chain Approach to Probabilistic Swarm Guidance'
    :param k: column stochastic, column major, square, proposal matrix
    :type: N-D numpy.array
    :param f: column stochastic, column major, square, acceptance matrix
    :type: N-D numpy.array
    :return: m: transition matrix used by each node for probabilistic swarm guidance
    :type N-D numpy.array
    """
    size = k.shape[0]
    m = np.zeros(shape=k.shape, dtype=np.float64, order='F')
    for i in range(size):
        for j in range(size):
            if i != j:
                m[i, j] = k[i, j] * f[i, j]
            else:
                m[i, j] = k[j, j] + _mh_weighted_sum(k, f, j)
    return m


def _mh_weighted_sum(k, f, j):
    """
    Performs summation of the m-h branch when indices of m[i, j] are the same, i.e.: when i=j
    :param k: column stochastic, column major, square, proposal matrix
    :type: N-D numpy.array
    :param f: column stochastic, column major, square, acceptance matrix
    :type: N-D numpy.array
    :return: result
    :type float
    """
    size = k.shape[0]
    result = 0
    for _ in range(size):
        if _ == j:
            continue
        result += (1 - f[_, j]) * k[_, j]


# endregion
