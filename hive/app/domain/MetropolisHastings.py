import sys
import logging
import numpy as np

from domain.exceptions.AlphaError import AlphaError
from domain.exceptions.MatrixNotSquareError import MatrixNotSquareError
from domain.exceptions.DistributionShapeError import DistributionShapeError


# region module public functions
def metropolis_algorithm(q, r, major='r', f_alpha=0.8):
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
def _construct_acceptance_matrix(q, r):
    """
    Constructs a acceptance matrix for a proposal matrix according to Behcet Acikmese & David S. Bayard in their paper:
    'A Markov Chain Approach to Probabilistic Swarm Guidance'
    :param q: column stochastic, column major, square, proposal matrix
    :type: N-D numpy.array
    :param r: distribution vector, irrelevant if it's column or row major
    :type: 1-D numpy.array
    :return:
    """
    size = q.shape[0]
    f = np.zeros(shape=q.shape)
    for i in range(size):
        for j in range(size):
            f[i, j] = min(1, (r[i] * q[i, i]) / (r[i]) * q[i, j])
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
def matrix_column_select_test():
    target = np.asarray([0.3, 0.2, 0.5])
    k = np.asarray([[0.3, 0.2, 0.5], [0.1, 0.2, 0.7], [0.2, 0.2, 0.6]]).transpose()
    print("matrix_column_select_test")
    print("expect:\n{}".format(str([0.3, 0.2, 0.5])))
    print("got:\n{}".format(k[:, 0]))
    print("accept: {}\n\n".format(np.array_equal(target, k[:, 0])))


def linalg_matrix_power_test():
    target = np.asarray([[0.201, 0.2, 0.599], [0.199, 0.2, 0.601], [0.2, 0.2, 0.6]]).transpose()
    kn = np.linalg.matrix_power(np.asarray([[0.3, 0.2, 0.5], [0.1, 0.2, 0.7], [0.2, 0.2, 0.6]]).transpose(), 3)
    print("linalg_matrix_power_test")
    print("expect:\n{}".format(target))
    print("got:\n{}".format(kn))
    print("accept: {}\n\n".format(np.allclose(target, kn)))


def matrix_converges_to_known_ddv_test():
    target = np.asarray([0.35714286, 0.27142857, 0.37142857])
    k_ = np.linalg.matrix_power(np.asarray([[0.3, 0.4, 0.3], [0.1, 0.2, 0.7], [0.6, 0.2, 0.2]]).transpose(), 25)
    print("matrix_converges_to_known_ddv_test")
    print("expect:\n{}".format(target))
    print("got:\n{}".format(k_[:, 0]))
    print("accept:{}\n\n".format(np.allclose(target, k_[:, 0])))


def arbitrary_matrix_converges_to_ddv():
    target = np.asarray([0.35714286, 0.27142857, 0.37142857])
    k = [[0.3, 0.3, 0.4], [0.2, 0.4, 0.4], [0.25, 0.5, 0.25]]
    metropolis_result = metropolis_algorithm(k, target)
    k_ = np.linalg.matrix_power(metropolis_result, 1000)
    print("metropols_algorithm_test")
    print("expect:\n{}".format(target))
    print("got:\n{}\nfrom:\n{}".format(k_[:, 0], k_))
    print("accept:{}\n\n".format(np.allclose(target, k_[:, 0])))
# endregion lame unit testing


if __name__ == "__main__":
    np.set_printoptions(threshold=sys.maxsize, precision=5)
    # matrix_column_select_test()
    # linalg_matrix_power_test()
    # matrix_converges_to_known_ddv_test()
    arbitrary_matrix_converges_to_ddv()