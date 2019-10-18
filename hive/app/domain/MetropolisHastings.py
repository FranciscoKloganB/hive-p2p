import sys
import logging
import numpy as np

from domain.exceptions.AlphaError import AlphaError
from domain.exceptions.MatrixNotSquareError import MatrixNotSquareError
from domain.exceptions.DistributionShapeError import DistributionShapeError


# region module public functions
def metropolis_algorithm(k, v, major='r', f_alpha=0.8):
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
        v = np.asarray(v)
        k = np.asarray(k)
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

    # Ensure that proposal matrix is column major
    if major == 'r':
        k = k.transpose()

    # size = k.shape[0]
    # smallest_positive = np.nextafter(0, 1)
    # for i in range(size):
    #    for j in range(size):
    #        if k[i, j] <= 0:
    #            k[i, j] = smallest_positive

    f = _construct_f_matrix(f_alpha, k, v)
    # Construct the transition matrix that will be returned to the user
    m = _construct_m_matrix(k, f)

    return m
# endregion


# region module private functions
def _construct_f_matrix(a, k, v):
    """
    Constructs a acceptance matrix for a proposal matrix according to Behcet Acikmese & David S. Bayard in their paper:
    'A Markov Chain Approach to Probabilistic Swarm Guidance'
    F must satisfy the following conditions to be valid:
    1. 0 <= F[i,j] <= min(1, R[i,j])
    2. F[j,i] = (1/R[i,j]) * F[i,j]
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
    f = np.zeros(shape=k.shape)
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
    # TODO fix div by zero errors not handled in papers (our fix is naive)
    smallest_positive = np.nextafter(0, 1)

    size = k.shape[0]
    r = np.zeros(shape=k.shape)
    for i in range(size):
        for j in range(size):
            try:
                r[i, j] = v[i] * k[j, i] / v[j] * k[i, j]
            except ZeroDivisionError as zde:
                r[i, j] = smallest_positive
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
    m = np.zeros(shape=k.shape)
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
    result = 0.0
    for _ in range(size):
        if _ == j:
            continue
        result += (1 - f[_, j]) * k[_, j]
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


