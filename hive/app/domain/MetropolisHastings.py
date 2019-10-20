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


# region lame unit testing
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


def construct_random_walk_test():
    target = np.asarray([[0.25, 0.25, 0.25, 0.25], [0.5, 0, 0.5, 0], [0.25, 0.25, 0.25, 0.25], [0, 0.5, 0.5, 0]])
    adj_matrix = np.asarray([[1, 1, 1, 1], [1, 0, 1, 0], [1, 1, 1, 1], [0, 1, 1, 0]])
    random_walk = _construct_random_walk_matrix(adj_matrix, adj_matrix.shape, adj_matrix.shape[0])
    print("matrix_converges_to_known_ddv_test")
    print("expect:\n{}".format(target))
    print("got:\n{}".format(random_walk))
    print("accept:{}\n\n".format(np.array_equal(target, random_walk)))


def construct_rejection_matrix_div_by_zero_error_exist_test():
    try:
        ddv = [0.1, 0.4, 0.3, 0.2]
        adj_matrix = np.asarray([[1, 1, 1, 1], [1, 0, 1, 0], [1, 1, 1, 1], [0, 1, 1, 0]])
        random_walk = _construct_random_walk_matrix(adj_matrix, adj_matrix.shape, adj_matrix.shape[0])
        rejection_matrix = _construct_rejection_matrix(ddv, random_walk, adj_matrix.shape, adj_matrix.shape[0])
        print("got:\n{}".format(rejection_matrix))
        print("accept:{}\n\n".format(True))
    except ZeroDivisionError:
        print("accept:{}\n\n".format(False))


def arbitrary_matrix_converges_to_ddv():
    target = [0.2, 0.3, 0.5, 0.0]
    adj = np.asarray([[1, 1, 0, 0], [1, 1, 1, 1], [0, 1, 1, 1], [0, 1, 1, 1]])
    mh = metropolis_algorithm(adj, ddv, column_major_in=False, column_major_out=True)
    mh_pow = np.linalg.matrix_power(mh, 1000)
    print("metropols_algorithm_test")
    print("expect:\n{}".format(target))
    print("got:\n{}\nfrom:\n{}".format(mh_pow[:, 0], mh_pow))
    print("accept:{}\n\n".format(np.allclose(target, mh_pow[:, 0])))


if __name__ == "__main__":
    np.set_printoptions(threshold=sys.maxsize, precision=5)
    # matrix_column_select_test()
    # linalg_matrix_power_test()
    # matrix_converges_to_known_ddv_test()
    # construct_random_walk_test()
    # construct_rejection_matrix_div_by_zero_error_exist_test()
    arbitrary_matrix_converges_to_ddv()
    """
    ddv = [0.2, 0.3, 0.5, 0.0]
    adj = np.asarray([[1, 1, 0, 0], [1, 1, 1, 1], [0, 1, 1, 1], [0, 1, 1, 1]])
    rw = _construct_random_walk_matrix(adj, adj.shape, adj.shape[0])
    print(rw)
    """

# endregion lame unit testing
