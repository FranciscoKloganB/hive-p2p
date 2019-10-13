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

    if f_alpha <= 0 or f_alpha > 1:
        raise AlphaError("(0, 1]")
    if v.shape[0] != k.shape[1]:
        raise DistributionShapeError("distribution shape: {}, proposal matrix shape: {}".format(v.shape, k.shape))
    if k.shape[0] != k.shape[1]:
        raise MatrixNotSquareError("rows: {}, columns: {}, expected square matrix".format(k.shape[0], k.shape[1]))

    if major == 'r':
        k.transpose()

    # Construct the acceptance_matrix according to Behcet Acikmese and David S. Bayard in their MCAtPSW paper
    acceptance_matrix = _construct_f_matrix(f_alpha, k, v)
    # Construct the transition_matrix that will be returned to the user
    transition_matrix = np.zeros(shape=k.shape, dtype=np.float64, order='F')

    return None
# endregion


# region module private functions
def _construct_f_matrix(a, k, v):
    """
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
            f[i][j] = a * min(1, r[i][j])
    return f


def _construct_r_matrix(k, v):
    """
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
            r[i][j] = v[i] * k[j][i] / v[j] * k[i][j]
    return r
# endregion
