import matlab
import cvxpy as cvx
import numpy as np

from typing import Tuple, Union, Any

from domain.exceptions.DistributionShapeError import DistributionShapeError
from domain.exceptions.MatrixNotSquareError import MatrixNotSquareError

OPTIMAL_STATUS = {cvx.OPTIMAL, cvx.OPTIMAL_INACCURATE}


# region Markov Matrix Constructors

def new_mh_transition_matrix(a: np.ndarray, v_: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Constructs a transition matrix using metropolis-hastings algorithm for the desired distribution vector.
    :param np.ndarray a: Symmetric unoptimized adjency matrix.
    :param np.ndarray v_: a stochastic desired distribution vector
    :returns Tuple[np.ndarray, float] (T, mrate): Transition Markov Matrix for the desired, possibly non-uniform, distribution vector ddv and respective mixing rate
    """
    t = _metropolis_hastings(a, v_)
    return t, get_markov_matrix_fast_mixing_rate(t)


def new_sdp_mh_transition_matrix(a: np.ndarray, v_: np.ndarray) -> Tuple[Union[None, np.ndarray], float]:
    """
    Constructs a transition matrix using metropolis-hastings algorithm for the desired distribution vector.
    The provided adjacency matrix A is first optimized with semi-definite programming techniques for the uniform distribution vector.
    :param np.ndarray a: Symmetric unoptimized adjency matrix.
    :param np.ndarray v_: a stochastic desired distribution vector
    :returns Tuple[np.ndarray, float] (T, mrate): Transition Markov Matrix for the desired, possibly non-uniform, distribution vector ddv and respective mixing rate
    """
    problem, a = __adjency_matrix_sdp_optimization(a)
    if problem.status in OPTIMAL_STATUS:
        t = _metropolis_hastings(a.value, v_)
        return t, get_markov_matrix_fast_mixing_rate(t)
    else:
        return None, float('inf')


def new_go_transition_matrix(a: np.ndarray, v_: np.ndarray) -> Tuple[Union[None, np.ndarray], float]:
    """
    Constructs a transition matrix using linear programming relaxations and convex envelope approximations for the desired distribution vector.
    Result is only trully optimal if normal(Tranistion Matrix Opt - Uniform Matrix, 2) is equal to the markov matrix eigenvalue.
    :param np.ndarray a: Symmetric unoptimized adjency matrix.
    :param np.ndarray v_: a stochastic desired distribution vector
    :returns Tuple[np.ndarray, float] (T, mrate): Transition Markov Matrix for the desired, possibly non-uniform, distribution vector ddv and respective mixing rate
    """
    # Allocate python variables
    n: int = a.shape[0]
    ones_vector: np.ndarray = np.ones(n)  # np.ones((3,1)) shape is (3, 1)... whereas np.ones(n) shape is (3,), the latter is closer to cvxpy representation of vector
    ones_matrix: np.ndarray = np.ones((n, n))
    zeros_matrix: np.ndarray = np.zeros((n, n))
    u: np.ndarray = np.ones((n, n)) / n

    # Specificy problem variables
    t: cvx.Variable = cvx.Variable((n, n))

    # Create constraints - Python @ is Matrix Multiplication (MatLab equivalent is *), # Python * is Element-Wise Multiplication (MatLab equivalent is .*)
    constraints = [
        t >= 0,  # Aopt entries must be non-negative
        (t @ ones_vector) == ones_vector,  # Aopt lines are stochastics, thus all entries in a line sum to one and are necessarely smaller than one
        cvx.multiply(t, ones_matrix - a) == zeros_matrix,  # optimized matrix has no new connections. It may have less than original adjencency matrix
        (v_ @ t) == v_,  # Resulting matrix must be a markov matrix.
    ]

    # Formulate and Solve Problem
    objective = cvx.Minimize(cvx.norm(t - u, 2))
    problem = cvx.Problem(objective, constraints)
    problem.solve()

    if problem.status in OPTIMAL_STATUS:
        return t.value.transpose(), get_markov_matrix_fast_mixing_rate(t.value)

    return None, float('inf')


def go_with_matlab_bmibnb_solver(a: np.ndarray, v_: np.ndarray, eng: matlab.engine.MatlabEngine) -> Tuple[Union[np.ndarray, None], float]:
    """
    Constructs a transition matrix using linear programming relaxations and convex envelope approximations for the desired distribution vector.
    Result is only trully optimal if normal(Tranistion Matrix Opt - Uniform Matrix, 2) is equal to the markov matrix eigenvalue.
    The code is run on a matlab engine because it provides a non-convex SDP solver BMIBNB.
    :param np.ndarray a: Symmetric unoptimized adjency matrix.
    :param np.ndarray v_: a stochastic desired distribution vector
    :param mleng.MatlabEngine eng: an instance of a running matlab engine.
    :returns Tuple[np.ndarray, float] (T, mrate): Transition Markov Matrix for the desired, possibly non-uniform, distribution vector ddv and respective mixing rate
    """
    a = matlab.double(a.tolist())
    v = matlab.double(v_.tolist())
    result = eng.matrixGlobalOpt(a, v, nargout=1)
    if result:
        t = np.array(result._data).reshape(result.size, order='F').T
        return t, get_markov_matrix_fast_mixing_rate(t)
    return None, float('inf')
# endregion


# region Optimization

def __adjency_matrix_sdp_optimization(a: np.ndarray) -> Tuple[cvx.Problem, cvx.Variable]:
    """
    Constructs an optimized adjacency matrix
    :param np.ndarray a: Any symmetric adjacency matrix. Matrix a should have no transient states/absorbent nodes, but this is not enforced or verified.
    :returns np.ndarray a_opt.value: an optimized adjacency matrix for the uniform distribution vector u, whose entries have value 1/n, where n is shape of a.
    """
    # Allocate python variables
    n: int = a.shape[0]
    ones_vector: np.ndarray = np.ones(n)  # np.ones((3,1)) shape is (3, 1)... whereas np.ones(n) shape is (3,), the latter is closer to cvxpy representation of vector
    ones_matrix: np.ndarray = np.ones((n, n))
    zeros_matrix: np.ndarray = np.zeros((n, n))
    u: np.ndarray = np.ones((n, n)) / n

    # Specificy problem variables
    a_opt: cvx.Variable = cvx.Variable((n, n), symmetric=True)
    t: cvx.Variable = cvx.Variable()
    i: np.ndarray = np.identity(n)

    # Create constraints - Python @ is Matrix Multiplication (MatLab equivalent is *), # Python * is Element-Wise Multiplication (MatLab equivalent is .*)
    constraints = [
        a_opt >= 0,  # a_opt entries must be non-negative
        (a_opt @ ones_vector) == ones_vector,  # a_opt lines are stochastics, thus all entries in a line sum to one and are necessarely smaller than one
        cvx.multiply(a_opt, ones_matrix - a) == zeros_matrix,  # optimized matrix has no new connections. It may have less than original adjencency matrix
        (a_opt - u) >> (-t * i),  # eigenvalue lower bound, cvxpy does not accept chained constraints, e.g.: 0 <= x <= 1
        (a_opt - u) << (t * i)  # eigenvalue upper bound
    ]

    # Formulate and Solve Problem
    objective = cvx.Minimize(t)
    problem = cvx.Problem(objective, constraints)
    problem.solve(solver=cvx.MOSEK)

    return problem, a_opt

# endregion


# region Metropolis Hastings Impl.

def _metropolis_hastings(a: np.ndarray, v_: np.ndarray, column_major_in: bool = False, column_major_out: bool = True) -> np.ndarray:
    """
    Constructs a transition matrix with desired distribution as steady state
    :param np.ndarray a: any symmetric adjacency matrix
    :param np.ndarray v_: a stochastic desired distribution vector
    :param bool column_major_in: indicates whether adj_matrix given in input is in row or column major form
    :param bool column_major_out: indicates whether to return transition_matrix output is in row or column major form
    :returns np.ndarray transition_matrix: unlabeled transition matrix
    """

    # Input checking
    if v_.shape[0] != a.shape[1]:
        raise DistributionShapeError("distribution shape: {}, proposal matrix shape: {}".format(v_.shape, a.shape))
    if a.shape[0] != a.shape[1]:
        raise MatrixNotSquareError("rows: {}, columns: {}, expected square matrix".format(a.shape[0], a.shape[1]))

    if column_major_in:
        a = a.transpose()

    shape: Tuple[int, int] = a.shape
    size: int = a.shape[0]

    rw: np.ndarray = _construct_random_walk_matrix(a, shape, size)
    r: np.ndarray = _construct_rejection_matrix(v_, rw, shape, size)

    transition_matrix: np.ndarray = np.zeros(shape=shape)

    for i in range(size):
        for j in range(size):
            if i != j:
                transition_matrix[i, j] = rw[i, j] * min(1, r[i, j])
        # after defining all p[i, j] we can safely defined p[i, i], i.e.: define p[i, j] when i = j
        transition_matrix[i, i] = _mh_summation(rw, r, i)

    return transition_matrix.transpose() if column_major_out else transition_matrix


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
        degree: Any = np.sum(adj_matrix[i, :])  # all possible states reachable from state i, including self
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
    for k in range(size):
        pii += rw[i, k] * (1 - min(1, r[i, k]))
    return pii

# endregion


# region Helpers

def get_markov_matrix_fast_mixing_rate(M: np.ndarray) -> float:
    """
    Returns the expected fast mixing rate of matrix M, i.e., the highest eigenvalue that is smaller than one. If returned value is 1.0 than M has transient states or absorbent nodes.
    :param np.ndarray M: A Markov Matrix, i.e., a square stochastic transition matrix.
    :returns float mixing_rate: The highest eigenvalue of matrix M that is smaller than one.
    """
    size = M.shape[0]

    if size != M.shape[1]:
        raise MatrixNotSquareError("get_max_eigenvalue can't compute eigenvalues/vectors with non-square matrix input.")

    eigenvalues, eigenvectors = np.linalg.eig(M - np.ones((size, size)) / size)
    mixing_rate = np.max(np.abs(eigenvalues))
    return mixing_rate.item()

# endregion
