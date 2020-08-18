"""Module used by :py:class:`~domain.cluster_groups.Cluster to create transition
matrices for the simulation.

You should implement your own metropolis-hastings or alternative algorithms
as well as any steady-state or transition matrix optimization algorithms in
this module.
"""

from typing import Tuple, Any, Optional

import random
import numpy as np
import cvxpy as cvx

from matlab.engine import EngineError
from scipy.sparse.csgraph import connected_components

from domain.helpers.matlab_utils import MatlabEngineContainer
from domain.helpers.exceptions import *
from utils.randoms import random_index

OPTIMAL_STATUS = {cvx.OPTIMAL, cvx.OPTIMAL_INACCURATE}


# region Markov Matrix Constructors
def new_mh_transition_matrix(
        a: np.ndarray, v_: np.ndarray) -> Tuple[np.ndarray, float]:
    """ Constructs a transition matrix using metropolis-hastings.

    Constructs a transition matrix using metropolis-hastings algorithm  for
    the specified steady state `v`.

    Note:
        The input Matrix hould have no transient states or absorbent nodes,
        but this is not enforced or verified.

    Args:
        a:
            A symmetric adjency matrix.
        v_:
            A stochastic steady state distribution vector.

    Returns:
        Markov Matrix with `v_` as steady state distribution and the
        respective mixing rate.
    """
    t = _metropolis_hastings(a, v_)
    return t, get_mixing_rate(t)


def new_sdp_mh_transition_matrix(
        a: np.ndarray, v_: np.ndarray) -> Tuple[Optional[np.ndarray], float]:
    """Constructs an optimized transition matrix using cvxpy and MOSEK solver.

    Constructs a transition matrix using metropolis-hastings algorithm  for
    the specified steady state `v`. The provided adjacency matrix A is first
    optimized with semi-definite programming techniques for the uniform
    distribution vector.

    Note:
        This function only works if you have a valid MOSEK license.

    Args:
        a:
            A non-optimized symmetric adjency matrix.
        v_:
            A stochastic steady state distribution vector.

    Returns:
        Markov Matrix with `v_` as steady state distribution and the
        respective mixing rate.
    """
    try:
        problem, a = _adjency_matrix_sdp_optimization(a)
        if problem.status in OPTIMAL_STATUS:
            t = _metropolis_hastings(a.value, v_)
            return t, get_mixing_rate(t)
        else:
            return None, float('inf')
    except (cvx.SolverError, cvx.DCPError):
        return None, float('inf')


def new_go_transition_matrix(
        a: np.ndarray, v_: np.ndarray) -> Tuple[Optional[np.ndarray], float]:
    """Constructs an optimized transition matrix using cvxpy and MOSEK solver.

    Constructs an optimized markov matrix using linear programming relaxations
    and convex envelope approximations for the specified steady state `v`.
    Result is only trully optimal if normal(Tranistion Matrix Opt - Uniform
    Matrix, 2) is equal to the markov matrix eigenvalue.

    Note:
        This function only works if you have a valid MOSEK license.

    Args:
        a:
            A non-optimized symmetric adjency matrix.
        v_:
            A stochastic steady state distribution vector.

    Returns:
        Markov Matrix with `v_` as steady state distribution and the
        respective mixing rate.
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
    try:
        objective = cvx.Minimize(cvx.norm(t - u, 2))
        problem = cvx.Problem(objective, constraints)
        problem.solve()

        if problem.status in OPTIMAL_STATUS:
            return t.value.transpose(), get_mixing_rate(t.value)
        else:
            return None, float('inf')
    except (cvx.SolverError, cvx.DCPError):
        return None, float('inf')


def new_mgo_transition_matrix(
        a: np.ndarray, v_: np.ndarray) -> Tuple[Optional[np.ndarray], float]:
    """Constructs an optimized transition matrix using the matlab engine.

    Constructs an optimized transition matrix using linear programming
    relaxations and convex envelope approximations for the specified steady
    state `v`. Result is only trully optimal if normal(Tranistion Matrix Opt
    - Uniform Matrix, 2) is equal to the markov matrix eigenvalue. The code
    is run on a matlab engine because it provides a non-convex SDP solver ç
    BMIBNB.

    Note:
        This function can only be invoked if you have a valid matlab license.

    Args:
        a:
            A non-optimized symmetric adjency matrix.
        v_:
            A stochastic steady state distribution vector.

    Returns:
        Markov Matrix with `v_` as steady state distribution and the
        respective mixing rate.
    """
    matlab_container = MatlabEngineContainer.get_instance()
    try:
        result = matlab_container.matrix_global_opt(a, v_)
        if result:
            t = np.array(result._data).reshape(result.size, order='F').T
            return t, get_mixing_rate(t)
        else:
            return None, float('inf')
    except EngineError:
        return None, float('inf')
# endregion


# region SDP Optimization
def _adjency_matrix_sdp_optimization(
        a: np.ndarray) -> Optional[Tuple[cvx.Problem, cvx.Variable]]:
    """Optimizes a symmetric adjacency matrix using Semidefinite Programming.

    The optimization is done with respect to the uniform stochastic vector
    with the the same length as the inputed symmetric matrix.

    Note:
        1. This function only works if you have a valid MOSEK license.
        2. The input Matrix hould have no transient states/absorbent nodes, but
        this is not enforced or verified.

    Args:
        a:
            Any symmetric adjacency matrix.

    Returns:
        The optimal matrix or None if the problem is unfeasible.
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


# region Metropolis Hastings
def _metropolis_hastings(a: np.ndarray,
                         v_: np.ndarray,
                         column_major_in: bool = False,
                         column_major_out: bool = True) -> np.ndarray:
    """ Constructs a transition matrix using metropolis-hastings algorithm.

    Note:
        The input Matrix hould have no transient states/absorbent nodes,
        but this is not enforced or verified.

    Args:
        a:
            A symmetric adjency matrix.
        v_:
            A stochastic vector that is the steady state of the resulting
            transition matrix.
        column_major_in:
            optional; Indicates whether adj_matrix given in input is in row
            or column major form.
        column_major_out:
            optional; Indicates whether to return transition_matrix output
            is in row or column major form.

    Returns:
        An unlabeled transition matrix with steady state v_.

    Raises:
        DistributionShapeError:
            When the length of v_ is not the same as the matrix a.
        MatrixNotSquareError:
            When matrix a is not a square matrix.
    """
    # Input checking
    if v_.shape[0] != a.shape[1]:
        raise DistributionShapeError(
            "distribution shape: {}, proposal matrix shape: {}".format(
                v_.shape, a.shape))
    if a.shape[0] != a.shape[1]:
        raise MatrixNotSquareError(
            "rows: {}, columns: {}, expected square matrix".format(
                a.shape[0], a.shape[1]))

    shape: Tuple[int, int] = a.shape
    size: int = a.shape[0]

    rw: np.ndarray = _construct_random_walk_matrix(a)
    r: np.ndarray = _construct_rejection_matrix(rw, v_)

    transition_matrix: np.ndarray = np.zeros(shape=shape)

    for i in range(size):
        for j in range(size):
            if i != j:
                transition_matrix[i, j] = rw[i, j] * min(1, r[i, j])
        # after defining all p[i, j] we can safely defined p[i, i], i.e.: define p[i, j] when i = j
        transition_matrix[i, i] = __get_diagonal_entry_probability(rw, r, i)

    if column_major_out:
        return transition_matrix.transpose()
    return transition_matrix


def _construct_random_walk_matrix(a: np.ndarray) -> np.ndarray:
    """Builds a random walk matrix over the given adjacency matrix

    Args:
        a:
            Any adjacency matrix.

    Returns:
        A matrix representing the performed random walk.
    """
    # Old Code
    shape = a.shape
    size = shape[0]
    rw: np.ndarray = np.zeros(shape=shape)
    for i in range(size):
        # all possible states reachable from state i, including self
        degree: Any = np.sum(a[i, :])
        for j in range(size):
            rw[i, j] = a[i, j] / degree
    return rw
    # return (a / np.sum(a, axis=0)).transpose()


def _construct_rejection_matrix(rw: np.ndarray, v_: np.array) -> np.ndarray:
    """Builds a rejection matrix for a given rejection matrix rw and vector v_.

    Args:
        v_:
            a stochastic desired distribution vector
        rw:
            a random_walk over an adjacency matrix

    Returns:
        A matrix whose entries are acceptance probabilities for the random walk.
    """
    shape = rw.shape
    size = shape[0]
    r = np.zeros(shape=shape)
    with np.errstate(divide='ignore', invalid='ignore'):
        for i in range(size):
            for j in range(size):
                r[i, j] = (v_[j] * rw[j, i]) / (v_[i] * rw[i, j])
    return r


def __get_diagonal_entry_probability(
        rw: np.ndarray, r: np.ndarray, i: int) -> np.int32:
    """Helper function used by _metropolis_hastings function.

    Calculates the value that should be assigned to the entry (i, i) of the
    transition matrix being calculated by the metropolis hastings algorithm
    by considering the rejection probability over the random walk that was
    performed on an adjacency matrix.

    Args:
        rw:
            A random_walk over an adjacency matrix.
        r:
            A matrix whose entries contain acceptance probabilities for rw.
        i:
            The diagonal-index of the random walk where summation needs to
            be performed on.

    Returns:
        A probability to be inserted at entry (i, i) of the transition matrix
        outputed by the _metropolis_hastings function.
    """
    size: int = rw.shape[0]
    pii: np.int32 = rw[i, i]
    for k in range(size):
        pii += rw[i, k] * (1 - min(1, r[i, k]))
    return pii
# endregion


# region Helpers
def get_mixing_rate(m: np.ndarray) -> float:
    """Calculats the fast mixing rate the input matrix.

    The fast mixing rate of matrix M is the highest eigenvalue that is
    smaller than one. If returned value is 1.0 than the matrix has transient
    states or absorbent nodes.

    Args:
        m:
            A matrix.

    Returns:
        The highest eigenvalue of `m` that is smaller than one or one.
    """
    size = m.shape[0]

    if size != m.shape[1]:
        raise MatrixNotSquareError(
            "Can not compute eigenvalues/vectors with non-square matrix")
    m = m - (np.ones((size, size)) / size)
    eigenvalues, eigenvectors = np.linalg.eig(m)
    mixing_rate = np.max(np.abs(eigenvalues))
    return mixing_rate.item()


def new_symmetric_matrix(
        size: int, allow_sloops: bool = True, force_sloops: bool = True
) -> np.ndarray:
    """Generates a random symmetric matrix.

     The generated adjacency matrix does not have transient state sets or
     absorbent nodes and can effectively represent a network topology
     with bidirectional connections between network nodes.

     Args:
         size:
            The length of the square matrix.
         allow_sloops:
            Indicates if the generated adjacency matrix allows diagonal
            entries representing self-loops. If false, then, all diagonal
                entries must be zeros. Otherwise, they can be zeros or ones (
                default is True).
         force_sloops:
            Indicates if the diagonal of the generated matrix should be
            filled with ones. If false, valid diagonal entries are decided by
            `allow_self_loops` param. Otherwise, diagonal entries are filled
            with ones. If `allow_self_loops` is False and `enforce_loops` is
            True, an error is raised (default is True).

    Returns:
        The adjency matrix representing the connections between a
        groups of network nodes.

    Raises:
        IllegalArgumentError:
            When `allow_self_loops` (False) conflicts with
            `enforce_loops` (True).
    """
    if not allow_sloops and force_sloops:
        raise IllegalArgumentError("Can not invoke new_symmetric_matrix with:\n"
                                   "    [x] allow_sloops=False\n"
                                   "    [x] force_sloops=True")
    secure_random = random.SystemRandom()
    m = np.zeros((size, size))
    for i in range(size):
        for j in range(i, size):
            if i == j:
                if not allow_sloops:
                    m[i, i] = 0
                elif force_sloops:
                    m[i, i] = 1
                else:
                    m[i, i] = __new_edge_val__(secure_random)
            else:
                m[i, j] = m[j, i] = __new_edge_val__(secure_random)
    return m


def new_symmetric_connected_matrix(
        size: int, allow_sloops: bool = True, force_sloops: bool = True
) -> np.ndarray:
    """Generates a random symmetric matrix which is also connected.

    See :py:func:`~domain.helpers.matrices.new_symmetric_matrix` and
    py:func:`~domain.helpers.matrices.make_connected`.

     Args:
         size:
            The length of the square matrix.
         allow_sloops:
            See :py:func:`~domain.helpers.matrices.new_symmetric_matrix`.
         force_sloops:
            See :py:func:`~domain.helpers.matrices.new_symmetric_matrix`.

    Returns:
        A matrix that represents an adjacency matrix that is also connected.
    """
    m = np.asarray(new_symmetric_matrix(size))
    if not is_connected(m):
        m = make_connected(m)
    return m


def make_connected(m: np.ndarray) -> np.ndarray:
    """Turns a matrix into a connected matrix that could represent a
    connected graph.

    Args:
        m: The matrix to be made connected.

    Returns:
        A connected matrix. If the inputed matrix was connected it will
        remain so.
    """
    size = m.shape[0]
    # Use guilty until proven innocent approach for both checks
    for i in range(size):
        is_absorbent_or_transient: bool = True
        for j in range(size):
            # Ensure state i can reach and be reached by some other state j
            if m[i, j] == 1 and i != j:
                is_absorbent_or_transient = False
                break
        if is_absorbent_or_transient:
            j = random_index(i, size)
            m[i, j] = m[j, i] = 1
    return m


def is_symmetric(m: np.ndarray, tol: float = 1e-8) -> bool:
    """Checks if a matrix is symmetric by comparing entries of a and a.T.

    Args:
        m:
            The matrix to be verified.
        tol:
            The tolerance used to verify the entries of the matrix (default
            is 1e-8).

    Returns:
        True if the matrix is symmetric, else False.
    """
    return np.all(np.abs(m - m.transpose()) < tol)


def is_connected(m: np.ndarray, directed: bool = False) -> bool:
    """Checks if a matrix is connected by counting the number of connected
    components.

    Args:
        m:
            The matrix to be verified.
        directed:
            If the matrix edges are directed, i.e., if the matrix is an adjency
            matrix are the edges bidirectional, where false means they are (
            default is false).

    Returns:
        True if the matrix is a connected graph, else False.
    """
    n, cc_labels = connected_components(m, directed=directed)
    return n == 1


def __new_edge_val__(random_generator: random.SystemRandom) -> np.float64:
    p = random_generator.uniform(0.0, 1.0)
    return np.ceil(p) if p >= 0.5 else np.floor(p)
# endregion
