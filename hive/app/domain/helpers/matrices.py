"""This module is used by :py:class:`~domain.cluster_groups.Hive to create transition matrices for the simulation.

You should implement your own metropolis-hastings or alternative algorithms
as well as any steady-state or transition matrix optimization algorithms in
this module.
"""

from typing import Tuple, Union, Any, Optional

import cvxpy as cvx
import numpy as np

from domain.exceptions import DistributionShapeError
from domain.exceptions import MatrixNotSquareError
from domain.helpers.matlab_utils import MatlabEngineContainer

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
    return t, get_markov_matrix_fast_mixing_rate(t)


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
    problem, a = _adjency_matrix_sdp_optimization(a)
    if problem.status in OPTIMAL_STATUS:
        t = _metropolis_hastings(a.value, v_)
        return t, get_markov_matrix_fast_mixing_rate(t)
    else:
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
    objective = cvx.Minimize(cvx.norm(t - u, 2))
    problem = cvx.Problem(objective, constraints)
    problem.solve()

    if problem.status in OPTIMAL_STATUS:
        return t.value.transpose(), get_markov_matrix_fast_mixing_rate(t.value)

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
    result = matlab_container.matrix_global_opt(a, v_)
    if result:
        t = np.array(result._data).reshape(result.size, order='F').T
        return t, get_markov_matrix_fast_mixing_rate(t)
    return None, float('inf')
# endregion


# region Optimization

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


# region Metropolis Hastings Impl.

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

    if column_major_in:
        a = a.transpose()

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
    shape = a.shape
    size = shape[0]
    rw: np.ndarray = np.zeros(shape=shape)
    for i in range(size):
        degree: Any = np.sum(a[i, :])  # all possible states reachable from state i, including self
        for j in range(size):
            rw[i, j] = a[i, j] / degree
    return rw


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

def get_markov_matrix_fast_mixing_rate(m: np.ndarray) -> float:
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
        raise MatrixNotSquareError("get_max_eigenvalue can't compute eigenvalues/vectors with non-square matrix input.")

    eigenvalues, eigenvectors = np.linalg.eig(m - np.ones((size, size)) / size)
    mixing_rate = np.max(np.abs(eigenvalues))
    return mixing_rate.item()

# endregion
