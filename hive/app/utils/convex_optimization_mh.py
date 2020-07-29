import os
import matlab as m
import matlab.engine as me
import cvxpy as cvx
import numpy as np

from typing import List, Tuple, Optional

# region Functions Under Testing


def __adjency_matrix_sdp_optimization(a: List[List[int]]) -> Optional[np.ndarray]:
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
    n: int = len(a)
    adj_matrix: np.ndarray = np.asarray(a)
    ones_vector: np.ndarray = np.ones(n)  # np.ones((3,1)) shape is (3, 1)... whereas np.ones(n) shape is (3,), the latter is closer to cvxpy representation of vector
    ones_matrix: np.ndarray = np.ones((n, n))
    zeros_matrix: np.ndarray = np.zeros((n, n))
    U: np.ndarray = np.ones((n, n)) / n

    # Specificy problem variables
    Aopt: cvx.Variable = cvx.Variable((n, n), symmetric=True)
    t: cvx.Variable = cvx.Variable()
    I: np.ndarray = np.identity(n)

    # Create constraints - Python @ is Matrix Multiplication (MatLab equivalent is *), # Python * is Element-Wise Multiplication (MatLab equivalent is .*)
    constraints = [
        Aopt >= 0,  # Aopt entries must be non-negative
        (Aopt @ ones_vector) == ones_vector,  # Aopt lines are stochastics, thus all entries in a line sum to one and are necessarely smaller than one
        cvx.multiply(Aopt, ones_matrix - adj_matrix) == zeros_matrix,  # optimized matrix has no new connections. It may have less than original adjencency matrix
        (Aopt - U) >> (-t * I),  # eigenvalue lower bound, cvxpy does not accept chained constraints, e.g.: 0 <= x <= 1
        (Aopt - U) << (t * I)  # eigenvalue upper bound
    ]
    # Formulate and Solve Problem
    objective = cvx.Minimize(t)
    problem = cvx.Problem(objective, constraints)
    problem.solve(solver=cvx.MOSEK)

    return Aopt.value


def __optimal_bilevel_mh_transition_matrix(
        a: np.ndarray, v_: np.ndarray
) -> np.ndarray:
    """Constructs an optimized adjacency matrix using Semidefinite Programming.

    Unlike adjency_matrix_sdp_optimization the transition matrix uses some
    relaxation constraints, i.e, it performs global optimization thus the
    returned matrix may not correspond to the best solution, but has a high
    probability of being close to optimal.

    Note:
        1. Solutions with more than 10 constraints are likely to be bad.
        2. This function only works if you have a valid MOSEK license.
        3. The input Matrix hould have no transient states/absorbent nodes, but
        this is not enforced or verified.

    Args:
        a:
            A symmetric unoptimized adjency matrix.
        v_:
            A stochastic vector that represents a steady state distribution.

    Returns:
        A transition Markov Matrix for the desired, possibly non-uniform,
        distribution vector v_ and respective mixing rate.
    """
    # Allocate python variables
    n: int = a.shape[0]
    ones_vector: np.ndarray = np.ones(n)  # np.ones((3,1)) shape is (3, 1)... whereas np.ones(n) shape is (3,), the latter is closer to cvxpy representation of vector
    ones_matrix: np.ndarray = np.ones((n, n))
    zeros_matrix: np.ndarray = np.zeros((n, n))
    U: np.ndarray = np.ones((n, n)) / n

    # Specificy problem variables
    Topt: cvx.Variable = cvx.Variable((n, n))

    # Create constraints - Python @ is Matrix Multiplication (MatLab equivalent is *), # Python * is Element-Wise Multiplication (MatLab equivalent is .*)
    constraints = [
        Topt >= 0,  # Aopt entries must be non-negative
        (Topt @ ones_vector) == ones_vector,  # Aopt lines are stochastics, thus all entries in a line sum to one and are necessarely smaller than one
        cvx.multiply(Topt, ones_matrix - a) == zeros_matrix,  # optimized matrix has no new connections. It may have less than original adjencency matrix
        (v_ @ Topt) == v_,  # Resulting matrix must be a markov matrix.
    ]

    # Formulate and Solve Problem
    objective = cvx.Minimize(cvx.norm(Topt - U, 2))
    problem = cvx.Problem(objective, constraints)
    problem.solve(solver=cvx.MOSEK)

    return Topt.value


def __metropolis_hastings(a: np.ndarray, v_: np.ndarray) -> np.ndarray:
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

    Returns:
        An unlabeled transition matrix with steady state v_.
    """

    shape: Tuple[int, int] = a.shape
    size: int = a.shape[0]

    rw: np.ndarray = np.zeros(shape=shape)
    for i in range(size):
        degree: np.int32 = np.sum(a[i, :])  # all possible states reachable from state i, including self
        for j in range(size):
            rw[i, j] = a[i, j] / degree

    r = np.zeros(shape=shape)
    with np.errstate(divide='ignore', invalid='ignore'):
        for i in range(size):
            for j in range(size):
                r[i, j] = (v_[j] * rw[j, i]) / (v_[i] * rw[i, j])

    transition_matrix: np.ndarray = np.zeros(shape=shape)

    for i in range(size):
        for j in range(size):
            if i != j:
                transition_matrix[i, j] = rw[i, j] * min(1, r[i, j])
        # after defining all p[i, j] we can safely defined p[i, i], i.e.: define p[i, j] when i = j
        transition_matrix[i, i] = __get_diagonal_entry_probability(rw, r, i)

    return transition_matrix.transpose()


def __get_diagonal_entry_probability(rw: np.ndarray, r: np.ndarray, i: int) -> np.int32:
    """ Helper function used by _metropolis_hastings function.

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

# region Main
def first_method(A: np.ndarray, v_: np.ndarray, U: np.ndarray) -> None:
    markov_matrix = __metropolis_hastings(A, v_)
    eigenvalues, eigenvectors = np.linalg.eig(markov_matrix - U)
    mixing_rate = np.max(np.abs(eigenvalues))
    print(f"Pure Metropolis-Hastings generation...\nMixing rate: {mixing_rate}\nResulting Markov Matrix is: \n{markov_matrix}")


def second_method(A: np.ndarray, v_: np.ndarray, U: np.ndarray) -> None:
    adj_matrix_optimized = __adjency_matrix_sdp_optimization(A)
    markov_matrix = __metropolis_hastings(adj_matrix_optimized, v_)
    eigenvalues, eigenvectors = np.linalg.eig(markov_matrix - U)
    mixing_rate = np.max(np.abs(eigenvalues))
    print(f"SDP Optimization before Metropolis-Hastings generation...\nMixing rate: {mixing_rate}\nResulting Markov Matrix is: \n{markov_matrix}")


def third_method(A: np.ndarray, v_: np.ndarray) -> None:
    matlab_scripts_dir = os.path.join(os.path.abspath(os.path.join(os.getcwd(), '..', 'scripts', 'matlabscripts')))
    try:
        eng = me.start_matlab()
        eng.cd(matlab_scripts_dir)
        mA = m.double(A.tolist())
        mv_ = m.double(v_.tolist())
        markov_matrix, mixing_rate = eng.matrixGlobalOpt(mA, mv_, nargout=2)
        # noinspection PyProtectedMember
        markov_matrix = np.array(markov_matrix._data).reshape(markov_matrix.size, order='F')
        print(f"Global Optimization generation with MatLab...\nMixing rate: {mixing_rate}\nResulting Markov Matrix is: \n{markov_matrix}")
    except me.EngineError as error:
        print(str(error))


def main() -> None:
    n = 4
    v_ = np.asarray([0.1, 0.3, 0.4, 0.2])
    A = np.asarray([[1, 0, 1, 0],
                    [0, 1, 1, 1],
                    [1, 1, 1, 0],
                    [0, 1, 0, 1]])
    third_method(A, v_)


if __name__ == "__main__":
    print(f"installed solvers: {cvx.installed_solvers()}")
    main()

# endregion
