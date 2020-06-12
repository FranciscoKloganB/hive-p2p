import os
import matlab
import matlab.engine
import cvxpy as cvx
import numpy as np

from typing import List, Tuple


# region Semidefinite Programming Optimization

def adjency_matrix_sdp_optimization(a: List[List[int]]) -> np.ndarray:
    """
    Constructs an optimized adjacency matrix
    :param List[List[int]] a: any symmetric adjacency matrix. Matrix a should have no transient states/absorbent nodes, but this is not enforced or verified.
    :returns  List[List[int]] adj_matrix_optimized: an optimized adjacency matrix for the uniform distribution vector u, whose entries have value 1/n, where n is shape of a.
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

# endregion


# region Global Optimizattion

def optimal_bilevel_mh_transition_matrix(A: np.ndarray, v_: np.ndarray) -> np.ndarray:
    """
    Constructs a transition matrix using linear programming relaxations and convex envelope approximations for the desired distribution vector.
    Result is only trully optimal if cvx.norm(Topt - U, 2) is equal to the markov matrix eigenvalue.
    :param np.ndarray A: Symmetric unoptimized adjency matrix.
    :param np.ndarray v_: a stochastic desired distribution vector
    :returns Tuple[np.ndarray, float] (T, mrate): Transition Markov Matrix for the desired, possibly non-uniform, distribution vector ddv and respective mixing rate
    """
    # Allocate python variables
    n: int = A.shape[0]
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
        cvx.multiply(Topt, ones_matrix - A) == zeros_matrix,  # optimized matrix has no new connections. It may have less than original adjencency matrix
        (v_ @ Topt) == v_,  # Resulting matrix must be a markov matrix.
    ]

    # Formulate and Solve Problem
    objective = cvx.Minimize(cvx.norm(Topt - U, 2))
    problem = cvx.Problem(objective, constraints)
    problem.solve(solver=cvx.MOSEK)

    return Topt.value

# endregion


# region Helpers

def __metropolis_hastings(A: np.ndarray, v_: np.ndarray) -> np.ndarray:
    shape: Tuple[int, int] = A.shape
    size: int = A.shape[0]

    rw: np.ndarray = np.zeros(shape=shape)
    for i in range(size):
        degree: np.int32 = np.sum(A[i, :])  # all possible states reachable from state i, including self
        for j in range(size):
            rw[i, j] = A[i, j] / degree

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
        transition_matrix[i, i] = __mh_summation(rw, r, i)

    return transition_matrix.transpose()


def __mh_summation(rw: np.ndarray, r: np.ndarray, i: int) -> np.int32:
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


# region Main
def first_method(A: np.ndarray, v_: np.ndarray, U: np.ndarray) -> None:
    markov_matrix = __metropolis_hastings(A, v_)
    eigenvalues, eigenvectors = np.linalg.eig(markov_matrix - U)
    mixing_rate = np.max(np.abs(eigenvalues))
    print(f"Pure Metropolis-Hastings generation...\nMixing rate: {mixing_rate}\nResulting Markov Matrix is: \n{markov_matrix}")


def second_method(A: np.ndarray, v_: np.ndarray, U: np.ndarray) -> None:
    adj_matrix_optimized = adjency_matrix_sdp_optimization(A)
    markov_matrix = __metropolis_hastings(adj_matrix_optimized, v_)
    eigenvalues, eigenvectors = np.linalg.eig(markov_matrix - U)
    mixing_rate = np.max(np.abs(eigenvalues))
    print(f"SDP Optimization before Metropolis-Hastings generation...\nMixing rate: {mixing_rate}\nResulting Markov Matrix is: \n{markov_matrix}")


def third_method(A: np.ndarray, v_: np.ndarray, U: np.ndarray) -> None:
    markov_matrix = optimal_bilevel_mh_transition_matrix(A, v_).transpose()
    eigenvalues, eigenvectors = np.linalg.eig(markov_matrix - U)
    mixing_rate = np.max(np.abs(eigenvalues))
    print(f"Global Optimization generation...\nMixing rate: {mixing_rate}\nResulting Markov Matrix is: \n{markov_matrix}")


def third_method_as_matlab(A: np.ndarray, v_: np.ndarray):
    matlab_scripts_dir = os.path.join(os.path.abspath(os.path.join(os.getcwd(), '..', 'scripts', 'matlabscripts')))
    try:
        eng = matlab.engine.start_matlab()
        eng.cd(matlab_scripts_dir)
        matlab_A = matlab.double(A.tolist())
        matlab_v_ = matlab.double(v_.tolist())
        Topt, mr = eng.matrixGlobalOpt(matlab_A, matlab_v_, nargout=2)
        print(f"Type: {type(mr)}, Mixing Rate: {mr}:\nType: {type(Topt)}, Topt:\n{Topt}")
    except matlab.engine.EngineError as error:
        print(str(error))


def main() -> None:
    n = 4
    v_ = np.asarray([0.1, 0.3, 0.4, 0.2])
    A = np.asarray([[1, 0, 1, 0], [0, 1, 1, 1], [1, 1, 1, 0], [0, 1, 0, 1]])
    # n = 8
    # v_ = np.asarray([0.13211647, 0.23120382, 0.03172534, 0.16644937, 0.26249457, 0.09474142, 0.04476315, 0.03650587])
    # A = np.asarray([[0, 1, 1, 1, 0, 1, 1, 0],
    #                 [1, 1, 1, 0, 0, 0, 1, 0],
    #                 [1, 1, 0, 1, 0, 1, 0, 1],
    #                 [1, 0, 1, 0, 0, 0, 0, 1],
    #                 [0, 0, 0, 0, 1, 0, 0, 1],
    #                 [1, 0, 1, 0, 0, 0, 1, 1],
    #                 [1, 1, 0, 0, 0, 1, 0, 1],
    #                 [0, 0, 1, 1, 1, 1, 1, 0]])
    U = np.ones((n, n)) / n
    # first_method(A, v_, U)
    # print("\n########\n")
    # second_method(A, v_, U)
    # print("\n########\n")
    third_method(A, v_, U)
    print("\n########\n")
    third_method_as_matlab(A, v_)


if __name__ == "__main__":
    # print(cvx.installed_solvers())
    main()

# endregion
