import cvxpy as cvx
import numpy as np

from typing import List, Tuple


def adjency_matrix_sdp_optimization(a: List[List[int]]) -> Tuple[float, np.ndarray]:
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

    return Aopt.value, problem.value


def optimal_bilevel_mh_transition_matrix(A: np.ndarray, v_: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Constructs a transition matrix using linear programming relaxations and convex envelope approximations for the desired distribution vector.
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
    Topt: cvx.Variable = cvx.Variable((n, n), symmetric=False)

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

    return Topt.value, problem.value


if __name__ == "__main__":
    Aunopt = np.asarray([[1, 0, 1, 0], [0, 1, 1, 1], [1, 1, 1, 0], [0, 1, 0, 1]])
    desired_distribution_ = np.asarray([0.1, 0.3, 0.4, 0.2])
    adj_matrix_optimized, eigenvalue = adjency_matrix_sdp_optimization(Aunopt)
    # markov_matrix = _metropolis_hastings(adj_matrix_optimized, desired_distribution_)
    print("Using Semidefinite Programming techniques...")
    print(f"The optimal eigenvalue is: {eigenvalue}")
    print(f"Aopt solution is: \n{adj_matrix_optimized}")
    # print(f"Resulting Markov Matrix is: \n{markov_matrix}")
    print("\n########\n########\n########\n########\n")
    print("Using Global Optimization techniques...")
    markov_matrix, eigenvalue = optimal_bilevel_mh_transition_matrix(Aunopt, desired_distribution_)
    print(f"The optimal eigenvalue is: {eigenvalue}")
    print(f"Resulting Markov Matrix is: \n{markov_matrix.transpose()}")

