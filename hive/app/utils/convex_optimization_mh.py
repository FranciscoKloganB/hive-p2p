import cvxpy as cp
import cvxopt as co

import numpy as np

from typing import List, Any


def optimize_adjency_matrix(a: List[List[int]]) -> Any:
    """
    Constructs an optimized adjacency matrix
    :param List[List[int]] a: any symmetric adjacency matrix. Matrix a should have no transient states/absorbent nodes, but this is not enforced or verified.
    :returns  List[List[int]] a_opt: an optimized adjacency matrix for the uniform distribution vector u, whose entries have value 1/n, where n is shape of a.
    """
    # Allocate python variables
    n: int = len(a)
    adj_matrix: np.ndarray = np.asarray(a)
    ones_vector: np.ndarray = np.ones(n)  # np.ones((3,1)) shape is (3, 1)... whereas np.ones(n) shape is (3,), the latter is closer to cvxpy representation of vector
    ones_matrix: np.ndarray = np.ones((n, n))
    zeros_matrix: np.ndarray = np.zeros((n, n))
    U: np.ndarray = np.ones((n, n)) / n

    # Specificy problem variables
    Aopt: cp.Variable = cp.Variable((n, n), symmetric=True)
    t: cp.Variable = cp.Variable()
    I: np.ndarray = np.identity(n)

    # Create constraints - Python @ is Matrix Multiplication (MatLab equivalent is *), # Python * is Element-Wise Multiplication (MatLab equivalent is .*)
    constraints = [
        Aopt >= 0,  # Aopt entries must be non-negative
        (Aopt @ ones_vector) == ones_vector,  # Aopt lines are stochastics, thus all entries in a line sum to one and are necessarely smaller than one
        (Aopt * (ones_matrix - adj_matrix)) == zeros_matrix,  # optimized matrix has no new connections. It may have less than original adjencency matrix
        (Aopt - U) >> (-t * I),  # eigenvalue lower bound, cvxpy does not accept chained constraints, e.g.: 0 <= x <= 1
        (Aopt - U) << (t * I)  # eigenvalue upper bound
    ]
    # Formulate and Solve Problem
    objective = cp.Minimize(t)
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.CVXOPT)

    print("The optimal value is", problem.value)
    print("A solution X is")
    print(Aopt.value)


if __name__ == "__main__":
    # from utils.matrices import new_symmetric_adjency_matrix
    # optimize_adjency_matrix(new_symmetric_adjency_matrix(5))
    optimize_adjency_matrix([[1, 0, 1, 0], [0, 1, 1, 1], [1, 1, 1, 0], [0, 1, 0, 1]])
