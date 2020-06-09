import cvxpy as cvx
import numpy as np
import mosek

from typing import List, Tuple


def optimize_adjency_matrix(a: List[List[int]]) -> Tuple[float, np.ndarray]:
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

    return problem.value, Aopt.value


if __name__ == "__main__":
    # from utils.matrices import new_symmetric_adjency_matrix
    # optimize_adjency_matrix(new_symmetric_adjency_matrix(5))
    adj_matrix_optimized, eigenvalue = optimize_adjency_matrix([[1, 0, 1, 0], [0, 1, 1, 1], [1, 1, 1, 0], [0, 1, 0, 1]])
    print(f"The optimal eigenvalue is: {eigenvalue}")
    print(f"Aopt solution is: \n{adj_matrix_optimized}")
