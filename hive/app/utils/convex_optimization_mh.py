import cvxpy as cp
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
    ones_vector: np.ndarray = np.ones((n, 1))
    ones_matrix: np.ndarray = np.ones((n, n))
    zeros_matrix: np.ndarray = np.zeros((n, n))
    U: np.ndarray = (1/n) * ones_matrix

    # Specificy problem variables
    Aopt: cp.Variable = cp.Variable((n, n), symmetric=True)
    t: cp.Variable = cp.Variable()
    I: np.ndarray = np.eye(n)
    # Create constraints - Python @ is Matrix Multiplication (MatLab equivalent is *), # Python * is Element-Wise Multiplication (MatLab equivalent is .*)
    constraints = [
        Aopt >= 0,  # a_opt must be a non negative matrix
        (Aopt @ ones_vector) == ones_vector,  # whose lines are stochastic
        (Aopt * (ones_matrix - adj_matrix)) == zeros_matrix,  # optimized matrix has no new connections. It may have less than original adjencency matrix
        -t * I <= Aopt - U, Aopt - U <= t * I  # define valid eigenvalues interval, cvxpy does not accept chained constraints, e.g.: 0 <= x <= 1
    ]
    # Formulate and Solve Problem
    objective = cp.Minimize(t)
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.SCS)
    print("The optimal value is", problem.value)
    print("A solution X is")
    print(Aopt.value)


if __name__ == "__main__":
    optimize_adjency_matrix([[0, 1, 1], [1, 1, 1], [1, 1, 0]])
