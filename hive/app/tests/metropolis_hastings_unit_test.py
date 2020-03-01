import sys
import numpy as np
import utils.metropolis_hastings as mH

from utils.printers import print_test, print_pow


# region unit tests
def matrix_column_select_test():
    target = np.asarray([0.3, 0.2, 0.5])
    k = np.asarray([[0.3, 0.2, 0.5], [0.1, 0.2, 0.7], [0.2, 0.2, 0.6]]).transpose()
    return print_test("matrix_column_select_test", str([0.3, 0.2, 0.5]), k[:, 0], np.array_equal(target, k[:, 0]))


def linalg_matrix_power_test():
    target = np.asarray([[0.201, 0.2, 0.599], [0.199, 0.2, 0.601], [0.2, 0.2, 0.6]]).transpose()
    kn = np.linalg.matrix_power(np.asarray([[0.3, 0.2, 0.5], [0.1, 0.2, 0.7], [0.2, 0.2, 0.6]]).transpose(), 3)
    return print_test("linalg_matrix_power_test", target, kn, np.allclose(target, kn))


def matrix_converges_to_known_ddv_test():
    target = np.asarray([0.35714286, 0.27142857, 0.37142857])
    k_ = np.linalg.matrix_power(np.asarray([[0.3, 0.4, 0.3], [0.1, 0.2, 0.7], [0.6, 0.2, 0.2]]).transpose(), 25)
    return print_test("matrix_converges_to_known_ddv_test", target, k_[:, 0], np.allclose(target, k_[:, 0]))


# noinspection PyProtectedMember
def construct_random_walk_test():
    target = np.asarray([[0.25, 0.25, 0.25, 0.25], [0.5, 0, 0.5, 0], [0.25, 0.25, 0.25, 0.25], [0, 0.5, 0.5, 0]])
    adj_matrix = np.asarray([[1, 1, 1, 1], [1, 0, 1, 0], [1, 1, 1, 1], [0, 1, 1, 0]])
    random_walk = mH._construct_random_walk_matrix(adj_matrix, adj_matrix.shape, adj_matrix.shape[0])
    print(target)
    print(random_walk)
    return print_test("construct_random_walk_test", target, random_walk, np.array_equal(target, random_walk))


# noinspection PyProtectedMember
def construct_rejection_matrix_div_by_zero_error_exist_test():
    try:
        ddv = [0.1, 0.4, 0.3, 0.2]
        adj_matrix = np.asarray([[1, 1, 1, 1], [1, 0, 1, 0], [1, 1, 1, 1], [0, 1, 1, 0]])
        random_walk = mH._construct_random_walk_matrix(adj_matrix, adj_matrix.shape, adj_matrix.shape[0])
        mH._construct_rejection_matrix(ddv, random_walk, adj_matrix.shape, adj_matrix.shape[0])
        return print_test("construct_rejection_matrix_div_by_zero_error_exist_test", "no excpetion", "no exception", True)
    except ZeroDivisionError:
        return print_test("construct_rejection_matrix_div_by_zero_error_exist_test", "no excpetion", "ZeroDivisionError", False)


def arbitrary_matrix_converges_to_ddv_1():
    target = [0.2, 0.3, 0.5, 0]
    adj = np.asarray([[1, 1, 0, 0], [1, 1, 1, 1], [0, 1, 1, 1], [0, 1, 1, 1]])
    mh = mH.metropolis_algorithm(adj, ddv=target, column_major_in=False, column_major_out=True)
    mh_pow = np.linalg.matrix_power(mh, 50)
    print_pow(mh_pow)
    for j in range(mh_pow.shape[1]):
        if not np.allclose(target, mh_pow[:, j]):
            return print_test("arbitrary_matrix_converges_to_ddv_1", target, mh_pow[:, j], False)
    return print_test("arbitrary_matrix_converges_to_ddv_1", target, mh_pow[:, 0], True)


def arbitrary_matrix_converges_to_ddv_2():
    target = [0.2, 0.3, 0.2, 0.3]
    adj = np.asarray([[1, 1, 0, 0], [1, 1, 1, 1], [0, 1, 1, 1], [0, 1, 1, 1]])
    mh = mH.metropolis_algorithm(adj, ddv=target, column_major_in=False, column_major_out=True)
    mh_pow = np.linalg.matrix_power(mh, 1000)
    print_pow(mh_pow)
    for j in range(mh_pow.shape[1]):
        if not np.allclose(target, mh_pow[:, j]):
            return print_test("arbitrary_matrix_converges_to_ddv_2", target, mh_pow[:, j], False)
    return print_test("arbitrary_matrix_converges_to_ddv_2", target, mh_pow[:, 0], True)


def arbitrary_matrix_converges_to_ddv_3():
    target = [0.2, 0.3, 0.5, 0]
    adj = np.asarray([[1, 1, 1, 1], [1, 1, 1, 1], [0, 1, 1, 1], [1, 1, 1, 1]])
    mh = mH.metropolis_algorithm(adj, ddv=target, column_major_in=False, column_major_out=True)
    mh_pow = np.linalg.matrix_power(mh, 1000)
    print_pow(mh_pow)
    for j in range(mh_pow.shape[1]):
        if not np.allclose(target, mh_pow[:, j]):
            return print_test("arbitrary_matrix_converges_to_ddv_3", target, mh_pow[:, j], False)
    return print_test("arbitrary_matrix_converges_to_ddv_3", target, mh_pow[:, 0], True)


def arbitrary_matrix_converges_to_ddv_4():
    target = [0.0, 0.1, 0.1, 0.8]
    adj = np.asarray([[1, 1, 0, 0], [1, 0, 0, 1], [0, 1, 1, 1], [0, 1, 1, 1]])
    mh = mH.metropolis_algorithm(adj, ddv=target, column_major_in=False, column_major_out=True)
    mh_pow = np.linalg.matrix_power(mh, 1000)
    print_pow(mh_pow)
    for j in range(mh_pow.shape[1]):
        if not np.allclose(target, mh_pow[:, j]):
            return print_test("arbitrary_matrix_converges_to_ddv_4", target, mh_pow[:, j], False)
    return print_test("arbitrary_matrix_converges_to_ddv_4", target, mh_pow[:, 0], True)


def arbitrary_matrix_converges_to_ddv_5():
    target = [0.2, 0.3, 0.5, 0.0]
    adj = np.asarray([[1, 1, 0, 0], [1, 0, 1, 0], [0, 1, 1, 0], [0, 1, 1, 0]])
    mh = mH.metropolis_algorithm(adj, ddv=target, column_major_in=False, column_major_out=True)
    mh_pow = np.linalg.matrix_power(mh, 1000)
    print_pow(mh_pow)
    for j in range(mh_pow.shape[1]):
        if not np.allclose(target, mh_pow[:, j]):
            return print_test("arbitrary_matrix_converges_to_ddv_5", target, mh_pow[:, j], False)
    return print_test("arbitrary_matrix_converges_to_ddv_5", target, mh_pow[:, 0], True)


def arbitrary_matrix_converges_to_ddv_6():
    target = [1, 0, 0, 0]
    adj = np.asarray([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0]])
    mh = mH.metropolis_algorithm(adj, ddv=target, column_major_in=False, column_major_out=True)
    mh_pow = np.linalg.matrix_power(mh, 1000)
    print_pow(mh_pow)
    for j in range(mh_pow.shape[1]):
        if not np.allclose(target, mh_pow[:, j]):
            return print_test("arbitrary_matrix_converges_to_ddv_6", target, mh_pow[:, j], False)
    return print_test("arbitrary_matrix_converges_to_ddv_6", target, mh_pow[:, 0], True)


def arbitrary_matrix_does_not_converges_to_ddv_1():
    target = [1, 0, 0, 0]
    adj = np.asarray([[0, 1, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0]])
    mh = mH.metropolis_algorithm(adj, ddv=target, column_major_in=False, column_major_out=True)
    mh_pow = np.linalg.matrix_power(mh, 1000)
    print_pow(mh_pow)
    for j in range(mh_pow.shape[1]):
        if j == 1 and np.allclose(target, mh_pow[:, 1]):
            return print_test("arbitrary_matrix_does_not_converges_to_ddv_1", [0, 1, 0, 0], mh_pow[:, 1], False)
        elif j != 1 and not np.allclose(target, mh_pow[:, j]):
            return print_test("arbitrary_matrix_does_not_converges_to_ddv_1", target, mh_pow[:, j], False)
    return print_test("arbitrary_matrix_does_not_converges_to_ddv_1", "_", "_", True)


def arbitrary_matrix_does_not_converges_to_ddv_2():
    target = [0.2, 0, 0.8, 0]
    adj = np.asarray([[0, 1, 0, 0], [0, 1, 1, 0], [1, 0, 0, 1], [1, 1, 0, 0]])
    mh = mH.metropolis_algorithm(adj, ddv=target, column_major_in=False, column_major_out=True)
    mh_pow = np.linalg.matrix_power(mh, 1500)
    print_pow(mh_pow)
    for j in range(mh_pow.shape[1]):
        if not np.allclose(target, mh_pow[:, j]):
            return print_test("arbitrary_matrix_converges_to_ddv_with_some_zero_entries_1", target, mh_pow[:, j], True)
    return print_test("arbitrary_matrix_converges_to_ddv_with_some_zero_entries_1", target, mh_pow[:, 0], False)

# endregion


if __name__ == "__main__":
    np.set_printoptions(threshold=sys.maxsize, precision=5)

    tests = [
        matrix_column_select_test,
        linalg_matrix_power_test,
        matrix_converges_to_known_ddv_test,
        construct_random_walk_test,
        construct_rejection_matrix_div_by_zero_error_exist_test,
        arbitrary_matrix_converges_to_ddv_1,
        arbitrary_matrix_converges_to_ddv_2,
        arbitrary_matrix_converges_to_ddv_3,
        arbitrary_matrix_converges_to_ddv_4,
        arbitrary_matrix_converges_to_ddv_5,
        arbitrary_matrix_converges_to_ddv_6,
        arbitrary_matrix_does_not_converges_to_ddv_1,
        arbitrary_matrix_does_not_converges_to_ddv_2
    ]

    passed = 0
    for test in tests:
        if test():
            passed += 1
    print("-------------\nPassed {} out of {} specified tests...\n-------------".format(passed, len(tests)))

# endregion lame unit testing
