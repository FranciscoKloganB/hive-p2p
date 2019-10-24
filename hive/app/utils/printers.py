def print_test(name, expect, result, accept):
    print(name)
    print("expect:\n{}".format(expect))
    print("got:\n{}".format(result))
    print("accept: {}\n".format(accept))
    return accept


def print_pow(matrix):
    print("---------------------------------\npowered matrix:")
    print(matrix)
