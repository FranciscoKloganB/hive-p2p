def print_test(name, expect, result, accept):
    print(name)
    print("expect:\n{}".format(expect))
    print("got:\n{}".format(result))
    print("accept: {}\n".format(accept))
    return accept
