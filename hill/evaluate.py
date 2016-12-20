from simulator import *
import hills
import sys


def evaluate(path):
    program = open(path).readlines()
    vector = prog_to_vector(program)
    af = 0
    for m in hills.hills_test:
        try:
            af += simulate(m, vector, verbose=True, graphics=False, delay=0.1,
                           max_iter=3000, trace=True, max_moves=3000)
        except KeyboardInterrupt:
            continue

    return af/len(hills.hills_test)

if __name__ == "__main__":
    print(evaluate(sys.argv[1]))
