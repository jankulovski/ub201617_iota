import numpy as np
import time
import random
import itertools
import os
import sys

# legal commands and tests
# add additional moves and tests
CMDS = [
    "nop()",
    "move_forward()",
    "move_backward()",
    "turn_left()",
    "turn_right()",
    "set_flag()",
    "clear_flag()",
    "mark_position()",
    "unmark_position()"]

TESTS = [
    "coverage_improved()",
    "can_move_forward",
    "can_move_backward",
    "marked_current",
    "flag"
]


def get_parser():
    parse_dict = dict()

    for i, c in enumerate(CMDS):
        parse_dict[i] = "%s" % c

    i += 1
    for j, t in enumerate([ti for ti in itertools.product(TESTS, CMDS)]):
        parse_dict[i + j] = "if %s: %s" % (t[0], t[1])

    i = i + j + 1
    for j, t in enumerate([ti for ti in itertools.product(TESTS, CMDS)]):
        parse_dict[i + j] = "if not %s: %s" % (t[0], t[1])

    return parse_dict


def prog_to_executable(p):
    out = []
    for l in p:
        for r in CMDS + TESTS:
            l = l.replace(' ' + r, " sim." + r)
            if l.startswith(r):
                l = 'sim.' + r + l[len(r):]
        out.append(l)
    return out


def prog_to_vector(p):
    v = np.zeros((len(p),), dtype=int)
    pd_inv = {v: k for k, v in pd.items()}
    for i, line in enumerate(p):
        line = line.strip()
        try:
            v[i] = pd_inv[line]
        except:
            raise Exception("Illegal program: %s" % line)
    return v


def vector_to_prog(v):
    try:
        return [pd[i] for i in v]
    except:
        raise Exception("Illegal vector")


pd = get_parser()


class Simulator:
    def __init__(self, input_hills, seed=0, max_moves=1000):
        # i: rows, j: columns
        self.hills = input_hills
        self.dim_i, self.dim_j = self.hills.shape
        self.covered_positions = np.zeros(self.hills.shape, dtype=int)
        self.marked_positions = np.zeros(self.hills.shape, dtype=int)
        self.max_moves = max_moves

        # initial positions
        self.move_counter = max_moves
        self.cur_dir = 'down'
        self.prev_i = 0
        self.prev_j = 0
        self.cur_i = 0
        self.cur_j = 0
        self.cost_a = 0
        self.cost_b = 0
        self.iterations = 0
        self.can_move_forward = True
        self.can_move_backward = self.flag = False

        self.steps = 0
        self.flag_a = True
        self.flag_b = True

        # available commands
        self.prev_coverage = self.coverage()
        self.update_marked_flags()
        self.update_can_move_forward_backward()
        self.update_covered_positions()

        # Initialize random generator
        # a fixed random generator
        self.rand = random.Random(seed)

        self.turn_directions = {
            'left': {
                'up': 'left',
                'left': 'down',
                'down': 'right',
                'right': 'up'
            },
            'right': {
                'up': 'right',
                'left': 'up',
                'down': 'left',
                'right': 'down'
            }
        }

    def update_marked_flags(self):
        # TO-DO: update marked flags
        pass

    def update_can_move_forward_backward(self):
        # TO-DO: update can move forward backward
        pass

    def move_forward_position(self):
        if self.cur_dir == 'up':
            self.cur_i += 1
        elif self.cur_dir == 'down':
            self.cur_i -= 1
        elif self.cur_dir == 'left':
            self.cur_j -= 1
        elif self.cur_dir == 'right':
            self.cur_j += 1

    def move_backward_position(self):
        if self.cur_dir == 'up':
            self.cur_i -= 1
        elif self.cur_dir == 'down':
            self.cur_i += 1
        elif self.cur_dir == 'left':
            self.cur_j += 1
        elif self.cur_dir == 'right':
            self.cur_j -= 1

    # Commands
    def move_forward(self):
        self.move_forward_position()
        self.update_move_counter()
        self.update_covered_positions()
        self.check_end()

    def move_backward(self):
        self.move_backward_position()
        self.update_move_counter()
        self.update_covered_positions()
        self.check_end()

    def turn_left(self):
        self.cur_dir = self.turn_directions['left'][self.cur_dir]

    def turn_right(self):
        self.cur_dir = self.turn_directions['right'][self.cur_dir]

    def set_flag(self):
        self.flag = True

    def clear_flag(self):
        self.flag = False

    def nop(self):
        pass

    def mark_position(self):
        # TO-DO: mark position
        pass

    def unmark_position(self):
        # TO-DO: unmark position
        pass

    # TODO: you can add more commands

    def coverage(self):
        return np.sum(self.covered_positions > 0)

    def cost(self, start_square, end_square):
        # TODO: experiment with cost function (fitness function)
        return 5 + 2 * (end_square - start_square) ** 2

    def update_covered_positions(self):
        self.covered_positions[self.cur_i][self.cur_j] += 1

    def update_move_counter(self):
        current_square, previous_square = self.hills[self.cur_i][self.cur_j], \
                                          self.hills[self.prev_i][self.prev_j]
        self.move_counter -= self.cost(previous_square, current_square)
        self.steps += 1

    def check_end(self):
        if self.move_counter <= 0:
            raise MaxMovesExceededException(self.fitness())

    # Fitness is the coverage
    def fitness(self):
        return self.coverage()


class MaxMovesExceededException(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


def simulate(input_hills, program, graphics=False, verbose=False, max_iter=100,
             max_len=100, delay=1.0, seed=0, max_moves=1000, trace=False):
    """
        program can be a path to file, string or vector
        return fitness value
    """
    sim = Simulator(input_hills, seed, max_moves)
    markers = {}

    if isinstance(program, str):
        if os.path.exists(program):
            prog = open(program).readlines()
        else:
            prog = program
            prog = map(lambda x: x + '\n', filter(None, prog.split('\n')))
    else:
        prog = vector_to_prog(program)

    v = prog_to_vector(prog)
    if len(v) > max_len:
        raise Exception("Illegal program length")

    prog = prog_to_executable(prog)  # add sim.

    try:
        one_step = compile("\n".join(prog) + "\n", "<string>", "exec")
    except:
        raise Exception("Compilation error")

    if graphics:
        import pylab as plt
        plt.ion()
        markers = {'up': '^', 'down': 'v', 'left': '<', 'right': '>'}

    # run simulation
    for step in range(max_iter):
        try:
            exec(one_step)
            sim.iterations += 1
            if verbose:
                # sys.stdout.write(
                #     "Coverage: %d Moves remaining: %d Iterations: %d\r" % (
                #     sim.coverage(), sim.move_counter, step))
                sys.stdout.write("Moves remaining: %d Iterations: %d\r" %
                                 (sim.move_counter, step))
                sys.stdout.flush()
        except MaxMovesExceededException:
            if verbose:
                print("Home. Fitness value:", sim.fitness())
            return sim.fitness()

        if graphics:
            plt.clf()
            plt.imshow(sim.hills, cmap="YlOrBr", interpolation='nearest')
            plt.ylim([-0.5, sim.hills.shape[1] - 0.5])
            plt.xlim([-0.5, sim.hills.shape[0] - 0.5])
            plt.plot(sim.marked_positions.T.nonzero()[0],
                     sim.marked_positions.T.nonzero()[1], 'cs', markersize=8.0,
                     markerfacecolor="c")
            plt.plot(sim.cur_j, sim.cur_i, markers[sim.cur_dir], markersize=8.0,
                     markerfacecolor="g")
            plt.title("Flag A:%d Flag B:%d cost A:%d cost B:%d " %
                      (sim.flag_a, sim.flag_b,
                       (sim.cost_a if sim.cost_a < sys.maxsize else -1),
                       (sim.cost_b if sim.cost_b < sys.maxsize else -1)
                       ))
            if trace:
                plt.imshow((sim.covered_positions > 0), cmap="Greys", alpha=0.2)

            plt.draw()
            time.sleep(delay)

    if verbose:
        print("Iteration limit exceed: Failed to find path through hills. "
              "Fitness: ", sim.fitness())

    return sim.fitness()
