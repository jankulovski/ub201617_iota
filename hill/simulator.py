import numpy as np
import time
import random
import itertools
import os
import sys
import pylab as plt
import matplotlib.pyplot as mat_plt
import plotly.plotly as py
import plotly.tools as tls

#set credentials for the plot to be opened in a browser
tls.set_credentials_file(username='OliveraPerunkovska', api_key='uJZCJEvvxZp5anzyIEed')

# legal commands and testsS
# add additional moves and tests
CMDS = [
    "nop()",
    "move_forward()",
    "move_backward()",
    "turn_left()",
    "turn_right()",
 #   "set_flag()",
 #   "clear_flag()",
    "mark_position()",
    "unmark_position()"]

TESTS = [
    "coverage_improved()",
    "can_move_forward",
    "can_move_backward",
    "marked_ahead",
    "marked_behind",
    "marked_current",
#    "flag"
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


def prog_to_executable(p, obj_str):
    out = []
    for l in p:
        for r in CMDS + TESTS:
            l = l.replace(' ' + r, " " + obj_str + "." + r)
            if l.startswith(r):
                l = obj_str + '.' + r + l[len(r):]
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


class Agent:
    def __init__(self, input_hills, program=None, seed=0, max_moves=1000):
        # i: rows, j: columns
        self.hills = input_hills
        self.dim_i, self.dim_j = self.hills.shape
        self.covered_positions = np.zeros(self.hills.shape, dtype=int)
        self.marked_positions = np.zeros(self.hills.shape, dtype=int)
        self.max_moves = max_moves

        # initial positions
        self.move_counter = max_moves
        self.cur_dir = 'down'
        self.cur_i = self.dim_i-1
        self.cur_j = self.dim_j-1
        self.prev_i = self.cur_i
        self.prev_j = self.cur_j
        self.start_i = self.cur_i
        self.start_j = self.cur_j
        self.cost_a = 0
        self.cost_b = 0
        self.iterations = 0
        self.can_move_forward = True
        self.can_move_backward = self.flag = False

        self.steps = 0
        self.flag_a = True
        self.flag_b = True
        self.marked_current = False
        self.marked_ahead = False
        self.marked_behind = False
        self.fmove = True

        # available commands
        self.prev_coverage = self.coverage()
        self.update_marked_flags()
        self.update_can_move_forward_backward()
        self.update_covered_positions()

        # Initialize random generator
        # a fixed random generator
        self.rand = random.Random(seed)

        # Program vector
        self.program = program

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

    def set_rand_seed(self, seed):
        self.rand = random.Random(seed)

    def set_max_moves(self, max_moves):
        self.max_moves = max_moves
        self.move_counter = max_moves

    def update_marked_flags(self):
        # TO-DO: update marked flags
        self.marked_ahead = False
        self.marked_behind = False

        if self.cur_dir == 'up':
            if self.cur_i != self.dim_i - 1:
                if self.marked_positions[self.cur_i+1][self.cur_j] == 1:
                    self.marked_ahead = True
            if self.cur_i != 0:
                if self.marked_positions[self.cur_i-1][self.cur_j] == 1:
                    self.marked_behind = True
        elif self.cur_dir == 'down':
            if self.cur_i != self.dim_i - 1:
                if self.marked_positions[self.cur_i+1][self.cur_j] == 1:
                    self.marked_behind = True
            if self.cur_i != 0:
                if self.marked_positions[self.cur_i-1][self.cur_j] == 1:
                    self.marked_ahead = True
        elif self.cur_dir == 'left':
            if self.cur_j != self.dim_j - 1:
                if self.marked_positions[self.cur_i][self.cur_j+1] == 1:
                    self.marked_behind = True
            if self.cur_j != 0:
                if self.marked_positions[self.cur_i][self.cur_j-1] == 1:
                    self.marked_ahead = True
        elif self.cur_dir == 'right':
            if self.cur_j != self.dim_j - 1:
                if self.marked_positions[self.cur_i][self.cur_j+1] == 1:
                    self.marked_ahead = True
            if self.cur_j != 0:
                if self.marked_positions[self.cur_i][self.cur_j-1] == 1:
                    self.marked_behind = True

    def update_can_move_forward_backward(self):

        self.can_move_forward = True
        self.can_move_backward = True

        if self.cur_dir == 'left':
            if self.cur_j == 0:
                self.can_move_forward = False
            elif self.cur_j == self.dim_j - 1:
                self.can_move_backward = False
        elif self.cur_dir == 'right':
            if self.cur_j == self.dim_j - 1:
                self.can_move_forward = False
            elif self.cur_j == 0:
                self.can_move_backward = False
        elif self.cur_dir == 'up':
            if self.cur_i == 0:
                self.can_move_backward = False
            elif self.cur_i == self.dim_i - 1:
                self.can_move_forward = False
        elif self.cur_dir == 'down':
            if self.cur_i == self.dim_i - 1:
                self.can_move_backward = False
            elif self.cur_i == 0:
                self.can_move_forward = False

    def move_forward_position(self):
        if self.cur_dir == 'up':
            self.prev_i = self.cur_i
            self.cur_i += 1
        elif self.cur_dir == 'down':
            self.prev_i = self.cur_i
            self.cur_i -= 1
        elif self.cur_dir == 'left':
            self.prev_j = self.cur_j
            self.cur_j -= 1
        elif self.cur_dir == 'right':
            self.prev_j = self.cur_j
            self.cur_j += 1

    def move_backward_position(self):
        if self.cur_dir == 'up':
            self.prev_i = self.cur_i
            self.cur_i -= 1
        elif self.cur_dir == 'down':
            self.prev_i = self.cur_i
            self.cur_i += 1
        elif self.cur_dir == 'left':
            self.prev_j = self.cur_j
            self.cur_j += 1
        elif self.cur_dir == 'right':
            self.prev_j = self.cur_j
            self.cur_j -= 1

    # Commands
    def move_forward(self):
        self.prev_coverage = self.coverage()
        self.move_forward_position()
        self.update_covered_positions()
        self.update_move_counter()
        self.update_marked_flags()
        self.check_end()
        self.update_can_move_forward_backward()

    def move_backward(self):
        self.prev_coverage = self.coverage()
        self.move_backward_position()
        self.update_covered_positions()
        self.update_move_counter()
        self.update_marked_flags()
        self.check_end()
        self.update_can_move_forward_backward()

    def turn_left(self):
        self.cur_dir = self.turn_directions['left'][self.cur_dir]
        self.update_can_move_forward_backward()

    def turn_right(self):
        self.cur_dir = self.turn_directions['right'][self.cur_dir]
        self.update_can_move_forward_backward()

    def set_flag(self):
        self.flag = True

    def clear_flag(self):
        self.flag = False

    def nop(self):
        pass

    def mark_position(self):
        # TO-DO: mark position
        self.marked_positions[self.cur_i][self.cur_j] = 1
        pass

    def unmark_position(self):
        # TO-DO: unmark position
        self.marked_positions[self.cur_i][self.cur_j] = 0
        pass

    def coverage_improved(self):
        # TO-DO: coverage improved
        if self.coverage() > self.prev_coverage: return True
        return False

    # TODO: you can add more commands

    def coverage(self):
        return np.sum(self.covered_positions > 0)

    def cost(self, start_square, end_square):
        # TODO: experiment with cost function (fitness function)
        return 5 + 2 * (end_square - start_square) ** 2

    def cost(self, start_square, end_square):
        # TODO: experiment with cost function (fitness function)
        return 5 + 2 * (end_square - start_square) ** 2

    def update_covered_positions(self):
        if self.covered_positions.shape[0] >= self.cur_i + 1 and self.covered_positions.shape[1] >= self.cur_j + 1:
            self.covered_positions[self.cur_i][self.cur_j] += 1
        else:
            raise OutOfBoundException

    def update_move_counter(self):
        if (self.hills.shape[0] > self.cur_i and
                    self.hills.shape[1] > self.cur_j and
                    self.hills.shape > (self.prev_i, self.prev_j) and
                (self.cur_i >= 0 and self.cur_j >= 0 and self.prev_i >= 0 and
                         self.prev_j >= 0)) or self.fmove:
            current_square, previous_square = \
                self.hills[self.cur_i][self.cur_j], \
                self.hills[self.prev_i][self.prev_j]
            self.move_counter -= self.cost(previous_square, current_square)
            self.steps += 1
            self.fmove = False
        else:
            raise OutOfBoundException

    def check_end(self):
        if self.move_counter <= 0:
            raise MaxMovesExceededException(self.fitness())

    # # Fitness is the coverage
    # def fitness(self):
    #     return self.coverage()
    # Fitness is the distance from starting point
    def fitness(self):
        f = abs(self.cur_i-self.start_i)+abs(self.cur_j-self.start_j)
        return f

    def set_hill(self, hill):
        self.hills = hill
        self.dim_i, self.dim_j = self.hills.shape

    def set_program(self, program):
        self.program = program

    def run(self, graphics=False, verbose=False, max_iter=100, max_len=200,
            delay=1.0, seed=0, max_moves=1000, trace=False):

        self.set_rand_seed(seed)
        self.set_max_moves(max_moves)

        markers = {}
        prog = vector_to_prog(self.program)

        v = prog_to_vector(prog)
        if len(v) > max_len:
            raise Exception("Illegal program length")

        prog = prog_to_executable(prog, 'self')  # add sim.

        try:
            one_step = compile("\n".join(prog) + "\n", "<string>", "exec")
        except:
            raise Exception("Compilation error")

        if graphics:
            plt.ion()
            markers = {'up': '^', 'down': 'v', 'left': '<', 'right': '>'}
            self.draw(delay, markers, trace)

        # run simulation
        for step in range(max_iter):
            try:
                ss = self.fitness()
                exec(prog[step])
                self.iterations += 1
                if verbose:
                    sys.stdout.write("Moves remaining: %d Iterations: %d\r" %
                                     (self.move_counter, step))
                    sys.stdout.flush()
            except (MaxMovesExceededException, OutOfBoundException):
                if graphics:
                    self.draw(delay, markers, trace)
                if verbose:
                    print("Home. Fitness value:", self.fitness())
                return self.fitness()

            if graphics:
                self.draw(delay,markers,trace)

        if verbose:
            print("Iteration limit exceed: Failed to find path through hills. "
                  "Fitness: ", self.fitness())

        return self.fitness()

    def draw(self,delay,markers,trace):
        plt.clf()
        plt.imshow(self.hills, cmap="YlOrBr", interpolation='nearest')
        plt.ylim([-0.5, self.hills.shape[1] - 0.5])
        plt.xlim([-0.5, self.hills.shape[0] - 0.5])
        plt.plot(self.marked_positions.T.nonzero()[0],
                 self.marked_positions.T.nonzero()[1], 'cs',
                 markersize=8.0,
                 markerfacecolor="c")
        plt.plot(self.cur_j, self.cur_i, markers[self.cur_dir],
                 markersize=8.0,
                 markerfacecolor="g")

        plt.title("Money:%d, Steps:%d, Fitness:%d" % (self.move_counter,self.steps, self.fitness()))
        if trace:
            plt.imshow((self.covered_positions > 0), cmap="BuPu",
                       alpha=0.3)

        plt.draw()
        plt.pause(delay)

#draws the average fitness per generation
def drawChartAvgFitness(gen):
    averageFit_agent = list()
    for key, value in gen.items():
        averageFit_agent.append(value["avr_fitness"])

    fig = mat_plt.gcf()
    x = np.arange(len(gen)) #the x axes values
    averageFit = tuple(averageFit_agent) #average fitness of the generation

    ax = mat_plt.subplot(111)
    ax.bar(x, averageFit, width=0.2, color='b')

    #set the values for the axes
    ax.set_ylabel('Fitness')
    ax.set_xlabel('Generation')
    ax.set_title('Scores for average fitness per generation')

    mat_plt.show()

    plot_url = py.plot_mpl(fig, filename='mpl-basic-bar')

#draws the best fitness per generation
def drawChartBestFitness(gen):
    bestFit_agent = list()
    for key, value in gen.items():
        bestFit_agent.append(value["best_fitness"])

    fig = mat_plt.gcf()
    x = np.arange(len(gen)) #the x axes values
    bestFit = tuple(bestFit_agent) #best fitness of the generation

    ax = mat_plt.subplot(111)
    ax.bar(x, bestFit, width=0.2, color='g')

    #set the values for the axes
    ax.set_ylabel('Fitness')
    ax.set_xlabel('Generation')
    ax.set_title('Scores for best fitness per generation')

    mat_plt.show()

    plot_url = py.plot_mpl(fig, filename='mpl-basic-bar')

#draws combine plot from the average fitness and best fitness in one chart per generation
def drawChart(gen):
    averageFit_agent = list()
    bestFit_agent = list()
    my_xticks_list = []
    for key, value in gen.items():
       # print(key, value)
        averageFit_agent.append(value["avr_fitness"])
        bestFit_agent.append(value["best_fitness"])
        my_xticks_list.append("G_" + repr(key))

    mpl_fig = mat_plt.figure()
    ax = mpl_fig.add_subplot(111)

    N = len(gen)

    average = tuple(averageFit_agent)
    best = tuple(bestFit_agent)

    ind = np.arange(N)  # the x locations for the groups
    width = 0.35  # the width of the bars: can also be len(x) sequence
    my_xticks = np.asarray(my_xticks_list)
    p1 = ax.bar(ind, average, width, color=(0.2588,0.4433,1.0))
    p2 = ax.bar(ind, best, width, color=(0.0, 0.5019607843137255, 0.0),
                bottom=average)

    #set the label text for the axes
    ax.set_ylabel('Fitness')
    ax.set_xlabel('Generation')
    ax.set_title('Scores for average and best fitness per generation')

    #ax.set_xticks(ind, my_xticks.all(ax))
    plotly_fig = tls.mpl_to_plotly( mpl_fig )
    # For Legend - plot
    plotly_fig["layout"]["showlegend"] = True
    plotly_fig["data"][0]["name"] = "Avg_fitness"
    plotly_fig["data"][1]["name"] = "Best_fitness"

    plot_url = py.plot(plotly_fig, filename='stacked-bar-chart')

# def drawChart1(gen):
#     averageFit_agent = list()
#     bestFit_agent = list()
#     my_xticks = []
#     for key, value in gen.items():
#         averageFit_agent.append(value["avr_fitness"])
#         bestFit_agent.append(value["best_fitness"])
#         my_xticks.append("G_" + repr(key))
#
#     multiple_bars = mat_plt.figure()
#
#     x = np.arange(len(gen))
#
#     y = tuple(averageFit_agent)
#     z = tuple(bestFit_agent)
#
#     ax = mat_plt.subplot(111)
#     ax.bar(x - 0.2, y, width=0.2, color='b', align='center')
#     ax.bar(x, z, width=0.2, color='g', align='center')
#
#     #set the values for the axes
#     ax.set_ylabel('Fitness')
#     ax.set_xlabel('Generation')
#     ax.set_title('Scores for average and best fitness per generation')
#
#     mat_plt.show()
#
#     plot_url = py.plot_mpl(multiple_bars, filename='mpl-multiple-bars')


class OutOfBoundException(Exception):
    pass


class MaxMovesExceededException(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)

class ChartProperties:
    def __init__(self):
        self.crossOverClassName = ""
        self.selectionClassName = ""

    def set_crossover(self, name):
        self.crossOverClassName = name

    def set_selection(self, name):
        self.selectionClassName = name

    def get_crossover(self):
        return self.crossOverClassName

    def get_selection(self):
        return self.selectionClassName
#
# def simulate(input_hills, program, graphics=False, verbose=False, max_iter=100,
#              max_len=100, delay=1.0, seed=0, max_moves=1000, trace=False):
#     """
#         program can be a path to file, string or vector
#         return fitness value
#     """
#     sim = Agent(input_hills, seed, max_moves)
#     markers = {}
#
#     if isinstance(program, str):
#         if os.path.exists(program):
#             prog = open(program).readlines()
#         else:
#             prog = program
#             prog = map(lambda x: x + '\n', filter(None, prog.split('\n')))
#     else:
#         prog = vector_to_prog(program)
#
#     v = prog_to_vector(prog)
#     if len(v) > max_len:
#         raise Exception("Illegal program length")
#
#     prog = prog_to_executable(prog, 'sim')  # add sim.
#
#     try:
#         one_step = compile("\n".join(prog) + "\n", "<string>", "exec")
#     except:
#         raise Exception("Compilation error")
#
#     if graphics:
#         plt.ion()
#         markers = {'up': '^', 'down': 'v', 'left': '<', 'right': '>'}
#
#     # run simulation
#     for step in range(max_iter):
#         try:
#             exec(one_step)
#             sim.iterations += 1
#             if verbose:
#                 # sys.stdout.write(
#                 #     "Coverage: %d Moves remaining: %d Iterations: %d\r" % (
#                 #     sim.coverage(), sim.move_counter, step))
#                 sys.stdout.write("Moves remaining: %d Iterations: %d\r" %
#                                  (sim.move_counter, step))
#                 sys.stdout.flush()
#         except MaxMovesExceededException:
#             if verbose:
#                 print("Home. Fitness value:", sim.fitness())
#             return sim.fitness()
#
#         if graphics:
#             plt.clf()
#             plt.imshow(sim.hills, cmap="YlOrBr", interpolation='nearest')
#             plt.ylim([-0.5, sim.hills.shape[1] - 0.5])
#             plt.xlim([-0.5, sim.hills.shape[0] - 0.5])
#             plt.plot(sim.marked_positions.T.nonzero()[0],
#                      sim.marked_positions.T.nonzero()[1], 'cs', markersize=8.0,
#                      markerfacecolor="c")
#             plt.plot(sim.cur_j, sim.cur_i, markers[sim.cur_dir],
#                      markersize=8.0,
#                      markerfacecolor="g")
#             plt.title("Flag A:%d Flag B:%d cost A:%d cost B:%d " %
#                       (sim.flag_a, sim.flag_b,
#                        (sim.cost_a if sim.cost_a < sys.maxsize else -1),
#                        (sim.cost_b if sim.cost_b < sys.maxsize else -1)
#                        ))
#             if trace:
#                 plt.imshow((sim.covered_positions > 0), cmap="Greys",
#                            alpha=0.2)
#
#             plt.draw()
#             time.sleep(delay)
#
#     if verbose:
#         print("Iteration limit exceed: Failed to find path through hills. "
#               "Fitness: ", sim.fitness())
#
#     return sim.fitness()
