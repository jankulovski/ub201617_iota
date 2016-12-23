from simulator import *
import hills
import numpy as np
import random
import math

# Load program file
# path = "examples/sample_program.py"

# Program is a list of commands. Each command ends with \n.
# Look at simulator.py for more info.
# program = open(path).read().splitlines()
# print("Program: ")
# for line in program:
#     print("\t%s" % line)

# Convert program to vector. Use vectors when performing searching,
# genetic algorithms, etc.
# vector is a list of command numbers 0-279
# The command will fail for illegal programs.
# vector = prog_to_vector(program)
# print("Vector: ", vector)

# You can convert vectors back to programs to manually examine solutions.
# The command will fail for illegal vectors.
# program2 = vector_to_prog(vector)

# Simulate and visualize some terrains.
# The function simulate can operate on files, programs or vectors.
# If you have problems with visualization,
# try running the script from the terminal instead of PyCharm.
# af = 0
# for m in hills.hills_train:
#     af += simulate(m, vector, verbose=True, graphics=True, delay=0,
#                    max_moves=500, max_iter=1000, trace=True)
# print("Average fitness: ", af/len(hills.hills_train))

_P_SAMPLES = list(get_parser().keys())


def generate_random_generation(samples, hill, length=1, size=1):
    """
    Generate random generation with N agents
    """
    generation = []
    for _ in range(size):
        generation.append(Agent(hill, program_combinations(samples, length)))

    return generation


def generate_pool(population):

    pool = []
    for a_index in range(len(population)):
        n = int((population[a_index].fitness() / len(population)) * 100)
        if n == 0:
            n = 1
        for _ in range(n):
            pool.append(a_index)

    return pool


def regenerate_generation(population, hill):
    """
    Generate generation
    """
    new_population = []
    pool = generate_pool(population)
    for _ in range(len(population)):
        # p1, p2 = wheel_selection(population, pool)
        p1, p2 = tourn_selection(population, pool)

        # child = crossover(p1, p2)
        child = crossover2(p1, p2)
        child.set_hill(hill)

        new_population.append(mutate(child, _P_SAMPLES, 0.1))

    return new_population


def wheel_selection(population, pool):
    """
    Roulette Wheel Selection
    """
    return population[random.choice(pool)], population[random.choice(pool)]


def tourn_selection(population, pool, k=10):
    """
    Tournament Selection
    """
    candidates = []
    for _ in range(k):
        candidates.append(population[random.choice(pool)])

    return sorted(candidates, key=lambda x: x.fitness(), reverse=True)[:2]


def crossover(a1, a2):
    """
    Crossover
    """
    child = Agent(hills.hills_train[0])
    m = int(random.choice(range(len(a1.program))) / 2)
    child.set_program(np.append(a1.program[:m], a2.program[m:]))
    where_meet(a1, a2)
    return child


def crossover2(a1, a2):
    """
    Crossover 2
    """
    child = Agent(hills.hills_train[0])

    meet_points = where_meet(a1, a2)
    if meet_points[0] != -1:
        a1sum, a2sum, a1num_steps, a2num_steps = \
            sum_meet(a1, a2, meet_points[0], meet_points[1])

        a = a1num_steps / math.gcd(a1num_steps, a2num_steps)
        b = a2num_steps / math.gcd(a1num_steps, a2num_steps)

        _max = len(a1.program) - 1
        x = int(1 + a * _max / (a + b))
        y = int(1 + b * _max / (a + b))

        if a1sum > a2sum:
            child.set_program(np.append(a1.program[:x],
                                        a2.program[x:]))
        else:
            child.set_program(np.append(a2.program[:x],
                                        a1.program[x:]))
    else:
        a = a1.fitness() / math.gcd(a1.fitness(), a2.fitness())
        b = a2.fitness() / math.gcd(a1.fitness(), a2.fitness())

        _max = len(a1.program) - 1
        x = int(1 + a * _max / (a + b))
        y = int(1 + b * _max / (a + b))

        if a > b:
            child.set_program(np.append(a1.program[:x], a2.program[x:]))
        else:
            child.set_program(np.append(a2.program[:x], a1.program[x:]))

    return child


def where_meet(a1, a2):
    for i in range(len(a1.covered_positions)-1, 0, -1):
        for j in range(len(a1.covered_positions[0])-1, 0, -1):
            if a1.covered_positions[i, j] != 0 and \
                            a2.covered_positions[i, j] != 0:
                return i, j
    return -1, -1


def sum_meet(a1, a2, i, j):
    a1sum, a2sum = 0, 0
    a1num_steps, a2num_steps = 0, 0
    for ni in range(i):
        for nj in range(j):
            a1sum += a1.covered_positions[ni, nj]
            a2sum += a2.covered_positions[ni, nj]
            if a1.covered_positions[ni, nj] != 0:
                a1num_steps += 1
            if a2.covered_positions[ni, nj] != 0:
                a2num_steps += 1

    return a1sum, a2sum, a1num_steps, a2num_steps


def mutate(agent, samples, rate=0.1):
    """
    Mutate
    """
    for i in range(len(agent.program)):
        if random.uniform(0, 1) < rate:
            agent.program[i] = random.choice(samples)

    return agent


def program_combinations(samples, length=1, size=1):
    """
    Program combinations
    """
    if size == 1:
        return np.random.choice(samples, length)

    programs = []
    for _ in range(size):
        programs.append(np.random.choice(samples, length))
    return programs


def store_output(output, filename):
    import json
    with open(filename, 'w') as file:
        file.write(json.dumps(output, ensure_ascii=False))

hill_index = 0

# for hill in hills.hills_train:

hill_index += 1
print("Train hill %d" % hill_index)

hill = hills.hills_train[0]
output = {}

population = generate_random_generation(_P_SAMPLES, hill, 50, 1500)
for agent in population:
    agent.run()

for gen in range(1, 100):
    avg_fit = 0
    best = 0
    population = regenerate_generation(population, hill)
    for agent in population:
        fit = agent.run()
        if fit > best:
            best = fit
        avg_fit += fit

    print("Generation: %d, Average Fitness: %d, Best %d" %
          (gen, avg_fit/len(population), best))

    output[gen] = {
        "agents": [
            {"program": agent.program.tolist(),
             "fitness": int(agent.fitness())}
            for agent in sorted(population, key=lambda x: x.fitness(),
                                reverse=True)[:10]
        ]
    }

store_output(output, "output/output.txt")
store_output(vector_to_prog(_P_SAMPLES), "output/cmds.txt")