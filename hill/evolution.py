from simulator import *
import hills
import numpy as np
import random
import math
from initialization import *


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

chartProp = ChartProperties()

_P_SAMPLES = list(get_parser().keys())
path = "examples/sample_program.py"

program = open(path).readlines()
print("Program: ")
for line in program:
    print("\t%s" % line)

vector = prog_to_vector(program)
print("Vector: ", vector)

# _P_SAMPLES = vector

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
        p1, p2 = wheel_selection(population, pool)
        # p1 = tourn_selection(population,tournament_size)
        # p2 = tourn_selection(population,tournament_size)

        # child = crossover(p1, p2)
        # child.set_hill(hill)
        child = crossover(p1, p2)

        new_population.append(mutate(child, _P_SAMPLES, mutation_rate))

    return new_population


def wheel_selection(population, pool):
    """
    Roulette Wheel Selection
    """
    chartProp.set_selection("weel")
    return population[random.choice(pool)], population[random.choice(pool)]


def tourn_selection(population, k):
    """
    Tournament Selection
    """
    chartProp.set_selection("turnament")
    candidates = []
    for _ in range(k):
        candidates.append(population[random.choice(range(population_size))])

    return sorted(candidates, key=lambda x: x.fitness(), reverse=True)[0]



def crossover(a1, a2):
    """
    Crossover
    """
    chartProp.set_crossover("crossover")
    child = Agent(hills.hills_train[0])
    m = int(random.choice(range(len(a1.program))) / 2)
    child.set_program(np.append(a1.program[:m], a2.program[m:]))
    where_meet(a1, a2)
    return child


def crossover2(a1, a2):
    """
    Crossover 2
    """
    chartProp.set_crossover("crossover2")
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
            child.set_program(np.append(a1.program[:x], a2.program[x:]))
        else:
            child.set_program(np.append(a2.program[:x], a1.program[x:]))
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

def crossover3(a1,a2,hill):
    chartProp.set_crossover("crossover3")
    x = np.random.randint(0,max_iter)
    prog = np.append(a1.program[:x], a2.program[x:])
    return Agent(hill,prog)

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


if __name__ == '__main__':

    output = {}
    agents_fitness={}


    #generates first population
    population = generate_random_generation(_P_SAMPLES, hill, max_iter, population_size)
    # for each agent run his program
    for agent in population:
        agent.run(max_iter=max_iter,max_moves=money)

    #
    for gen in range(1, num_of_generations):
        avg_fit = 0
        best = 0
        population = regenerate_generation(population, hill)
        for agent in population:
            fit = agent.run(max_iter=max_iter,max_moves=money)
            if fit > best:
                best = fit
            avg_fit += fit
        # all_averageFitness.append(avg_fit/len(population))
        print("Generation: %d, Average Fitness: %d, Best %d" %
              (gen, avg_fit/len(population), best))

        agents_fitness[gen] = {
        "avr_fitness": avg_fit/len(population),
        "best_fitness": best
        }

        output[gen] = {
            "agents": [
                {"program": agent.program.tolist(),
                 "fitness": int(agent.fitness())}
                for agent in sorted(population, key=lambda x: x.fitness(),
                                    reverse=True)[:]
            ]
        }

    output["crossoverClassName"] = chartProp.get_crossover()
    output["selectionClassName"] = chartProp.get_selection()
    store_output(output, "output/output.txt")
    store_output(vector_to_prog(list(get_parser().keys())), "output/cmds.txt")

    #draw a chart with average fitness per generation.
    drawChartAvgFitness(agents_fitness)
    # draw a chart with best fitness per generation.
    drawChartBestFitness(agents_fitness)
    #draw a chart with combine average fit and best fit per generation
    drawChart(agents_fitness)