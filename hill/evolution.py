from simulator import *
import hills
import numpy as np
import random

# Load program file
path = "examples/sample_program.py"

# Program is a list of commands. Each command ends with \n.
# Look at simulator.py for more info.
program = open(path).read().splitlines()
print("Program: ")
for line in program:
    print("\t%s" % line)

# Convert program to vector. Use vectors when performing searching,
# genetic algorithms, etc.
# vector is a list of command numbers 0-279
# The command will fail for illegal programs.
vector = prog_to_vector(program)
print("Vector: ", vector)

# You can convert vectors back to programs to manually examine solutions.
# The command will fail for illegal vectors.
program2 = vector_to_prog(vector)

# Simulate and visualize some terrains.
# The function simulate can operate on files, programs or vectors.
# If you have problems with visualization,
# try running the script from the terminal instead of PyCharm.
af = 0
for m in hills.hills_train:
    af += simulate(m, vector, verbose=True, graphics=True, delay=0,
                   max_moves=500, max_iter=1000, trace=True)

print("Average fitness: ", af/len(hills.hills_train))


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
        child = crossover(population[random.choice(pool)],
                          population[random.choice(pool)])

        child.set_hill(hill)

        # TODO: replace the int list with a list of program samples
        new_population.append(mutate(child, [1, 2, 3, 4, 5, 6, 7, 8, 9]))

    return new_population


def crossover(a1, a2):
    """
    Crossover
    """
    child = Agent()
    m = int(random.choice(range(len(a1.program))) / 2)
    child.set_program(a1.program[:m] + a2.program[m:])

    return child


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
