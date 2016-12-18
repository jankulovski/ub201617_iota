from simulator import *
import hills

# Load program file
path = "examples/sample_program.py"

# Program is a list of commands. Each command ends with \n. Look at simulator.py for more info.
program = open(path).readlines()
print("Program: ")
for line in program:
    print("\t%s" % line)

# Convert program to vector. Use vectors when performing searching, genetic algorithms, etc.
# vector is a list of command numbers 0-279
# The command will fail for illegal programs.
vector = prog_to_vector(program)
print
print("Vector: ", vector)

# You can convert vectors back to programs to manually examine solutions.
# The command will fail for illegal vectors.
program2 = vector_to_prog(vector)

# Simulate and visualize some terrains.
# The function simulate can operate on files, programs or vectors.
# If you have problems with visualization,
#  try running the script from the terminal instead of PyCharm.
af = 0
# for m in hills.hills_train:
#     af += simulate(m, vector, verbose=True, graphics=True, delay=0.2, max_moves=500, max_iter=1, trace=True)




af += simulate(hills.hills_train[0], vector, verbose=True, graphics=True, delay=0.2, max_moves=100, max_iter=1, trace=True,population = 5)



print("Average fitness: ", af/len(hills.hills_train))
