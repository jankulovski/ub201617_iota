import json
import hills
from simulator import *
from initialization import *

file = open("output/output.txt")
line = file.readline()
output = json.loads(line)

file = open("output/cmds.txt")
line = file.readline()
cmds = json.loads(line)

plot_type = 2

if plot_type==1:
    # plot best agent in last x generations
    x = 5
    for i in range(num_of_generations-x,num_of_generations):
        prg = output[str(i)]["agents"][0]["program"]
        agent = Agent(hill, prg)
        agent.run(max_iter=max_iter,max_moves=money,graphics=True,delay=0.01,trace = True)

elif plot_type ==2:
    # in each plot we can see the area covered by one generation
    # the animation shows progress through generations
    for i in range(1,num_of_generations):
        covered_positions = np.zeros(hill.shape, dtype=int)
        for j in range(population_size):
            prg = output[str(i)]["agents"][j]["program"]
            agent = Agent(hill, prg)
            agent.run(max_iter=max_iter,max_moves=money,graphics=False,delay=0.1)
            covered_positions += agent.covered_positions

        plt.ion()
        plt.clf()
        plt.title("Generation number " + str(i))
        plt.imshow(hill, cmap="YlOrBr", interpolation='nearest')
        plt.ylim([-0.5, hill.shape[1] - 0.5])
        plt.xlim([-0.5, hill.shape[0] - 0.5])
        plt.imshow((covered_positions > 0), cmap="BuPu",
                    alpha=0.3)
        plt.draw()
        plt.pause(0.5)

