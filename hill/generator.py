import random,math
import matplotlib.pyplot as plt
import numpy as np


class HillGrid:

    def __init__(self, KRADIUS = (1.0/5.0), ITER=200, SIZE=40):
        self.KRADIUS = KRADIUS
        self.ITER = ITER
        self.SIZE = SIZE

        self.grid = [[0 for x in range(self.SIZE)] for y in range(self.SIZE)]

        self.MAX = self.SIZE * self.KRADIUS
        for i in range(self.ITER):
            self.step()

    def dump(self):
        for ele in self.grid:
            s = ''
            for alo in ele:
                s += '%s ' % str(alo)
            print(s)

    def __getitem__(self, n):
        return self.grid[n]

    def step(self):
        point = [random.randint(0, self.SIZE-1),random.randint(0,self.SIZE-1)]
        radius = random.uniform(0, self.MAX)

        startX = point[0] - (radius/2)
        startY = point[1] - (radius/2)

        if startX < 0: startX = 0
        if startY < 0: startY = 0

        x2 = point[0]
        y2 = point[1]

        for x in range(self.SIZE):
            for y in range(self.SIZE):

                z = (radius**2) - (math.pow(x2-x,2) + math.pow(y2-y,2))
                if z >= 0:
                    self.grid[x][y] += int(z)


if __name__ == '__main__':
    for d in ["training", "test"]:
        for i in range(30):
            print(d, i)
            h = HillGrid(ITER=50, SIZE=15)
            g = np.array(h.grid)
            np.savetxt("hills/%s/hill_%d.txt" % (d, i), g, fmt='%d')
            plt.figure()
            plt.imshow(g, cmap="YlOrBr", interpolation='nearest')
            plt.savefig("hills/%s/hill_%d.png" % (d, i))
            plt.close()
