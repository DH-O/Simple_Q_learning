import numpy as np

class OneLineToCell:
    def __init__(self, x_dim, y_dim):
        self.maze = np.zeros([x_dim, y_dim])
        self.state = 0
        self.observation_space_dim = self.maze.size
        self.row = len(self.maze)
        self.col = len(self.maze[0])
    
    def FillGridByOneLineArray(self, array):
        for i in range(self.row):
            for j in range(self.col):
                self.maze[i][j] = array[i*self.row + j]
        return self.maze