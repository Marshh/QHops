import numpy as np

GRID_SIZE = 7
BLOCK_MAP = {
	"air" : 0,
	"gold_block" : 1,
	"emerald_block" : 1,
	"lapis_block" : 2,
}

class Grid:
	total_size = GRID_SIZE * GRID_SIZE

	def __init__(self):
		self.str_len = 0 #for formatting spaces
		self.grid = [] #for visualization purposes
		for i in range(GRID_SIZE):
			self.grid.append(["null"] * GRID_SIZE)
		self.arr = np.zeros(self.total_size) #for neural network

	def update(self, json_grid):
		self.str_len = 0
		n = GRID_SIZE * GRID_SIZE
		for i, s in enumerate(json_grid):
			s = str(s)
			self.str_len = max(self.str_len, len(s))

			v = BLOCK_MAP.get(s, -1)
			self.arr[self.total_size-1-i] = v

			gi = i // GRID_SIZE
			gj = i % GRID_SIZE
			self.grid[GRID_SIZE-1-gi][GRID_SIZE-1-gj] = s

	def print(self):
		for row in self.grid:
			for s in row:
				print(s.rjust(self.str_len), end = " ")
			print()
