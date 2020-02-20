import numpy as np

GRID_SIZE = 7 # just keep it at 7 for now

# any other blocks will be bedrock
BLOCK_MAP = {
	"air" : 0,
	"gold_block" : 1,
	"emerald_block" : 1,
	"lapis_block" : 2,
}

# values are the center of observation area
# maybe add values for 22.5 degree angles increments later
OBS_RANGE = {
	   0 : (1, 3),
	  45 : (1, 5),
	  90 : (3, 5),
	 135 : (5, 5),
	 180 : (5, 3),
	-135 : (5, 1),
	 -90 : (3, 1),
	 -45 : (1, 1)
}

# x x o o o x x
# x x o o o x x
# x x o o o x x
# x x x A x x x
# x x x x x x x
# x x x x x x x
# x x x x x x x

class Grid:
	def __init__(self):
		self.grid = np.zeros((GRID_SIZE, GRID_SIZE))

	def get(self, row, col):
		return self.grid[row][col]

	def update(self, json_grid):
		for i, s in enumerate(json_grid):
			v = BLOCK_MAP.get(s, -1)
			gi = i // GRID_SIZE
			gj = i % GRID_SIZE
			self.grid[GRID_SIZE-1-gi][GRID_SIZE-1-gj] = v

	def getCenter(self):
		c = GRID_SIZE // 2
		return self.grid[c][c]

	# returns the 3x3 grid of blocks directly in front of the agent
	# the 3x3 area location will depend on agent's direction
	def getObsBlocks(self, yaw):
		obs = []
		ci, cj = OBS_RANGE[yaw]
		for i in range(ci-1, ci+2):
			for j in range(cj-1, cj+2):
				obs.append(self.grid[i][j])
		return obs

	def print(self):
		print(self.grid)
