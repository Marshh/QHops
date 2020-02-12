from __future__ import print_function
from __future__ import division
# ------------------------------------------------------------------------------------------------
# Copyright (c) 2016 Microsoft Corporation
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
# associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute,
# sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all copies or
# substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
# NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# ------------------------------------------------------------------------------------------------

# Tutorial sample #4: Challenge - get to the centre of the sponge

from builtins import range
from past.utils import old_div
import MalmoPython
import os
import sys
import time
import json
import numpy
import pathlib
import platform

import grid

level_path = str(pathlib.Path().absolute())
if platform.system() == "Windows":
    level_path += "\\EasyWalk"
else:
    level_path += "/EasyWalk"

if sys.version_info[0] == 2:
	sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)  # flush print output immediately
else:
	import functools
	print = functools.partial(print, flush=True)

missionXML = '''<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
<Mission xmlns="http://ProjectMalmo.microsoft.com" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
	<About>
		<Summary>Hello world!</Summary>
	</About>
	<ServerSection>
		<ServerInitialConditions>
			<Time>
				<StartTime>1000</StartTime>
				<AllowPassageOfTime>false</AllowPassageOfTime>
			</Time>
			<Weather>clear</Weather>
		</ServerInitialConditions>
		<ServerHandlers>
			<FileWorldGenerator src="''' + level_path + '''"/>
				<ServerQuitFromTimeUp timeLimitMs="15000"/>
				<ServerQuitWhenAnyAgentFinishes/>
			</ServerHandlers>
	</ServerSection>
	<AgentSection mode="Survival">
		<Name>MalmoTutorialBot</Name>
		<AgentStart>
			<Placement x="592.0" y="4.0" z="72.0" yaw="0"/>
		</AgentStart>
		<AgentHandlers>
			<ObservationFromFullStats/>
			<ObservationFromGrid>
				<Grid name="floor7x7">
					<min x="-3" y="-1" z="-3"/>
					<max x="3" y="-1" z="3"/>
				</Grid>
			</ObservationFromGrid>
			<ContinuousMovementCommands turnSpeedDegs="900"/>
			<InventoryCommands/>
			<AgentQuitFromReachingPosition>
				<Marker x="-26.5" y="40" z="0.5" tolerance="0.5" description="Goal_found"/>
			</AgentQuitFromReachingPosition>
		</AgentHandlers>
	</AgentSection>
</Mission>'''

class Grid:
	offset = {
		"NORTH" : (-1, 0),
		"SOUTH" : (1, 0),
		"EAST" : (0, 1),
		"WEST" : (0, -1)
	}
	yaw_lookup = {
		"NORTH" : 0,
		"SOUTH" : 180,
		"EAST" : 90,
		"WEST": -90
	}

	def __init__(self):
		self.grid = [["NULL"] * 7 for i in range(7)]
		self.dir = "NORTH"

	def updateGrid(self, json_grid):
		for index, s in enumerate(json_grid):
			i = index // 7
			j = index % 7
			self.grid[6-i][6-j] = s

	def updateDir(self, json_yaw):
		deg = int(json_yaw)
		if deg > -45 and deg < 45:
			self.dir = "NORTH"
		elif deg > 45 and deg < 135:
			self.dir = "EAST"
		elif deg < -45 and deg > -135:
			self.dir = "WEST"
		else:
			self.dir = "SOUTH"

	# "front", "left", "right", "black"
	def check(self, direction):
		g = self.grid
		i, j = self.offset[self.dir]
		if direction == "left":
			i, j = -j, i
		elif direction == "right":
			i, j = j, -i
		elif direction == "back":
			i, j = -i, -j

		li = []
		for x in range(1, 4):
			li.append(g[3+i*x][3+j*x])
		return li

	def print(self):
		print("Direction:", self.dir)
		for row in self.grid:
			print(row)


# Create default Malmo objects:

agent_host = MalmoPython.AgentHost()
try:
	agent_host.parse( sys.argv )
except RuntimeError as e:
	print('ERROR:',e)
	print(agent_host.getUsage())
	exit(1)
if agent_host.receivedArgument("help"):
	print(agent_host.getUsage())
	exit(0)

my_mission = MalmoPython.MissionSpec(missionXML, True)
my_mission_record = MalmoPython.MissionRecordSpec()

# Attempt to start a mission:
max_retries = 3
for retry in range(max_retries):
	try:
		agent_host.startMission( my_mission, my_mission_record )
		break
	except RuntimeError as e:
		if retry == max_retries - 1:
			print("Error starting mission:",e)
			exit(1)
		else:
			time.sleep(2)

# Loop until mission starts:
print("Waiting for the mission to start ", end=' ')
world_state = agent_host.getWorldState()
while not world_state.has_mission_begun:
	print(".", end="")
	time.sleep(0.1)
	world_state = agent_host.getWorldState()
	for error in world_state.errors:
		print("Error:",error.text)

print()
print("Mission running ", end=' ')

#THE PLAN
# observe a 7x7 grid under player
# if the two tiles in front of player are 'air' then turn
# if the two tiles are 'air' followed by 'gold_block' then jump
# TODO: make sure to figure out agent's angle to determine which blocks are in front of it

time.sleep(1) #to make sure everything is spawned in; might remove later
my_grid = grid.Grid()

agent_host.sendCommand("move 1")

agent_state = "walking"
turning = 0
turn_target = 0


while world_state.is_mission_running:
	# print(".", end="")
	# time.sleep(0.1)

	world_state = agent_host.getWorldState()
	for error in world_state.errors:
		print("Error:",error.text)
	for error in world_state.errors:
		print("Error:",error.text)
	if world_state.number_of_observations_since_last_state > 0:
		msg = world_state.observations[-1].text
		observations = json.loads(msg)

		my_grid.update(observations.get(u'floor7x7', 0))

		my_grid.print()
		print(my_grid.arr)

		# if agent_state == "walking":
		# 	front = grid.check("front")
		# 	if front[-1] == "air" and front[-2] == "air":
		# 		#decide which way to turn
		# 		left = grid.check("left")
		# 		right = grid.check("right")
		# 		lcount = left.count("air")
		# 		rcount = right.count("air")
		# 		if lcount < rcount:
		# 			agent_host.sendCommand("turn -1")
		# 		else:
		# 			agent_host.sendCommand("turn 1")
		# 		agent_state = "turning"
		# elif agent_state == "turning":
		# 	agent_host.sendCommand("turn 0")
		# 	agent_state = "walking"


	time.sleep(0.5)
			

print()
print("Mission ended")
# Mission has ended.
