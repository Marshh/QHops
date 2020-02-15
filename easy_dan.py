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

from future import standard_library
standard_library.install_aliases()
from builtins import object
from builtins import range
from past.utils import old_div
import MalmoPython
import os
import sys
import time
import json
import numpy
import logging
import time
import random
import pathlib
import platform
import math

from grid import Grid

level_path = str(pathlib.Path().absolute())
if platform.system() == "Windows":
	level_path += "\\EasyWalk"
else:
	level_path += "/EasyWalk"

if sys.version_info[0] == 2:
	# Workaround for https://github.com/PythonCharmers/python-future/issues/262
	import Tkinter as tk
else:
	import tkinter as tk
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
			<Placement x="592.0" y="2.0" z="72.0" yaw="0"/>
		</AgentStart>
		<AgentHandlers>
			<ObservationFromFullStats/>
			<ObservationFromGrid>
				<Grid name="floor7x7">
					<min x="-3" y="-1" z="-3"/>
					<max x="3" y="-1" z="3"/>
				</Grid>
			</ObservationFromGrid>
			<ContinuousMovementCommands turnSpeedDegs="720"/>
			<RewardForTouchingBlockType>
				<Block reward="-1000" type="bedrock" behaviour="onceOnly"/>
				<Block reward="1000" type="lapis_block" behaviour="onceOnly"/>
			</RewardForTouchingBlockType>
			<RewardForSendingCommand reward="-1" />
			<AgentQuitFromTouchingBlockType>
				<Block type="bedrock" />
				<Block type="lapis_block" />
			</AgentQuitFromTouchingBlockType>
			<InventoryCommands/>
			<AgentQuitFromReachingPosition>
				<Marker x="-26.5" y="40" z="0.5" tolerance="0.5" description="Goal_found"/>
			</AgentQuitFromReachingPosition>
			<MissionQuitCommands quitDescription="manual_quit"/>
		</AgentHandlers>
	</AgentSection>
</Mission>'''

# angles will be in multiples of 90 for now
# increase the number of slices when testing more complex courses
def nearest_angle(theta):
	a = round(theta / 90) * 90
	if a == -180:
		return -a
	else:
		return a

def angle_distance(initial, target):
	a = target - initial
	if a > 180.0:
		return a - 360.0
	elif a < -180.0:
		return a + 360.0
	else:
		return a

class TabQAgent(object):
	"""Tabular Q-learning agent for discrete state/action spaces."""

	def __init__(self):
		self.alpha = 1 # learning rate; set it to 1 the environment if determininistic
		self.gamma = 0.2 # discount value; 0 will make the agent greedy
		self.epsilon = 0.1 # chance of taking a random action instead of the best

		self.logger = logging.getLogger(__name__)
		if False: # True if you want to see more information
			self.logger.setLevel(logging.DEBUG)
		else:
			self.logger.setLevel(logging.INFO)
		self.logger.handlers = []
		self.logger.addHandler(logging.StreamHandler(sys.stdout))

		self.grid = Grid()

		# get rid of move actions because the bot should always be moving forward
		# get rid of jump 0 because it's automatically called in the next frame
		self.actions = ["turn 90", "turn -90", "forward"] # add jump 1 back in later
		self.q_table = {}
		self.canvas = None
		self.root = None

		self.jumping = False
		self.turning = False

	def updateState(self, json_obs):
		self.xpos = json_obs[u'XPos']
		self.ypos = json_obs[u'YPos']
		self.zpos = json_obs[u'ZPos']
		self.yaw = json_obs[u'Yaw']
		self.pitch = json_obs[u'Pitch']
		self.grid.update(json_obs[u'floor7x7'])

	def is_grounded(self):
		y = self.ypos
		return y - math.floor(y) <= 0.01 and self.grid.grid[3][3] != "air"

	def updateQTable( self, reward, current_state ):
		"""Change q_table to reflect what we have learnt."""
		
		# retrieve the old action value from the Q-table (indexed by the previous state and the previous action)
		old_q = self.q_table[self.prev_s][self.prev_a]
		
		# TODO: what should the new action value be?
		max_q = max(self.q_table[current_state][:])
		new_q = old_q + self.alpha * (reward + self.gamma * max_q - old_q)
		
		# assign the new action value to the Q-table
		self.q_table[self.prev_s][self.prev_a] = new_q
		
	def updateQTableFromTerminatingState( self, reward ):
		"""Change q_table to reflect what we have learnt, after reaching a terminal state."""
		
		# retrieve the old action value from the Q-table (indexed by the previous state and the previous action)
		old_q = self.q_table[self.prev_s][self.prev_a]
		
		# assign the new action value to the Q-table
		# adapted from tabular_q_learning.py
		self.q_table[self.prev_s][self.prev_a] = old_q + self.alpha * (reward - old_q)
		
	def act(self, world_state, agent_host, current_r ):
		"""take 1 action in response to the current world state"""
		
		obs_text = world_state.observations[-1].text
		obs = json.loads(obs_text) # most recent observation
		self.logger.debug(obs)
		if not u'XPos' in obs or not u'ZPos' in obs:
			self.logger.error("Incomplete observation received: %s" % obs_text)
			return 0

		# current_s = "%d:%d" % (int(obs[u'XPos']), int(obs[u'ZPos']))
		current_s = (int(obs[u'XPos']), int(obs[u'ZPos']), nearest_angle(obs[u'Yaw'])), 

		# self.logger.debug("State: %s (x = %.2f, z = %.2f)" % (current_s, float(obs[u'XPos']), float(obs[u'ZPos'])))
		if current_s not in self.q_table:
			self.q_table[current_s] = ([0] * len(self.actions))

		# update Q values
		if self.prev_s is not None and self.prev_a is not None:
			self.updateQTable( current_r, current_s )

		# self.drawQ( curr_x = int(obs[u'XPos']), curr_y = int(obs[u'ZPos']) )

		# select the next action
		rnd = random.random()
		if rnd < self.epsilon:
			a = random.randint(0, len(self.actions) - 1)
			self.logger.info("Random action: %s" % self.actions[a])
		else:
			m = max(self.q_table[current_s])
			self.logger.debug("Current values: %s" % ",".join(str(x) for x in self.q_table[current_s]))
			l = list()
			for x in range(0, len(self.actions)):
				if self.q_table[current_s][x] == m:
					l.append(x)
			y = random.randint(0, len(l)-1)
			a = l[y]
			self.logger.info("Taking q action: %s" % self.actions[a])

		# try to send the selected action, only update prev_s if this succeeds
		try:
			action = self.actions[a]
			# specialized turning action to help the agent face the right angle
			if action[:4] == "turn":
				val = int(action[5:])
				initial = self.yaw
				target = nearest_angle(initial) + val
				theta = angle_distance(initial, target)
				theta = theta / 90.0 * 0.5
				agent_host.sendCommand("turn " + str(theta))
				print("TURNING", self.yaw, target, theta)
				self.turning = True
			elif action == "jump 1":
				agent_host.sendCommand("jump 1")
				self.jumping = True
			elif action == "forward":
				pass #do nothing; just keep going forward
			else:
				agent_host.sendCommand(action)
			
			self.prev_s = current_s
			self.prev_a = a

		except RuntimeError as e:
			self.logger.error("Failed to send command: %s" % e)

		return current_r


	def run(self, agent_host):
		"""run the agent on the world"""

		total_reward = 0
		
		self.prev_s = None
		self.prev_a = None

		observed = set()
		# when the agent doesn't observe any new blocks for
		# a few cycles in the row, then just abort this run
		max_timeout = 2
		timeout = max_timeout


		agent_host.sendCommand("move 1")
		
		# main loop:
		world_state = agent_host.getWorldState()

		# first action
		if world_state.is_mission_running:
			# wait until have received a valid observation
			current_r = 0
			while True:
				time.sleep(0.1)
				world_state = agent_host.getWorldState()
				for error in world_state.errors:
					self.logger.error("Error: %s" % error.text)
				for reward in world_state.rewards:
					current_r += reward.getValue()
				if world_state.is_mission_running and len(world_state.observations)>0 and not world_state.observations[-1].text=="{}":
					msg = world_state.observations[-1].text
					observations = json.loads(msg)
					self.updateState(observations)

					# add all initial blocks to the observed set without rewards
					x, z = int(self.xpos), int(self.zpos)
					gr = self.grid.grid
					for i in range(7):
						for j in range(7):
							bl = gr[i][j]
							if bl == "gold_block" or bl == "lapis_block" or bl == "emerald_block":
								coord = (x + j - 3, z + i - 3)
								observed.add(coord)	


					total_reward += self.act(world_state, agent_host, 0)
					break
				if not world_state.is_mission_running:
					break

		# next actions
		while world_state.is_mission_running:
			# wait for non-zero reward
			# while world_state.is_mission_running and current_r == 0:
			# 	time.sleep(0.1)
			# 	world_state = agent_host.getWorldState()
			# 	for error in world_state.errors:
			# 		self.logger.error("Error: %s" % error.text)
			# 	for reward in world_state.rewards:
			# 		current_r += reward.getValue()
			# allow time to stabilise after action
			current_r = 0
			while True: # keep waiting until there is a new observation?

				time.sleep(0.05)
				if self.jumping:
					agent_host.sendCommand("jump 0")
					self.jumping = False
		
				time.sleep(0.20)
				if self.turning:
					agent_host.sendCommand("turn 0")
					self.turning = False
					time.sleep(0.1)

				# ~0.25 second delay


				world_state = agent_host.getWorldState()
				for error in world_state.errors:
					self.logger.error("Error: %s" % error.text)
				for reward in world_state.rewards:
					current_r += reward.getValue()
				if world_state.is_mission_running and len(world_state.observations)>0 and not world_state.observations[-1].text=="{}":
					# why is the agent acting twice?
					# total_reward += self.act(world_state, agent_host, current_r)

					msg = world_state.observations[-1].text
					observations = json.loads(msg)
					self.updateState(observations)

					# don't do anything if not grounded
					if self.is_grounded():
						observed_new_block = False

						x, z = int(self.xpos), int(self.zpos)
						single_r = 0
						gr = self.grid.grid
						for i in range(7):
							for j in range(7):
								bl = gr[i][j]
								if bl == "gold_block" or bl == "lapis_block" or bl == "emerald_block":
									coord = (x + j - 3, z + i - 3)
									if coord in observed:
										single_r -= 2
									else:
										observed.add(coord)
										single_r += 10
										observed_new_block = True

						if observed_new_block:
							timeout = max_timeout
						else:
							timeout -= 1
							if timeout <= 0:
								# just stop the run because we're not getting anywhere
								current_r -= 1000
								print("TIMEOUT")
								agent_host.sendCommand("quit")
								break

						print("Observed Reward:", single_r)
						current_r += single_r
						total_reward += self.act(world_state, agent_host, current_r)
					break
				if not world_state.is_mission_running:
					break

		# process final reward
		self.logger.info("Final reward: %d" % current_r)
		total_reward += current_r

		# update Q values
		if self.prev_s is not None and self.prev_a is not None:
			self.updateQTableFromTerminatingState( current_r )
			
		# self.drawQ()
	
		return total_reward
		
	# def drawQ( self, curr_x=None, curr_y=None ):
	#     scale = 40
	#     world_x = 6
	#     world_y = 14
	#     if self.canvas is None or self.root is None:
	#         self.root = tk.Tk()
	#         self.root.wm_title("Q-table")
	#         self.canvas = tk.Canvas(self.root, width=world_x*scale, height=world_y*scale, borderwidth=0, highlightthickness=0, bg="black")
	#         self.canvas.grid()
	#         self.root.update()
	#     self.canvas.delete("all")
	#     action_inset = 0.1
	#     action_radius = 0.1
	#     curr_radius = 0.2
	#     action_positions = [ ( 0.5, action_inset ), ( 0.5, 1-action_inset ), ( action_inset, 0.5 ), ( 1-action_inset, 0.5 ) ]
	#     # (NSWE to match action order)
	#     min_value = -20
	#     max_value = 20
	#     for x in range(world_x):
	#         for y in range(world_y):
	#             s = "%d:%d" % (x,y)
	#             self.canvas.create_rectangle( x*scale, y*scale, (x+1)*scale, (y+1)*scale, outline="#fff", fill="#000")
	#             for action in range(4):
	#                 if not s in self.q_table:
	#                     continue
	#                 value = self.q_table[s][action]
	#                 color = int( 255 * ( value - min_value ) / ( max_value - min_value )) # map value to 0-255
	#                 color = max( min( color, 255 ), 0 ) # ensure within [0,255]
	#                 color_string = '#%02x%02x%02x' % (255-color, color, 0)
	#                 self.canvas.create_oval( (x + action_positions[action][0] - action_radius ) *scale,
	#                                          (y + action_positions[action][1] - action_radius ) *scale,
	#                                          (x + action_positions[action][0] + action_radius ) *scale,
	#                                          (y + action_positions[action][1] + action_radius ) *scale, 
	#                                          outline=color_string, fill=color_string )
	#     if curr_x is not None and curr_y is not None:
	#         self.canvas.create_oval( (curr_x + 0.5 - curr_radius ) * scale, 
	#                                  (curr_y + 0.5 - curr_radius ) * scale, 
	#                                  (curr_x + 0.5 + curr_radius ) * scale, 
	#                                  (curr_y + 0.5 + curr_radius ) * scale, 
	#                                  outline="#fff", fill="#fff" )
	#     self.root.update()
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
if sys.version_info[0] == 2:
	sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)  # flush print output immediately
else:
	import functools
	print = functools.partial(print, flush=True)

agent = TabQAgent()
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

# -- set up the mission -- #

max_retries = 3

if agent_host.receivedArgument("test"):
	num_repeats = 1
else:
	num_repeats = 150

cumulative_rewards = []
for i in range(num_repeats):

	print()
	print('Repeat %d of %d' % ( i+1, num_repeats ))
	
	my_mission_record = MalmoPython.MissionRecordSpec()

	for retry in range(max_retries):
		try:
			agent_host.startMission( my_mission, my_mission_record )
			break
		except RuntimeError as e:
			if retry == max_retries - 1:
				print("Error starting mission:",e)
				exit(1)
			else:
				time.sleep(2.5)

	print("Waiting for the mission to start", end=' ')
	world_state = agent_host.getWorldState()
	while not world_state.has_mission_begun:
		print(".", end="")
		time.sleep(0.1)
		world_state = agent_host.getWorldState()
		for error in world_state.errors:
			print("Error:",error.text)
	print()

	# -- run the agent in the world -- #
	cumulative_reward = agent.run(agent_host)
	print('Cumulative reward: %d' % cumulative_reward)
	cumulative_rewards += [ cumulative_reward ]

	# -- clean up -- #
	time.sleep(0.5) # (let the Mod reset)

print("Done.")

print()
print("Cumulative rewards for all %d runs:" % num_repeats)
print(cumulative_rewards)
