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
from grid2 import Grid
from keras.model import Sequential
from keras.model import Dense

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
	if a > 180:
		return a - 360
	elif a < -180:
		return a + 360
	else:
		return a
class DQNAgent(object):
	"""DQN agent for discrete state/action spaces."""
	def __init__(self):
		self.alpha = 1
		self.gamma = .2
		self.epsilon = .1
		self.logger = logging.getLogger(__name__)
		if False: # True if you want to see more information
			self.logger.setLevel(logging.DEBUG)
		else:
			self.logger.setLevel(logging.INFO)
		self.logger.handlers = []
		self.logger.addHandler(logging.StreamHandler(sys.stdout))

		self.grid = Grid()
		self.actions = ["turn 90", "turn -90", "forward"] # add jump 1 back in later
		self.q = Sequential()
		self.q.add(Dense(3,activation='relu'))
		self.q.add(Dense(3,activation='relu'))
		self.q.add(Dense(3,activation='sigmoid'))
		self.q.compile(loss='bibary_crossentropy',optimizer='adam',metric=['accuracy'])
		self.canvas = None
		self.root = None

		self.jumping = False
		self.turning = False

	def updateState(self, json_obs):
		self.xpos = json_obs[u'XPos']
		self.ypos = json_obs[u'YPos']
		self.zpos = json_obs[u'ZPos']

		# for some reason the yaw goes beyond (-180, 180] range
		yaw = json_obs[u'Yaw'] 
		if yaw > 180:
			yaw -= 360
		elif yaw < -180:
			yaw += 360
		self.yaw = yaw

		self.pitch = json_obs[u'Pitch']
		self.grid.update(json_obs[u'floor7x7'])

	def is_grounded(self):
		y = self.ypos
		return y - math.floor(y) <= 0.01 and self.grid.getCenter() > 0

	def updateQ( self, reward, current_state ):
		old_q = self.q.predict(self.prev_s)
		max_q = max(self.q.predict(current_state)
		new_q = old_q + self.alpha * (reward + self.gamma * max_q - old_q)
		self.q.fit(new_q,reward,epochs=1)
	def updateQFromTerminatingState( self, reward ):
		old_q = self.q.predict(self.prev_s)
		# assign the new action value to the Q-table
		# adapted from tabular_q_learning.py
		new_q = old_q + self.alpha * (reward - old_q)
		self.q.fit(new_q,reward,epochs=1)
	def act(self, world_state, agent_host, current_r ):
		"""take 1 action in response to the current world state"""
		obs_text = world_state.observations[-1].text
		obs = json.loads(obs_text) # most recent observation
		self.logger.debug(obs)
		if not u'XPos' in obs or not u'ZPos' in obs:
			self.logger.error("Incomplete observation received: %s" % obs_text)
			return 0
		current_s = (int(obs[u'XPos']), int(obs[u'ZPos']), nearest_angle(obs[u'Yaw'])), 
		if self.prev_s is not None and self.prev_a is not None:
			self.updateQ( current_r, current_s )

		# select the next action
		rnd = random.random()
		if rnd < self.epsilon:
			a = random.randint(0, len(self.actions) - 1)
			self.logger.info("Random action: %s" % self.actions[a])
		else:
			m = max(self.q.predict(current_s)
			self.logger.debug("Current values: %s" % ",".join(str(x) for x in self.q.predict(current_s)))
			l = list()
			for x in range(0, len(self.actions)):
				if self.q.predict(current_s)[x] == m:
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
			self.prev_action_string = action

		except RuntimeError as e:
			self.logger.error("Failed to send command: %s" % e)

		return current_r


	def run(self, agent_host):
		"""run the agent on the world"""

		total_reward = 0
		
		self.prev_s = None
		self.prev_a = None
		self.prev_action_string = "NULL"

		# when the agent doesn't observe any new blocks for
		# a few cycles in the row, then just abort this run
		max_timeout = 2
		timeout = max_timeout


		agent_host.sendCommand("move 1")
		
		# main loop:
		world_state = agent_host.getWorldState()

		#I don't get why the first action is isolated from the rest
		while world_state.is_mission_running:
			current_r = 0

			time.sleep(0.05)
			if self.jumping:
				agent_host.sendCommand("jump 0")
				self.jumping = False
	
			time.sleep(0.20)
			if self.turning:
				agent_host.sendCommand("turn 0")
				self.turning = False

			while True: # keep waiting until there is a new observation?
				time.sleep(0.1)

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
						single_r = 0

						# positive reward based on how many blocks
						# in front of the agent
						x, z = int(self.xpos), int(self.zpos)
						print(self.yaw)
						theta = nearest_angle(self.yaw)
						blocks = self.grid.getObsBlocks(theta)
						for v in blocks:
							single_r += v

						# negative reward if the agent decides
						# to turn or jump
						if self.prev_action_string[:4] == "turn":
							single_r -= 3

						print("Reward for this step:", single_r)
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
			self.updateQFromTerminatingState( current_r )
	
		return total_reward
		
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
	num_repeats = 300

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
