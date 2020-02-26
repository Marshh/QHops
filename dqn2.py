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
import numpy as np
import logging
import time
import random
import pathlib
import platform
import math
from grid2 import Grid
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *

import random, numpy, math, gym, sys
from keras import backend as K

import tensorflow as tfimport random, numpy, math, gym, sys
from keras import backend as K

import tensorflow as tf

LEARNING_RATE = 0.00025
MEMORY_CAPACITY = 100000
BATCH_SIZE = 64

GAMMA = 0.99

MAX_EPSILON = 1
MIN_EPSILON = 0.01
LAMBDA = 0.001      # speed of decay

UPDATE_TARGET_FREQUENCY = 1000

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
def huber_loss(y_true, y_pred):
    err = y_true - y_pred

    cond = K.abs(err) < HUBER_LOSS_DELTA
    L2 = 0.5 * K.square(err)
    L1 = HUBER_LOSS_DELTA * (K.abs(err) - 0.5 * HUBER_LOSS_DELTA)

    loss = tf.where(cond, L2, L1)   # Keras does not cover where function in tensorflow :-(

    return K.mean(loss)


class Brain:
    def __init__(self, stateCnt, actionCnt):
        self.stateCnt = stateCnt
        self.actionCnt = actionCnt

        self.model = self._createModel()
        self.model_ = self._createModel() 

    def _createModel(self):
        model = Sequential()

        model.add(Dense(units=64, activation='relu', input_dim=stateCnt))
        model.add(Dense(units=actionCnt, activation='linear'))

        opt = RMSprop(lr=LEARNING_RATE)
        model.compile(loss=huber_loss, optimizer=opt)

        return model

    def train(self, x, y, epochs=1, verbose=0):
        self.model.fit(x, y, batch_size=64, epochs=epochs, verbose=verbose)

    def predict(self, s, target=False):
        if target:
            return self.model_.predict(s)
        else:
            return self.model.predict(s)

    def predictOne(self, s, target=False):
        return self.predict(s.reshape(1, self.stateCnt), target=target).flatten()

    def updateTargetModel(self):
        self.model_.set_weights(self.model.get_weights())


class Memory:   # stored as ( s, a, r, s_ )
    samples = []

    def __init__(self, capacity):
        self.capacity = capacity

    def add(self, sample):
        self.samples.append(sample)        

        if len(self.samples) > self.capacity:
            self.samples.pop(0)

    def sample(self, n):
        n = min(n, len(self.samples))
        return random.sample(self.samples, n)

    def isFull(self):
        return len(self.samples) >= self.capacity



class DQNAgent(object):
	"""DQN agent for discrete state/action spaces."""
	def __init__(self,stateCnt,actionCnt):
		self.stateCnt = stateCnt
		self.actionCnt = actionCnt
		self.brain = Brain(stateCnt,actionCnt)
		
		self.memory = Memory(MEMORY_CAPACITY)
	
 	def act(self, s):
 		if random.random() < self.epsilon:
            return random.randint(0, self.actionCnt-1)
        else:
            return numpy.argmax(self.brain.predictOne(s))
  	def observe(self, sample):  # in (s, a, r, s_) format
        self.memory.add(sample)        

        if self.steps % UPDATE_TARGET_FREQUENCY == 0:
            self.brain.updateTargetModel()

        # debug the Q function in poin S
        if self.steps % 100 == 0:
            S = numpy.array([-0.01335408, -0.04600273, -0.00677248, 0.01517507])
            pred = agent.brain.predictOne(S)
            print(pred[0])
            sys.stdout.flush()

        # slowly decrease Epsilon based on our eperience
        self.steps += 1
        self.epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * math.exp(-LAMBDA * self.steps)

    def replay(self):    
        batch = self.memory.sample(BATCH_SIZE)
        batchLen = len(batch)

        no_state = numpy.zeros(self.stateCnt)

        states = numpy.array([ o[0] for o in batch ])
        states_ = numpy.array([ (no_state if o[3] is None else o[3]) for o in batch ])

        p = self.brain.predict(states)
        p_ = self.brain.predict(states_, target=True)

        x = numpy.zeros((batchLen, self.stateCnt))
        y = numpy.zeros((batchLen, self.actionCnt))
        
        for i in range(batchLen):
            o = batch[i]
            s = o[0]; a = o[1]; r = o[2]; s_ = o[3]
            
            t = p[i]
            if s_ is None:
                t[a] = r
            else:
                t[a] = r + GAMMA * numpy.amax(p_[i])

            x[i] = s
            y[i] = t

        self.brain.train(x, y)
	
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
						total_reward += self.act(world_state)
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

agent = DQNAgent()
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
HUBER_LOSS_DELTA = 1.0
