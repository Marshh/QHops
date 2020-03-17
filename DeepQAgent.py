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
import matplotlib.pyplot as plt
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
from keras.models import Sequential
from keras.layers import Input, Dense, Activation
from keras.models import Model, model_from_json
from keras.optimizers import Adam, RMSprop

level_path = str(pathlib.Path().absolute())
if platform.system() == "Windows":
    level_path += "\\Easy2"
else:
    level_path += "/Easy2"

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

missionXML ='''
<?xml version="1.0" encoding="UTF-8" standalone="no"?>
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
        <FlatWorldGenerator generatorString="2;7;1;"/>
        <DrawingDecorator>
        </DrawingDecorator>
        <ServerQuitFromTimeUp timeLimitMs="15000"/>
        <ServerQuitWhenAnyAgentFinishes/>
    </ServerHandlers>
    </ServerSection>
    <AgentSection mode="Survival">
    <Name>MalmoTutorialBot</Name>
    <AgentStart>
        <Placement x="0.5" y="1.0" z="0.5" yaw="0"/>
    </AgentStart>
    <AgentHandlers>
        <ContinuousMovementCommands/>
        <ObservationFromFullStats/>
        <ObservationFromGrid>
            <Grid name="layer_0">
                <min x="-2" y="-1" z="-2"/>
                <max x="2" y="-1" z="2"/>
            </Grid>
            <Grid name="layer_1">
                <min x="-2" y="0" z="-2"/>
                <max x="2" y="0" z="2"/>
            </Grid>
        </ObservationFromGrid>
        <ContinuousMovementCommands turnSpeedDegs="360"/>
        <RewardForTouchingBlockType>
        <Block reward="-100" type="bedrock" behaviour="onceOnly"/>
        <Block reward="100" type="lapis_block" behaviour="onceOnly"/>
        </RewardForTouchingBlockType>
        <RewardForSendingCommand reward="-1"/>
        <AgentQuitFromTouchingBlockType>
        <Block type="bedrock"/>
        <Block type="lapis_block"/>
        </AgentQuitFromTouchingBlockType>
        <InventoryCommands/>
        <AgentQuitFromReachingPosition>
        <Marker x="-26.5" y="40" z="0.5" tolerance="0.5" description="Goal_found"/>
        </AgentQuitFromReachingPosition>
        <MissionQuitCommands quitDescription="manual_quit"/>
    </AgentHandlers>
    </AgentSection>
</Mission>
'''
block_values = {
    "air":0,
    "gold_block":1,
    "lapis_block":2,
    "bedrock":3
}
course_length = 10

goal_x = 0
goal_z = 0
goal_y = 0

def convert_grid(grid):
    for i in range(len(grid)):
        grid[i] = block_values[grid[i]]

def generate_jump_course(length,  mission):
    global goal_x
    global goal_z
    global goal_y

    my_mission.drawBlock(0, 0, 0, "gold_block")

    ## current y not needed atm for this function since it's a flat course
    current_x = 0
    current_y = 0
    current_z = 0

    for i in range(length):
        if random.randint(1,2) == 1:
            current_x += 2
        else:
            current_z += 2

        current_y +=1

        my_mission.drawBlock(current_x, current_y, current_z, "gold_block")

    if random.randint(1,2) == 1:
        current_x += 2
    else:
        current_z += 2 
    goal_x = current_x
    goal_z = current_z
    goal_y = current_y
    my_mission.drawBlock(current_x, current_y, current_z, "lapis_block")

def clear_jump_course(length,mission):
    current_y = 0
    current_x = 0
    current_z = 0
    for i in range(length):
        for j in range(length):
            for k in range(length):
                my_mission.drawBlock(current_x+i,current_y+k , current_z+j, "bedrock")
    my_mission.drawBlock(0, 0, 0, "gold_block")  

        
class DQNAgent(object):
    """Deep Q-learning agent for continuous state/action spaces."""

    def __init__(self):
        self.epsilon = 0.01 # chance of taking a random action instead of the best

        self.logger = logging.getLogger(__name__)
        if False: # True if you want to see more information
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)
        self.logger.handlers = []
        self.logger.addHandler(logging.StreamHandler(sys.stdout))

        # get rid of move -1 because parkour players don't move backwards
        self.actions = ["move 1", "move 0", "turn .5", "turn -.5", "turn 0", "jump 0", "jump 1"]
        self.q_table = {}
        self.canvas = None
        self.root = None

        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.action_size = 9

        self.model = Sequential()

        self.model.add(Dense(33, input_dim=4, activation='sigmoid'))
        self.model.add(Dense(9, activation='sigmoid'))
        self.model.add(Dense(9, activation='sigmoid'))
        self.model.add(Dense(7, activation='sigmoid'))
        self.model.compile(loss="mse", optimizer=Adam(lr=self.learning_rate))
    
    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state))

    def act(self, world_state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action
        

    def replay(self, batch_size):
    minibatch = random.sample(self.memory, batch_size)
    for state, action, reward, next_state in minibatch:
        target = reward
        if not done:
          target = reward + self.gamma * \
                   np.amax(self.model.predict(next_state)[0])
        target_f = self.model.predict(state)
        target_f[0][action] = target
        self.model.fit(state, target_f, epochs=1, verbose=0)
    if self.epsilon > self.epsilon_min:
        self.epsilon *= self.epsilon_decay

    def run(self, agent_host):
        """run the agent on the world"""

        total_reward = 0
        
        self.prev_s = None
        self.prev_a = None
        
        is_first_action = True
        
        # main loop:
        world_state = agent_host.getWorldState()
        while world_state.is_mission_running:

            current_r = 0
            
            if is_first_action:
                # wait until have received a valid observation
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
                        state = []                        
                        state.extend(observations.get(u'layer_0', 0))
                        state.extend(observations.get(u'layer_1', 0))
                        convert_grid(state)
                        new_yaw = observations.get(u'Yaw')   
                        state += new_yaw

                        # self.grid.print()

                        ## Figure out good amount to reward based on position and observations
                        self.act(world_state, agent_host, current_r)
                        break
                    if not world_state.is_mission_running:
                        break
                is_first_action = False
            else:
                # wait for non-zero reward
                while world_state.is_mission_running and current_r == 0:
                    time.sleep(0.1)
                    world_state = agent_host.getWorldState()
                    for error in world_state.errors:
                        self.logger.error("Error: %s" % error.text)
                    for reward in world_state.rewards:
                        current_r += reward.getValue()
                # allow time to stabilise after action
                while True:
                    time.sleep(0.1)
                    world_state = agent_host.getWorldState()
                    for error in world_state.errors:
                        self.logger.error("Error: %s" % error.text)
                    for reward in world_state.rewards:
                        current_r += reward.getValue()
                    if world_state.is_mission_running and len(world_state.observations)>0 and not world_state.observations[-1].text=="{}":
                        
                        self.act(world_state, agent_host, current_r)
                        break
                    if not world_state.is_mission_running:
                        break

        # process final reward
        self.logger.debug("Final reward: %d" % current_r)
        total_reward += current_r

        # update Q values
        if self.prev_s is not None and self.prev_a is not None:
            self.updateQTableFromTerminatingState( current_r )
            
        # self.drawQ()
    
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
clear_flat_course(course_length, my_mission)
generate_jump_course(course_length,my_mission)


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
    num_repeats = 250

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
    clear_flat_course(course_length, my_mission)
    generate_jump_course(course_length, my_mission)
clear_flat_course(course_length, my_mission)    
xi = [x for x in range(len(cumulative_rewards))]
plt.figure()
plt.xlabel("Episode")
plt.ylabel("Cumulative Rewards")
plt.plot(xi, cumulative_rewards)
plt.savefig("DiscreteRandomNoJumps.png")
print("Done.")

print()
print("Cumulative rewards for all %d runs:" % num_repeats)
print(cumulative_rewards)
