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
        <DiscreteMovementCommands/>
        <ObservationFromFullStats/>
        <ObservationFromGrid>
            <Grid name="layer_0">
                <min x="-1" y="-1" z="-1"/>
                <max x="1" y="-1" z="1"/>
            </Grid>
            <Grid name="layer_1">
                <min x="-1" y="0" z="-1"/>
                <max x="1" y="0" z="1"/>
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

def convert_grid(grid):
    for i in range(len(grid)):
        grid[i] = block_values[grid[i]]

def generate_flat_course(length,  mission):
    my_mission.drawBlock(0, 0, 0, "gold_block")

    ## current y not needed atm for this function since it's a flat course
    current_x = 0
    current_y = 0
    current_z = 0

    for i in range(length):
        if random.randint(1,2) == 1:
            current_x +=1
        else:
            current_z += 1
        if random.randint(1,2)==1:
            current_y+=1
        else:
            current_y-=1
        if current_y<0:
            current_y=0
        my_mission.drawBlock(current_x, current_y, current_z, "gold_block")

    if random.randint(1,2) == 1:
        current_x +=1
    else:
        current_z += 1
    if random.randint(1,2)==1:
        current_y+=1
    else:
        current_y-=1
        if current_y<0:
            current_y=0
    goal_x = current_x
    goal_z = current_z
    my_mission.drawBlock(current_x, current_y, current_z, "lapis_block")

def clear_flat_course(length,mission):
    current_x = 0
    current_z = 0
    for i in range(length):
        for j in range(length):
            my_mission.drawBlock(current_x+i, 0, current_z+j, "bedrock")
    my_mission.drawBlock(0, 0, 0, "gold_block")  

        
class TabQAgent(object):
    """Tabular Q-learning agent for discrete state/action spaces."""

    def __init__(self):
        self.epsilon = 0.5 # chance of taking a random action instead of the best

        self.logger = logging.getLogger(__name__)
        if False: # True if you want to see more information
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)
        self.logger.handlers = []
        self.logger.addHandler(logging.StreamHandler(sys.stdout))

        # get rid of move -1 because parkour players don't move backwards
        self.actions = ["movenorth", "movesouth", "moveeast", "movewest","jumpnorth 1", "jumpsouth 1", "jumpeast 1", "jumpwest 1"]
        self.q_table = {}
        self.canvas = None
        self.root = None
        

    def updateQTable( self, reward, current_state ):
        """Change q_table to reflect what we have learnt."""
        
        # retrieve the old action value from the Q-table (indexed by the previous state and the previous action)
        old_q = self.q_table[self.prev_s][self.prev_a]
        
        # TODO: what should the new action value be?
        alpha = 1.0
        gamma = 0.8

        max_q = max(self.q_table[current_state][:])
        
        new_q = old_q + alpha * (reward + gamma * max_q - old_q)
        self.epsilon*=.99
        
        # assign the new action value to the Q-table
        self.q_table[self.prev_s][self.prev_a] = new_q
        
    def updateQTableFromTerminatingState( self, reward ):
        """Change q_table to reflect what we have learnt, after reaching a terminal state."""
        
        # retrieve the old action value from the Q-table (indexed by the previous state and the previous action)
        old_q = self.q_table[self.prev_s][self.prev_a]
        
        # TODO: what should the new action value be?
        new_q = reward
        self.epsilon*=.99
        # assign the new action value to the Q-table
        self.q_table[self.prev_s][self.prev_a] = new_q
        
    def act(self, world_state, agent_host, current_r ):
        """take 1 action in response to the current world state"""
        
        obs_text = world_state.observations[-1].text
        obs = json.loads(obs_text) # most recent observation
        self.logger.debug(obs)
        if not u'XPos' in obs or not u'ZPos' in obs:
            self.logger.error("Incomplete observation received: %s" % obs_text)
            return 0
        
        new_layers = []                        
        new_layers.extend(obs.get(u'layer_0', 0))
        new_layers.extend(obs.get(u'layer_1', 0))
        convert_grid(new_layers)
        print(new_layers)
        current_s = str(new_layers)

        #self.logger.debug("State: %s (x = %.2f, z = %.2f)" % (current_s, float(obs[u'XPos']), float(obs[u'ZPos'])))
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
            agent_host.sendCommand(self.actions[a])
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
        
        is_first_action = True

        visited = {}
        
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
                        new_layers = []                        
                        new_layers.extend(observations.get(u'layer_0', 0))
                        new_layers.extend(observations.get(u'layer_1', 0))
                        new_yaw = observations.get(u'Yaw')   


                        # self.grid.print()

                        ## Figure out good amount to reward based on position and observations
                        total_reward += self.act(world_state, agent_host, current_r)
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
                        total_reward += self.act(world_state, agent_host, current_r)
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
generate_flat_course(course_length,my_mission)


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
    generate_flat_course(course_length, my_mission)
clear_flat_course(course_length, my_mission)    
xi = [x for x in range(len(cumulative_rewards))]
plt.figure()
plt.xlabel("Episode")
plt.ylabel("Cumulative Rewards")
plt.plot(xi, cumulative_rewards)
plt.savefig("DiscreteRandomJumps.png")
print("Done.")

print()
print("Cumulative rewards for all %d runs:" % num_repeats)
print(cumulative_rewards)
