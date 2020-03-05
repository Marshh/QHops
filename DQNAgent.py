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

## DQN Agent

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
import keras
from keras.models import Sequential
from keras.layers import Input, Dense, Activation
from keras.models import Model, model_from_json
from keras.optimizers import Adam, RMSprop
from collections import deque
from grid2 import Grid

level_path = str(pathlib.Path().absolute())
if platform.system() == "Windows":
    level_path += "\\FlatWorld"
else:
    level_path += "/FlatWorld"

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

missionXML = '''
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
        <Placement x="0.0" y="2.0" z="0.0" yaw="0"/>
    </AgentStart>
    <AgentHandlers>
        <ObservationFromFullStats/>
        <ObservationFromGrid>
            <Grid name="layer_0">
                <min x="-3" y="-1" z="-3"/>
                <max x="3" y="-1" z="3"/>
            </Grid>
            <Grid name="layer_1">
                <min x="-3" y="0" z="-3"/>
                <max x="3" y="0" z="3"/>
            </Grid>
        </ObservationFromGrid>
        <ContinuousMovementCommands turnSpeedDegs="360"/>
        <RewardForTouchingBlockType>
        <Block reward="-1000" type="bedrock" behaviour="onceOnly"/>
        <Block reward="1000" type="lapis_block" behaviour="onceOnly"/>
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

EPISODES = 1000
memory = deque(maxlen=2000)

gamma = 0.90    # discount rate
epsilon = 1.0  # exploration rate
epsilon_min = 0.001
epsilon_decay = 0.999
batch_size = 64
state_size = 99
action_size = 9
train_start = 1000


block_values = {
    "air":0,
    "gold_block":1,
    "lapis_block":2
}

## Initialize DQN Model
model = Sequential()
model.add(Dense(99, input_dim=4, activation='relu'))
model.add(Dense(9, activation='sigmoid'))
model.compile(loss="mse", optimizer=RMSprop(lr=0.00025, rho=0.95, epsilon=0.01), metrics=["accuracy"])

## Determine what action to take based on the DQN Output
def action(action):
    if action == 0:
        agent_host.sendCommand("turn .25")
        print("Turning Right")

    elif action == 1:
        agent_host.sendCommand("turn -.25")
        print("Turning Left")

    elif action == 2:
        agent_host.sendCommand("turn 0")
        print("Stop Turning")

    elif action == 3:
        agent_host.sendCommand("move -1")
        print("Moving Backwards")

    elif action == 4:
        agent_host.sendCommand("move 1")
        print("Moving Forward")

    elif action == 5:
        agent_host.sendCommand("move 0")
        print("Stop Moving")

    elif action == 6:
        agent_host.sendCommand("jump 0")
        print("Stop Jumping")

    elif action == 7:
        agent_host.sendCommand("jump 1")
        print("Start Jump")

    elif action == 8:
        agent_host.sendCommand("jumpmove 1")
        print("Right Jump")

    else:
        print("Error: Unknown Action")

def remember(state, action, reward, next_state, done):
        memory.append((state, action, reward, next_state, done))
        if len(memory) > train_start:
            if epsilon > epsilon_min:
                epsilon *= epsilon_decay

def act(state):
    if np.random.random() <= epsilon:
        return random.randrange(action_size)
    else:
        return np.argmax(model.predict(state))

def replay():
        if len(memory) < train_start:
            return
        # Randomly sample minibatch from the memory
        minibatch = random.sample(memory, min(len(memory), batch_size))


        state = np.zeros((batch_size, state_size))
        next_state = np.zeros((batch_size, state_size))
        action, reward, done = [], [], []

        # do this before prediction
        # for speedup, this could be done on the tensor level
        # but easier to understand using a loop
        for i in range(batch_size):
            state[i] = minibatch[i][0]
            action.append(minibatch[i][1])
            reward.append(minibatch[i][2])
            next_state[i] = minibatch[i][3]
            done.append(minibatch[i][4])

        # do batch prediction to save speed
        target = model.predict(state)
        target_next = model.predict(next_state)

        for i in range(batch_size):
            # correction on the Q value for the action used
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                # Standard - DQN
                # DQN chooses the max Q value among next actions
                # selection and evaluation of action is on the target Q Network
                # Q_max = max_a' Q_target(s', a')
                target[i][action[i]] = reward[i] + gamma * (np.amax(target_next[i]))

        # Train the Neural Network with batches
        model.fit(state, target, batch_size=batch_size, verbose=0)


def load(name):
    model = load_model(name)

def save(name):
    model.save(name)

## Convert Grid data from observation to values for DQN inputs
def convert_grid(grid):
    for i in range(len(grid)):
        grid[i] = block_values[grid[i]]

# Generate a flat course with jumps given length of blocks
def generate_flat_course(length,  mission):
    my_mission.drawBlock(0, 1, 0, "emerald_block")

    ## current y not needed atm for this function since it's a flat course
    current_x = 0
    current_y = 1
    current_z = 0

    ## Flag to ensure no gap of 2 blocks
    jump = False

    counter = length

    while(counter > 0):
        if(not jump):
            ## Roll to see if there will be a jump
            if(random.random() > .1):
                jump = True
                counter -= 1
                variance = random.random()
                if(variance < .5):
                    current_x += 1
                else:
                    current_z += 1
                pass
            else:
                variance = random.random()
                if(variance < .5):
                    current_x += 1
                else:
                    current_z += 1

                my_mission.drawBlock(current_x, 1, current_z, "gold_block")
                counter -=1
        else:
            ## Do not roll to jump if previously jumped
            jump = False
            variance = random.random()
            if(variance < .5):
                current_x += 1
            else:
                current_z += 1

            my_mission.drawBlock(current_x, 1, current_z, "gold_block")

            counter -=1
    my_mission.drawBlock(current_x+1, 1, current_z, "lapis_block")

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
my_mission.drawBlock( 594,1,0,"lava")
my_mission_record = MalmoPython.MissionRecordSpec()

generate_flat_course(30, my_mission)

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
for e in range(EPISODES):
    time.sleep(0.1)
    world_state = agent_host.getWorldState()
    for error in world_state.errors:
        print("Error:",error.text)
    if world_state.number_of_observations_since_last_state > 0: 
        msg = world_state.observations[-1].text                 
        observations = json.loads(msg)  
        layers = []                        
        layers.extend(observations.get(u'layer_0', 0))
        layers.extend(observations.get(u'layer_1', 0))
        yaw = observations.get(u'Yaw')   

        layers.append(yaw)
        state = layers
        done = False    
    while world_state.is_mission_running:
        #print(".", end="")
        
            
        i = 0
        while not done:
            ## DQN returns index of maxd output value
            predicted_action = act(state)
            agent_host.sendCommand(action(predicted_action))
            new_world_state = agent_host.getWorldState()
            if world_state.number_of_observations_since_last_state > 0: 
                msg = world_state.observations[-1].text                 
                observations = json.loads(msg)  
                new_layers = []                        
                new_layers.extend(observations.get(u'layer_0', 0))
                new_layers.extend(observations.get(u'layer_1', 0))
                new_yaw = observations.get(u'Yaw')   
                new_layers.append(new_yaw)
            next_state = new_layers
            reward = 0
            for reward in new_world_state.rewards:
                reward += reward.getValue()
            if(reward == -1001 or reward == 999):
                done = True
            else:
                done = False
            next_state = np.reshape(next_state, [1, state_size])
            if not done or i == env._max_episode_steps-1:
                reward = reward
            else:
                reward = -100
            remember(state, action, reward, next_state, done)
            state = next_state
            i += 1
            if done:                   
                print("episode: {}/{}, score: {}, e: {:.2}".format(e, EPISODES, i, epsilon))
                if i == 500:
                    print("Saving trained model as parkour-dqn.h5")
                    save("parkour-dqn.h5")
                    break
            replay()
        break



print()
print("Mission ended")
# Mission has ended.
