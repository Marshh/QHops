## QHops

A Minecraft agent that learns to complete parkour courses through reinforcement learning <br>
Group 49 <br>
Interview time: 1:45pm on Thursday March 5th, 2020<br> 
Members:<br>
Daniel Chiu <br>
Joseph Lopez <br>
Marshall Nguyen <br>

https://marshh.github.io/QHops/

### Summary

Minecraft parkour maps are a fun mode of gameplay wherein players navigate through courses composed of a multitude of obstacles with combinations of crouching, jumping, and sprinting motions. Players are not allowed to use items or place blocks onto the map while traversing a parkour course. The blocks the player must land on vary in height, distance,surface area and block type. A parkour course has the player start at a particular spot within the course and said player must reach a particular spot to complete the course. 

For our proposal, we plan on training an agent to traverse Minecraft parkour courses by utilizing commonly used obstacles found within user-made courses and have said obstacles be generated randomly within courses for our agent to traverse. Our agent will take as input the percepts a regular minecraft player has access to and will the best action given a particular state within a parkour course.

We plan on utilizing deep q learning with convolutional layers and also evolutionary strategies for our agent. We aspire to train an agent to generalize to any parkour course we can fathom being generated ad hoc. Deep q learning with convolutions and evolutionary strategies are the gold standards for agents like self driving cars and has a multitude of applications within the fields of robotics and computer science. We wish to see if an ES agent would converge faster than a conv DQL agent if the primary heuristic is time and visuals. In theory, conv DQL should converge faster than ES as conv DQL is off policy, however we believe that testing both paradigms will be advantageous for our learning nonetheless.  

### Evaluation
 
We will utilize the primary metric of discrete and continuous times for our agent as it traverses through a minecraft course. The evaluation data will consist of edge-case parkour courses the agent will not train on to see if it generalizes to the nuances of multiple parkour obstacles we wish to have as our building blocks for our parkour courses.
 
To analyze our agent qualitatively we will have our agent traverse a multitude of courses to demonstrate its generalizability. We will handcraft parkour courses which will highlight the primary challenges present for each obstacle type. We will visualize the internals of our agent by saving iterations of our agent and compare their functionalities within courses. Our moonshot case is to be able to generalize to parkour courses we handpick from professional parkour course websites. 

### Goals

Minimum Goal: Basic parkour agent 
Basic Q learning agent to traverse basic parkour courses 
Basic Genetic Algo agent to traverse basic parkour courses 

Realistic Goal: Intermediate parkour agent 
Basic Deep Q learning agent to traverse courses we can generate ad hoc

Ambitious Goal: An agent that can complete popular public parkour courses
An agent which can generalize to unseen courses  

### Completed goals

Agent solving simple premade parkour course with discrete movement (no jumps) using Tabular Q-Learning <br> 
Agent solving simple premade parkour course with discrete movement (with jumps) using Tabular Q-Learning <br> 
(Both adapted cliffwalking challenge from Barto and Sutton) <br> 
<br> 
Agent learning generic and randomly made parkour courses with discrete movement (no jumps) using Tabular Q-Learning<br> 
Agent learning generic and randomly made parkour courses with discrete movement (with jumps) using Tabular Q-Learning<br> 
<br> 
Agen solving simple premade parkour course with limited continuous movement (controlled turning) using Tabular Q-Learning with grid-based perception and discrete X/Y/Z states<br> 

### Challenges faced:
Agent does not feasibly learn to complete parkour courses with continuous movement using Tabular Q-Learning (Too many states and high entropy)<br> 
<br> 
Limited command set (cannot sprint, limited novel policies to learn)<br> 
<br> 
### Current task:
Implement an agent that uses Deep Q-Learning in order to solve courses with continuous movement commands<br> 
Experiment with different neural network architectures and hyper-parameter tuning (current architecture is a an input layer of size 99, a single hidden layer of size 24, and an output layer of size 9 for each possible continuous command)<br> 
Implement Soft Actor Critic<br> 


### Dependencies
Tensorflow
Keras

### How to use code
If malmo is compiled from source:
Have malmo installed <br>
Install QHops <br>
Have a minecraft client running <br>
Run the python files (BendJump.py and BendNoJump.py are agents with discrete movement commands, easy_dan is continuous) <br>
<br> else: <br>
Install the most applicable precompiled release and unzip<br>
https://github.com/Microsoft/malmo/releases <br>
Download Qhops<br>
Move QHops into the unzipped malmo folder <br>
Open Python_examples and copy MalmoPython.so into QHops <br>
Have a minecraft client running <br>

