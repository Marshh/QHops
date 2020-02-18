## QHops

A Minecraft agent that learns to complete parkour courses through reinforcement learning <br>
Group 49 <br>
Interview time: 1:45pm <br> 
Members:<br>
Daniel Chiu <br>
Joseph Lopez <br>
Marshall Nguyen <br>

### Summary

Minecraft parkour maps are a fun alternative mode of gameplay where players must navigate through a course composed of a multitude of obstacles with combinations of crouching, jumping, and sprinting motions. Players are not allowed to use items or place blocks while traversing a parkour course. The blocks the player must land on vary in height, distance, and surface area. The parkour course has the player start at a particular spot within the course and said player must reach a particular spot to complete the course. For our proposal, we plan on training an agent to traverse Minecraft parkour courses. We plan on utilizing commonly used obstacles found within most user-made courses and have said obstacles be generated randomly within courses for our agent to traverse. Our agent will take as input the percepts a regular minecraft player has access (visuals and time) to and will output moves.

We plan on utilizing deep q learning with convolutional layers and also evolutionary strategies for our agent. We aspire to train an agent to generalize to any parkour course we can fathom being generated ad hoc. Deep q learning with convolutions and evolutionary strategies are the gold standards for agents like self driving cars and has a multitude of applications within the fields of robotics and computer science. We wish to see if an ES agent would converge faster than a conv DQL agent if the primary heuristic is time and visuals. In theory, conv DQL should converge faster than ES as conv DQL is off policy, however we believe that testing both paradigms will be advantageous for our learning nonetheless.  

### Evaluation
 
We will utilize the primary metric of time for our agent as it traverses through a minecraft course. The evaluation data will consist of edge-case parkour courses the agent will not train on to see if it generalizes to the nuances of multiple parkour obstacles we wish to have as our building blocks for our parkour courses.
 
To analyze our agent qualitatively we will have our agent traverse a multitude of courses to demonstrate its generalizability. We will handcraft parkour courses which will highlight the primary challenges present for each obstacle type. We will visualize the internals of our agent by saving iterations of our agent and compare their functionalities within courses. Our moonshot case is to be able to generalize to parkour courses we handpick from professional parkour course websites. 

### Goals

Minimum Goal: Basic parkour agent <b>
Basic Q learning agent to traverse basic parkour courses <b>
Basic Genetic Algo agent to traverse basic parkour courses <b>

Realistic Goal: Intermediate parkour agent <b>
Basic Q learning agent to traverse courses we can generate ad hoc<b>
Genetic Algo agent to traverse courses we can generate ad hoc <b>

Ambitious Goal: An agent that can complete popular public parkour courses<b>
An agent which can generalize to unseen courses <b> </b>


### How to use code

Have malmo installed <br>
Have a minecraft client running <br>
Have the BendJump, Easy2, and EasyWalk world folders in the same directory as BendJump.py, BendNoJump.py, and easy_dan.py <br>
Run the python files (BendJump.py and BendNoJump.py are agents with discrete movement commands, easy_dan is continuous) <br>
