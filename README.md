## QHops

A Minecraft agent that learns to complete parkour courses through reinforcement learning /n
Group 49 /n
Interview time: 1:45pm /n 
Members:/n 
Daniel Chiu /n
Joseph Lopez /n
Marshall Nguyen /n

### Summary

Minecraft parkour maps are a fun alternative mode of gameplay where players must navigate through a course composed of a multitude of obstacles with combinations of crouching, jumping, and sprinting motions. Players are not allowed to use items or place blocks while traversing a parkour course. The blocks the player must land on vary in height, distance, and surface area. The parkour course has the player start at a particular spot within the course and said player must reach a particular spot to complete the course. For our proposal, we plan on training an agent to traverse Minecraft parkour courses. We plan on utilizing commonly used obstacles found within most user-made courses and have said obstacles be generated randomly within courses for our agent to traverse. Our agent will take as input the percepts a regular minecraft player has access (visuals and time) to and will output moves.

We plan on utilizing deep q learning with convolutional layers and also evolutionary strategies for our agent. We aspire to train an agent to generalize to any parkour course we can fathom being generated adhawk. Deep q learning with convolutions and evolutionary strategies are the gold standards for agents like self driving cars and has a multitude of applications within the fields of robotics and computer science. We wish to see if an ES agent would converge faster than a conv DQL agent if the primary heuristic is time and visuals. In theory, conv DQL should converge faster than ES as conv DQL is off policy, however we believe that testing both paradigms will be advantageous for our learning nonetheless.  

### Evaluation
 
We will utilize the primary metric of time for our agent as it traverses through a minecraft course. The evaluation data will consist of edge-case parkour courses the agent will not train on to see if it generalizes to the nuances of multiple parkour obstacles we wish to have as our building blocks for our parkour courses.
 
To analyze our agent qualitatively we will have our agent traverse a multitude of courses to demonstrate its generalizability. We will handcraft parkour courses which will highlight the primary challenges present for each obstacle type. We will visualize the internals of our agent by saving iterations of our agent and compare their functionalities within courses. Our moonshot case is to be able to generalize to parkour courses we handpick form professional parkour course websites. 

### Goals

Minimum Goal:
Basic Q learning agent to traverse basic parkour courses
Basic Genetic Algo agent to traverse basic parkour courses

Realistic Goal:
Basic Q learning agent to traverse courses we can generate ad hawk
Genetic Algo agent to traverse courses we can generate ad hawk

Ambitious Goal:
An agent which can generalize to unseen courses

