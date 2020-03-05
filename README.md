# Emergence of Recursive Teaching Protocol between Theory of Mind Agents

This is the code repo for the referential games described in the paper "Emergence of Recursive Teaching Protocol between Theory of Mind Agents." In this work, we applied ToM to a cooperative multiagent pedagogical environment and developed an efficient communication protocol which improves the agents' performance in completing referential tasks. We proposed an adaptive algorithm to learn the ToM protocol and showed that the teaching complexity in the protocol is an empirical approximation of the recursive teaching dimension (RTD).

## Code Structure

The code consists of three main parts:  learning modules, agents, and concept definitions for number set, 3D shapes and ImageNet referential games.

### Modules

`perception.py`: contains the visual encoders used in image processing of 3D shapes and ImageNet.

`meditation.py`: contains the belief update module along with three submodules: explicit blief update, implicit belief update, and order free belief update.

`value.py`: contains the Q-network.

### Agents

 `teacher.py`: implements the agent teacher with the message selection function and belief update function. The teacher chooses messages according to the Q-function, which is learned through Q-learning.
 
 `student.py`: implements the agent student and its belief update function. The student's policy is learned through policy gradient.

### Number_Set, Geometry3D_4, and ImageNet

Each contains the concept class definition for the specified game setting.

## Running the Code

Under the `Experiments` folder, create a new folder with a config file specifying the desired settings. Alternatively, you can modify the corresponding config file in `Experiments/Number_Set_Example`, `Experiments/Geometry3D_Example`, or `Experiments/ImageNet_Example` to reproduce the referential game in the paper.

### Pretrain 

cd to the directory containing  `train.py`,  run  `train.py` with the folder name in the previous step and "pretrain" as the arguments. For example:
```
$ python3 train.py Number_Set_Example pretrain
```
Note: if no pretrain is needed, please comment out the below lines (line 225 and 226) in `train.py`.
```
teacher.pretrain_bayesian_belief_update(concept_generator)
student.pretrain_bayesian_belief_update(concept_generator)
```

### Training 

cd to the directory containing  `train.py`,  run  `train.py` with the folder name in the previous step as the only argument. For example:
```
$ python3 train.py Number_Set_Example
```
You can modify the continue steps and the checkpoints to be loaded in each phase in  `train.py`.

### Testing 

cd to the directory containing  `test.py`,  run  `test.py`  with the folder name in the previous step as the only argument. For example:
```
$ python3 test.py Number_Set_Example
```
You can change the checkpoints to be tested in  `test.py`.

## Datasets

Currently, only number set referential game can run without datasets. 

Datasets for 3D shapes and ImageNet will be update later, so they are not runnable at present.
