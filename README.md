# DDQN_BeamSelection
Joint Deep Reinforcement Learning and Unfolding: Beam Selection and Precoding for mmWave Multiuser MIMO With Lens Arrays

This repository contains the entire code for our work "Joint deep reinforcement learning and unfolding: Beam selection and precoding for mmWave multiuser MIMO with lens arrays", available at: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9448095 and has been accepted for publication in IEEE JOURNAL ON SELECTED AREAS IN COMMUNICATIONS (JSAC).

For any reproduce, further research or development, please kindly cite our JSAC Journal paper:

`Q. Hu, Y. Liu, Y. Cai, G. Yu, and Z. Ding, “Joint deep reinforcement learning and unfolding: Beam selection and precoding for mmWave multiuser MIMO with lens arrays,” IEEE J. Sel. Areas Commun., vol. 39, no. 8, pp. 2289–2304, Aug. 2021.`

# Requirements
The following versions have been tested: Python 3.6 + Pytorch 1.9.0. But newer versions should also be fine.

## Training and Testing
Run the main program "`joint_trainer.py.py`".

## The introduction of each file
`joint_trainer.py`：main function，run this file to train jointly the DDQN and deep-unfolding neural network;

`Config.py`: System parameters;

DDQN:

`Net_module.py`：The architecture (dimension of trainable parameters) of DDQN;

`Dueling_DDQN.py`: The dueling architecture of DDQN;

`Base_Agent.py`: Define the class of agent that contains the basic functions of the DQN;

`DQN.py`: Define the class of DQN and it inherits from the `Base_Agent`;

`DDQN.py`: The class that inherits from the `DQN` and add the function of DDQN;

`my_DQN.py`: The class that inherits from the `DQN` and add some functions to deal with our problem, e.g., the computation of the reward function; 

`Replay_Buffer.py`: The replay buffer of the DDQN;

The folder `data_structures` and `exploration_strategies` denote the data structure and exploration strategies (e.g., noise net), respectively;

Deep-unfolding:

`WMMSE.py`: Iterative WMMSE algorithm for digital precoding; 

`unfolder.py` & `model.py`：Deep-unfolding neural network for digital precoding (unfold the iterative WMMSE algorithm);

`complex_matrix.py`: Some complex matrix operations, e.g., the matrix inversion and determinant of complex matrix;

`Beamspace_channel.py`：Beamspace channel model.
