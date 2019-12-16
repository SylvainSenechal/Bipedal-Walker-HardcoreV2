# Bipedal-Walker-HardcoreV2
Trying to solve the hardcore bipedal walker environnement, using [POET](https://arxiv.org/abs/1901.01753) architecture

## Installation
- install OpenAI [Gym](https://gym.openai.com/) with : `pip install 'gym[all]'`
- install [Pycma](https://github.com/CMA-ES/pycma)

## Running
`python main.py`

## Relevant Folders & Files descriptions
### Files
- main.py : this contains the POET algorithm, and comparisons with other algorithms
- PairAgentEnv.py : A class to create a pair of Agent / Environnement, including optimizing and benchmarking functions
- environment.py : A modified version of the [hardcoreBipedalWalker](https://gym.openai.com/envs/BipedalWalkerHardcore-v2/) from Gym, so we can change the difficulty with parameters
- plotter.py : Various functions for displaying results
### Folders
- images : Storing the results (plots)
- savedAgent : Storing the brain (neuralNetwork) of our trained agents
- savedScores : Storing the scores of our agents
