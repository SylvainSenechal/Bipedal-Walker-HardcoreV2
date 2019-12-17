import multiprocessing
import numpy as np
import random

from environment import BipedalWalker
import cma # https://github.com/CMA-ES/pycma
from cma.fitness_transformations import EvalParallel

POPULATION_SIZE = 100
SIGMA_INIT = 0.3 # Otimum weigths shoudl be within +/- 3*SIGMA
DEFAULT_MAX_ITERATION_CMAES = 100
ITERATIONS_STEPS_LEARNING = 300
ITERATIONS_STEPS_TESTING = 2000
CPU_COUNT = multiprocessing.cpu_count()
NB_ENV_BENCHMARK = 1 # How many environnement simulation are run to mesure the quality of an agent ?

INPUT_SIZE = 24
OUTPUT_SIZE = 4
HIDDEN_LAYER_SIZE = 20

BIAS_VERSION = False # Are we using a neural network with bias ?


model = {}
################ NO BIAS VERSION ################
if (BIAS_VERSION == False):
    model['W1'] = np.random.randn(INPUT_SIZE, HIDDEN_LAYER_SIZE) / np.sqrt(INPUT_SIZE)
    model['W2'] = np.random.randn(HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE) / np.sqrt(HIDDEN_LAYER_SIZE)
    model['W3'] = np.random.randn(HIDDEN_LAYER_SIZE, OUTPUT_SIZE) / np.sqrt(HIDDEN_LAYER_SIZE)
    SIZE_BRAIN = INPUT_SIZE*HIDDEN_LAYER_SIZE + HIDDEN_LAYER_SIZE*HIDDEN_LAYER_SIZE + HIDDEN_LAYER_SIZE * OUTPUT_SIZE # + HIDDEN_LAYER_SIZE + OUTPUT_SIZE
################ BIAS VERSION ################
else:
    model['W1'] = np.random.randn(INPUT_SIZE, HIDDEN_LAYER_SIZE) / np.sqrt(INPUT_SIZE)
    model['W2'] = np.random.randn(HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE) / np.sqrt(HIDDEN_LAYER_SIZE)
    model['B1'] = np.random.randn(HIDDEN_LAYER_SIZE) / np.sqrt(HIDDEN_LAYER_SIZE)
    model['B2'] = np.random.randn(OUTPUT_SIZE) / np.sqrt(OUTPUT_SIZE)
    SIZE_BRAIN = INPUT_SIZE*HIDDEN_LAYER_SIZE + HIDDEN_LAYER_SIZE * OUTPUT_SIZE + HIDDEN_LAYER_SIZE + OUTPUT_SIZE

class PairAgentEnv:
    def __init__(self,
                    difficultySTAIRS = 0,
                    difficultySTUMP = 0,
                    difficultyHEIGHT = 0,
                    brain = None,
                    iterationCMAES = DEFAULT_MAX_ITERATION_CMAES
                ):
        self.difficultySTAIRS = difficultySTAIRS,
        self.difficultySTUMP = difficultySTUMP,
        self.difficultyHEIGHT = difficultyHEIGHT,
        self.brain = brain if brain else np.random.randn(SIZE_BRAIN) / 3
        self.iterationCMAES = iterationCMAES

    def optimize(self, fromSolution = None):
        if (fromSolution):
            es = cma.CMAEvolutionStrategy(
                fromSolution,
                SIGMA_INIT,
                {
                    'popsize': POPULATION_SIZE,
                    'maxiter': self.iterationCMAES
                })
        else:
            es = cma.CMAEvolutionStrategy(
                # SIZE_BRAIN * [0],
                self.brain,
                SIGMA_INIT,
                {
                    'popsize': POPULATION_SIZE,
                    'maxiter': self.iterationCMAES
                })

        with EvalParallel(CPU_COUNT-2) as eval_all:
            while not es.stop():
                noisySolutions = es.ask()
                # es.tell(noisySolutions, eval_all(evaluateBrain, noisySolutions))
                # Parameters of fitness function (ie : evaluateBrain) can only be passed as a tuple in pycma..
                es.tell(noisySolutions, eval_all(evaluateBrain, noisySolutions, tuple([self.difficultySTAIRS, self.difficultySTUMP, self.difficultyHEIGHT])))
                es.disp()

        # help(cma.CMAEvolutionStrategy) # Show documentatio
        res = es.result
        # es.result_pretty()
        self.brain = res.xfavorite # Updating brain with best solution

    def mutate(self):
        if random.uniform(0, 1) > 0.4:
            newDiff = self.difficultySTAIRS[0] + (- 0.02 + random.uniform(0, 0.1)) # + [-0.02, +0.08] added to difficulty
            self.difficultySTAIRS = max(min(1, newDiff), 0), # constraint in [0, 1]
        if random.uniform(0, 1) > 0.4:
            newDiff = self.difficultySTUMP[0] + (- 0.02 + random.uniform(0, 0.1))
            self.difficultySTUMP = max(min(1, newDiff), 0),
        if random.uniform(0, 1) > 0.4:
            newDiff = self.difficultyHEIGHT[0] + (- 0.02 + random.uniform(0, 0.1))
            self.difficultyHEIGHT = max(min(1, newDiff), 0),
        return self
    def saveBrain(self, filename):
        with open("savedAgent/" + filename, "w+") as file:
            for weight in self.brain:
                file.write(str(weight) + "\n")

    def loadBrain(self, filename):
        brain = []
        with open("savedAgent/" + filename, "r") as file:
            for weight in file:
                brain.append(float(weight.strip()))
        self.brain = np.array(brain)

    def benchmark(self): # This is used for displaying the agent capacity
        for i_episode in range(20):
            print(-evaluateBrain(self.brain, self.difficultySTAIRS, self.difficultySTUMP, self.difficultyHEIGHT, ITERATIONS_STEPS_TESTING, False))
        for i_episode in range(5):
            print(-evaluateBrain(self.brain, self.difficultySTAIRS, self.difficultySTUMP, self.difficultyHEIGHT, ITERATIONS_STEPS_TESTING, True))
        print('##### End benchmark #####')
    def benchmarkAverage(self, nbSimulationBenchmark = 10): # This one is for computing the average quality of an agent and plotting it latter
        scores = []
        for i in range(nbSimulationBenchmark):
            scores.append(-evaluateBrain(self.brain, self.difficultySTAIRS, self.difficultySTUMP, self.difficultyHEIGHT, ITERATIONS_STEPS_TESTING, False))
        averageScore = (sum(scores) - max(scores) - min(scores)) / (nbSimulationBenchmark - 2) # Mean value excluding the best and the worst score
        print("Average Score : ", averageScore)
        print('##### End benchmark #####')
        return averageScore


#######################################################################################################
## NB : These method are not in the class because pycma wont accept instanceMethods, only function.. ##
#######################################################################################################
def actionFromBrain(state, brain):
    ################ NO BIAS VERSION ################
    if (BIAS_VERSION == False):
        hiddenLayer1 = np.matmul(state, brain['W1'])
        hiddenLayer1 = np.tanh(hiddenLayer1)
        hiddenLayer2 = np.matmul(hiddenLayer1, brain['W2'])
        hiddenLayer2 = np.tanh(hiddenLayer2)
        action = np.matmul(hiddenLayer2, brain['W3'])
        action = np.tanh(action)
    ################ BIAS VERSION ################
    else:
        hiddenLayer1 = np.matmul(state, brain['W1'])
        hiddenLayer1 = np.add(hiddenLayer1, brain['B1'])
        hiddenLayer1 = np.tanh(hiddenLayer1)
        action = np.matmul(hiddenLayer1, brain['W2'])
        action = np.add(action, brain['B2'])
        action = np.tanh(action)
    return action

def evaluateBrain(neuralNetwork, difficultySTAIRS, difficultySTUMP, difficultyHEIGHT, iterationsSteps=ITERATIONS_STEPS_LEARNING, render=False):
    brain = {}
    ################ NO BIAS VERSION ################
    if (BIAS_VERSION == False):
        brain['W1'] = neuralNetwork[0 :
                                    INPUT_SIZE * HIDDEN_LAYER_SIZE
                                    ].reshape(INPUT_SIZE, HIDDEN_LAYER_SIZE)
        brain['W2'] = neuralNetwork[INPUT_SIZE * HIDDEN_LAYER_SIZE :
                                    INPUT_SIZE * HIDDEN_LAYER_SIZE + HIDDEN_LAYER_SIZE * HIDDEN_LAYER_SIZE
                                    ].reshape(HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE)
        brain['W3'] = neuralNetwork[INPUT_SIZE * HIDDEN_LAYER_SIZE + HIDDEN_LAYER_SIZE * HIDDEN_LAYER_SIZE :
                                    INPUT_SIZE * HIDDEN_LAYER_SIZE + HIDDEN_LAYER_SIZE * HIDDEN_LAYER_SIZE + HIDDEN_LAYER_SIZE * OUTPUT_SIZE
                                    ].reshape(HIDDEN_LAYER_SIZE, OUTPUT_SIZE)
    ################ BIAS VERSION ################
    else:
        brain['W1'] = neuralNetwork[0 :
                                    INPUT_SIZE * HIDDEN_LAYER_SIZE
                                    ].reshape(INPUT_SIZE, HIDDEN_LAYER_SIZE)
        brain['W2'] = neuralNetwork[INPUT_SIZE * HIDDEN_LAYER_SIZE :
                                    INPUT_SIZE * HIDDEN_LAYER_SIZE + HIDDEN_LAYER_SIZE * OUTPUT_SIZE
                                    ].reshape(HIDDEN_LAYER_SIZE, OUTPUT_SIZE)
        brain['B1'] = neuralNetwork[INPUT_SIZE * HIDDEN_LAYER_SIZE + HIDDEN_LAYER_SIZE * OUTPUT_SIZE :
                                    INPUT_SIZE * HIDDEN_LAYER_SIZE + HIDDEN_LAYER_SIZE * OUTPUT_SIZE + HIDDEN_LAYER_SIZE]
        brain['B2'] = neuralNetwork[INPUT_SIZE * HIDDEN_LAYER_SIZE + HIDDEN_LAYER_SIZE * OUTPUT_SIZE + HIDDEN_LAYER_SIZE :
                                    INPUT_SIZE * HIDDEN_LAYER_SIZE + HIDDEN_LAYER_SIZE * OUTPUT_SIZE + HIDDEN_LAYER_SIZE + OUTPUT_SIZE]

    # env = gym.make('BipedalWalker-v2')
    env = BipedalWalker(difficultySTAIRS[0], difficultySTUMP[0], difficultyHEIGHT[0])
    # env.seed()
    total_reward = 0
    for i in range(NB_ENV_BENCHMARK):

        state = env.reset()
        for t in range(iterationsSteps):
            if render: env.render()

            action = actionFromBrain(state, brain)
            state, reward, done, info = env.step(action)
            total_reward += reward
            if done:
                break
    return - total_reward # CMA_ES is minimizing so we minimize (- reward)
