import cma # https://github.com/CMA-ES/pycma
import gym
import numpy as np
import multiprocessing
from cma.fitness_transformations import EvalParallel

from environment import BipedalWalker


### CREATIN AN ENVIRONNEMENT :
# env = BipedalWalker(PIT, STUMP, HEIGHT)
# PIT, STUMP, HEIGHT = [0, 1] : 0 => EASY, 1 => HARD
# PIT : Trous
# STUMP : rectangle verticaux <=> murs
# HEIGHT : Variation d'Altitude du terrain
# counter : ecart en obstacles (TODO?)

POPULATION_SIZE = 25
SIGMA_INIT = 0.3
MAX_ITERATION_CMAES = 100
ITERATIONS_STEPS_LEARNING = 300
ITERATIONS_STEPS_TESTING = 1600
CPU_COUNT = multiprocessing.cpu_count()
NB_ENV_BENCHMARK = 1
HARDCORE = False

INPUT_SIZE = 24
OUTPUT_SIZE = 4
HIDDEN_LAYER_SIZE = 40

BIAS_VERSION = True


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

def actionFromBrain(state, brain):
    ################ NO BIAS VERSION ################
    if (BIAS_VERSION == False):
        hl = np.matmul(state, brain['W1'])
        hl = np.tanh(hl)

        hl2 = np.matmul(hl, brain['W2'])
        hl2 = np.tanh(hl)

        action = np.matmul(hl2, brain['W3'])
        action = np.tanh(action)
    ################ BIAS VERSION ################
    else:
        hl = np.matmul(state, brain['W1'])
        hl = np.add(hl, brain['B1'])
        hl = np.tanh(hl)
        action = np.matmul(hl, brain['W2'])
        action = np.add(action, brain['B2'])
        action = np.tanh(action)

    return action

def evaluateNeuralNetwork(neuralNetowrk, iterationsSteps=ITERATIONS_STEPS_LEARNING, render=False):
    nn = {}
    ################ NO BIAS VERSION ################
    if (BIAS_VERSION == False):
        nn['W1'] = neuralNetowrk[0 :
                                INPUT_SIZE * HIDDEN_LAYER_SIZE
                                ].reshape(INPUT_SIZE, HIDDEN_LAYER_SIZE)
        nn['W2'] = neuralNetowrk[INPUT_SIZE * HIDDEN_LAYER_SIZE :
                                INPUT_SIZE * HIDDEN_LAYER_SIZE + HIDDEN_LAYER_SIZE * HIDDEN_LAYER_SIZE
                                ].reshape(HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE)
        nn['W3'] = neuralNetowrk[INPUT_SIZE * HIDDEN_LAYER_SIZE + HIDDEN_LAYER_SIZE * HIDDEN_LAYER_SIZE :
                                INPUT_SIZE * HIDDEN_LAYER_SIZE + HIDDEN_LAYER_SIZE * HIDDEN_LAYER_SIZE + HIDDEN_LAYER_SIZE * OUTPUT_SIZE
                                ].reshape(HIDDEN_LAYER_SIZE, OUTPUT_SIZE)
    ################ BIAS VERSION ################
    else:
        nn['W1'] = neuralNetowrk[0 :
                                INPUT_SIZE * HIDDEN_LAYER_SIZE
                                ].reshape(INPUT_SIZE, HIDDEN_LAYER_SIZE)
        nn['W2'] = neuralNetowrk[INPUT_SIZE * HIDDEN_LAYER_SIZE :
                                INPUT_SIZE * HIDDEN_LAYER_SIZE + HIDDEN_LAYER_SIZE * OUTPUT_SIZE
                                ].reshape(HIDDEN_LAYER_SIZE, OUTPUT_SIZE)
        nn['B1'] = neuralNetowrk[INPUT_SIZE * HIDDEN_LAYER_SIZE + HIDDEN_LAYER_SIZE * OUTPUT_SIZE :
                                INPUT_SIZE * HIDDEN_LAYER_SIZE + HIDDEN_LAYER_SIZE * OUTPUT_SIZE + HIDDEN_LAYER_SIZE]
        nn['B2'] = neuralNetowrk[INPUT_SIZE * HIDDEN_LAYER_SIZE + HIDDEN_LAYER_SIZE * OUTPUT_SIZE + HIDDEN_LAYER_SIZE :
                                INPUT_SIZE * HIDDEN_LAYER_SIZE + HIDDEN_LAYER_SIZE * OUTPUT_SIZE + HIDDEN_LAYER_SIZE + OUTPUT_SIZE]

    # env = gym.make('BipedalWalker-v2')
    env = BipedalWalker(0, 0, 0)
    # env.seed()
    total_reward = 0
    for i in range(NB_ENV_BENCHMARK):

        state = env.reset()
        for t in range(iterationsSteps):
            if render: env.render()

            action = actionFromBrain(state, nn)
            state, reward, done, info = env.step(action)
            total_reward += reward
            if done:
                break
    return - total_reward # CMA_ES is minimizing so we minimize (- reward)


def main():
    es = cma.CMAEvolutionStrategy(
        SIZE_BRAIN * [0],
        SIGMA_INIT,
        {
            'popsize': POPULATION_SIZE,
            'maxiter': MAX_ITERATION_CMAES
        }
    )


    with EvalParallel(CPU_COUNT) as eval_all:
        while not es.stop():
            noisySolutions = es.ask()
            es.tell(noisySolutions, eval_all(evaluateNeuralNetwork, noisySolutions))
            # es.logger.add()
            es.disp()
    # help(cma.CMAEvolutionStrategy)
    res = es.result
    print('REEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEESULT')
    print(res.xbest)
    print(res.xfavorite)
    print(len(res.xbest))
    print('PRETTTTYYYYYYYYYYYYYYy')
    es.result_pretty()
    # model = res.xbest
    model = res.xfavorite

    es = cma.CMAEvolutionStrategy(
        model,
        SIGMA_INIT,
        {
            'popsize': POPULATION_SIZE,
            'maxiter': MAX_ITERATION_CMAES
        }
    )


    with EvalParallel(CPU_COUNT) as eval_all:
        while not es.stop():
            noisySolutions = es.ask()
            es.tell(noisySolutions, eval_all(evaluateNeuralNetwork, noisySolutions))
            # es.logger.add()
            es.disp()
    # help(cma.CMAEvolutionStrategy)
    res = es.result
    print('REEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEESULT')
    print(res.xbest)
    print(res.xfavorite)
    print(len(res.xbest))
    print('PRETTTTYYYYYYYYYYYYYYy')
    es.result_pretty()
    # model = res.xbest
    model = res.xfavorite



    iter_num = 1600
    for i_episode in range(20):
        print(-evaluateNeuralNetwork(model, ITERATIONS_STEPS_TESTING, False))
    for i_episode in range(5):
        print(-evaluateNeuralNetwork(model, ITERATIONS_STEPS_TESTING, True))

if __name__== "__main__":
    main()
