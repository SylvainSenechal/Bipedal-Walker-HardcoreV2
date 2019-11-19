import gym
import numpy as np
from fixed_structure_nn_numpy import SimpleNeuralControllerNumpy
import random
from deap import creator, base, tools, algorithms, cma

# env = gym.make('CartPole-v1')
env = gym.make('BipedalWalker-v2')

def eval_nn(genotype):
    total_reward=0
    # print(genotype)
    nn=SimpleNeuralControllerNumpy(24,4,3,5)
    nn.set_parameters(genotype)
    observation = env.reset()
    for t in range(1600):
        action=nn.predict(observation)
        print(action)
        observation, reward, done, info = env.step(action)
        total_reward+=reward
        if done:
            # print("Episode finished after %d timesteps"%(t+1))
            break
    return total_reward,


### A completer pour optimiser les parametres du reseau de neurones avec CMA-ES ###
def cmaEs():
    np.random.seed(128)
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    toolbox = base.Toolbox()
    toolbox.register("evaluate", eval_nn)

    nn=SimpleNeuralControllerNumpy(24,4,3,5)
    size=len(nn.get_parameters())

    strategy = cma.Strategy(centroid=[0.0]*size, sigma=0.3)
    toolbox.register("generate", strategy.generate, creator.Individual)
    toolbox.register("update", strategy.update)

    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    algorithms.eaGenerateUpdate(toolbox, ngen=20, stats=stats, halloffame=hof)

    return hof[0]

def benchmarkBestPlayer(bestPlayer):
    observation = env.reset()
    for t in range(1000):
        env.render()
        # print(observation)
        action=nn.predict(observation)

        observation, reward, done, info = env.step(action)
        print(reward)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
    env.close()

bestPlayer = cmaEs()
nn=SimpleNeuralControllerNumpy(24,4,3,5)
nn.set_parameters(bestPlayer)

benchmarkBestPlayer(nn)
