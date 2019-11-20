# Evolution Strategies BipedalWalker-v2
# https://blog.openai.com/evolution-strategies/
# gives good solution at around iter 100 in 5 minutes
# for testing model set reload=True

# dans le papier : 24 => 40 => 40 => 4 tanh
# 512 pop


import cma


def square(x):
    return (2-x[0])*(2-x[0])



import gym
import numpy as np
import sys

env = gym.make('BipedalWalker-v2')
np.random.seed(10)

hiddenLayerSize = 100
sizePopulation = 20
sigma = 0.1
alpha = 0.03
iter_num = 300
aver_reward = None
allow_writing = True
reload = False



model = {}
model['W1'] = np.random.randn(24, hiddenLayerSize) / np.sqrt(24)
model['W2'] = np.random.randn(hiddenLayerSize, 4) / np.sqrt(hiddenLayerSize)
size = 24*hiddenLayerSize + hiddenLayerSize * 4


def get_action(state, model):
    hl = np.matmul(state, model['W1'])
    hl = np.tanh(hl)
    action = np.matmul(hl, model['W2'])
    action = np.tanh(action)

    # hl = np.matmul(state, model['W1'])
    # hl[hl<0] = 0 # np.tanh(hl)
    # action = np.matmul(hl, model['W2'])
    # action[action<0] = 0 #np.tanh(action)

    return action

def evaluateNeuralNetwork(neuralNetowrk, render=False):
    nn = {}
    nn['W1'] = neuralNetowrk[:24*hiddenLayerSize].reshape(24, hiddenLayerSize)
    nn['W2'] = neuralNetowrk[-hiddenLayerSize*4:].reshape(hiddenLayerSize, 4)

    state = env.reset()
    total_reward = 0
    for t in range(iter_num):
        if render: env.render()

        action = get_action(state, nn)
        state, reward, done, info = env.step(action)
        total_reward += reward

        if done:
            break
    return -total_reward

es = cma.CMAEvolutionStrategy(size * [0], 0.5, {'popsize': 100})
for i in range(150):
    noisySolutions = es.ask()
    es.tell(noisySolutions, [evaluateNeuralNetwork(solution) for solution in noisySolutions])
    es.logger.add()
    es.disp()
# help(cma.CMAEvolutionStrategy)
# es.optimize(evaluateNeuralNetwork)
res = es.result
print('REEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEESULT')
print(res.xbest)
print(len(res.xbest))
print('PRETTTTYYYYYYYYYYYYYYy')
es.result_pretty()
model = res.xbest

# for i in range(100):
#     N = {}
#     for k, v in model.items():
#         N[k] = np.random.randn(sizePopulation, v.shape[0], v.shape[1])
#     R = np.zeros(sizePopulation)
#
#     for j in range(sizePopulation):
#         model_try = {}
#         for k, v in model.items():
#             model_try[k] = v + sigma*N[k][j]
#         R[j] = evaluateNeuralNetwork(model_try)
#
#     A = (R - np.mean(R)) / np.std(R)
#     for k in model:
#         model[k] = model[k] + alpha/(sizePopulation*sigma) * np.dot(N[k].transpose(1, 2, 0), A)
#
#     cur_reward = evaluateNeuralNetwork(model)
#     aver_reward = aver_reward * 0.9 + cur_reward * 0.1 if aver_reward != None else cur_reward
#     print('iter %d, cur_reward %.2f, aver_reward %.2f' % (i, cur_reward, aver_reward))


iter_num = 1600
for i_episode in range(5):
    print(evaluateNeuralNetwork(model, True))
sys.exit('demo finished')
