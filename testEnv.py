from environment import BipedalWalker

def evaluateNeuralNetwork(iterationsSteps=2000, render=True):

    env = BipedalWalker(0, 0, 0)
    total_reward = 0

    state = env.reset()
    for t in range(iterationsSteps):
        if render: env.render()

        # action = actionFromBrain(state, nn)
        action = [0,0,0,0]
        state, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            break
    return - total_reward # CMA_ES is minimizing so we minimize (- reward)

evaluateNeuralNetwork()
