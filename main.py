from PairAgentEnv import PairAgentEnv
from plotter import plotSpider, plotScoresOverDifficulty
# nb : install gym with : pip install 'gym[all]'

### CREATING AN ENVIRONNEMENT :
# env = BipedalWalker(PIT, STUMP, HEIGHT)
# PIT, STUMP, HEIGHT = [0, 1] : 0 => EASY, 1 => HARD
###
# STAIRS : stairs height & number of stairs
# STUMP : vertical walls
# HEIGHT : Terrain altitude variation
# Attention reward modifiee

## REMETTRE CPU COUNT
## attention hidden layer

thresholdTooEasy = 200
thresholdTooHard = 50
# We will compare a classic raw learning with a Handmade curriculum and "POET" learning
def poetLearning(nbIteration = 10, maxEnvironnement = 5, mutationInterval = 2, transferInterval = 2):
    listPair = []
    pair = PairAgentEnv(difficultySTAIRS = 0, difficultySTUMP = 0, difficultyHEIGHT = 0)
    listPair.append(pair)

    for iteration in range(nbIteration):
        if (iteration > 0 and (iteration % mutationInterval) == 0): # Creating new environnements
            mutateEnv()
        pairSize = len(listPair)
        for pairID in range(pairSize): # Optimizing each pair
            listPair[pairID].optimize()
        for pairID in range(pairSize): # Attempting transfers
            if (pairSize >= 2 and (iteration % transferInterval) == 0): # We need at least 2 pairs to make a transfer
                listPair[pairID].brain = bestBrain(listPair, pairID)


def bestBrain(listPair, pairID): # We are picking the best brain for the environment in the pairID
    diffStair = listPair[pairID].difficultySTAIRS
    diffStump = listPair[pairID].difficultySTUMP
    diffHeight = listPair[pairID].difficultyHEIGHT
    bestScore = - 1000
    bestBrain = []
    for brain in range(len(listPair)):
        pair = PairAgentEnv(difficultySTAIRS = diffStair, difficultySTUMP = diffStump, difficultyHEIGHT = diffHeight)
        pair.brain = listPair[brain]
        brainQuality = pair.benchmarkAverage()
        if brainQuality > bestScore:
            bestScore = brainQuality
            bestBrain = pair.brain
    return bestBrain

def curriculumLearning(targetStairs, targetStump, targetHeight):

    pass

def directLearning(): # 2 hours to run on 12 cores
    nbBenchmark = 20

    scoresSTAIRS = []
    difficultiesSTAIRS = []
    for difficulty in range(0, nbBenchmark):
        pair =
        pair.optimize()
        scoresSTAIRS.append(pair.benchmarkAverage())
        difficultiesSTAIRS.append(difficulty/(nbBenchmark-1))
    with open("savedScores/stairs3.txt", "w+") as file:
        for score in scoresSTAIRS:
            file.write(str(score) + "\n")

    scoresSTUMP = []
    difficultiesSTUMP = []
    for difficulty in range(0, nbBenchmark):
        pair = PairAgentEnv(difficultySTAIRS = 0, difficultySTUMP = difficulty/(nbBenchmark-1), difficultyHEIGHT = 0)
        pair.optimize()
        scoresSTUMP.append(pair.benchmarkAverage())
        difficultiesSTUMP.append(difficulty/(nbBenchmark-1))
    with open("savedScores/stump3.txt", "w+") as file:
        for score in scoresSTUMP:
            file.write(str(score) + "\n")

    scoresHEIGHT = []
    difficultiesHEIGHT = []
    for difficulty in range(0, nbBenchmark):
        pair = PairAgentEnv(difficultySTAIRS = 0, difficultySTUMP = 0, difficultyHEIGHT = difficulty/(nbBenchmark-1))
        pair.optimize()
        scoresHEIGHT.append(pair.benchmarkAverage())
        difficultiesHEIGHT.append(difficulty/(nbBenchmark-1))
    with open("savedScores/height3.txt", "w+") as file:
        for score in scoresHEIGHT:
            file.write(str(score) + "\n")

    plotScoresOverDifficulty(scoresSTAIRS, difficultiesSTAIRS, scoresSTUMP, difficultiesSTUMP, scoresHEIGHT, difficultiesHEIGHT)

# data = [['STAIRS', 'STUMP', 'HEIGHT'],
#         ('', [
#             [0.2, 0.2, 0.8],
#             [0.4, 0.0, 0.5],
#             [0.6, 0.4, 0.9]])]
# plotSpider(data)

def environnementDifficulty(difficultySTAIRS, difficultySTUMP, difficultyHEIGHT): # used for plotting latter
    return (difficultySTAIRS * 2 + difficultySTUMP * 2 + difficultyHEIGHT) / 5 # (normalized [0, 1], height count less than other)

def main():
    # pair = PairAgentEnv(difficultySTAIRS = 1, difficultySTUMP = 0, difficultyHEIGHT = 0)
    # pair.optimize()
    # pair.benchmarkAverage()
    # pair.benchmark()

    directLearning()

    # pair.saveBrain("brain1.txt")
    # pair2 = PairAgentEnv(difficultyPIT = 0, difficultySTUMP = 0, difficultyHEIGHT = 1)
    # pair2.loadBrain("brain1.txt")
    # pair2.benchmark()

if __name__== "__main__":
    main()
