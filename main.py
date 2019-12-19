from PairAgentEnv import PairAgentEnv
from plotter import plotSpider, plotScoresOverDifficulty, plotDifficultyReached, plotListDifficulties
import copy


THRESHOLD_TOO_HARD = 50
THRESHOLD_TOO_EASY = 200
THRESHOLD_MUTATE = 100
MAX_ACTIVE_PAIRS = 5
VERSION = 7 # Writting to new files for new algorithm

########################################
# In this work we compare 3 learning technics on a custom version of Gym BipedalWalker environment :
# - a classic direct learning witch CMA-ES
# - a curriculum learning
# - the PÖET algorithm from https://arxiv.org/pdf/1901.01753.pdf

def poetLearning(mutationInterval = 1, transferInterval = 1): # NB : 4/5 hours on 12 cores for decent results
    # NB : intervals = 1 here, since we already do several iterations (25) of CMAES in optimize() function
    # We initialize our algorithm with one easy pair
    listPairs = []
    pair = PairAgentEnv(difficultySTAIRS = 0, difficultySTUMP = 0, difficultyHEIGHT = 0, iterationCMAES = 25)
    listPairs.append(pair)

    iteration = 0
    while True:
        iteration += 1
        print('#########################################################################################################################################################')
        print('############################################################ POET iteration n°', iteration, '############################################################')
        print('#########################################################################################################################################################')
        for nb, pair in enumerate(listPairs):
        ########################### Saving functions ###########################
            print('##### Difficulty pair n°', nb)
            print('stairs : ', pair.difficultySTAIRS[0])
            print('stumps : ', pair.difficultySTUMP[0])
            print('height : ', pair.difficultyHEIGHT[0])
            pair.saveBrain("brainPOET_V" + str(VERSION) + "_" + str(nb) + ".txt")
            pair.saveLastDifficulty("difficultyLastPOET_V" + str(VERSION) + "_" + str(nb) + ".txt")
            if (nb == (len(listPairs) - 1)):
                pair.saveDifficulty("difficultyPOET_V" + str(VERSION) + "_" + str(nb) + ".txt", True)
                pair.saveDifficulty("difficultyPOET_V" + str(VERSION) + "_" + ".txt", True)
            else:
                pair.saveDifficulty("difficultyPOET_V" + str(VERSION) + "_" + str(nb) + ".txt", False)
                pair.saveDifficulty("difficultyPOET_V" + str(VERSION) + "_" + ".txt", False)
            pair.addListDifficulty(iteration)
        ########################### Actual algorithm ###########################
        if (iteration > 0 and (iteration % mutationInterval) == 0): # Creating new environnements by mutation
            listPairs = mutateEnvironment(listPairs)
        for pair in listPairs: # Optimizing each pair
            pair.optimize()
        for pair in listPairs: # Attempting transfers
            if (len(listPairs) >= 2 and (iteration % transferInterval) == 0): # We need at least 2 pairs to make a transfer
                pair.brain = bestBrain(listPairs, pair)


def bestBrain(listPairs, pair): # We are picking the best brain in the list for the environment in the given pair
    diffStair = pair.difficultySTAIRS[0]
    diffStump = pair.difficultySTUMP[0]
    diffHeight = pair.difficultyHEIGHT[0]
    bestScore = - 1000
    bestBrain = []
    for pair in listPairs:
        localPair = PairAgentEnv(difficultySTAIRS = diffStair, difficultySTUMP = diffStump, difficultyHEIGHT = diffHeight)
        localPair.brain = pair.brain
        brainQuality = localPair.benchmarkAverage(nbSimulationBenchmark = 5) # Computing the quality of each brain ..
        if brainQuality > bestScore: # .. And keeping the best
            bestScore = brainQuality
            bestBrain = localPair.brain
    return copy.deepcopy(bestBrain)

def mutateEnvironment(listPairs):
    # FIRST : We evaluate each pair, and create a parentList of environments eligible to reproduce,
    # when their agent have a certain score above a threshold
    parentEnvironments = []
    for pair in listPairs:
        score = pair.benchmarkAverage(nbSimulationBenchmark = 5)
        if score > THRESHOLD_MUTATE:
            parentEnvironments.append(pair)
    # SECOND : Selected parents are then mutated into child
    childList = []
    for parent in parentEnvironments:
        childList.append(copy.deepcopy(parent).mutate())
    # THIRD : Generated childs are then filtered so we only keep environments with the right difficulty
    childListFiltered = []
    for child in childList:
        score = child.benchmarkAverage(nbSimulationBenchmark = 5)
        # if (score > THRESHOLD_TOO_HARD and score < THRESHOLD_TOO_EASY):
        if (score > THRESHOLD_TOO_HARD):
            childListFiltered.append(child)
    # FOURTH : We add the child to the orignial listPairs
    for child in childListFiltered:
        child.listDifficulty = []
        listPairs.append(child)
    # FIFTH : We remove some pair if the list is too big
    pairSize = len(listPairs)
    while pairSize > MAX_ACTIVE_PAIRS:
            listPairs[0].saveListDifficulty("listDifficulty_V" + str(VERSION) + ".txt")
            listPairs.pop(0) # Removing oldest pair when there are too many pairs
            pairSize -= 1
    print('new listPairs : ', listPairs)
    return listPairs

# In this algorithm we optimize several times our agent starting from an easy environment and increasing slowly the difficulty
# 1,5 hours on 12 cores
def curriculumLearning():
    maxIteration = 30 # Giving 30 mutation to the curriculum algorithm
    pair = PairAgentEnv(difficultySTAIRS = 0, difficultySTUMP = 0, difficultyHEIGHT = 0)
    for i in range(maxIteration):
        pair.optimize()
        pair.saveBrain("brainCURRICULUM_V" + str(VERSION) + "_" + ".txt")
        pair.addListDifficulty(i)
        pair.saveLastDifficulty("difficultyLastCURRICULUM_V" + str(VERSION) + "_" + ".txt")
        score = pair.benchmarkAverage(nbSimulationBenchmark = 5)
        if score > THRESHOLD_MUTATE: # Increasing difficulty when the agent has a good enough score
            pair.mutate()
    pair.saveListDifficulty("listDifficultyCURRICULUM_V" + str(VERSION) + ".txt")

# In this algorithm we try to optimize our agent directly and check their results depending on the difficulty of the environment
def directLearning(): # 2 hours to run on 12 cores
    nbBenchmark = 15

    scoresSTAIRS = []
    difficultiesSTAIRS = []
    for difficulty in range(0, nbBenchmark):
        pair = PairAgentEnv(difficultySTAIRS = difficulty/(nbBenchmark-1), difficultySTUMP = 0, difficultyHEIGHT = 0)
        pair.optimize()
        scoresSTAIRS.append(pair.benchmarkAverage())
        difficultiesSTAIRS.append(difficulty/(nbBenchmark-1))
    with open("savedScores/stairs4.txt", "w+") as file:
        for score in scoresSTAIRS:
            file.write(str(score) + "\n")

    scoresSTUMP = []
    difficultiesSTUMP = []
    for difficulty in range(0, nbBenchmark):
        pair = PairAgentEnv(difficultySTAIRS = 0, difficultySTUMP = difficulty/(nbBenchmark-1), difficultyHEIGHT = 0)
        pair.optimize()
        scoresSTUMP.append(pair.benchmarkAverage())
        difficultiesSTUMP.append(difficulty/(nbBenchmark-1))
    with open("savedScores/stump4.txt", "w+") as file:
        for score in scoresSTUMP:
            file.write(str(score) + "\n")

    scoresHEIGHT = []
    difficultiesHEIGHT = []
    for difficulty in range(0, nbBenchmark):
        pair = PairAgentEnv(difficultySTAIRS = 0, difficultySTUMP = 0, difficultyHEIGHT = difficulty/(nbBenchmark-1))
        pair.optimize()
        scoresHEIGHT.append(pair.benchmarkAverage())
        difficultiesHEIGHT.append(difficulty/(nbBenchmark-1))
    with open("savedScores/height4.txt", "w+") as file:
        for score in scoresHEIGHT:
            file.write(str(score) + "\n")

    plotScoresOverDifficulty(scoresSTAIRS, difficultiesSTAIRS, scoresSTUMP, difficultiesSTUMP, scoresHEIGHT, difficultiesHEIGHT)


def main():
    # plotDifficultyReached()
    # plotListDifficulties()

    pair = PairAgentEnv(difficultySTAIRS = 0.6, difficultySTUMP = 0.6, difficultyHEIGHT = 1)
    pair.loadBrain("brainPOET_V2_0.txt")
    # pair.loadBrain("brainPOET0.txt")
    # pair.loadBrain("brainCURRICULUM_V6_.txt")
    # pair.optimize()
    pair.benchmarkAverage()
    pair.benchmark()

    # poetLearning()
    # curriculumLearning()
    # directLearning()

if __name__== "__main__":
    main()
