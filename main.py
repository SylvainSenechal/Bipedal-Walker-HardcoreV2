from PairAgentEnv import PairAgentEnv
from plotter import plotSpider, plotScoresOverDifficulty
# nb : install gym with : pip install 'gym[all]'

### CREATING AN ENVIRONNEMENT :
# env = BipedalWalker(PIT, STUMP, HEIGHT)
# PIT, STUMP, HEIGHT = [0, 1] : 0 => EASY, 1 => HARD
###
# PIT : holes
# STUMP : vertical walls
# HEIGHT : Terrain altitude variation
# Attention reward modifiee




thresholdTooEasy = 200
thresholdTooHard = 50

def POET():
    pass
def rawLearning(): # 12/15 minutes to run on 12 cores
    scores = []
    difficulties = []
    nbBenchmark = 11
    for PIT in range(0, nbBenchmark):
        pair = PairAgentEnv(difficultyPIT = PIT/(nbBenchmark-1), difficultySTUMP = 0, difficultyHEIGHT = 0)
        pair.optimize()
        scores.append(pair.benchmarkAverage())
        difficulties.append(PIT/(nbBenchmark-1))
    plotScoresOverDifficulty(scores, difficulties)

# data = [['PIT', 'STUMP', 'HEIGHT'],
#         ('Basecase', [
#             [0.2, 0.2, 0.8],
#             [0.4, 0.0, 0.5],
#             [0.6, 0.4, 0.9]])]
# plotSpider(data)

def environnementDifficulty(difficultyPIT, difficultySTUMP, difficultyHEIGHT): # used for plotting latter
    return (difficultyPIT * 2 + difficultySTUMP * 2 + difficultyHEIGHT) / 5 # (normalized [0, 1], height count less than other)

def main():
    # pair = PairAgentEnv(difficultyPIT = 0, difficultySTUMP = 0, difficultyHEIGHT = 1)
    # pair.optimize()
    # pair.benchmarkAverage()
    # pair.benchmark()

    rawLearning()

    # pair.saveBrain("brain1.txt")
    # pair2 = PairAgentEnv(difficultyPIT = 0, difficultySTUMP = 0, difficultyHEIGHT = 1)
    # pair2.loadBrain("brain1.txt")
    # pair2.benchmark()

if __name__== "__main__":
    main()
