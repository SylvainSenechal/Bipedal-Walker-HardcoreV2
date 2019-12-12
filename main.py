from PairAgentEnv import PairAgentEnv
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

def environnementDifficulty(difficultyPIT, difficultySTUMP, difficultyHEIGHT): # used for plotting latter
    return (difficultyPIT * 2 + difficultySTUMP * 2 + difficultyHEIGHT) / 5 # (normalized [0, 1], height count less than other)

def main():
    pair = PairAgentEnv(difficultyPIT = 0, difficultySTUMP = 0, difficultyHEIGHT = 1)
    pair.optimize()
    pair.benchmark()

if __name__== "__main__":
    main()
