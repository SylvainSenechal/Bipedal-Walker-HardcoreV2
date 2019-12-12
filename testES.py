from PairAgentEnv import PairAgentEnv
# nb : install gym with : pip install 'gym[all]'

### CREATING AN ENVIRONNEMENT :
# env = BipedalWalker(PIT, STUMP, HEIGHT)
# PIT, STUMP, HEIGHT = [0, 1] : 0 => EASY, 1 => HARD
###
# PIT : holes
# STUMP : vertical walls
# HEIGHT : Terrain altitude variation


def main():
    pair = PairAgentEnv()
    pair.optimize()
    pair.benchmark()

if __name__== "__main__":
    main()
