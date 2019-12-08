# python -m cProfile -s time testES.py

# installation : pip install 'gym[all]'
# making a custom env : https://github.com/openai/gym/blob/master/docs/creating-environments.md
# bipedal walker source code : https://github.com/openai/gym/blob/master/gym/envs/box2d/bipedal_walker.py

# code https://gym.openai.com/evaluations/eval_QjN2JTCQRFu7rNgm0iGy4A/

"""
Usefull ressources :
http://blog.otoro.net/2017/10/29/visual-evolution-strategies/
http://blog.otoro.net/2017/11/12/evolving-stable-strategies/
https://openai.com/blog/evolution-strategies/
"""

# algo génétique +/- double
# - travail a ameliorer les jouers
# - travail a ameliorer les environnements
# dans le papier de recheche le nnet : 24 => 40 => 40 => 4 tanh
# 512 pop

"""
IMPORTANT PARAMETERS :

TERRAIN_GRASS : Distance of GRASS between 2 types of obstacles (stump, pit, ..)

For the STAIRS :
stair_height = +1 if self.np_random.rand() > 0.5 else -1
stair_width = self.np_random.randint(4, 5)
stair_steps = self.np_random.randint(3, 5)
"""

""" Environnement bipedalWalker :
-   24 observations <=> state (including 10 form lidar)
-   4 actions possible :
    - hip 1/2
    - knee 1/2
- Reward :
    - + 300 when reaching the end
    - - 100 when robot falls
    - motor torque costs a bit


"""
