# installation : pip install 'gym[all]'
# making a custom env : https://github.com/openai/gym/blob/master/docs/creating-environments.md
# bipedal walker source code : https://github.com/openai/gym/blob/master/gym/envs/box2d/bipedal_walker.py

# algo génétique double !
# - travail a ameliorer les jouers
# - travail a ameliorer les environnements

# Important parameters :
"""
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




# import gym
# env = gym.make('BipedalWalkerHardcore-v2')
# for i_episode in range(1):
#     observation = env.reset()
#     for t in range(1000):
#         env.render()
#         # print(observation)
#         action = env.action_space.sample()
#         print(action)
#         observation, reward, done, info = env.step(action)
#         if done:
#             print("Episode finished after {} timesteps".format(t+1))
#             break
# env.close()


# import gym
# from agent import Agent
#
# environment = gym.make('BipedalWalker-v2')
# a = Agent(environment)
# print(a.environment)
