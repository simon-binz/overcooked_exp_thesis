import random

import pygame

from gym_cooking.environment.game.game import Game
#from gym_cooking.environment.game.failure_message import Game
import numpy as np
from gym_cooking.environment import cooking_zoo
import action_mapping



n_agents = 2
num_humans = 1
max_steps = 100
render = True

level = "tomato_salad"
level1 = 'multiagent1'
level2 = 'multiagent2'
level3 = 'multiagent3'
level4 = 'multiagent4'
seed = 1
record = False
max_num_timesteps = 100
#recipe = ["LettuceOnionSalad"]
recipe = ["TomatoLettuceSalad"]
recipes = ["TomatoLettuceSalad", "TomatoLettuceSalad"]
parallel_env = cooking_zoo.parallel_env(level=level, num_agents=1, record=record, obs_spaces=["numeric"],
                                        max_steps=max_num_timesteps, recipes=recipes)
parallel_env1 = cooking_zoo.parallel_env(level=level1, num_agents=n_agents, record=record, obs_spaces=["numeric"],
                                        max_steps=max_num_timesteps, recipes=recipes)
parallel_env2 = cooking_zoo.parallel_env(level=level2, num_agents=n_agents, record=record, obs_spaces=["numeric"],
                                        max_steps=max_num_timesteps, recipes=recipes)
parallel_env3 = cooking_zoo.parallel_env(level=level3, num_agents=n_agents, record=record, obs_spaces=["numeric"],
                                        max_steps=max_num_timesteps, recipes=recipes)
parallel_env4 = cooking_zoo.parallel_env(level=level4, num_agents=n_agents, record=record, obs_spaces=["numeric"],
                                        max_steps=max_num_timesteps, recipes=recipes)

game = Game(parallel_env, 1, [], max_steps)
store = game.on_execute()



class Passiv_agent():
    def __init__(self, name):
        self.actions_space = [0]
        self.name = name
    def get_action(self, obs):
        return 0
agent = Passiv_agent('player_0')
agent2 = Passiv_agent('player_1')

game = Game(parallel_env1, num_humans, [agent], max_steps)
store = game.on_execute()
game = Game(parallel_env2, num_humans, [agent], max_steps)
store = game.on_execute()
game = Game(parallel_env3, num_humans, [agent], max_steps)
store = game.on_execute()
game = Game(parallel_env4, num_humans, [agent], max_steps)
store = game.on_execute()
