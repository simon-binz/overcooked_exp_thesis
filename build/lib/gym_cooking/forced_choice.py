import pygame
import sys

from gym_cooking.environment import cooking_zoo
from gym_cooking.environment.game import forced_choice_query

level = 'tomato_salad_chopped'
level2 = 'tomato_salad'

record = False
max_steps = 100
recipe = "TomatoLettuceSalad"
recipes = ["TomatoLettuceSalad","TomatoLettuceSalad"]

base_env = cooking_zoo.parallel_env(level=level, num_agents=1, record=record,
                                              max_steps=max_steps, recipes=recipes, obs_spaces=['numeric'])

def make_query():
    forced_q = forced_choice_query.Game(base_env)
    return forced_q.on_execute()
