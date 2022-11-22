import gym
import numpy as np
import random
from gym_cooking.environment import cooking_zoo
from gym_cooking.environment.game.game import Game
import gym_cooking.environment.game.shield_query as query
import gym_cooking.environment.game.shield_trainer as trainer
from q_learning_agent import Q_Learning_agent

level = 'tomato_salad_chopped'
level2 = 'tomato_salad'
level = 'tomato_salad'
random.seed = 1
record = False
max_steps = 100
recipe = "TomatoLettuceSalad"
recipes = ["TomatoLettuceSalad","TomatoLettuceSalad"]

env = gym.envs.make("gym_cooking:cookingEnv-v1", level=level, record=record, max_steps=max_steps, recipe=recipe,
                    obs_spaces=["numeric"])
parallel_env = cooking_zoo.parallel_env(level=level, num_agents=1, record=record,
                                        max_steps=max_steps, recipes=recipes, obs_spaces=['numeric'])




QL_agent = Q_Learning_agent(env)
QL_agent.train(250)
QL_agent.plot()
game = Game(parallel_env, 0, [QL_agent], max_steps, render = True)
store = game.on_execute_ai_only_with_delay()
#print(QL_agent.generate_shields(level2))
#actions = [1,2,3,4]
#index = 0
#plan_finished = False
#while not plan_finished:
#    print(actions[index])
#    if index == len(actions)-1 or actions[index] < 8:
#        actions = actions + [5,6,7,8]
#    else:
#        plan_finished = True
#    index += 1
