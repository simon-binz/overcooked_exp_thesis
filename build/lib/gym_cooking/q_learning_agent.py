import gym
import numpy as np
import random
from gym_cooking.environment import cooking_zoo
from gym_cooking.environment.game.game import Game
import gym_cooking.environment.game.shield_query as query
import gym_cooking.environment.game.shield_trainer as trainer
import action_mapping
import matplotlib.pyplot as plt
from copy import deepcopy

level = 'tomato_salad_chopped'
level2 = 'tomato_salad'
random.seed = 1
np.random.seed(1)
record = False
max_steps = 100
recipe = "TomatoLettuceSalad"
recipes = ["TomatoLettuceSalad","TomatoLettuceSalad"]

env = gym.envs.make("gym_cooking:cookingEnv-v1", level=level, record=record, max_steps=max_steps, recipe=recipe,
                    obs_spaces=["numeric"])
env2 = gym.envs.make("gym_cooking:cookingEnv-v1", level=level2, record=record, max_steps=max_steps, recipe=recipe,
                    obs_spaces=["numeric"])

parallel_env = cooking_zoo.parallel_env(level=level2, num_agents=1, record=record,
                                        max_steps=max_steps, recipes=recipes, obs_spaces=['numeric'])

action_space = env.action_space


class Q_Learning_agent:
    def __init__(self, env):
        self.use_shields = False
        self.alpha = 0.1
        self.epsilon = 0.3
        self.gamma = 0.9
        self.Q_values = {}
        self.env = env
        #self.actions = range(len(action_mapping.Actions))
        self.actions = [0,1,3,4,14]
        self.action_map = action_mapping.ActionMapping()
        self.rewards = []
        #shields as a list of ditionaries
        self.shields = []

    def train(self, n_episodes):
        self.evaluate()
        for i in range(n_episodes):
            #print("training: ", i)
            done = False
            observation = self.env.reset()
            while not done:
                action = self.learn_action(observation)
                old_obs = observation
                agent_pos = old_obs['agent_location']
                env_state = old_obs['symbolic_observation']
                if self.use_shields:
                    action = self.shield_action(action, env_state, agent_pos)
                high_level_action = action_mapping.Actions(action)
                #print(high_level_action, agent_pos, env_state)
                #print("high level action: ", high_level_action)
                self.action_map.motion_generator.reset(env_state, agent_pos)
                low_level_action = self.action_map.map_action(env_state, agent_pos, high_level_action)
                #print("low level action: ", low_level_action)
                observation, reward, done, info = self.env.step(low_level_action)
                self.update_policy(old_obs, observation, reward, action)
            self.evaluate()

    def generate_shields(self, level):
        shield_env = cooking_zoo.parallel_env(level=level, num_agents=1, record=record,
                                        max_steps=max_steps, recipes=recipes, obs_spaces=['numeric'])
        shield_trainer = trainer.Game(shield_env, 0, [self], max_steps, render=False)
        shield_dict = shield_trainer.on_execute_ai_only_with_delay()
        #print(shield_dict)
        shield_query = query.Game(shield_env, 0, shield_dict, [self], max_steps, render=True)
        shields = shield_query.on_execute()
        return shields

    def get_action(self, obs, return_high_level_action = False):
        state = self.convert_obs_to_tuple(obs)
        viable_actions = random.sample(list(self.actions), len(self.actions))
        # check if the obs is known
        if self.check_if_obs_in_Q_values(state):
            current_reward = float('-inf')
            for current_action in viable_actions:
                if self.Q_values[state][current_action] >= current_reward:
                    action = current_action
                    current_reward = self.Q_values[state][current_action]
        else:
            action = viable_actions[0]
        try:
            agent_pos = obs['player_0']['agent_location']
            env_state = obs['player_0']['symbolic_observation']
        except:
            agent_pos = obs['agent_location']
            env_state = obs['symbolic_observation']
        if self.use_shields:
            action = self.shield_action(action, env_state, agent_pos)
        self.action_map.motion_generator.reset(env_state, agent_pos)
        low_level_action = self.action_map.map_action(env_state, agent_pos, action)
        if return_high_level_action:
            return (action, low_level_action)
        return low_level_action


    def learn_action(self, obs):
        exploration = np.random.uniform(0, 1) <= self.epsilon
        action = None
        if exploration:
            action = random.choice(self.actions)
        else:
            state = self.convert_obs_to_tuple(obs)
            viable_actions = random.sample(list(self.actions), len(self.actions))
            #check if the obs is known
            if self.check_if_obs_in_Q_values(state):
                current_reward = float('-inf')
                for current_action in viable_actions:
                    if self.Q_values[state][current_action] >= current_reward:
                        action = current_action
                        current_reward = self.Q_values[state][current_action]
            else:
                action = viable_actions[0]
        return action


    def update_policy(self, old_obs, new_obs, reward, action):
        old_state = self.convert_obs_to_tuple(old_obs)
        next_state = self.convert_obs_to_tuple(new_obs)
        current_q_value = 0
        if self.check_if_obs_in_Q_values(old_state):
            current_q_value = self.Q_values[old_state][action]
        max_q = float('-inf')
        for a in self.actions:
            if self.check_if_obs_in_Q_values(next_state):
                if self.Q_values[next_state][a] >= max_q:
                    max_q = self.Q_values[next_state][a]
            else:
                max_q = 0
        q_hat = reward + self.gamma * max_q
        new_q_value = current_q_value + self.alpha * (q_hat - current_q_value)
        new_q_value = round(new_q_value, 3)
        self.Q_values[old_state][action] = new_q_value

    def shield_action(self, action, env_state, agent_pos):
        #Tomato_Plate
        #for shield in self.shields:
        #    if action
        if action == 2:
            if self.action_map.map_action(env_state, agent_pos, action) == 0:
                #1 is equal to CHOP_TOMATO
                print("shielding")
                return 1
        return action

    def evaluate(self):
        done = False
        observation = self.env.reset()
        rewards_this_game = []
        while not done:
            action = self.get_action(observation)
            observation, reward, done, info = self.env.step(action)
            rewards_this_game.append(reward)
        self.rewards.append(np.sum(rewards_this_game))


    def convert_obs_to_tuple(self, obs):
        player_obs = None
        try:
            player_obs = obs['player_0']['symbolic_observation']
        except:
            player_obs = obs['symbolic_observation']
        #print(tuple((map(tuple, np.vstack(player_obs)))))
        #array_of_tuples = map(tuple, player_obs)
        #tuple_of_tuples = tuple(array_of_tuples)
        #return tuple_of_tuples
        return tuple((map(tuple, np.vstack(player_obs))))


    #check if the given obs is in the Q_values, else append it to the Q_values
    def check_if_obs_in_Q_values(self, obs):
        known_states = self.Q_values.keys()
        state_in_Q_values = False
        for state in known_states:
            if (obs == state):
                state_in_Q_values = True
                break
        if state_in_Q_values:
            return True
        else:
            #add state obs to Q-Values:
            self.Q_values[obs] = {}
            for action in self.actions:
                self.Q_values[obs][action] = 0
        return False

    def plot(self, title = None):
        x = range(len(self.rewards))
        plt.plot(x, self.rewards, label = title)
        plt.legend()
        if title is not None:
            plt.title(title)
            plt.savefig(title)
        else:
            plt.show()

#QL_agent = Q_Learning_agent(env)
#QL_agent.train(100)
#QL_agent.env = env2
#QL_agent.rewards = []
#QL_agent2 = deepcopy(QL_agent)
#QL_agent2.use_shields = True
#print("1")
#QL_agent.train(100)
#print("2")
#QL_agent2.train(100)
#QL_agent.plot("no shields")
#QL_agent2.plot("shields")
#QL_agent.generate_shields(level2)
#input("start?")

#game = Game(parallel_env, 0, [QL_agent], max_steps, render=True)
#store = game.on_execute_ai_only_with_delay()

