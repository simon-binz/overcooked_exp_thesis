import os

from gym_cooking.environment.game import graphic_pipeline

from gym_cooking.misc.game.utils import *

import pygame

import os.path
from collections import defaultdict
from datetime import datetime
from time import sleep

from gym_cooking.utils.utils import correct_fm_tensor, MAP_NUMBER_TO_OBJECT

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = 'hide'


class Game:

    def __init__(self, env, num_humans, ai_policies, max_steps=100, render=False, idx_human=0, carpet = False):
        self._running = True
        self.env = env
        self.play = bool(num_humans)
        self.render = render or self.play
        # Visual parameters
        self.graphics_pipeline = graphic_pipeline.GraphicPipeline(env, carpet, self.render)
        self.save_dir = 'misc/game/screenshots'
        self.store = defaultdict(list)
        self.num_humans = num_humans
        self.idx_human = idx_human
        self.ai_policies = ai_policies
        self.max_steps = max_steps
        self.current_step = 0
        self.last_obs = env.reset()
        self.step_done = False
        self.yielding_action_dict = {}
        self.carpet = carpet
        assert len(ai_policies) == len(env.unwrapped.world.agents) - num_humans
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def on_init(self):
        pygame.init()
        self.graphics_pipeline.on_init()
        return True

    def on_event(self, event):
        self.step_done = False
        if event.type == pygame.QUIT:
            self._running = False
        elif event.type == pygame.KEYDOWN:
            # exit the game
            if event.key == pygame.K_ESCAPE:
                self._running = False
            # Save current image
            if event.key == pygame.K_RETURN:
                image_name = '{}_{}.png'.format(self.env.unwrapped.filename, datetime.now().strftime('%m-%d-%y_%H-%M-%S'))
                pygame.image.save(self.graphics_pipeline.screen, '{}/{}'.format(self.save_dir, image_name))
                print('Saved image {} to {}'.format(image_name, self.save_dir))
                return

            # Control current human agent
            if event.key in KeyToTuple_human1 and self.num_humans > 0:
                store_action_dict = {}
                action = KeyToTuple_human1[event.key]
                self.env.unwrapped.world.agents[self.idx_human].action = action
                store_action_dict[self.env.unwrapped.world.agents[self.idx_human]] = action
                self.store["observation"].append(self.last_obs)
                self.store["agent_states"].append([agent.location for agent in self.env.unwrapped.world.agents])
                for idx, agent in enumerate(self.env.unwrapped.world.agents):
                    if idx != self.idx_human:
                        ai_policy = self.ai_policies[idx - self.num_humans]
                        env_agent = self.env.unwrapped.world_agent_to_env_agent_mapping[agent]
                        # last_obs_raw = self.last_obs[env_agent]
                        last_obs_raw = self.last_obs
                        ai_action = ai_policy.get_action(last_obs_raw)
                        store_action_dict[agent] = ai_action
                        self.env.unwrapped.world.agents[idx].action = ai_action

                self.yielding_action_dict = {agent: self.env.unwrapped.world_agent_mapping[agent].action
                                             for agent in self.env.agents}
                observations, rewards, dones, infos = self.env.step(self.yielding_action_dict)
                print(tuple(observations['player_0']['agent_location']))
                if self.carpet:
                    if tuple(observations['player_0']['agent_location']) == (2,1):
                        self._running = False
                self.store["actions"].append(store_action_dict)
                self.store["info"].append(infos)
                self.store["rewards"].append(rewards)
                self.store["done"].append(dones)

                if all(dones.values()):
                    self._running = False

                self.last_obs = observations
                self.step_done = True

            # print_game_sao(self)

    def ai_only_event(self):
        self.step_done = False

        store_action_dict = {}

        self.store["observation"].append(self.last_obs)
        self.store["agent_states"].append([agent.location for agent in self.env.unwrapped.world.agents])
        for idx, agent in enumerate(self.env.unwrapped.world.agents):
            if idx >= self.num_humans:
                ai_policy = self.ai_policies[idx - self.num_humans]
                env_agent = self.env.unwrapped.world_agent_to_env_agent_mapping[agent]
                # last_obs_raw = self.last_obs[env_agent]
                last_obs_raw = self.last_obs
                ai_action = ai_policy.get_action(last_obs_raw)
                store_action_dict[agent] = ai_action
                self.env.unwrapped.world.agents[idx].action = ai_action

        self.yielding_action_dict = {agent: self.env.unwrapped.world_agent_mapping[agent].action
                                     for agent in self.env.agents}
        observations, rewards, dones, infos = self.env.step(self.yielding_action_dict)
        if self.carpet:
            if tuple(observations['player_0']['agent_location']) == (2,1):
                self._running = False
        print(rewards)
        self.store["actions"].append(store_action_dict)
        self.store["info"].append(infos)
        self.store["rewards"].append(rewards)
        self.store["done"].append(dones)


        if all(dones.values()):
            self._running = False

        self.last_obs = observations
        self.step_done = True

    def on_execute(self):
        self._running = self.on_init()

        while self._running:
            for event in pygame.event.get():
                self.on_event(event)
            self.on_render()
        self.on_cleanup()

        return self.store

    def on_execute_yielding(self):
        self._running = self.on_init()

        while self._running:
            for event in pygame.event.get():
                self.on_event(event)
            self.on_render()
            if self.step_done:
                self.step_done = False
                yield self.store["observation"][-1], self.store["done"][-1], self.store["info"][-1], \
                      self.store["rewards"][-1], self.yielding_action_dict
        self.on_cleanup()

    def on_execute_ai_only_with_delay(self):
        self._running = self.on_init()

        while self._running:
            sleep(0.2)
            self.ai_only_event()
            self.on_render()
        sleep(2)
        self.on_cleanup()

        return self.store

    def on_render(self):
        self.graphics_pipeline.on_render()

    @staticmethod
    def on_cleanup():
        # pygame.display.quit()
        #pygame.quit()
        pass

    def get_image_obs(self):
        return self.graphics_pipeline.get_image_obs()

    def save_image_obs(self, t):
        self.graphics_pipeline.save_image_obs(t)


def print_game_sao(game):
    """
    Prints the current state, last taken actions and last observation
    """
    # Print State, Actions and Observations
    tensor_information = game.env.unwrapped.get_tensor_representation()

    # for state as feature map and oriented as shown in the game-GUI, transpose tensore
    state = correct_fm_tensor(tensor_information)

    # TODO are actions in the right order?
    actions = game.store['actions']
    if actions:
        actions = actions[-1]
    # TODO observations
    obs = game.store['observation']
    if obs:
        obs = obs[-1]

    agent_states = game.store['agent_states']
    if agent_states:
        agent_states = agent_states[-1]

    # print state
    agent_map = 0
    for i in range(state.shape[2]):
        if i in [16, 15 + len(game.env.unwrapped.world.agents)]:
            print(f"Feature Map - Agent {agent_map}:")
            agent_map += 1
        else:
            print(f"Feature Map - {MAP_NUMBER_TO_OBJECT[i - agent_map]}:")
        print(state[:, :, i])

    print("\n")
    for i, agent_state in enumerate(agent_states):
        print(f"Agent {i} is at {agent_state}")
    print(actions)
    print(obs)

