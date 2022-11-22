import os

from gym_cooking.environment.game import replay_graphic_pipeline

from gym_cooking.misc.game.utils import *

import pygame

import os.path
from collections import defaultdict
from datetime import datetime
from time import sleep

from gym_cooking.utils.utils import correct_fm_tensor, MAP_NUMBER_TO_OBJECT

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = 'hide'


class Game:

    def __init__(self, env, num_humans, ai_policies, max_steps=100, render=True, idx_human=0, carpet = False):
        self._running = True
        self.env = env
        self.play = bool(num_humans)
        self.render = render or self.play
        # Visual parameters
        action_history = ai_policies[0].action_history
        self.graphics_pipeline = replay_graphic_pipeline.GraphicPipeline(env,action_history, carpet = False, display= self.render)
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
        self.hlas = []
        assert len(ai_policies) == len(env.unwrapped.world.agents) - num_humans
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)


    def on_init(self):
        pygame.init()
        self.graphics_pipeline.on_init()
        return True

    def ai_only_event(self):
        self.step_done = False

        store_action_dict = {}

        self.store["observation"].append(self.last_obs)
        self.store["agent_states"].append([agent.location for agent in self.env.unwrapped.world.agents])
        for idx, agent in enumerate(self.env.unwrapped.world.agents):
            if idx >= self.num_humans:
                ai_policy = self.ai_policies[idx - self.num_humans]
                env_agent = self.env.unwrapped.world_agent_to_env_agent_mapping[agent]
                last_obs_raw = self.last_obs
                ai_action = ai_policy.get_action(last_obs_raw)
                hla = ai_policy.current_hla
                self.graphics_pipeline.hla = hla
                if len(self.hlas) == 0:
                    self.hlas.append(hla)
                if self.hlas[-1:][0] != hla:
                    self.hlas.append(hla)
                if ai_action == 'failure':
                    self._running = False
                    self.env.unwrapped.world.agents[idx].action = 0 #todobugtest this
                    if len(self.ai_policies) == 1:
                        self.graphics_pipeline.action_stack_buttons[-1:][0].background_color = Color.RED
                    break
                store_action_dict[agent] = ai_action
                self.env.unwrapped.world.agents[idx].action = ai_action

        self.yielding_action_dict = {agent: self.env.unwrapped.world_agent_mapping[agent].action
                                     for agent in self.env.agents}
        observations, rewards, dones, infos = self.env.step(self.yielding_action_dict)
        if self.carpet:
            if tuple(observations['player_0']['agent_location']) == (2,1):
                self._running = False
        #print(rewards)
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
            sleep(0.4)
            self.ai_only_event()
            self.on_render()
        sleep(2)
        self.on_cleanup()

        return self.hlas

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

