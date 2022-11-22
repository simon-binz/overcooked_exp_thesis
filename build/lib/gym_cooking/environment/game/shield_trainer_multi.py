import os

from gym_cooking.environment.game import graphic_pipeline
from gym_cooking.misc.game.utils import *
import action_mapping
import pygame

import os.path
from collections import defaultdict
from datetime import datetime
from time import sleep
from copy import deepcopy
import action_mapping
from gym_cooking.utils.utils import correct_fm_tensor, MAP_NUMBER_TO_OBJECT

#os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = 'hide'


class Game:

    def __init__(self, env, num_humans, ai_policies, action_history1, action_history2,max_steps=100, render=False, idx_human=0):
        self._running = True
        self.env = env
        self.play = bool(num_humans)
        self.render = render or self.play
        # Visual parameters
        #self.graphics_pipeline = graphic_pipeline.GraphicPipeline(env, self.render)
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
        self.action_history1 = action_history1
        self.action_history2 = action_history2
        self.current_hlas = []
        #dict of worlds, high level actions, low level actions
        self.failure_dict = {'worlds': [], 'high level actions': [], 'low level actions': []}
        assert len(ai_policies) == len(env.unwrapped.world.agents) - num_humans
        #if not os.path.exists(self.save_dir):
        #    os.makedirs(self.save_dir)

    def on_init(self):
        pygame.init()
        #self.graphics_pipeline.on_init()
        return True

    def ai_only_event(self):
        self.step_done = False
        self.failure_dict['worlds'].append(deepcopy(self.env.unwrapped.world))
        for idx, agent in enumerate(self.env.unwrapped.world.agents):
            if idx >= self.num_humans:
                #get high level action
                if idx == 0:
                    actions = self.action_history1.pop(0)
                else:
                    actions = self.action_history2.pop(0)
                high_level_action = action_mapping.Actions(actions[0])
                self.current_hlas.append(high_level_action)
                if actions[1] == 'failure':
                    self._running = False
                    #continue
                else:
                    for action in actions[1]:
                        if idx == 0:
                            self.env.unwrapped.world.agents[0].action = action
                            self.env.unwrapped.world.agents[1].action = 0
                        else:
                            self.env.unwrapped.world.agents[0].action = 0
                            self.env.unwrapped.world.agents[1].action = action
                        self.yielding_action_dict = {agent: self.env.unwrapped.world_agent_mapping[agent].action
                                                         for agent in self.env.agents}
                        observations, rewards, dones, infos = self.env.step(self.yielding_action_dict)
            self.failure_dict['high level actions'].append(self.current_hlas)
            self.current_hlas = []
        self.step_done = True

    def on_execute_ai_only_with_delay(self):
        self._running = self.on_init()

        while self._running:
            self.ai_only_event()
            self.on_render()
        self.on_cleanup()
        return self.failure_dict

    def on_render(self):
        pass

    @staticmethod
    def on_cleanup():
        #pygame.display.quit()
        #pygame.quit()
        pass
    pass

    def get_image_obs(self):
        #return self.graphics_pipeline.get_image_obs()
        pass

    def save_image_obs(self, t):
        #self.graphics_pipeline.save_image_obs(t)
        pass

    def check_failure(self, low_level_action, action):
        high_level_action = action_mapping.Actions(action)
        if high_level_action is not(action_mapping.Actions.AVOID):
            if low_level_action == 0:
                print("failure", high_level_action, low_level_action)
                self._running = False
                return True
        return False


