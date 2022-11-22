import os

from gym_cooking.environment.game import shield_display
from gym_cooking.misc.game.utils import *

import pygame

import os.path
from collections import defaultdict
from datetime import datetime
from time import sleep
from copy import deepcopy
from gym_cooking.utils.utils import correct_fm_tensor, MAP_NUMBER_TO_OBJECT

#os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = 'hide'


class Game:

    def __init__(self, env, num_humans, world_dict, hlas, ai_policies, max_steps=100, render=False, idx_human=0):
        self._running = True
        self.env = env
        self.play = bool(num_humans)
        self.render = render or self.play
        # Visual parameters
        self.graphics_pipeline = shield_display.GraphicPipeline(env, world_dict, hlas, self.render)
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
        self.world_dict = world_dict
        self.env.unwrapped.world = world_dict['worlds'][0]
        self.aborted = False
        assert len(ai_policies) == len(env.unwrapped.world.agents) - num_humans
        #if not os.path.exists(self.save_dir):
        #    os.makedirs(self.save_dir)


    def on_init(self):
        pygame.init()
        self.graphics_pipeline.on_init()
        return True

    def on_event(self, event):
        self.step_done = False
        self.checkForButtonClicks(event)
        if event.type == pygame.QUIT:
            self._running = False
        elif event.type == pygame.KEYDOWN:
            # exit the game
            if event.key == pygame.K_ESCAPE:
                #self._running = False
                pass
            # Control current human agent
            

    def on_execute(self):
        self._running = self.on_init()

        while self._running:
            for event in pygame.event.get():
                self.on_event(event)
            self.on_render()
        self.on_cleanup()
        if self.aborted:
            print("aborted")
            return None
        return self.get_shield()


    def on_render(self):
        self.graphics_pipeline.on_render()

    @staticmethod
    def on_cleanup():
        #pygame.display.quit()
        #pygame.quit()
        pass

    def get_image_obs(self):
        return self.graphics_pipeline.get_image_obs()

    def save_image_obs(self, t):
        self.graphics_pipeline.save_image_obs(t)

    def checkForButtonClicks(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            x, y = pygame.mouse.get_pos()
            if (self.graphics_pipeline.isValidButton(x, y)):
                button = self.graphics_pipeline.getButton(x, y)
                for i in range(len(self.graphics_pipeline.action_buttons)):
                    if button.text == 'Abort':
                        self._running = False
                        self.aborted = True
                    if button is self.graphics_pipeline.action_buttons[i]:
                        self.env.unwrapped.world = self.world_dict['worlds'][i]
                        self.on_render()
                self.graphics_pipeline.process_button_click(button)
                #end if
                if button is self.graphics_pipeline.confirm_button:
                    self._running = False

    def get_shield(self):
        for button in self.graphics_pipeline.action_buttons:
            if button.is_clicked:
                failure = button.text
        for button in self.graphics_pipeline.shields_buttons:
            if button.is_clicked:
                shield_type = button.text
        for button in self.graphics_pipeline.action_selection_buttons:
            if button.is_clicked:
                shield_action = button.text
        return {'Failure': failure, 'Shield type': shield_type, 'Shield action': shield_action}
