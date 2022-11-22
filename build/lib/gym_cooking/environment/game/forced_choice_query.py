import os

from gym_cooking.environment.game import forced_choice_display
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

    def __init__(self, env, num_humans=0, idx_human=0, ai_policies = None, max_steps =100, render=True):
        self._running = True
        self.env = env
        self.play = bool(num_humans)
        self.render = render or self.play
        # Visual parameters
        self.graphics_pipeline = forced_choice_display.GraphicPipeline(env, self.render)
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
        self.aborted = False
        self.user_text = ""
        self.clicked_button_text = ""


    def on_init(self):
        pygame.init()
        self.graphics_pipeline.on_init()
        return True

    def on_event(self, event):
        self.step_done = False
        self.checkForButtonClicks(event)
        if event.type == pygame.QUIT:
            self._running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN:
                self.user_text = self.user_text.strip()
                self._running = False
            if event.key == pygame.K_BACKSPACE:
                self.user_text = self.user_text[:-1]
            else:
                if event.key != pygame.K_RETURN:
                    self.user_text += event.unicode
            

    def on_execute(self):
        self._running = self.on_init()

        while self._running:
            for event in pygame.event.get():
                self.on_event(event)
                self.graphics_pipeline.user_text = self.user_text
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
                self.graphics_pipeline.process_button_click(button)
            self.clicked_button_text = ""
            for button in self.graphics_pipeline.shields_buttons:
                if button.is_clicked:
                    self.clicked_button_text = button.text

    def get_shield(self):
        return [self.clicked_button_text, self.user_text]
