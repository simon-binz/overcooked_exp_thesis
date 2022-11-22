import os
from gym_cooking.environment import cooking_zoo
from gym_cooking.environment.game import message_displayer
import pygame
import os.path
from collections import defaultdict
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = 'hide'


class Game:

    def __init__(self, env, message,  max_steps=100, render=True, idx_human=0, carpet = False):
        self._running = True
        self.env = env
        self.render = render
        # Visual parameters
        #self.graphics_pipeline = graphic_pipeline.GraphicPipeline(env, carpet, self.render)
        self.graphics_pipeline = message_displayer.GraphicPipeline(env, message, display=self.render)
        self.save_dir = 'misc/game/screenshots'
        self.store = defaultdict(list)
        self.idx_human = idx_human
        self.max_steps = max_steps
        self.current_step = 0
        self.last_obs = env.reset()
        self.step_done = False
        self.yielding_action_dict = {}
        self.carpet = carpet
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
            if event.key == pygame.K_ESCAPE or pygame.K_KP_ENTER:
                self._running = False
            # Save current image

    def on_execute(self):
        self._running = self.on_init()

        while self._running:
            for event in pygame.event.get():
                self.on_event(event)
            self.on_render()
        self.on_cleanup()

        return self.store

    def on_render(self):
        self.graphics_pipeline.on_render()

    @staticmethod
    def on_cleanup():
        pass

    def get_image_obs(self):
        return self.graphics_pipeline.get_image_obs()

    def save_image_obs(self, t):
        self.graphics_pipeline.save_image_obs(t)


