import os


from gym_cooking.environment.game import failure_message_multi_display
from gym_cooking.misc.game.utils import *

import pygame

import os.path
from collections import defaultdict
from datetime import datetime
from time import sleep

from gym_cooking.utils.utils import correct_fm_tensor, MAP_NUMBER_TO_OBJECT

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = 'hide'


class Game:

    def __init__(self, env, num_humans, hlas1, hlas2, failure, max_steps=100, render=False, idx_human=0, carpet = False):
        self._running = True
        self.env = env
        self.play = bool(num_humans)
        self.render = render or self.play
        # Visual parameters
        #self.graphics_pipeline = graphic_pipeline.GraphicPipeline(env, carpet, self.render)
        self.graphics_pipeline = failure_message_multi_display.GraphicPipeline(env, hlas1= hlas1, hlas2 = hlas2,
                                                                               failure = failure, display=self.render)
        self.save_dir = 'misc/game/screenshots'
        self.store = defaultdict(list)
        self.num_humans = num_humans
        self.idx_human = idx_human
        self.hlas1 = hlas1
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

