import pygame
from gym_cooking.cooking_world.world_objects import *
from gym_cooking.misc.game.utils import *
from collections import defaultdict, namedtuple
from gym_cooking.utils.utils import HIGH_LEVEL_ACTION_IMAGE_MAP
import numpy as np
import pathlib
import os.path
import math


COLORS = ['blue', 'magenta', 'yellow', 'green']

_image_library = {}


def get_image(path):
    global _image_library
    image = _image_library.get(path)
    if image == None:
        canonicalized_path = path.replace('/', os.sep).replace('\\', os.sep)
        image = pygame.image.load(canonicalized_path)
        _image_library[path] = image
    return image


GraphicsProperties = namedtuple("GraphicsProperties", ["pixel_per_tile", "holding_scale", "container_scale",
                                                       "width_pixel", "height_pixel", "tile_size", "holding_size",
                                                       "container_size", "holding_container_size"])

class Button():
    def __init__(self, loc, color= Color.BLACK, image = None, text = None, background_color = Color.LIGHTBLUE):
        #unscaled location (without excess width or height)
        self.loc = loc
        self.base_color = color
        self.color = color
        self.is_clicked = False
        # String describing the button
        self.text = text
        #button image
        self.image = image
        self.background_color = background_color
    #toggle button
    def on_click(self):
        if self.is_clicked == False:
            self.is_clicked = True
            self.color = Color.GREEN
        else:
            self.is_clicked = False
            self.color = self.base_color

class GraphicPipeline:

    PIXEL_PER_TILE = 80
    HOLDING_SCALE = 0.5
    CONTAINER_SCALE = 0.7

    def __init__(self, env, world_dict, hlas, display=False):
        self.env = env
        self.display = display
        self.screen = None
        self.graphics_dir = 'misc/game/graphics'
        self.graphics_properties = GraphicsProperties(self.PIXEL_PER_TILE, self.HOLDING_SCALE, self.CONTAINER_SCALE,
                                                      self.PIXEL_PER_TILE * self.env.unwrapped.world.width,
                                                      self.PIXEL_PER_TILE * self.env.unwrapped.world.height,
                                                      (self.PIXEL_PER_TILE, self.PIXEL_PER_TILE),
                                                      (self.PIXEL_PER_TILE * self.HOLDING_SCALE,
                                                       self.PIXEL_PER_TILE * self.HOLDING_SCALE),
                                                      (self.PIXEL_PER_TILE * self.CONTAINER_SCALE,
                                                       self.PIXEL_PER_TILE * self.CONTAINER_SCALE),
                                                      (self.PIXEL_PER_TILE * self.CONTAINER_SCALE * self.HOLDING_SCALE,
                                                       self.PIXEL_PER_TILE * self.CONTAINER_SCALE * self.HOLDING_SCALE))
        my_path = os.path.realpath(__file__)
        dir_name = os.path.dirname(my_path)
        path = pathlib.Path(dir_name)
        self.root_dir = path.parent.parent
        pygame.init()
        self.infoObject = pygame.display.Info()
        self.excess_width = (self.infoObject.current_w - (
                self.graphics_properties.width_pixel + 2 * self.PIXEL_PER_TILE)) // 2
        self.excess_height = (self.infoObject.current_h - (
                self.graphics_properties.height_pixel +
                self.PIXEL_PER_TILE)) // 2

        #shield selection after clicking the failure
        self.shield_selection_mode = False

        #action selection after clicking both buttons
        self.action_selection_mode = False

        #after clicking all 3 buttons
        self.confirm_mode = False

        self.action_selection_buttons_normal = []
        for i, key in enumerate(hlas):
            text = HIGH_LEVEL_ACTION_IMAGE_MAP[key]
            if len(hlas)>5:
                button = Button((-0.75*self.excess_width + (i)*self.PIXEL_PER_TILE + 0.5*self.PIXEL_PER_TILE, -0.5*self.excess_height+(2)*self.PIXEL_PER_TILE+self.graphics_properties.height_pixel), image=text, text=text)
            else:
                button = Button((-0.5 * self.excess_width + (i) * self.PIXEL_PER_TILE+0.5*self.PIXEL_PER_TILE,
                                 -0.5 * self.excess_height + (
                                     2) * self.PIXEL_PER_TILE + self.graphics_properties.height_pixel), image=text,
                                text=text)
            self.action_selection_buttons_normal.append(button)

        self.action_selection_buttons_gray = []
        for button in self.action_selection_buttons_normal:
            button_g = Button(button.loc, image=button.image + '_gray', text=button.text)
            self.action_selection_buttons_gray.append(button_g)
        self.action_selection_buttons = self.action_selection_buttons_gray
        self.action_stack1 = []
        self.action_stack2 = []

        #the actions buttons which led to failure
        for i in range(len(world_dict['high level actions'])):
            text = HIGH_LEVEL_ACTION_IMAGE_MAP[world_dict['high level actions'][i]]
            if i == len(world_dict['high level actions'])-1:
                button = Button((-0.5 * self.excess_width + self.PIXEL_PER_TILE* (i+1),
                                 -0.5 * self.excess_height ), image=text, text=text, background_color=Color.RED)
            else:
                button = Button((-0.5*self.excess_width+ self.PIXEL_PER_TILE* (i+1), -0.5*self.excess_height), image= text, text = text, background_color=Color.WHITE)
            self.action_stack1.append(button)



        self.action_buttons = self.action_stack1 + self.action_stack2
        self.agent_button = Button((-0.5 * self.excess_width, -0.5*self.excess_height), image='agent-blue', background_color=Color.WHITE)



        #the shield text buttons (insert, replace, forbid)
        self.font = pygame.font.SysFont('timesnewroman', 28)
        self.small_font = pygame.font.SysFont('timesnewroman', 18)

        self.describe_text1 = self.font.render('1: Select which action needs to be modified to avoid the failure:', True,
                                               Color.BLACK, Color.WHITE)
        self.describe_text_rect1 = self.describe_text1.get_rect()
        self.describe_text_rect1.center = (0.5 * self.excess_width + self.PIXEL_PER_TILE * 3,
                                           0.5 * self.excess_height + - self.PIXEL_PER_TILE * 0.5)

        self.insert_text = self.small_font.render('Insert', True, Color.BLACK, Color.WHITE)
        self.insert_text_g = self.small_font.render('Insert', True, Color.BLACK, Color.LIGHTBLUE)
        self.insert_text_rect = self.insert_text.get_rect()
        self.insert_text_rect.center = (0.5*self.excess_width + self.PIXEL_PER_TILE * 1.5,
                                        0.5*self.excess_height + self.graphics_properties.height_pixel + self.PIXEL_PER_TILE /4)

        self.action_text = self.small_font.render('Action', True, Color.BLACK, Color.WHITE)
        self.action_text_g = self.small_font.render('Action', True, Color.BLACK, Color.LIGHTBLUE)
        self.action_text2 = self.small_font.render('Action:', True, Color.BLACK, Color.WHITE)
        self.action_text2_g = self.small_font.render('Action:', True, Color.BLACK, Color.LIGHTBLUE)
        self.a1_text_rect = self.action_text.get_rect()
        self.a1_text_rect.center = (0.5 * self.excess_width + self.PIXEL_PER_TILE * 1.5,
                                        0.5 * self.excess_height + self.graphics_properties.height_pixel + self.PIXEL_PER_TILE / 2)

        self.before_text = self.small_font.render('Before:', True, Color.BLACK, Color.WHITE)
        self.before_text_g = self.small_font.render('Before:', True, Color.BLACK, Color.LIGHTBLUE)
        self.before_text_rect = self.before_text.get_rect()
        self.before_text_rect.center = (0.5*self.excess_width + self.PIXEL_PER_TILE * 1.5,
                                        0.5*self.excess_height + self.graphics_properties.height_pixel + self.PIXEL_PER_TILE * (3/4))

        self.insert_button = Button((-0.5*self.excess_width +self.PIXEL_PER_TILE, -0.5*self.excess_height+self.graphics_properties.height_pixel), text='insert before')

        self.inserta_text = self.small_font.render('Insert', True, Color.BLACK, Color.WHITE)
        self.inserta_text_g = self.small_font.render('Insert', True, Color.BLACK, Color.LIGHTBLUE)
        self.inserta_text_rect = self.inserta_text.get_rect()
        self.inserta_text_rect.center = (0.5*self.excess_width + self.PIXEL_PER_TILE * 2.5,
                                        0.5*self.excess_height + self.graphics_properties.height_pixel + self.PIXEL_PER_TILE / 4)

        self.a2_text_rect = self.action_text.get_rect()
        self.a2_text_rect.center = (0.5 * self.excess_width + self.PIXEL_PER_TILE * 2.5,
                                    0.5 * self.excess_height + self.graphics_properties.height_pixel + self.PIXEL_PER_TILE / 2)

        self.after_text = self.small_font.render('After:', True, Color.BLACK, Color.WHITE)
        self.after_text_g = self.small_font.render('After:', True, Color.BLACK, Color.LIGHTBLUE)
        self.after_text_rect = self.after_text.get_rect()
        self.after_text_rect.center = (0.5*self.excess_width + self.PIXEL_PER_TILE * 2.5,
                                         0.5*self.excess_height + self.graphics_properties.height_pixel + self.PIXEL_PER_TILE * (3/4))
        self.inserta_button = Button((-0.5*self.excess_width +self.PIXEL_PER_TILE*2, -0.5*self.excess_height+self.graphics_properties.height_pixel), text='insert after')

        self.replace_text = self.small_font.render('Replace', True, Color.BLACK, Color.WHITE)
        self.replace_text_g = self.small_font.render('Replace', True, Color.BLACK, Color.LIGHTBLUE)
        self.replace_text_rect = self.replace_text.get_rect()
        self.replace_text_rect.center = (0.5*self.excess_width + self.PIXEL_PER_TILE * 3.5,
                                        0.5*self.excess_height + self.graphics_properties.height_pixel + self.PIXEL_PER_TILE *1/3)

        self.a3_text_rect = self.action_text2.get_rect()
        self.a3_text_rect.center = (0.5 * self.excess_width + self.PIXEL_PER_TILE * 3.5,
                                    0.5 * self.excess_height + self.graphics_properties.height_pixel + self.PIXEL_PER_TILE * 2/3)

        self.replace_button = Button((-0.5*self.excess_width +self.PIXEL_PER_TILE*3, -0.5*self.excess_height+self.graphics_properties.height_pixel), text='replace')

        self.forbid_text = self.small_font.render('Forbid', True, Color.BLACK, Color.WHITE)
        self.forbid_text_g = self.small_font.render('Forbid', True, Color.BLACK, Color.LIGHTBLUE)
        self.forbid_text_rect = self.forbid_text.get_rect()
        self.forbid_text_rect.center = (0.5*self.excess_width + self.PIXEL_PER_TILE * 4.5,
                                         0.5*self.excess_height + self.graphics_properties.height_pixel + self.PIXEL_PER_TILE * 1/3)
        self.a4_text_rect = self.action_text2.get_rect()
        self.a4_text_rect.center = (0.5 * self.excess_width + self.PIXEL_PER_TILE * 4.5,
                                    0.5 * self.excess_height + self.graphics_properties.height_pixel + self.PIXEL_PER_TILE * 2 / 3)

        self.forbid_button = Button((-0.5*self.excess_width +self.PIXEL_PER_TILE * 4, -0.5*self.excess_height+self.graphics_properties.height_pixel),text='forbid')

        self.shields_buttons = [self.insert_button, self.inserta_button, self.replace_button, self.forbid_button]
        self.shields_texts_w = [self.insert_text, self.before_text, self.inserta_text, self.after_text, self.replace_text,
                              self.forbid_text, self.action_text, self.action_text, self.action_text2, self.action_text2]
        self.shields_texts_g = [self.insert_text_g, self.before_text_g, self.inserta_text_g, self.after_text_g, self.replace_text_g,
                              self.forbid_text_g, self.action_text_g, self.action_text_g, self.action_text2_g,
                              self.action_text2_g]
        self.shields_texts = self.shields_texts_w
        self.shields_texts_rects = [self.insert_text_rect, self.before_text_rect, self.inserta_text_rect,
                                    self.after_text_rect, self.replace_text_rect, self.forbid_text_rect,
                                    self.a1_text_rect, self.a2_text_rect, self.a3_text_rect, self.a4_text_rect]

        """
        #the button to not make a shield
        self.abort_text = self.font.render('Abort', True, Color.BLACK, Color.WHITE)
        self.abort_text_rect = self.abort_text.get_rect()
        self.abort_text_rect.center = (self.excess_width*2 + self.graphics_properties.width_pixel + self.PIXEL_PER_TILE*1.5,
                                         self.PIXEL_PER_TILE / 2)
        self.abort_button = Button((self.excess_width+ self.graphics_properties.width_pixel +self.PIXEL_PER_TILE, -self.excess_height),text='Abort')
        """

        #possibly add confirm
        self.confirm_text = self.small_font.render('Confirm', True, Color.BLACK, Color.LIGHTBLUE)
        self.confirm_rect = self.confirm_text.get_rect()
        self.confirm_rect.center = (
        self.excess_width * 0.5 + self.PIXEL_PER_TILE * 0.5,
        self.excess_height + self.graphics_properties.height_pixel + 2.5*self.PIXEL_PER_TILE)
        self.confirm_button = Button((-0.5*self.excess_width, self.graphics_properties.height_pixel+ 2*self.PIXEL_PER_TILE),text='Confirm')

        self.floor = []
        for i in range(2):
            button = Button((self.excess_width*0.5+self.graphics_properties.width_pixel*0.5 +self.PIXEL_PER_TILE*(i+1), self.PIXEL_PER_TILE),
                            background_color=Color.FLOOR, color=Color.FLOOR )
            self.floor.append(button)

        self.misc_buttons = []
        #self.misc_texts = [self.abort_text]
        #self.misc_texts_rect = [self.abort_text_rect]
        """
        self.all_action_buttons = []
        for i in range(2):
            button = Button((self.excess_width*(-0.5), i*self.PIXEL_PER_TILE), image= HIGH_LEVEL_ACTION_IMAGE_MAP[i])
            self.all_action_buttons.append(button)
        for i in range(4,6):
            button = Button((0, (i-4) * self.PIXEL_PER_TILE), image=HIGH_LEVEL_ACTION_IMAGE_MAP[i])
            self.all_action_buttons.append(button)
        for i in range(9,11):
            button = Button((self.graphics_properties.width_pixel, (i-9) * self.PIXEL_PER_TILE), image=HIGH_LEVEL_ACTION_IMAGE_MAP[i])
            self.all_action_buttons.append(button)

        for i in range(13,15):
            button = Button((self.graphics_properties.width_pixel, (i-11) * self.PIXEL_PER_TILE), image=HIGH_LEVEL_ACTION_IMAGE_MAP[i])
            self.all_action_buttons.append(button)

        for i in range(18,20):
            button = Button((self.graphics_properties.width_pixel+self.excess_width*0.5, (i-18) * self.PIXEL_PER_TILE), image=HIGH_LEVEL_ACTION_IMAGE_MAP[i])
            self.all_action_buttons.append(button)
        """

    def on_init(self):
        if self.display:
            self.screen = pygame.display.set_mode((self.infoObject.current_w,
                                                   self.infoObject.current_h))
        else:
            # Create a hidden surface
            self.screen = pygame.Surface((self.graphics_properties.width_pixel, self.graphics_properties.height_pixel))
        self.screen = self.screen
        return True

    def on_render(self):

        self.screen.fill(Color.WHITE)
        self.draw_static_objects()
        for button in self.floor:
            self.draw_button(button)
        self.draw_agents()
        self.change_button_color_to_highlight()
        self.draw_seperation()
        #self.draw_misc_buttons()
        self.draw_button(self.agent_button)
        self.draw_dynamic_objects()
        self.draw_shield_buttons()
        self.draw_action_buttons()
        self.draw_action_selection_buttons()

        if self.confirm_mode:
            self.draw_button(self.confirm_button)
            self.screen.blit(self.confirm_text, self.confirm_rect)
        self.highlight_clickable_buttons()
        if self.display:
            pygame.display.flip()
            pygame.display.update()

    def draw_square(self):
        pass

    def draw_static_objects(self):
        objects = self.env.unwrapped.world.get_object_list()
        static_objects = [obj for obj in objects if isinstance(obj, StaticObject)]
        for static_object in static_objects:
            self.draw_static_object(static_object)

    def draw_static_object(self, static_object: StaticObject):
        sl = self.scaled_location(static_object.location)
        fill = pygame.Rect(sl[0] + self.PIXEL_PER_TILE*2 + self.excess_width*1.5, sl[1] + self.excess_height, self.graphics_properties.pixel_per_tile,
                           self.graphics_properties.pixel_per_tile)
        if isinstance(static_object, Counter):
            pygame.draw.rect(self.screen, Color.COUNTER, fill)
            pygame.draw.rect(self.screen, Color.COUNTER_BORDER, fill, 1)
        elif isinstance(static_object, DeliverSquare):
            pygame.draw.rect(self.screen, Color.DELIVERY, fill)
            self.draw(static_object.file_name(), self.graphics_properties.tile_size, sl)
        elif isinstance(static_object, CutBoard):
            pygame.draw.rect(self.screen, Color.COUNTER, fill)
            pygame.draw.rect(self.screen, Color.COUNTER_BORDER, fill, 1)
            self.draw(static_object.file_name(), self.graphics_properties.tile_size, sl)
        elif isinstance(static_object, Blender):
            pygame.draw.rect(self.screen, Color.COUNTER, fill)
            pygame.draw.rect(self.screen, Color.COUNTER_BORDER, fill, 1)
            self.draw(static_object.file_name(), self.graphics_properties.tile_size, sl)
        # elif isinstance(static_object, Floor):
        #     pygame.draw.rect(self.screen, Color.FLOOR, fill)

    def draw_dynamic_objects(self):
        objects = self.env.unwrapped.world.get_object_list()
        dynamic_objects = [obj for obj in objects if isinstance(obj, DynamicObject)]
        dynamic_objects_grouped = defaultdict(list)
        for obj in dynamic_objects:
            dynamic_objects_grouped[obj.location].append(obj)
        for location, obj_list in dynamic_objects_grouped.items():
            if any([agent.location == location for agent in self.env.unwrapped.world.agents]):
                self.draw_dynamic_object_stack(obj_list, self.graphics_properties.holding_size,
                                               self.holding_location(location),
                                               self.graphics_properties.holding_container_size,
                                               self.holding_container_location(location))
            else:
                self.draw_dynamic_object_stack(obj_list, self.graphics_properties.tile_size,
                                               self.scaled_location(location),
                                               self.graphics_properties.container_size,
                                               self.container_location(location))

    def draw_dynamic_object_stack(self, dynamic_objects, base_size, base_location, holding_size, holding_location):
        highest_order_object = self.env.unwrapped.world.get_highest_order_object(dynamic_objects)
        if isinstance(highest_order_object, Container):
            self.draw(highest_order_object.file_name(), base_size, base_location)
            rest_stack = [obj for obj in dynamic_objects if obj != highest_order_object]
            if rest_stack:
                self.draw_food_stack(rest_stack, holding_size, holding_location)
        else:
            self.draw_food_stack(dynamic_objects, base_size, base_location)

    def draw_agents(self):
        for agent in self.env.unwrapped.world.agents:
            self.draw('agent-{}'.format(agent.color), self.graphics_properties.tile_size,
                      self.scaled_location(agent.location))
            if agent.orientation == 1:
                file_name = "arrow_left"
                location = self.scaled_location(agent.location)
                location = (location[0], location[1] + self.graphics_properties.tile_size[1] // 4)
                size = (self.graphics_properties.tile_size[0] // 4, self.graphics_properties.tile_size[1] // 4)
            elif agent.orientation == 2:
                file_name = "arrow_right"
                location = self.scaled_location(agent.location)
                location = (location[0] + 3 * self.graphics_properties.tile_size[0] // 4,
                            location[1] + self.graphics_properties.tile_size[1] // 4)
                size = (self.graphics_properties.tile_size[0] // 4, self.graphics_properties.tile_size[1] // 4)
            elif agent.orientation == 3:
                file_name = "arrow_down"
                location = self.scaled_location(agent.location)
                location = (location[0] + self.graphics_properties.tile_size[0] // 4,
                            location[1] + 3 * self.graphics_properties.tile_size[1] // 4)
                size = (self.graphics_properties.tile_size[0] // 4, self.graphics_properties.tile_size[1] // 4)
            elif agent.orientation == 4:
                file_name = "arrow_up"
                location = self.scaled_location(agent.location)
                location = (location[0] + self.graphics_properties.tile_size[0] // 4, location[1])
                size = (self.graphics_properties.tile_size[0] // 4, self.graphics_properties.tile_size[1] // 4)
            else:
                raise ValueError(f"Agent orientation invalid ({agent.orientation})")
            self.draw(file_name, size, location)

    def draw(self, path, size, location):
        image_path = f'{self.root_dir}/{self.graphics_dir}/{path}.png'
        image = pygame.transform.scale(get_image(image_path), (int(size[0]), int(size[1])))
        location = tuple((location[0] + self.PIXEL_PER_TILE*2 + self.excess_width*1.5,
                          location[1]+ self.excess_height))
        self.screen.blit(image, location)

    def draw_food_stack(self, dynamic_objects, base_size, base_loc):
        tiles = int(math.floor(math.sqrt(len(dynamic_objects) - 1)) + 1)
        size = (base_size[0] // tiles, base_size[1] // tiles)
        for idx, obj in enumerate(dynamic_objects):
            location = (base_loc[0] + size[0] * (idx % tiles), base_loc[1] + size[1] * (idx // tiles))
            self.draw(obj.file_name(), size, location)


    def scaled_location(self, loc):
        """Return top-left corner of scaled location given coordinates loc, e.g. (3, 4)"""
        return tuple(self.graphics_properties.pixel_per_tile * np.asarray(loc))

    def holding_location(self, loc):
        """Return top-left corner of location where agent holding will be drawn (bottom right corner)
        given coordinates loc, e.g. (3, 4)"""
        scaled_loc = self.scaled_location(loc)
        return tuple((np.asarray(scaled_loc) + self.graphics_properties.pixel_per_tile *
                      (1 - self.HOLDING_SCALE)).astype(int))

    def container_location(self, loc):
        """Return top-left corner of location where contained (i.e. plated) object will be drawn,
        given coordinates loc, e.g. (3, 4)"""
        scaled_loc = self.scaled_location(loc)
        return tuple((np.asarray(scaled_loc) + self.graphics_properties.pixel_per_tile *
                      (1 - self.CONTAINER_SCALE) / 2).astype(int))
    
    def holding_container_location(self, loc):
        """Return top-left corner of location where contained, held object will be drawn
        given coordinates loc, e.g. (3, 4)"""
        scaled_loc = self.scaled_location(loc)
        factor = (1 - self.HOLDING_SCALE) + (1 - self.CONTAINER_SCALE) / 2 * self.HOLDING_SCALE
        return tuple((np.asarray(scaled_loc) + self.graphics_properties.pixel_per_tile * factor).astype(int))

    def get_image_obs(self):
        self.on_render()
        img_int = pygame.PixelArray(self.screen)
        img_rgb = np.zeros([img_int.shape[1], img_int.shape[0], 3], dtype=np.uint8)
        for i in range(img_int.shape[0]):
            for j in range(img_int.shape[1]):
                color = pygame.Color(img_int[i][j])
                img_rgb[j, i, 0] = color.g
                img_rgb[j, i, 1] = color.b
                img_rgb[j, i, 2] = color.r
        return img_rgb

    def save_image_obs(self, t):
        game_record_dir = 'misc/game/record/example/'
        self.on_render()
        pygame.image.save(self.screen, '{}/t={:03d}.png'.format(game_record_dir, t))

    def draw_shield_buttons(self):
        for button in self.shields_buttons:
            self.draw_button(button)
        for i in range(len(self.shields_texts)):
            self.screen.blit(self.shields_texts[i], self.shields_texts_rects[i])
        if self.shield_selection_mode:
            message = self.font.render('2: Choose how to avoid the action which lead to the failure:', True,
                                                   Color.BLACK, Color.WHITE)
            message_rect = message.get_rect()
            message_rect.center = (0.5 * self.excess_width + self.PIXEL_PER_TILE * 3,
                                               0.5 * self.excess_height + self.graphics_properties.height_pixel - self.PIXEL_PER_TILE * 0.5)
            self.screen.blit(message, message_rect)
        else:
            message = self.font.render('These are the possible ways to avoid the failure:', True,
                                       Color.BLACK, Color.WHITE)
            message_rect = message.get_rect()
            message_rect.center = (0.5 * self.excess_width + self.PIXEL_PER_TILE * 3,
                                   0.5 * self.excess_height + self.graphics_properties.height_pixel - self.PIXEL_PER_TILE * 0.5)
            self.screen.blit(message, message_rect)


    def draw_misc_buttons(self):
        for button in self.misc_buttons:
            self.draw_button(button)
        for i in range(len(self.misc_texts)):
            self.screen.blit(self.misc_texts[i], self.misc_texts_rect[i])

    def draw_action_buttons(self):
        for button in self.action_buttons:
            self.draw_button(button)
        self.screen.blit(self.describe_text1, self.describe_text_rect1)

    def draw_action_selection_buttons(self):
        for button in self.action_selection_buttons:
            self.draw_button(button)
        text = None
        for button in self.shields_buttons:
            if button.is_clicked:
                text = button.text
        for button in self.action_buttons:
            if button.is_clicked:
                text2 = button.text
        if self.action_selection_mode:
            if text != "forbid":
                message_text= self.font.render("3: Select an action to " + text + " " + text2 + " to avoid the failure:", True, Color.BLACK, Color.WHITE)
            else:
                message_text = self.font.render(
                    "3: Select an action to " + text + " to avoid the failure:", True, Color.BLACK,
                    Color.WHITE)
            message_text_rect= message_text.get_rect()
            message_text_rect.center = (0.5 * self.excess_width + self.PIXEL_PER_TILE * 3,
                                               0.5 * self.excess_height + self.graphics_properties.height_pixel + self.PIXEL_PER_TILE * 1.5)
            self.screen.blit(message_text, message_text_rect)
        else:
            message_text = self.font.render("These actions are available: ", True, Color.BLACK,
                                            Color.WHITE)
            message_text_rect = message_text.get_rect()
            message_text_rect.center = (0.5 * self.excess_width + self.PIXEL_PER_TILE * 3,
                                        0.5 * self.excess_height + self.graphics_properties.height_pixel + self.PIXEL_PER_TILE * 1.5)
            self.screen.blit(message_text, message_text_rect)

    def draw_button(self, button:Button):
        fill = pygame.Rect(button.loc[0] +self.excess_width, button.loc[1]+self.excess_height, self.graphics_properties.pixel_per_tile,
                           self.graphics_properties.pixel_per_tile)
        pygame.draw.rect(self.screen, button.background_color, fill)
        pygame.draw.rect(self.screen, button.color, fill, 1)
        if button.image is not None:
            image_path = f'{self.root_dir}/{self.graphics_dir}/{button.image}.png'
            image = pygame.transform.scale(get_image(image_path), (self.PIXEL_PER_TILE-2, self.PIXEL_PER_TILE-2))
            location = tuple((button.loc[0]+1 +  self.excess_width,
                              button.loc[1] +1+ self.excess_height))
            self.screen.blit(image, location)

    def process_button_click(self, button:Button):
        if button in self.shields_buttons:
            for shield_button in self.shields_buttons:
                if shield_button.is_clicked and shield_button is not(button):
                    shield_button.is_clicked = False
                    shield_button.color = Color.BLACK
        if button in self.action_buttons:
            for action_button in self.action_buttons:
                if action_button.is_clicked and action_button is not(button):
                    action_button.is_clicked = False
                    action_button.color = Color.BLACK
        if button in self.action_selection_buttons:
            for action_button in self.action_selection_buttons:
                if action_button.is_clicked and action_button is not(button):
                    action_button.is_clicked = False
                    action_button.color = Color.BLACK
        button.on_click()
        self.shield_selection_mode = self.failure_clicked()
        if not self.shield_selection_mode:
            for button in self.shields_buttons + self.action_selection_buttons:
                button.is_clicked = False
                button.color = Color.BLACK
        self.action_selection_mode = self.failure_and_shield_clicked()
        if not self.action_selection_mode:
            self.action_selection_buttons = self.action_selection_buttons_gray
            for button in self.action_selection_buttons:
                button.is_clicked = False
                button.color = Color.BLACK
        else:
            self.action_selection_buttons = self.action_selection_buttons_normal
        self.confirm_mode = self.all_clicked()
        self.on_render()

    def isValidButton(self, x, y):
        #descale x and y
        dx = (x - self.excess_width)
        dy = (y - self.excess_height)
        for button in self.misc_buttons:
            x = button.loc[0]
            y = button.loc[1]
            if (x < dx < x + self.graphics_properties.pixel_per_tile and y < dy < y + self.graphics_properties.pixel_per_tile):
                return True
        if self.confirm_mode:
            x = self.confirm_button.loc[0]
            y = self.confirm_button.loc[1]
            if (x < dx < x + self.graphics_properties.pixel_per_tile and y < dy < y + self.graphics_properties.pixel_per_tile):
                return True
        if self.action_selection_mode:
            for button in (self.action_buttons + self.shields_buttons + self.action_selection_buttons):
                x = button.loc[0]
                y = button.loc[1]
                if (x<dx<x+self.graphics_properties.pixel_per_tile and y<dy<y+self.graphics_properties.pixel_per_tile):
                    return True
        if self.shield_selection_mode:
            for button in (self.action_buttons + self.shields_buttons):
                x = button.loc[0]
                y = button.loc[1]
                if (x<dx<x+self.graphics_properties.pixel_per_tile and y<dy<y+self.graphics_properties.pixel_per_tile):
                    return True
        if (not self.shield_selection_mode) and (not self.action_selection_mode):
            for button in (self.action_buttons):
                x = button.loc[0]
                y = button.loc[1]
                if (x<dx<x+self.graphics_properties.pixel_per_tile and y<dy<y+self.graphics_properties.pixel_per_tile):
                    return True
        return False



    #return the button at the given location
    def getButton(self, x, y):
        #descale x and y
        dx = (x-self.excess_width)
        dy = (y-self.excess_height)
        for button in self.misc_buttons:
            x = button.loc[0]
            y = button.loc[1]
            if (x < dx < x + self.graphics_properties.pixel_per_tile and y < dy < y + self.graphics_properties.pixel_per_tile):
                return button
        if self.confirm_mode:
            x = self.confirm_button.loc[0]
            y = self.confirm_button.loc[1]
            if (x < dx < x + self.graphics_properties.pixel_per_tile and y < dy < y + self.graphics_properties.pixel_per_tile):
                return self.confirm_button
        if self.action_selection_mode:
            for button in (self.action_buttons + self.shields_buttons + self.action_selection_buttons):
                x = button.loc[0]
                y = button.loc[1]
                if (x<dx<x+ self.graphics_properties.pixel_per_tile and y<dy<y+ self.graphics_properties.pixel_per_tile):
                    return button
        if self.shield_selection_mode:
            for button in (self.action_buttons + self.shields_buttons):
                x = button.loc[0]
                y = button.loc[1]
                if (x<dx<x+ self.graphics_properties.pixel_per_tile and y<dy<y+ self.graphics_properties.pixel_per_tile):
                    return button
        if (not self.shield_selection_mode) and (not self.action_selection_mode):
            for button in (self.action_buttons):
                x = button.loc[0]
                y = button.loc[1]
                if (x<dx<x+ self.graphics_properties.pixel_per_tile and y<dy<y+ self.graphics_properties.pixel_per_tile):
                    return button

    def all_clicked(self):
        shield_action = None
        shield = None
        action = None
        for shield_action_button in self.action_selection_buttons:
            if shield_action_button.is_clicked:
                shield_action = shield_action_button
        for shield_button in self.shields_buttons:
            if shield_button.is_clicked:
                shield = shield_button
        for action_button in self.action_buttons:
            if action_button.is_clicked:
                action = action_button
        if shield is not None and action is not None and shield_action is not None:
            return True
        return False

    # return False

    def failure_and_shield_clicked(self):
        #if not self.action_selection_mode:
            shield = None
            action = None
            for shield_button in self.shields_buttons:
                if shield_button.is_clicked:
                    shield = shield_button
            for action_button in self.action_buttons:
                if action_button.is_clicked:
                    action = action_button
            if shield is not None and action is not None:
                return True
            return False
        #return False

    def failure_clicked(self):
        action = None
        for action_button in self.action_buttons:
            if action_button.is_clicked:
                action = action_button
        if action is not None:
            return True
        return False

    def change_button_color_to_highlight(self):
        for button in self.shields_buttons:
            if not self.shield_selection_mode:
                button.color = Color.BLACK
                button.background_color = Color.WHITE
                self.shields_texts = self.shields_texts_w
            else:
                button.color = Color.BLACK
                button.background_color = Color.LIGHTBLUE
                if button.is_clicked:
                    button.color = Color.GREEN
                self.shields_texts = self.shields_texts_g
        for button in self.action_selection_buttons:
            if not self.action_selection_mode:
                button.color = Color.BLACK
                button.background_color = Color.WHITE
            else:
                button.color = Color.BLACK
                button.background_color = Color.LIGHTBLUE
                if button.is_clicked:
                    button.color = Color.GREEN

    def highlight_clickable_buttons(self):
        if not self.shield_selection_mode:
            s = pygame.Surface((self.PIXEL_PER_TILE * len(self.shields_buttons), self.PIXEL_PER_TILE))
            s.set_alpha(128)
            s.fill(Color.GREY)
            self.screen.blit(s, (self.insert_button.loc[0]+self.excess_width,self.insert_button.loc[1]+self.excess_height))
        if not self.action_selection_mode:
            s = pygame.Surface((self.PIXEL_PER_TILE * len(self.action_selection_buttons), self.PIXEL_PER_TILE))
            s.set_alpha(128)
            s.fill(Color.GREY)
            self.screen.blit(s, (
            self.action_selection_buttons[0].loc[0]+ self.excess_width, self.action_selection_buttons[0].loc[1] + self.excess_height))

    def draw_seperation(self):
        fill = pygame.Rect(self.excess_width * 1.5 + self.PIXEL_PER_TILE, 0,
                           1,
                           self.excess_height * 2 + self.graphics_properties.height_pixel + self.PIXEL_PER_TILE)
        pygame.draw.rect(self.screen, Color.BLACK, fill, 1)








