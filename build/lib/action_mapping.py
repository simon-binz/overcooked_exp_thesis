from enum import IntEnum, Enum
import numpy as np

from gym_cooking.utils.utils import ACTION_MAP, MAP_OBJECT_TO_NUMBER, MAP_NUMBER_TO_OBJECT
from motion_generator import MotionGenerator, get_neighbors


# Reduced high level action set for learning
class Actions(IntEnum):
    CHOP_LETTUCE = 0
    CHOP_TOMATO = 1
    CHOP_ONION = 2
    CHOP_CARROT = 3

    TOMATO_PLATE = 4
    LETTUCE_PLATE = 5
    ONION_PLATE = 6
    CARROT_PLATE = 7
    TOMATO_LETTUCE_PLATE = 8

    GIVE_TOMATO = 9
    GIVE_LETTUCE = 10
    GIVE_ONION = 11
    GIVE_CARROT = 12
    GIVE_CHOPPED_TOMATO = 13
    GIVE_CHOPPED_LETTUCE = 14
    GIVE_CHOPPED_ONION = 15
    GIVE_CHOPPED_CARROT = 16
    GIVE_PLATE = 17

    DELIVER = 18

    # DO_NOTHING = 11
    AVOID = 19

ACTIONS_NUMBERS_MAP = {}
for i in range(len(Actions)):
    ACTIONS_NUMBERS_MAP[i] = Actions(i)


class ActionTypes(Enum):
    PROCESS = {Actions.CHOP_LETTUCE, Actions.CHOP_TOMATO, Actions.CHOP_ONION, Actions.CHOP_CARROT}

    PLATE = {Actions.TOMATO_PLATE, Actions.LETTUCE_PLATE, Actions.ONION_PLATE, Actions.CARROT_PLATE, Actions.TOMATO_LETTUCE_PLATE}

    DELIVER = {Actions.DELIVER}

    GIVE = {Actions.GIVE_TOMATO, Actions.GIVE_LETTUCE, Actions.GIVE_ONION, Actions.GIVE_CARROT,
            Actions.GIVE_CHOPPED_TOMATO, Actions.GIVE_CHOPPED_LETTUCE, Actions.GIVE_CHOPPED_ONION, Actions.GIVE_CHOPPED_CARROT,
            Actions.GIVE_PLATE}


PROCESS_ACTIONS = {
    Actions.CHOP_LETTUCE: 'ChoppedTomato',
    Actions.CHOP_TOMATO: 'ChoppedTomato',
    Actions.CHOP_ONION: 'ChoppedOnion',
    Actions.CHOP_CARROT: 'ChoppedCarrot'
}

MAP_ACTIONS_TO_OBJECTS = {
    # needed tool, object, processed object
    Actions.CHOP_LETTUCE: ['CutBoard', 'Lettuce', 'ChoppedLettuce'],
    Actions.CHOP_TOMATO: ['CutBoard', 'Tomato', 'ChoppedTomato'],
    Actions.CHOP_ONION: ['CutBoard', 'Onion', 'ChoppedOnion'],
    Actions.CHOP_CARROT: ['CutBoard', 'Carrot', 'ChoppedCarrot'],

    # plate, ingredients
    Actions.TOMATO_PLATE: ['Plate', 'ChoppedTomato'],
    Actions.LETTUCE_PLATE: ['Plate', 'ChoppedLettuce'],
    Actions.ONION_PLATE: ['Plate', 'ChoppedOnion'],
    Actions.CARROT_PLATE: ['Plate', 'ChoppedCarrot'],
    Actions.TOMATO_LETTUCE_PLATE: ['Plate', 'ChoppedTomato', 'ChoppedLettuce'],

    # deliverSquare, plate
    Actions.DELIVER: ['DeliverSquare', 'Plate'],

    # counter, item to give
    Actions.GIVE_TOMATO: ['Counter', 'Tomato'],
    Actions.GIVE_LETTUCE: ['Counter', 'Lettuce'],
    Actions.GIVE_ONION: ['Counter', 'Onion'],
    Actions.GIVE_CARROT: ['Counter', 'Carrot'],
    Actions.GIVE_CHOPPED_TOMATO: ['Counter', 'ChoppedTomato'],
    Actions.GIVE_CHOPPED_LETTUCE: ['Counter', 'ChoppedLettuce'],
    Actions.GIVE_CHOPPED_ONION: ['Counter', 'ChoppedOnion'],
    Actions.GIVE_CHOPPED_CARROT: ['Counter', 'ChoppedCarrot'],
    Actions.GIVE_PLATE: ['Counter', 'Plate'],

    # Actions.DO_NOTHING: [],
    Actions.AVOID: ['Agent']
}


class ActionMapping:

    def __init__(self):
        self.motion_generator = MotionGenerator()

    def map_action(self, env_state, agent_pos, high_level_action):
        """ returns a low level action """

        # all actions that process the ingredients
        if high_level_action in ActionTypes.PROCESS.value:
            #print('CHOP')
            objects = MAP_ACTIONS_TO_OBJECTS[high_level_action]

            interaction_object_plane = env_state[:, :, MAP_OBJECT_TO_NUMBER[objects[0]]]
            #print(MAP_OBJECT_TO_NUMBER[objects[0]], MAP_OBJECT_TO_NUMBER[objects[1]])
            #print(interaction_object_plane)
            object_plane_index = MAP_OBJECT_TO_NUMBER[objects[1]]
            object_plane = env_state[:, :, object_plane_index]
            processed_object_plane = env_state[:, :, MAP_OBJECT_TO_NUMBER[objects[2]]]

            # if agent is holding something, check if he is holding the desired object
            if self._check_multiple_objects_at_position(env_state, agent_pos):
                held_object = self._get_object_held_by_agent(env_state, agent_pos)
                if held_object != objects[1]:
                    interaction_object_plane = self._get_free_counters(env_state)

            # if task is not already fulfilled
            if not self.check_objects_at_same_position(processed_object_plane, interaction_object_plane):
                return self._get_next_action(env_state, agent_pos, object_plane_index, interaction_object_plane)

            else:
                return ACTION_MAP['nop']

        # all actions that put ingredients on a plate
        elif high_level_action in ActionTypes.PLATE.value:
            # print('PLATE')
            # feature planes
            objects = MAP_ACTIONS_TO_OBJECTS[high_level_action]
            interaction_plane = env_state[:, :, MAP_OBJECT_TO_NUMBER[objects[0]]]

            # if agent is holding a plate or a object that is not needed, let him place it on a free counter
            if self._check_multiple_objects_at_position(env_state, agent_pos):
                held_object = self._get_object_held_by_agent(env_state, agent_pos)
                if held_object == 'Plate' or held_object not in objects:
                    interaction_plane = self._get_free_counters(env_state)

            for i in range(1, len(objects)):
                object_plane_index = MAP_OBJECT_TO_NUMBER[objects[1]]
                object_plane = env_state[:, :, object_plane_index]
                if not self.check_objects_at_same_position(object_plane, interaction_plane):
                    return self._get_next_action(env_state, agent_pos, object_plane_index, interaction_plane)

            return ACTION_MAP['nop']

        # all actions that deliver the plate
        elif high_level_action in ActionTypes.DELIVER.value:
            # print('DELIVER')
            objects = MAP_ACTIONS_TO_OBJECTS[high_level_action]
            interaction_object_plane = env_state[:, :, MAP_OBJECT_TO_NUMBER[objects[0]]]
            object_plane_index = MAP_OBJECT_TO_NUMBER[objects[1]]
            object_plane = env_state[:, :, object_plane_index]

            # if agent is holding something, check if he is holding the desired object
            if self._check_multiple_objects_at_position(env_state, agent_pos):
                held_object = self._get_object_held_by_agent(env_state, agent_pos)
                if held_object != objects[1]:
                    interaction_object_plane = self._get_free_counters(env_state)

            return self._get_next_action(env_state, agent_pos, object_plane_index, interaction_object_plane)
        elif high_level_action in ActionTypes.GIVE.value:
            # print('GIVE')
            objects = MAP_ACTIONS_TO_OBJECTS[high_level_action]
            object_plane_index = MAP_OBJECT_TO_NUMBER[objects[1]]

            mutually_available_counters = self._get_mutual_available_counters(env_state)

            # reduce to only free counters
            mutually_available_free_counters = []
            for c in mutually_available_counters:
                if np.sum(env_state[c]) == 1:
                    mutually_available_free_counters.append(c)

            mutually_available_free_counter_plane = self._create_plane_from_positions(mutually_available_free_counters,
                                                                                      np.shape(env_state[:, :, 0]))
            return self._get_next_action(env_state, agent_pos, object_plane_index,
                                         mutually_available_free_counter_plane)
        elif high_level_action == Actions.AVOID:
            # print("AVOID")
            objects = MAP_ACTIONS_TO_OBJECTS[high_level_action]
            agent_plane = env_state[:, :, MAP_OBJECT_TO_NUMBER[objects[0]]]
            neighbors = get_neighbors(agent_pos)

            move = False
            for neighbor in neighbors:
                # check if an important object (tool or ingredient) is nearby
                if np.sum(env_state[neighbor[0], neighbor[1], 2:15]) >= 1:
                    neighbors.remove(neighbor)
                    move = True
                # check if other agent blocks neighbor field
                elif agent_plane[tuple(neighbor)] == 1:
                    neighbors.remove(neighbor)

            # move to neighbor position, first free neighbor in list
            if move:
                self.motion_generator.reset(env_state, tuple(agent_pos))
                path = self.motion_generator.shortest_path(tuple(agent_pos), neighbors[0])
                return ACTION_MAP[path[1][0]]
            else:
                return ACTION_MAP['nop']

        # elif high_level_action == Actions.DO_NOTHING:
        #     return ACTION_MAP['nop']

        # other high level actions not implemented yet
        else:
            return ACTION_MAP['nop']

    #returns a list of low level actions to perform a high level action
    def map_action_to_plan(self, env_state, agent_pos, high_level_action):
        """ returns a low level action """

        # all actions that process the ingredients
        if high_level_action in ActionTypes.PROCESS.value:
            #print('CHOP')
            objects = MAP_ACTIONS_TO_OBJECTS[high_level_action]

            interaction_object_plane = env_state[:, :, MAP_OBJECT_TO_NUMBER[objects[0]]]
            #print(MAP_OBJECT_TO_NUMBER[objects[0]], MAP_OBJECT_TO_NUMBER[objects[1]])
            #print(interaction_object_plane)
            object_plane_index = MAP_OBJECT_TO_NUMBER[objects[1]]
            object_plane = env_state[:, :, object_plane_index]
            processed_object_plane = env_state[:, :, MAP_OBJECT_TO_NUMBER[objects[2]]]

            # if agent is holding something, check if he is holding the desired object
            if self._check_multiple_objects_at_position(env_state, agent_pos):
                held_object = self._get_object_held_by_agent(env_state, agent_pos)
                if held_object != objects[1]:
                    interaction_object_plane = self._get_free_counters(env_state)

            # if task is not already fulfilled
            if not self.check_objects_at_same_position(processed_object_plane, interaction_object_plane):
                plan = self.plan_next_actions(env_state, agent_pos, object_plane_index, interaction_object_plane)

                actions = []
                for action in plan:
                    actions.append(ACTION_MAP[action])
                return actions

            else:
                return [ACTION_MAP['nop']]

        # all actions that put ingredients on a plate
        elif high_level_action in ActionTypes.PLATE.value:
            # print('PLATE')
            # feature planes
            objects = MAP_ACTIONS_TO_OBJECTS[high_level_action]
            interaction_plane = env_state[:, :, MAP_OBJECT_TO_NUMBER[objects[0]]]

            # if agent is holding a plate or a object that is not needed, let him place it on a free counter
            if self._check_multiple_objects_at_position(env_state, agent_pos):
                held_object = self._get_object_held_by_agent(env_state, agent_pos)
                if held_object == 'Plate' or held_object not in objects:
                    interaction_plane = self._get_free_counters(env_state)

            for i in range(1, len(objects)):
                object_plane_index = MAP_OBJECT_TO_NUMBER[objects[1]]
                object_plane = env_state[:, :, object_plane_index]
                if not self.check_objects_at_same_position(object_plane, interaction_plane):
                    plan = self.plan_next_actions(env_state, agent_pos, object_plane_index, interaction_plane)
                    actions = []
                    for action in plan:
                        actions.append(ACTION_MAP[action])
                    return actions

                else:
                    return [ACTION_MAP['nop']]


        # all actions that deliver the plate
        elif high_level_action in ActionTypes.DELIVER.value:
            # print('DELIVER')
            objects = MAP_ACTIONS_TO_OBJECTS[high_level_action]
            interaction_object_plane = env_state[:, :, MAP_OBJECT_TO_NUMBER[objects[0]]]
            object_plane_index = MAP_OBJECT_TO_NUMBER[objects[1]]
            object_plane = env_state[:, :, object_plane_index]

            # if agent is holding something, check if he is holding the desired object
            if self._check_multiple_objects_at_position(env_state, agent_pos):
                held_object = self._get_object_held_by_agent(env_state, agent_pos)
                if held_object != objects[1]:
                    interaction_object_plane = self._get_free_counters(env_state)

            plan = self.plan_next_actions(env_state, agent_pos, object_plane_index, interaction_object_plane)
            actions = []
            for action in plan:
                actions.append(ACTION_MAP[action])
            return actions
        elif high_level_action in ActionTypes.GIVE.value:
            # print('GIVE')
            objects = MAP_ACTIONS_TO_OBJECTS[high_level_action]
            object_plane_index = MAP_OBJECT_TO_NUMBER[objects[1]]

            mutually_available_counters = self._get_mutual_available_counters(env_state)

            # reduce to only free counters
            mutually_available_free_counters = []
            for c in mutually_available_counters:
                if np.sum(env_state[c]) == 1:
                    mutually_available_free_counters.append(c)

            mutually_available_free_counter_plane = self._create_plane_from_positions(mutually_available_free_counters,
                                                                                      np.shape(env_state[:, :, 0]))
            plan = self.plan_next_actions(env_state, agent_pos, object_plane_index, mutually_available_free_counter_plane)
            actions = []
            for action in plan:
                actions.append(ACTION_MAP[action])
            return actions
        elif high_level_action == Actions.AVOID:
            # print("AVOID")
            objects = MAP_ACTIONS_TO_OBJECTS[high_level_action]
            agent_plane = env_state[:, :, MAP_OBJECT_TO_NUMBER[objects[0]]]
            neighbors = get_neighbors(agent_pos)

            move = False
            for neighbor in neighbors:
                # check if an important object (tool or ingredient) is nearby
                if np.sum(env_state[neighbor[0], neighbor[1], 2:15]) >= 1:
                    neighbors.remove(neighbor)
                    move = True
                # check if other agent blocks neighbor field
                elif agent_plane[tuple(neighbor)] == 1:
                    neighbors.remove(neighbor)

            # move to neighbor position, first free neighbor in list
            if move:
                self.motion_generator.reset(env_state, tuple(agent_pos))
                path = self.motion_generator.shortest_path(tuple(agent_pos), neighbors[0])
                actions = []
                for action in path[1]:
                    actions.append(ACTION_MAP[action])
                return actions
            else:
                return [ACTION_MAP['nop']]

        # elif high_level_action == Actions.DO_NOTHING:
        #     return ACTION_MAP['nop']

        # other high level actions not implemented yet
        else:
            return [ACTION_MAP['nop']]

    def _get_next_action(self, env_state, agent_pos, object_plane_index, interaction_object_plane):
        object_plane = env_state[:, :, object_plane_index]
        if not self.check_object_is_reachable(env_state, object_plane_index, agent_pos):
            return ACTION_MAP['nop']

        # if the object is in the level
        goal_positions = self._get_object_positions(object_plane)

        # if agent already holds the object
        agent_holding_object = False
        agent_holding_target = False
        if self._check_multiple_objects_at_position(env_state, agent_pos):
            # find the position of the interaction object
            # TODO utilize to get agent to drop object before switching goal
            agent_holding_object = True
            if self._check_agent_at_goal_position(tuple(agent_pos), self._get_object_positions(object_plane)):
                agent_holding_target = True
            goal_positions = self._get_object_positions(interaction_object_plane)

        # generate a path
        shortest_path = (None, ['nop'])
        shortest_path_len = np.inf
        for goal in goal_positions:
            self.motion_generator.reset(env_state, tuple(agent_pos))
            path = self.motion_generator.shortest_path(tuple(agent_pos), goal)

            # path is not (None, 'nop')
            if path[0] is not None:
                new_path_length = len(path[1])
                if new_path_length < shortest_path_len:
                    shortest_path = path
                    shortest_path_len = new_path_length

            else:
                # if agent is directly next to goal
                if self._check_fields_next_to_each_other(goal, agent_pos):
                    return ACTION_MAP['interact']

        # return the first action in the path
        return ACTION_MAP[shortest_path[1][0]]

    def plan_next_actions(self, env_state, agent_pos, object_plane_index, interaction_object_plane):
        object_plane = env_state[:, :, object_plane_index]
        if not self.check_object_is_reachable(env_state, object_plane_index, agent_pos):
            return ['nop']

        # if the object is in the level
        goal_positions = self._get_object_positions(object_plane)

        # if agent already holds the object
        agent_holding_object = False
        agent_holding_target = False
        if self._check_multiple_objects_at_position(env_state, agent_pos):
            # find the position of the interaction object
            # TODO utilize to get agent to drop object before switching goal
            agent_holding_object = True
            if self._check_agent_at_goal_position(tuple(agent_pos), self._get_object_positions(object_plane)):
                agent_holding_target = True
            goal_positions = self._get_object_positions(interaction_object_plane)

        # generate a path
        shortest_path = (None, ['nop'])
        shortest_path_len = np.inf
        for goal in goal_positions:
            self.motion_generator.reset(env_state, tuple(agent_pos))
            path = self.motion_generator.shortest_path(tuple(agent_pos), goal)

            # path is not (None, 'nop')
            if path[0] is not None:
                new_path_length = len(path[1])
                if new_path_length < shortest_path_len:
                    shortest_path = path
                    shortest_path_len = new_path_length

            else:
                # if agent is directly next to goal
                if self._check_fields_next_to_each_other(goal, agent_pos):
                    return ['interact']

        # return the first action in the path
        return shortest_path[1]


    @staticmethod
    def check_objects_at_same_position(object_plane, interaction_object_plane):
        """
        checks if a given object is placed on the interaction object
        TODO: how to deal with multiple plates and multiple of the same objects?
        :param object_plane: object feature plane
        :param interaction_object_plane: interaction object feature plane
        :return: True if object is on the same position as the interaction object
        """
        plane1 = np.equal(object_plane, np.ones_like(object_plane))
        plane2 = np.equal(interaction_object_plane, np.ones_like(interaction_object_plane))
        comb = np.logical_and(plane1, plane2)
        return np.sum(comb) > 0

    @staticmethod
    def _check_multiple_objects_at_position(env_state, position):
        """Returns true if there are multiple objects at the given position, e.g. an agent holding something. False otherwise."""
        return np.sum(env_state[tuple(position)][2:16]) > 1

    def _get_object_held_by_agent(self, env_state, agent_position):
        """Returns a string describing the object held by the agent at the given position. None otherwise."""
        if not self._check_multiple_objects_at_position(env_state, agent_position):
            return None
        else:
            objects_held = []
            for i in range(4, 15):
                if env_state[tuple(agent_position)][i] == 1:
                    objects_held.append(MAP_NUMBER_TO_OBJECT[i])
            return 'Plate' if 'Plate' in objects_held else objects_held[0]

    @staticmethod
    def _check_object_on_a_plate(env_state, object_position):
        """Returns true if the object at the given object_position is on a plate"""
        return env_state[tuple(object_position)][MAP_OBJECT_TO_NUMBER['Plate']]

    @staticmethod
    def check_any_object_on_a_plate(env_state, object_plane_index):
        """Returns true if any object of the given object_plane is on a plate"""
        object_plane = env_state[:, :, object_plane_index]
        return np.sum(np.logical_and(env_state[:, :, MAP_OBJECT_TO_NUMBER['Plate']], object_plane)) > 0

    @staticmethod
    def _check_agent_at_goal_position(agent_pos, goal_positions):
        """
        checks if the agent is at one of the goal positions
        :param np.array agent_pos: current position of the array
        :param list goal_positions: list with all goal positions
        :return: True if agent is at one of the goal positions
        """
        return tuple(agent_pos) in goal_positions

    @staticmethod
    def _check_object_exists(feature_plane):
        """
        checks if an entry of the feature plane is 1 and the object therefore exists
        :param feature_plane: feature plane
        :return: True if object exists
        """
        return 1 in feature_plane

    @staticmethod
    def _get_object_positions(feature_plane):
        """
        returns all positions where the entry in the feature plane is 1
        :param feature_plane: feature plane
        :return: list with all positions
        """
        goal_positions = []
        for (x, y), value in np.ndenumerate(feature_plane):
            if value == 1:
                goal_positions.append((x, y))
        return goal_positions

    def _get_mutual_available_counters(self, env_state):
        """Returns a list of all counter positions that are reachable by at least two agents. Agents are ignored as obstacles so a path may not exist,
        even if a field is available."""
        mutual_fields = []
        counter_plane = env_state[:, :, MAP_OBJECT_TO_NUMBER['Counter']]
        # check all fields except those at the wall
        for x in range(1, counter_plane.shape[0] - 1):
            for y in range(1, counter_plane.shape[1] - 1):
                if counter_plane[x, y] == 1:
                    # check if the counter is free
                    # print(x, y)
                    # print(env_state[x, y])
                    # if np.sum(env_state[x, y]) == 1:
                    if self._check_field_is_mutually_available(env_state, (x, y)):
                        mutual_fields.append((x, y))
        return mutual_fields

    def check_object_is_on_mutually_available_field(self, env_state, object_plane):
        """Returns true if one object from the given object plane is available for at least two agents"""
        obj_positions = self._get_object_positions(object_plane)
        for obj_pos in obj_positions:
            if self._check_field_is_mutually_available(env_state, obj_pos):
                return True
        return False

    def _check_field_is_mutually_available(self, env_state, position):
        """Returns true if the given position is basically reachable for at least two agents. The Path may still be blocked by
        other agents as these are ignored!"""
        agent_positions = self._get_object_positions(env_state[:, :, MAP_OBJECT_TO_NUMBER['Agent']])
        n_agents_reaching_position = 0

        self.motion_generator.reset(env_state, self.motion_generator.agent_position, ignore_agents_as_obstacles=True)
        for agent_pos in agent_positions:
            if self._check_field_is_reachable(position, agent_pos):
                n_agents_reaching_position += 1

        return n_agents_reaching_position > 1

    @staticmethod
    def _check_fields_next_to_each_other(field_pos1, field_pos2):
        """Returns true if the two given fields are neighbors"""
        diff = np.abs(np.array(field_pos1) - np.array(field_pos2))
        return diff[0] == 0 and diff[1] == 1 or diff[0] == 1 and diff[1] == 0

    def _check_field_is_reachable(self, goal_pos, start_pos):
        """Returns true if the given goal_pos is reachable from position start_pos. Agents are seens as obstacles as well."""
        goal_pos = tuple(goal_pos)
        start_pos = tuple(start_pos)
        return self.motion_generator.shortest_path(start_pos, goal_pos)[0] is not None \
               or self._check_fields_next_to_each_other(goal_pos, start_pos) or goal_pos == start_pos

    def check_object_is_reachable(self, env_state, object_plane_index, start_pos):
        """Returns true if an object from the given object plane is reachable from the start_position.
        ATTENTION: if an object is on a plate, it is not 'reachable' in that sense, since only the plate is reachable then.
        ##In the same sense a cutboard is only reachable if it is not occupied!"""
        object_plane = env_state[:, :, object_plane_index]
        floor_plane = env_state[:, :, MAP_OBJECT_TO_NUMBER['Floor']]

        if not self._check_object_exists(object_plane):
            return False

        # if start_pos is on object plane, e.g. agent is holding the object
        if object_plane[tuple(start_pos)] == 1: #and not self._check_object_on_a_plate(env_state, start_pos):
            if object_plane_index == MAP_OBJECT_TO_NUMBER['Plate'] or not self._check_object_on_a_plate(env_state, start_pos):
                return True

        object_positions = self._get_object_positions(object_plane)
        for obj_pos in object_positions:

            # continue if object is not on a counter, e.g. other agent is holding object
            if floor_plane[tuple(obj_pos)] == 1:
                continue

            # continue if object is on a plate
            if object_plane_index != MAP_OBJECT_TO_NUMBER['Plate']:
                if self._check_object_on_a_plate(env_state, obj_pos):
                    continue

            # continue if the object is a cutboard and occupied
            if object_plane_index == MAP_OBJECT_TO_NUMBER['CutBoard']:
                if self._check_multiple_objects_at_position(env_state, obj_pos):
                    continue


            # return true if object is not on a plate and reachable
            if self._check_field_is_reachable(obj_pos, start_pos):
                return True
        return False

    @staticmethod
    def _create_plane_from_positions(positions, plane_shape):
        """Returns a (feature) plane of the given shape, with 1s on all of the given positions."""
        plane = np.zeros(plane_shape)
        for pos in positions:
            plane[pos] = 1
        return plane

    @staticmethod
    def _get_free_counters(env_state):
        """Returns a feature map representing all free counters."""
        counter_plane = env_state[:, :, MAP_OBJECT_TO_NUMBER['Counter']]
        occupied_fields = np.sum(env_state[:, :, 2:], axis=2)
        occupied_fields = np.clip(occupied_fields, 0, 1)
        free_counters = np.logical_and(counter_plane, np.logical_not(occupied_fields))
        return free_counters

