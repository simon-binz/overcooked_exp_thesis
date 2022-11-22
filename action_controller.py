from action_mapping import ActionMapping, Actions, MAP_OBJECT_TO_NUMBER


class ActionController:

    def __init__(self):
        self.action_mapping = ActionMapping()
        self.current_high_level_action = 0

    def get_high_level_action(self, env_state, agent_pos, belief_subgoals=None, recipe=None):
        self.action_mapping.motion_generator.reset(env_state, agent_pos)

        # for now pre programmed actions
        if self.action_mapping.check_object_is_reachable(env_state, MAP_OBJECT_TO_NUMBER['DeliverSquare'],
                                                         agent_pos) and \
                self.action_mapping.check_any_object_on_a_plate(env_state, MAP_OBJECT_TO_NUMBER['ChoppedTomato']) and \
                self.action_mapping.check_any_object_on_a_plate(env_state, MAP_OBJECT_TO_NUMBER['ChoppedLettuce']) and \
                self.action_mapping.check_object_is_reachable(env_state, MAP_OBJECT_TO_NUMBER['Plate'], agent_pos):
            self.current_high_level_action = Actions.DELIVER
        elif self.action_mapping.check_object_is_reachable(env_state, MAP_OBJECT_TO_NUMBER['ChoppedTomato'],
                                                           agent_pos) and \
                self.action_mapping.check_object_is_reachable(env_state, MAP_OBJECT_TO_NUMBER['ChoppedLettuce'],
                                                              agent_pos) and \
                self.action_mapping.check_object_is_reachable(env_state, MAP_OBJECT_TO_NUMBER['Plate'], agent_pos):
            self.current_high_level_action = Actions.TOMATO_LETTUCE_PLATE
        elif self.action_mapping.check_object_is_reachable(env_state, MAP_OBJECT_TO_NUMBER['ChoppedTomato'],
                                                           agent_pos) and \
                self.action_mapping.check_object_is_reachable(env_state, MAP_OBJECT_TO_NUMBER['Plate'], agent_pos):
            self.current_high_level_action = Actions.TOMATO_PLATE
        elif self.action_mapping.check_object_is_reachable(env_state, MAP_OBJECT_TO_NUMBER['ChoppedLettuce'],
                                                           agent_pos) and \
                self.action_mapping.check_object_is_reachable(env_state, MAP_OBJECT_TO_NUMBER['Plate'], agent_pos):
            self.current_high_level_action = Actions.LETTUCE_PLATE
        elif self.action_mapping.check_object_is_reachable(env_state, MAP_OBJECT_TO_NUMBER['ChoppedTomato'],
                                                           agent_pos) and \
                not self.action_mapping.check_object_is_on_mutually_available_field(env_state,
                                                                                    env_state[:, :,
                                                                                    MAP_OBJECT_TO_NUMBER[
                                                                                        'ChoppedTomato']]):
            self.current_high_level_action = Actions.GIVE_CHOPPED_TOMATO
        elif (self.action_mapping.check_object_is_reachable(env_state, MAP_OBJECT_TO_NUMBER['Tomato'], agent_pos) and
              self.action_mapping.check_object_is_reachable(env_state, MAP_OBJECT_TO_NUMBER['CutBoard'], agent_pos)) or \
                (self.action_mapping.check_objects_at_same_position(env_state[:, :, MAP_OBJECT_TO_NUMBER['Tomato']],
                                                                    env_state[:, :,
                                                                    MAP_OBJECT_TO_NUMBER['CutBoard']]) and
                 self.action_mapping.check_object_is_reachable(env_state, MAP_OBJECT_TO_NUMBER['Tomato'], agent_pos)):
            # TODO agents "prefer" to chop tomatoes since they appear first in this if-clause here
            self.current_high_level_action = Actions.CHOP_TOMATO
        elif self.action_mapping.check_object_is_reachable(env_state, MAP_OBJECT_TO_NUMBER['ChoppedLettuce'],
                                                           agent_pos) and \
                not self.action_mapping.check_object_is_on_mutually_available_field(env_state, env_state[:, :,
                                                                                               MAP_OBJECT_TO_NUMBER[
                                                                                                   'ChoppedLettuce']]):
            self.current_high_level_action = Actions.GIVE_CHOPPED_LETTUCE
        elif (self.action_mapping.check_object_is_reachable(env_state, MAP_OBJECT_TO_NUMBER['Lettuce'], agent_pos) and
              self.action_mapping.check_object_is_reachable(env_state, MAP_OBJECT_TO_NUMBER['CutBoard'], agent_pos)) or \
                (self.action_mapping.check_objects_at_same_position(env_state[:, :, MAP_OBJECT_TO_NUMBER['Lettuce']],
                                                                    env_state[:, :,
                                                                    MAP_OBJECT_TO_NUMBER['CutBoard']]) and
                 self.action_mapping.check_object_is_reachable(env_state, MAP_OBJECT_TO_NUMBER['Lettuce'], agent_pos)):
            self.current_high_level_action = Actions.CHOP_LETTUCE
        elif self.action_mapping.check_object_is_reachable(env_state, MAP_OBJECT_TO_NUMBER['Tomato'], agent_pos) and \
                not self.action_mapping.check_object_is_on_mutually_available_field(env_state,
                                                                                    env_state[:, :,
                                                                                    MAP_OBJECT_TO_NUMBER['Tomato']]):
            self.current_high_level_action = Actions.GIVE_TOMATO
        elif self.action_mapping.check_object_is_reachable(env_state, MAP_OBJECT_TO_NUMBER['Lettuce'], agent_pos) and \
                not self.action_mapping.check_object_is_on_mutually_available_field(env_state,
                                                                                    env_state[:, :,
                                                                                    MAP_OBJECT_TO_NUMBER['Lettuce']]):
            self.current_high_level_action = Actions.GIVE_LETTUCE
        elif self.action_mapping.check_object_is_reachable(env_state, MAP_OBJECT_TO_NUMBER['Plate'], agent_pos) and \
                not self.action_mapping.check_object_is_on_mutually_available_field(env_state,
                                                                                    env_state[:, :,
                                                                                    MAP_OBJECT_TO_NUMBER['Plate']]):
            self.current_high_level_action = Actions.GIVE_PLATE
        else:
            # self.current_high_level_action = Actions.DO_NOTHING
            self.current_high_level_action = Actions.AVOID

        return self.current_high_level_action

    def get_low_level_action(self, env_state, agent_pos, belief_subgoals=None, recipe=None):
        high_level_action = self.get_high_level_action(env_state, agent_pos)
        low_level_action = self.action_mapping.map_action(env_state, agent_pos, high_level_action)
        # print('Low level action:', low_level_action)
        return low_level_action
