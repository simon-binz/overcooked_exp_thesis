class Replay_agent():
    def __init__(self, action_history):
        self.action_history = action_history
        self.current_hla = None
        self.action_stack = []

    def get_action(self, obs):
        if len(self.action_stack) == 0:
            actions = self.action_history.pop(0)
            self.current_hla = actions[0]
            self.action_stack = actions[1]
            if self.action_stack == 'failure':
                return self.action_stack
        return self.action_stack.pop(0)