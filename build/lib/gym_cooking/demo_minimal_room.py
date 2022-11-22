from gym_cooking.environment.game.game import Game
from time import sleep
from gym_cooking.environment import cooking_zoo

n_agents = 1
num_humans = 0
max_steps = 100
render = False

level = 'level_evaluation_4x4_open'
seed = 1
record = False
max_num_timesteps = 1000
recipes = ["TomatoLettuceSalad", 'TomatoLettuceSalad']

parallel_env = cooking_zoo.parallel_env(level=level, num_agents=n_agents, record=record,
                                        max_steps=max_num_timesteps, recipes=recipes, obs_spaces=['numeric'])

action_spaces = parallel_env.action_spaces
player_2_action_space = action_spaces["player_0"]


class CookingAgent:

    def __init__(self, action_space):
        self.action_space = action_space

    def get_action(self, observation) -> int:
        return self.action_space.sample()


# for playing by yourself with one agent:
cooking_agent = CookingAgent(player_2_action_space)
game = Game(parallel_env, num_humans, [cooking_agent], max_steps, render=True)
store = game.on_execute_ai_only_with_delay()
done = False

while not done:
    action = player_2_action_space.sample()
    observation, reward, done, info = parallel_env.step(action)

# for letting only the agents play:
# num_humans = 0
# cooking_agents = [CookingAgent(player_2_action_space), DeterministicProgrammedAgent(player_2_action_space)]
# game = Game(parallel_env, num_humans, cooking_agents, max_steps, render=True)
# store = game.on_execute_ai_only_with_delay()

print("done")
