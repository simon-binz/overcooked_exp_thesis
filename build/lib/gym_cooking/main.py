import pygame.display

from experiments.expirement1.experiment1_interactive import exp as tutorial
from experiments.experiment2.experiment2_interactive import exp as exp1
from experiments.experiment4.experiment4_interactive import exp as exp2
from experiments.experiment5.experiment5_interactive import exp as exp3
import startscreen as code_entry
import forced_choice as forced_choice
import open_question as open_question
import environment.game.messenger as msger

from gym_cooking.environment import cooking_zoo

level = 'tomato_salad'
record = False
max_steps = 100
recipes = ["TomatoLettuceSalad","TomatoLettuceSalad"]
base_env = cooking_zoo.parallel_env(level=level, num_agents=1, record=record,
                                      max_steps=max_steps, recipes=recipes, obs_spaces=['numeric'])


user_code = code_entry.Startscreen().make_screen()
msger.Game(base_env, "We start with the tutorial.")
shield_tutorial = tutorial()
msger.Game(base_env, "That was the the tutorial. If you have any questions, feel free to ask them.")
shield_exp1 = exp1()
shield_exp2 = exp2()
msger.Game(base_env, "That was the first part of the experiment. Please read the second part of the instructions now.")
#ToDo: Test this on linux
with open("results/"+ user_code + '.txt', "w") as text_file:
    text_file.write("Shield tutorial: Failure: " + shield_tutorial["Failure"] + " Shield type: " +
                    shield_tutorial["Shield type"]+ " Shield action: " + shield_tutorial["Shield action"] + "\n")
    text_file.write("Shield exp1: Failure: " + shield_exp1["Failure"] + " Shield type: " +
                    shield_exp1["Shield type"] + " Shield action: " + shield_exp1["Shield action"] + "\n")
    text_file.write("Shield exp2: Failure: " + shield_exp2["Failure"] + " Shield type: " +
                    shield_exp2["Shield type"] + " Shield action: " + shield_exp2["Shield action"] + "\n")
pygame.display.quit()
