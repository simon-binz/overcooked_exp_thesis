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
shield_exp3 = exp3()
open_answer = open_question.Startscreen().make_screen()
forced_choice_answer = forced_choice.make_query()
msger.Game(base_env, "Thank you for your participation")
with open("results/"+ user_code + '.txt', "a") as text_file:
    text_file.write("Shield exp3: Failure: " + shield_exp3["Failure"] + " Shield type: " +
                    shield_exp3["Shield type"] + " Shield action: " + shield_exp3["Shield action"] + "\n")
    text_file.write("Open answer: " + open_answer + "\n")
    text_file.write("Forced choice answer: " + forced_choice_answer[0] + " and " + forced_choice_answer[1])
pygame.display.quit()