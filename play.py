import os
import subprocess
import sys
import traceback

import click
import torch

from my_agent import SequenceAgent
from sequence_game.sequence_evironment import SequenceEnvironment

agent = SequenceAgent(2, train_mode=False)
env = SequenceEnvironment(2, -1000, 5, 1000)
env.reset()
torch.set_default_dtype(torch.float64)

directory = env.RENDER_DIR

if os.path.exists(directory):
    subprocess.call(['open', directory])
else:
    print('Directory does not exist')


@click.command()
@click.option('--player', type=click.Choice(['1', '2']), prompt='Choose player 1 or player 2')
def play(player):
    """
    Prompt the user to choose between being Player 1 or Player 2.
    """
    if player == '1':
        model_number = 1
        click.echo('You have chosen to be Player 1.')
    else:
        model_number = 0
        click.echo('You have chosen to be Player 2.')

    agent.read_model_from_disk(model_number)

    # initialize game variables
    done = False
    state = env.reset()

    while not done:
        # get user input
        try:
            action = agent.select_action(state)
            state, reward, done, info = env.step(action)
            env.render()
            y, z, x = [int(x) for x in input("Enter 3 integers separated by spaces: ").split(',')]
            action = (x, y, z)
            state, reward, done, info = env.step(action)
        except Exception as e:
            i = sys.exc_info()
            traceback.print_exception(*i)
            print("Invalid input. Please enter 3 integers separated by spaces.")
            break

    if done:
        env.render()
        my_dict = env.formed_sequences
        max_key = max(my_dict, key=lambda k: len(my_dict[k]))
        print(f'Game Over, Player: {max_key} Won')


play()
