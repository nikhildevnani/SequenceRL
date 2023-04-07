import click
import torch

from my_agent import DQNAgent
from sequence_game.sequence_evironment import SequenceEnvironment

agent = DQNAgent(2)
env = SequenceEnvironment(2, -1000, 5, 1000)
torch.set_default_dtype(torch.float64)


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
            x, y, z = [int(x) for x in input("Enter 3 integers separated by spaces: ").split(',')]
            action = (x, y, z)
            state, reward, done, info = env.step(action)
        except:
            print("Invalid input. Please enter 3 integers separated by spaces.")
            continue

    if done:
        my_dict = env.formed_sequences
        max_key = max(my_dict, key=lambda k: len(my_dict[k]))
        print(f'Game Over, Player: {max_key} Won')


play()
