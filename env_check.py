import random

import numpy as np

from sequence_game.sequence_evironment import SequenceEnvironment


def get_a_valid_move(observation):
    card_index = random.randint(0, 6)
    card_values = observation[card_index]
    card_positions = list(map(tuple, np.argwhere(card_values == 1)))
    if not card_positions:
        return get_a_valid_move(observation)

    position = random.sample(card_positions, 1)[0]
    return card_index, position[0], position[1]


sequence_env = SequenceEnvironment(2, -1000, 5, 1000)
(player_board_positions, players_hand, is_card_one_eyed_jack) = sequence_env.reset()
end = False
try:
    index = 0
    while not end:
        move = get_a_valid_move(players_hand)
        (player_board_positions, players_hand, is_card_one_eyed_jack), reward, end, info = sequence_env.step(move)
        print('reward obtained:', reward, 'at turn:', index)
        index += 1
        sequence_env.render()
except:
    print("FAILED")
    print((player_board_positions, players_hand, is_card_one_eyed_jack), info)
print(info)
# sequence_env.render()

# print(obv, reward, end, info)
