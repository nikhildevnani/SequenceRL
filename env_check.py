import random

import numpy as np

from sequence_game.sequence_evironment import SequenceEnvironment


def get_a_valid_move(observation):
    card_index = random.randint(2, 8)
    card_values = observation[card_index]
    card_positions = list(map(tuple, np.argwhere(card_values == 1)))
    if not card_positions:
        return get_a_valid_move(observation)

    position = random.sample(card_positions, 1)[0]
    return card_index - 2, position[0], position[1]


sequence_env = SequenceEnvironment(2, -1000, 5, 1000)
obv = sequence_env.reset()
end = False
try:
    while not end:
        move = get_a_valid_move(obv)
        obv, reward, end, info = sequence_env.step(move)
        print('reward obtained:', reward)
        sequence_env.render()
except:
    print("FAILED")
    print(obv, info)
print(info)
# sequence_env.render()

# print(obv, reward, end, info)
