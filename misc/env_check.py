import logging

logging.basicConfig(level=logging.ERROR)
from helper_functions import get_a_valid_move
from sequence_game.sequence_evironment import SequenceEnvironment

sequence_env = SequenceEnvironment(2, -1000, 5, 1000)
(player_board_positions, players_hand, is_card_one_eyed_jack) = sequence_env.reset()
end = False

try:
    index = 0
    while not end:
        move = get_a_valid_move(players_hand)
        (player_board_positions, players_hand, is_card_one_eyed_jack), reward, end, info = sequence_env.step(move)
        logging.info(f'reward obtained:{reward}, at turn: {index}')
        index += 1
        sequence_env.render()
except:
    print("FAILED")
    print((player_board_positions, players_hand, is_card_one_eyed_jack), info)
print(info)
# sequence_env.render()

# print(obv, reward, end, info)
