import os
import random
from functools import cache

import numpy as np
import pandas as pd
import torch


def get_indices_from_given_data(idx_arr, data_arr, value):
    """
    Get all the indices from the array where the values are the given value and are also in the given list of indices
    :param idx_arr:
    :param data_arr:
    :param value:
    :return:
    """
    result = []
    for index in idx_arr:
        if data_arr[index[0], index[1]] == value:
            result.append(index)
    return result


def get_number_of_cards_for_players(players):
    """
    Returns the number of cards that each player gets in the game based on number of players
    """
    if players == 2:
        return 7
    return 6


def convert_to_numeric_tuples(lst):
    """
    Given a list of string represnting tuples, convert them to numeric to tuples
    :param lst:
    :return:
    """
    result = []
    for s in lst:
        tup = tuple(map(int, s.split(',')))
        result.append(tup)
    return result


@cache
def get_card_positions_on_board():
    """
    Reads the card mapping file and gets every card's position on the board
    :return: dictionary containing each card's positions on the board
    """
    card_positions_df = pd.read_csv('card_mapping.csv')
    values = card_positions_df.values.tolist()

    # Create an empty dictionary
    card_positions_dict = {}

    # Loop through the values and add them to the dictionary
    for value in values:
        key = value[1]
        value = convert_to_numeric_tuples(value[2:])
        card_positions_dict[key] = value

    return card_positions_dict


def get_cards_name_positions():
    return {(0, 1): 'spade_2', (8, 6): 'spade_2', (0, 2): 'spade_3', (8, 5): 'spade_3', (0, 3): 'spade_4',
            (8, 4): 'spade_4', (0, 4): 'spade_5', (8, 3): 'spade_5', (0, 5): 'spade_6', (8, 2): 'spade_6',
            (0, 6): 'spade_7', (8, 1): 'spade_7', (0, 7): 'spade_8', (7, 1): 'spade_8', (0, 8): 'spade_9',
            (6, 1): 'spade_9', (1, 9): 'spade_10', (5, 1): 'spade_10', (2, 9): 'spade_queen', (4, 1): 'spade_queen',
            (3, 9): 'spade_king', (3, 1): 'spade_king', (4, 9): 'spade_ace', (2, 1): 'spade_ace', (8, 0): 'club_ace',
            (7, 5): 'club_ace', (1, 4): 'club_2', (3, 6): 'club_2', (1, 3): 'club_3', (3, 5): 'club_3',
            (1, 2): 'club_4', (3, 4): 'club_4', (1, 1): 'club_5', (3, 3): 'club_5', (1, 0): 'club_6', (3, 2): 'club_6',
            (2, 0): 'club_7', (4, 2): 'club_7', (3, 0): 'club_8', (5, 2): 'club_8', (4, 0): 'club_9', (6, 2): 'club_9',
            (5, 0): 'club_10', (7, 2): 'club_10', (6, 0): 'club_queen', (7, 3): 'club_queen', (7, 0): 'club_king',
            (7, 4): 'club_king', (1, 5): 'heart_ace', (4, 6): 'heart_ace', (8, 7): 'heart_2', (5, 4): 'heart_2',
            (8, 8): 'heart_3', (5, 5): 'heart_3', (7, 8): 'heart_4', (4, 5): 'heart_4', (6, 8): 'heart_5',
            (4, 4): 'heart_5', (5, 8): 'heart_6', (4, 3): 'heart_6', (4, 8): 'heart_7', (5, 3): 'heart_7',
            (3, 8): 'heart_8', (6, 3): 'heart_8', (2, 8): 'heart_9', (6, 4): 'heart_9', (1, 8): 'heart_10',
            (6, 5): 'heart_10', (1, 7): 'heart_queen', (6, 6): 'heart_queen', (1, 6): 'heart_king',
            (5, 6): 'heart_king', (9, 1): 'diamond_ace', (7, 6): 'diamond_ace', (5, 9): 'diamond_2',
            (2, 2): 'diamond_2', (6, 9): 'diamond_3', (2, 3): 'diamond_3', (7, 9): 'diamond_4', (2, 4): 'diamond_4',
            (8, 9): 'diamond_5', (2, 5): 'diamond_5', (9, 8): 'diamond_6', (2, 6): 'diamond_6', (9, 7): 'diamond_7',
            (2, 7): 'diamond_7', (9, 6): 'diamond_8', (3, 7): 'diamond_8', (9, 5): 'diamond_9', (4, 7): 'diamond_9',
            (9, 4): 'diamond_10', (5, 7): 'diamond_10', (9, 3): 'diamond_queen', (6, 7): 'diamond_queen',
            (9, 2): 'diamond_king', (7, 7): 'diamond_king'}


def get_card_number_to_name_mapping():
    return {1: 'spade_2', 2: 'spade_3', 3: 'spade_4', 4: 'spade_5', 5: 'spade_6', 6: 'spade_7', 7: 'spade_8',
            8: 'spade_9', 9: 'spade_10', 10: 'spade_queen', 11: 'spade_king', 12: 'spade_ace', 13: 'club_ace',
            14: 'club_2', 15: 'club_3', 16: 'club_4', 17: 'club_5', 18: 'club_6', 19: 'club_7', 20: 'club_8',
            21: 'club_9', 22: 'club_10', 23: 'club_queen', 24: 'club_king', 25: 'heart_ace', 26: 'heart_2',
            27: 'heart_3', 28: 'heart_4', 29: 'heart_5', 30: 'heart_6', 31: 'heart_7', 32: 'heart_8', 33: 'heart_9',
            34: 'heart_10', 35: 'heart_queen', 36: 'heart_king', 37: 'diamond_ace', 38: 'diamond_2', 39: 'diamond_3',
            40: 'diamond_4', 41: 'diamond_5', 42: 'diamond_6', 43: 'diamond_7', 44: 'diamond_8', 45: 'diamond_9',
            46: 'diamond_10', 47: 'diamond_queen', 48: 'diamond_king', 49: 'two_eyed_jack', 50: 'one_eyed_jack'}


def get_card_name_to_number_mapping():
    return {'spade_2': 1, 'spade_3': 2, 'spade_4': 3, 'spade_5': 4, 'spade_6': 5, 'spade_7': 6, 'spade_8': 7,
            'spade_9': 8, 'spade_10': 9, 'spade_queen': 10, 'spade_king': 11, 'spade_ace': 12, 'club_ace': 13,
            'club_2': 14, 'club_3': 15, 'club_4': 16, 'club_5': 17, 'club_6': 18, 'club_7': 19, 'club_8': 20,
            'club_9': 21, 'club_10': 22, 'club_queen': 23, 'club_king': 24, 'heart_ace': 25, 'heart_2': 26,
            'heart_3': 27, 'heart_4': 28, 'heart_5': 29, 'heart_6': 30, 'heart_7': 31, 'heart_8': 32, 'heart_9': 33,
            'heart_10': 34, 'heart_queen': 35, 'heart_king': 36, 'diamond_ace': 37, 'diamond_2': 38, 'diamond_3': 39,
            'diamond_4': 40, 'diamond_5': 41, 'diamond_6': 42, 'diamond_7': 43, 'diamond_8': 44, 'diamond_9': 45,
            'diamond_10': 46, 'diamond_queen': 47, 'diamond_king': 48, 'two_eyed_jack': 49, 'one_eyed_jack': 50}


def generate_the_card_deck_and_index():
    """
    Reads the file for different card locations and generates an iterator representing the list of all the cards
    :return:
    """
    total_cards = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 50, 50, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14,
                   15, 15, 16, 16, 17, 17, 18, 18, 19, 19, 20, 20, 21, 21, 22, 22, 23, 23, 24, 24, 25, 25, 26, 26, 27,
                   27, 28, 28, 29, 29, 30, 30, 31, 31, 32, 32, 33, 33, 34, 34, 35, 35, 36, 36, 37, 37, 38, 38, 39, 39,
                   40, 40, 41, 41, 42, 42, 43, 43, 44, 44, 45, 45, 46, 46, 47, 47, 48, 48, 50, 50, 49, 49, 49, 49]
    # # shuffle the card
    random.shuffle(total_cards)
    return iter(total_cards)


def fill_locations_with_ones_in_3d_array(arr, ones):
    """
    Fill a 3D array with ones at specified 2D indices on every item.

    Parameters:
        arr (numpy.ndarray): 3D array to modify
        ones (list): List of 2D indices to fill with ones
    """
    for idx in ones:
        arr[:, idx[0], idx[1]] = 1


def get_number_of_sequences_to_build(players):
    """
    Returns the number of sequences to build based on the number of players
    :param players:
    :return:
    """
    if players == 2:
        return 2
    return 1


def get_all_positions():
    """
    :return: a list of tuples representing all possible positions on the board
    """
    return [(x, y) for x in range(10) for y in range(10)]


def fill_2d_array_with_value(array, value, indices):
    """
    Fills the given array with the given value at the indices
    :param array: 2d array to be filled
    :param value: value to be assigned
    :param indices: indices where the values are to be filled
    """
    for index in indices:
        array[index[0]][index[1]] = value


def clear_directory(path):
    # Create the directory if it doesn't exist
    if not os.path.exists(path):
        os.makedirs(path)

    # Remove all files in the directory
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)


def get_a_valid_move(observation, depth=0):
    """
    Looks at the hand position of a player and returns a random move that is valid
    :param observation: expects a torch tensor representing the hand of a player
    :return: (tuple) containing the move that is supposed to be player (card_number, row_number, col_number)
    """
    card_index = random.randint(0, 6)
    card_values = observation[card_index]
    card_positions = torch.nonzero(card_values == 1)

    if card_positions.nelement() == 0:
        if depth == 2:
            for card in observation:
                card_positions = torch.nonzero(card == 1)
                if card_positions.nelement() == 0:
                    continue
                random_index = torch.randint(0, len(card_positions), (1,))
                position = card_positions[random_index][0]
                return card_index, position[0].item(), position[1].item()
            return 0, 0, 0  # no valid move left, just end the game
        return get_a_valid_move(observation, depth + 1)

    random_index = torch.randint(0, len(card_positions), (1,))
    position = card_positions[random_index][0]
    return card_index, position[0].item(), position[1].item()


def get_negative_array(array):
    return np.where(array == 0, -1, array)


def get_other_positions_dictionary():
    data_dict = dict()
    data = pd.read_csv('card_mapping.csv')
    col1, col2 = convert_to_numeric_tuples(data['position1']), convert_to_numeric_tuples(data['position2'])
    for pos1, pos2 in zip(col1, col2):
        data_dict[pos1] = pos2
        data_dict[pos2] = pos1
    return data_dict
