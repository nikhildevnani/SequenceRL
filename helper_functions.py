import os
import random

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


def dataframe_to_dict(df):
    """
    Converts a pandas DataFrame into a dictionary using the second column as the key
    and the following columns as the values.

    Parameters:
        df (pandas.DataFrame): The DataFrame to convert.

    Returns:
        dict: A dictionary with the second column as the key and the following columns as the values.
    """
    # Get the values from the DataFrame
    values = df.values.tolist()

    # Create an empty dictionary
    result = {}

    # Loop through the values and add them to the dictionary
    for value in values:
        key = value[1]
        value = convert_to_numeric_tuples(value[2:])
        result[key] = value

    return result


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


def get_card_positions_on_board():
    """
    Reads the card mapping file and gets every card's position on the board
    :return: dictionary containing each card's positions on the board
    """
    card_positions_df = pd.read_csv('card_mapping.csv')
    card_positions_dict = dataframe_to_dict(card_positions_df)
    return card_positions_dict


def generate_the_card_deck_and_index():
    """
    Reads the file for different card locations and generates an iterator representing the list of all the cards
    :return:
    """
    card_positions_df = pd.read_csv('card_mapping.csv')
    total_cards = [item for item in card_positions_df['card_number'] for _ in range(2)]
    total_cards.extend([48] * 4)  # two eyed jack
    total_cards.extend([49] * 4)  # one eyed jack
    # shuffle the card
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
