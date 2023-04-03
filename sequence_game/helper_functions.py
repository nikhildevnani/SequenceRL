import random

import numpy as np
import pandas as pd


def get_zero_indices_from_given_data(idx_arr, data_arr):
    result = []
    for index in idx_arr:
        if data_arr[index] == 0:
            result.append(index)
    return result


def get_one_indices_from_given_data(idx_arr, data_arr):
    result = []
    for index in idx_arr:
        if data_arr[index] == 1:
            result.append(index)
    return result


def get_one_indices(arr):
    return np.where(arr == 1)[0]


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
        value = convert_to_tuples(value[2:])
        result[key] = value

    return result


def convert_to_tuples(lst):
    result = []
    for s in lst:
        tup = tuple(map(int, s.split(',')))
        result.append(tup)
    return result


def get_card_positions_on_board():
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
    if players == 2:
        return 2
    return 1


def get_all_positions():
    return [(x, y) for x in range(10) for y in range(10)]

def fill_2d_array_with_value(array, value, indices):
    for index in indices:
        array[index[0]][index[1]] = value
