import random

import numpy as np
import pandas as pd


def get_zero_indices_from_given_data(idx_arr, data_arr):
    values = data_arr[tuple(idx_arr.T)]
    return idx_arr[values == 0]


def get_one_indices_from_given_data(idx_arr, data_arr):
    values = data_arr[tuple(idx_arr.T)]
    return idx_arr[values == 1]


def get_one_indices(arr):
    return np.where(arr == 1)[0]


def get_number_of_cards_for_players(players):
    """
    Returns the number of cards that each player gets in the game based on number of players
    """
    if players == 2:
        return 7
    return 6


def df_to_dict(df):
    """
    Given a pandas DataFrame where the second column is the key and the
    remaining columns form the values, returns a dictionary where each
    key maps to a list of values.
    """
    keys = df.iloc[:, 1].tolist()
    values = df.iloc[:, 2:].values.tolist()
    values = np.array(values)
    return dict(zip(keys, values))


def get_card_positions_on_board():
    card_positions_df = pd.read_csv('../card_mapping.csv')
    card_positions_dict = df_to_dict(card_positions_df)
    return card_positions_dict


def generate_the_card_deck_and_index():
    """
    Reads the file for different card locations and generates an iterator representing the list of all the cards
    :return:
    """
    card_positions_df = pd.read_csv('../card_mapping.csv')
    card_index = dict()
    for index, card in enumerate(card_positions_df['card']):
        card_index[card] = index
    card_index['one_eyed_jack'] = 48
    card_index['two_eyed_jack'] = 49
    total_cards = [item for item in card_positions_df['card'] for _ in range(2)]
    total_cards.extend(['one_eyed_jack'] * 4)
    total_cards.extend(['two_eyed_jack'] * 4)
    # shuffle the card
    random.shuffle(total_cards)
    return iter(total_cards), card_index


def fill_locations_with_ones(arr, ones):
    """
    Fill a 3D array with ones at specified 2D indices on every item.

    Parameters:
        arr (numpy.ndarray): 3D array to modify
        ones (list): List of 2D indices to fill with ones

    Returns:
        numpy.ndarray: The modified array with specified 2D indices filled with ones on every item
    """
    # Create a copy of the input array to avoid modifying the original array
    arr_copy = np.copy(arr)

    # Fill specified 2D indices with ones on every item
    for idx in ones:
        arr_copy[:, idx[0], idx[1]] = 1

    return arr_copy


def get_number_of_sequences_to_build(players):
    if players == 2:
        return 2
    return 1


def get_all_positions():
    # Define the ranges for each dimension
    range_x = np.arange(0, 10)
    range_y = np.arange(0, 10)

    # Generate the grid of points
    xx, yy = np.meshgrid(range_x, range_y)

    # Combine the points into tuples
    points = np.vstack((xx.ravel(), yy.ravel())).T

    return points



