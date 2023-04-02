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


def get_max_sequence_length_and_direction(matrix, position, sequences):
    """
    Get the maximum length of a sequence of 1 that can be formed given a matrix containing 1 and 0,
    a position where the new 1 is to be placed, a list of 2d indices which cannot be included in the sequence.
    The sequence can be horizontal or vertical or diagonal.

    Parameters:
        matrix (numpy.ndarray): The input matrix.
        position (tuple): A tuple representing the position where the new 1 is to be placed.
        sequences (list(list(tuple))): a list of existing sequences

    Returns:
        int: The maximum length of a sequence of 1 that can be formed.

    """
    # Define the directions to search in
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (-1, 1), (1, -1)]

    # Initialize the maximum sequence length
    max_seq_len = 0

    sequence_lengths = []
    # Iterate over the directions
    for index, direction in enumerate(directions):
        # Get the starting position
        i, j = position

        # Initialize the sequence length
        seq_len = 0
        number_of_existing_sequence_points_used = 0

        while (0 <= i < matrix.shape[0] and 0 <= j < matrix.shape[1] and matrix[i][j] == 1):
            if (i, j) in sequences:
                number_of_existing_sequence_points_used += 1

            # if we are using more than 1 sequence point, we are extending in the same direction are an existing
            # sequence, so stop it
            if number_of_existing_sequence_points_used == 2:
                seq_len -= 1
                break
            seq_len += 1
            i += direction[0]
            j += direction[1]

        # Update the maximum sequence length if needed
        sequence_lengths.append(seq_len)

    vertical_length = sequence_lengths[0] + sequence_lengths[1]
    horizontal_length = sequence_lengths[2] + sequence_lengths[3]
    top_left_to_bottom_right_length = sequence_lengths[4] + sequence_lengths[5]
    bottom_left_to_top_right_length = sequence_lengths[6] + sequence_lengths[7]

    return max(vertical_length, horizontal_length, top_left_to_bottom_right_length, bottom_left_to_top_right_length)
