import logging
import os
from collections import defaultdict, deque
from typing import Optional

import gym
import numpy as np
import torch
from gym.core import ActType
from gym.spaces import Box, Dict, MultiDiscrete
from matplotlib.colors import ListedColormap

from helper_functions import get_number_of_cards_for_players, generate_the_card_deck_and_index, \
    get_indices_from_given_data, \
    fill_locations_with_ones_in_3d_array, get_number_of_sequences_to_build, get_all_positions, \
    get_card_positions_on_board, \
    fill_2d_array_with_value, clear_directory, get_negative_array, get_other_positions_dictionary, \
    get_card_number_to_name_mapping
from misc.image_render import generate_image

CORNER_LOCATIONS = [(0, 0), (0, 9), (9, 0), (9, 9)]
ALL_POSITIONS = get_all_positions()
RENDER_OUTPUTS = './render_outputs'


class SequenceEnvironment(gym.Env):

    def __init__(self, players, invalid_move_reward, sequence_length_reward_multiplier, final_sequence_reward):
        colors = ['white', 'red', 'blue', 'green'][:players + 1]
        self.cmap = ListedColormap(colors)
        self.card_mapping_dict = get_card_number_to_name_mapping()
        self.final_sequence_reward = final_sequence_reward
        self.sequence_length_reward_multiplier = sequence_length_reward_multiplier
        self.invalid_move_reward = invalid_move_reward
        self.state = dict()
        self.other_positions = get_other_positions_dictionary()
        # first player is player 0
        self.current_player = 0
        self.players = players
        self.number_of_cards_per_player = get_number_of_cards_for_players(players)
        self.max_sequences_to_build = get_number_of_sequences_to_build(players)
        self.sequence_locations = set()
        self.render_count = 0

        self.observation_space = Dict({
            # information about where each player's cards are located on the board
            'player_board_positions': Box(low=0, high=1, shape=(self.players, 10, 10)),
            # where each card held by the player can be placed
            'hand_positions': Box(low=0, high=1, shape=(self.players, self.number_of_cards_per_player, 10, 10)),
            # is the card held by a player one-eyed jack or not, important because it changes how handle the step
            'is_card_one_eyed_jack': Box(low=0, high=1, shape=(self.players, self.number_of_cards_per_player)),
            # entire state of the board, to know which locations are free and which are not
            'board_occupied_locations': Box(low=0, high=1, shape=(self.players, 10, 10)),
            # how many sequences are built by each player
            'sequences_built': Box(low=0, high=self.max_sequences_to_build, shape=(self.players,)),
            # where are other player's cards located
            'other_player_board_positions': Box(low=0, high=1, shape=(self.players, 10, 10)),
            'actual_card_hand_positions': Box(low=0, high=1, shape=(self.players, self.number_of_cards_per_player))})

        # what card from a player's hand was used and where is it being placed
        self.action_space = MultiDiscrete([7, 10, 10])

    def get_current_players_observation(self):
        """
        Gets the observation of the game from current player's POV

        :returns
        one matrix containing all the player's positions on the board and the current player's hand positions
        """

        return torch.from_numpy(get_negative_array(self.state['player_board_positions'])), \
            torch.from_numpy(get_negative_array(self.state["hand_positions"][self.current_player])), \
            torch.from_numpy(get_negative_array(self.state['is_card_one_eyed_jack'][self.current_player]))

    def update_observation_for_regular_cards(self, position_placed):
        """
        Updates the game's state based on the position of the card played for all cards other than one eyed jack
        :param position_placed:
        """
        # take the action for the current player, update the observations
        # first update the player's position on board
        current_player_positions = self.state['player_board_positions'][self.current_player]
        current_player_positions[position_placed] = 1
        # update the entire board
        entire_board_positions = self.state['board_occupied_locations']
        entire_board_positions[position_placed] = 1

        # update other players locations
        other_player_board_positions = self.state['other_player_board_positions']
        for player in range(self.players):
            if player == self.current_player:
                continue
            other_player_board_positions[player][position_placed] = 1

        return

    def update_observation_for_one_eyed_jacks(self, position_placed):
        """
        Updates the game's state based on the position of one eyed jack played
        :param position_placed:
        """
        other_player_board_positions = self.state['other_player_board_positions']
        other_player_board_positions[:, position_placed[0], position_placed[1]] = 0

        player_board_positions = self.state['player_board_positions']
        player_board_positions[:, position_placed[0], position_placed[1]] = 0

        entire_board_positions = self.state['board_occupied_locations']
        entire_board_positions[position_placed[0], position_placed[1]] = 0

    def step(self, action: ActType):
        """
        Updates games state based on the current action, and calculates new observations/rewards, it also updates the current player
        :param action (tuple): tuple containing 3 elements: the card position in hand that was played, the row and column where the card is to be played
        :return: returns a tuple containing (new observation, reward for the action, has the game ended, info about the actions)
        """
        card_played, row, col = action
        position_placed = (row, col)
        logging.info(f'playing:{action}, for player:{self.current_player}')
        is_one_eyed_jack = self.state['is_card_one_eyed_jack'][self.current_player][card_played]
        if not self.is_action_valid(is_one_eyed_jack, position_placed, card_played):
            return self.get_current_players_observation(), self.invalid_move_reward, True, {
                'reason': 'player played an invalid move'}

        if is_one_eyed_jack:
            self.update_observation_for_one_eyed_jacks(position_placed)
        else:
            self.update_observation_for_regular_cards(position_placed)

        # update the player's hand, give them the next card
        next_card = self.get_next_card()
        self.give_player_card_at_hand_position(self.current_player, next_card, card_played)

        self.update_hand_positions(position_placed, is_one_eyed_jack)
        self.drop_dead_cards()

        length_factor = self.check_for_sequences(position_placed)
        number_of_sequences_so_far = len(self.formed_sequences[self.current_player])
        reward = self.sequence_length_reward_multiplier * length_factor + self.final_sequence_reward * number_of_sequences_so_far

        # move to the next player
        self.current_player += 1
        self.current_player %= self.players
        end = number_of_sequences_so_far == self.max_sequences_to_build
        return self.get_current_players_observation(), reward, end, {
            'reason': 'Max Sequences Formed' if end else 'Game continues'}

    def render(self):
        """
        Renders the current state of the board
        """

        arr = self.state['player_board_positions']
        new_image_arr = np.full((10, 10), -1)

        for row in range(10):
            for col in range(10):
                for player in range(self.players):
                    if (row, col) in CORNER_LOCATIONS:
                        continue
                    if arr[player][row][col] == 1:
                        new_image_arr[row][col] = player
        image_path = os.path.join(RENDER_OUTPUTS, f"{self.render_count}.png")
        generate_image(new_image_arr, image_path, self.state['actual_card_hand_positions'][self.current_player])

        print('Card Numbers', [(index, self.card_mapping_dict[card_number]) for index, card_number in
                               enumerate(self.state['actual_card_hand_positions'][self.current_player])])
        self.render_count += 1

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[dict] = None,
    ):
        """
        Sets up the environment
        :param seed:
        :param options:
        :return:
        """
        self.move_to_initial_state()
        clear_directory(RENDER_OUTPUTS)
        return self.get_current_players_observation()

    def distribute_the_cards(self):
        """
        :return: a dictionary where key is a player, and value is a list of cards at hand
        """
        player_cards = dict()
        for player in range(self.players):
            cards = [next(self.card_deck) for _ in range(self.number_of_cards_per_player)]
            player_cards[player] = cards
        return player_cards

    def move_to_initial_state(self):
        """
        Generates the initial state of the board and updates self.state, does not return anything
        :param hand_cards:
        """
        self.formed_sequences = defaultdict(list)
        self.card_deck = generate_the_card_deck_and_index()
        self.player_cards = self.distribute_the_cards()

        # initializing an empty board
        player_board_positions = np.zeros((self.players, 10, 10))
        # entire board is empty at the start
        board_occupied_locations = np.zeros((10, 10))
        fill_2d_array_with_value(board_occupied_locations, 1, CORNER_LOCATIONS)
        self.state['board_occupied_locations'] = board_occupied_locations

        # everyone gets corners by default
        fill_locations_with_ones_in_3d_array(player_board_positions, CORNER_LOCATIONS)
        self.state['player_board_positions'] = player_board_positions

        # no one has cards on the board at start
        self.state['other_player_board_positions'] = np.zeros((self.players, 10, 10))

        self.state['is_card_one_eyed_jack'] = np.zeros((self.players, self.number_of_cards_per_player))
        hand_positions = np.zeros((self.players, self.number_of_cards_per_player, 10, 10))
        self.state['actual_card_hand_positions'] = np.zeros((self.players, self.number_of_cards_per_player))
        self.state['hand_positions'] = hand_positions
        for player, cards in self.player_cards.items():
            for card_position_in_hand, card in enumerate(cards):
                self.give_player_card_at_hand_position(player, card, card_position_in_hand)

    def get_valid_locations_for_card(self, card_number, player):
        """
        Looks at the current board positions and returns a list of positions where the card can be placed on the
        board, returns empty list if card cannot be placed anywhere
        """
        entire_board_occupied_locations = self.state['board_occupied_locations']
        others_card_locations = self.state['other_player_board_positions'][player]
        card_positions_on_board = get_card_positions_on_board()
        if card_number < 0:
            return []

        if card_number < 49:  # specific positions are possible for cards, except jacks
            card_locations_possible = card_positions_on_board[card_number]
        else:  # all locations are possible for jacks
            card_locations_possible = ALL_POSITIONS

        # filter positions based what can actually be filled
        if card_number == 50:  # one eyed jack can be placed only on already filled locations
            fillable_positions = get_indices_from_given_data(card_locations_possible, others_card_locations, 1)
        else:
            fillable_positions = get_indices_from_given_data(card_locations_possible,
                                                             entire_board_occupied_locations, 0)

        # remove corner locations since we can never place cards there
        fillable_positions = [x for x in fillable_positions if x not in CORNER_LOCATIONS]
        return fillable_positions

    def is_location_in_any_sequence(self, location):
        """
        Checks if the given location is occupied by any of the already formed sequences
        :param location:
        :return: True if location is in an existing sequence, false if not
        """
        for player, sequences in self.formed_sequences.items():
            for sequence in sequences:
                if location in sequence:
                    return True
        return False

    def is_action_valid(self, is_one_eyed_jack, position_placed, card_played):
        """
        Returns True if the given action is valid else False
        """
        # check if it is a corner location
        if position_placed in CORNER_LOCATIONS:
            return False

        # check if it is in a formed sequence
        if self.is_location_in_any_sequence(position_placed):
            return False

        # check if the card is allowed to be placed there based on card value
        hand_positions = self.state['hand_positions'][self.current_player]
        valid_positions_for_card = hand_positions[card_played]
        if valid_positions_for_card[position_placed[0], position_placed[1]] != 1:
            return False

        # check if the card is allowed to be placed there based on board values
        board_occupied_locations = self.state['board_occupied_locations']
        others_occupied_locations = self.state['other_player_board_positions'][self.current_player]
        if is_one_eyed_jack:
            # check if the location has someone else's card otherwise no point using a one eyed jack
            return others_occupied_locations[position_placed] == 1
        else:
            # check if the position is empty
            return board_occupied_locations[position_placed] == 0

    def get_next_card(self):
        """
        Gets the next card from the card deck if there is any
        :return: card number that is picked up, if no cards are left, -1
        """
        try:
            next_card = next(self.card_deck)
            return next_card
        except:
            return -1

    def check_for_sequences(self, position):
        """
        Checks if the position is filled, are there any sequences, and what are the lengths, also updates the formed sequences list if there are any fully formed sequences
        :param position:
        :return:
        """
        matrix = self.state['player_board_positions'][self.current_player]
        # Define the directions to search in
        directions = [(1, 0), (0, 1), (1, 1), (-1, 1)]
        max_sequence_length = 0
        placed_row, placed_col = position
        for index, direction in enumerate(directions):
            dr, dc = direction
            sequence_length = 0
            row, col = placed_row - (dr * 4), placed_col - (dc * 4)
            curr_points_under_consideration = deque([])
            for _ in range(10):
                if row < 0 or col < 0 or row >= len(matrix) or col >= len(matrix[0]):
                    row += dr
                    col += dc
                    continue

                if sequence_length == 5:
                    can_add = True
                    for formed_sequence in self.formed_sequences[self.current_player]:
                        common_points_len = 0
                        for point in curr_points_under_consideration:
                            if point in formed_sequence:
                                common_points_len += 1
                        if common_points_len == 5:
                            can_add = False
                    if not can_add:
                        continue
                    self.formed_sequences[self.current_player].append(list(curr_points_under_consideration))
                    break

                if len(curr_points_under_consideration) == 5:
                    pop_row, pop_col = curr_points_under_consideration.popleft()
                    sequence_length -= matrix[pop_row, pop_col]

                curr_points_under_consideration.append([row, col])
                sequence_length += matrix[row, col]
                max_sequence_length = max(max_sequence_length, sequence_length)

                row += dr
                col += dc

        return max_sequence_length

    def give_player_card_at_hand_position(self, player, card, card_number_in_hand):
        """

        :param player:
        :param card:
        :param card_number_in_hand:
        """
        actual_card_hand_positions = self.state['actual_card_hand_positions']
        actual_card_hand_positions[player][card_number_in_hand] = card
        hand_positions = self.state['hand_positions']
        is_one_eyed_jack_dict = self.state['is_card_one_eyed_jack']
        player_hand_position = hand_positions[player]
        placeable_card_locations = self.get_valid_locations_for_card(card, player)
        new_card_positions = np.zeros((10, 10))
        fill_2d_array_with_value(new_card_positions, 1, placeable_card_locations)
        player_hand_position[card_number_in_hand] = new_card_positions
        is_one_eyed_jack_dict[player][card_number_in_hand] = card == 50

    def update_hand_positions(self, position_placed, one_eyed_jack_played):
        """
        Updates the cards in the hands of the players, to see where they can be placed
        :param position_placed:
        """
        hand_positions = self.state['hand_positions']
        is_card_in_hand_one_eyed_jack = self.state['is_card_one_eyed_jack']

        for player in range(self.players):
            for card in range(self.number_of_cards_per_player):
                if is_card_in_hand_one_eyed_jack[player][card]:
                    # one eyed jack in players hand
                    if one_eyed_jack_played:  # one eyed jack played
                        hand_positions[player][card][position_placed[0]][position_placed[1]] = 0
                    else:  # regular card was played
                        if self.current_player == player:
                            continue
                        hand_positions[player][card][position_placed[0]][position_placed[1]] = 1
                else:
                    # regular card in players hand
                    if one_eyed_jack_played:
                        # one eyed jack played
                        other_position = self.other_positions[position_placed]
                        if hand_positions[player][card][other_position[0]][other_position[1]] == 1:
                            hand_positions[player][card][position_placed[0]][position_placed[1]] = 1
                    else:
                        # regular card played
                        hand_positions[player][card][position_placed[0]][position_placed[1]] = 0

        # if card was removed

    def drop_dead_cards(self):
        hand_positions = self.state['hand_positions']
        is_one_eyed_jack = self.state['is_card_one_eyed_jack']
        for player in range(self.players):
            is_one_eyed_jack_player = is_one_eyed_jack[player]
            for card in range(self.number_of_cards_per_player):
                if np.sum(hand_positions[player][card]) == 0 and not is_one_eyed_jack_player[card]:
                    next_card = self.get_next_card()
                    if next_card == -1:
                        return
                    self.give_player_card_at_hand_position(player, next_card, card)
