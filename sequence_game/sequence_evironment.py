from collections import defaultdict
from typing import Optional

import gym
import matplotlib.pyplot as plt
import numpy as np
from gym.core import ActType
from gym.spaces import Box, Dict, MultiDiscrete

from sequence_game.helper_functions import get_number_of_cards_for_players, generate_the_card_deck_and_index, \
    get_one_indices_from_given_data, get_zero_indices_from_given_data, \
    fill_locations_with_ones_in_3d_array, get_number_of_sequences_to_build, get_all_positions, \
    get_card_positions_on_board, \
    fill_2d_array_with_value

CORNER_LOCATIONS = [(0, 0), (0, 9), (9, 0), (9, 9)]
ALL_POSITIONS = get_all_positions()


class SequenceEnvironment(gym.Env):

    def __init__(self, players, invalid_move_reward, sequence_length_reward_multiplier, final_sequence_reward):

        self.formed_sequences = defaultdict(dict)
        self.final_sequence_reward = final_sequence_reward
        self.sequence_length_reward_multiplier = sequence_length_reward_multiplier
        self.invalid_move_reward = invalid_move_reward
        self.state = dict()
        # first player is player 0
        self.current_player = 0
        self.players = players
        self.number_of_cards_per_player = get_number_of_cards_for_players(players)
        self.max_sequences_to_build = get_number_of_sequences_to_build(players)
        self.sequence_locations = set()

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
            'other_player_board_positions': Box(low=0, high=1, shape=(self.players, 10, 10))})

        # what card from a player's hand was used and where is it being placed
        self.action_space = MultiDiscrete([7, 10, 10])

    def get_current_players_observation(self):
        """
        Gets the observation of the game from current player's POV

        :returns
        one matrix containing all the player's positions on the board and the current player's hand positions
        """
        players_hand = self.state["hand_positions"][self.current_player]
        total_observation = np.concatenate((self.state['player_board_positions'], players_hand))
        return total_observation

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

    def update_observation_for_one_eyed_jacks(self, position_placed):
        """
        Updates the game's state based on the position of one eyed jack played
        :param position_placed:
        """
        other_player_board_positions = self.state['other_player_board_positions']
        for player in range(self.players):
            other_player_board_positions[player] = 0

        player_board_positions = self.state['player_board_positions']
        for player in range(self.players):
            player_board_positions[player] = 0

        entire_board_positions = self.state['board_occupied_locations']
        entire_board_positions[position_placed] = 0

    def step(self, action: ActType):
        """
        Updates games state based on the current action, and calculates new observations/rewards, it also updates the current player
        :param action (tuple): tuple containing 3 elements: the card position in hand that was played, the row and column where the card is to be played
        :return: returns a tuple containing (new observation, reward for the action, has the game ended, info about the actions)
        """
        card_played, row, col = action
        position_placed = (row, col)
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

        length_factor = self.check_for_sequences(position_placed)
        number_of_sequences_so_far = len(self.formed_sequences[self.current_player])
        reward = self.sequence_length_reward_multiplier ** length_factor + self.final_sequence_reward ** number_of_sequences_so_far

        # move to the next player
        self.current_player += 1
        self.current_player = self.current_player % self.players
        end = number_of_sequences_so_far == self.max_sequences_to_build
        return self.get_current_players_observation(), reward, end, {
            'reason': 'Max Sequences Formed' if end else 'Game continues'}

    def render(self):
        """
        Renders the current state of the board
        """
        arr = self.state['player_board_positions']

        # create a 2D array with random numbers
        arr = np.argmax(arr, axis=0)

        # create a color map with a range of colors
        cmap = plt.cm.get_cmap('jet', 256)

        # create the image from the 2D array using the color map
        img = cmap(arr)

        # display the image
        plt.imshow(img)
        plt.show()

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

        self.state['is_card_one_eyed_jack'] = defaultdict(dict)
        hand_positions = np.zeros((self.players, self.number_of_cards_per_player, 10, 10))
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

        if card_number < 48:  # specific positions are possible for cards, except jacks
            card_locations_possible = card_positions_on_board[card_number]
        else:  # all locations are possible for jacks
            card_locations_possible = ALL_POSITIONS

        # filter positions based what can actually be filled
        if card_number == 49:  # one eyed jack can be placed only on already filled locations
            fillable_positions = get_one_indices_from_given_data(card_locations_possible, others_card_locations)
        else:
            fillable_positions = get_zero_indices_from_given_data(card_locations_possible,
                                                                  entire_board_occupied_locations)

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
                if location in sequences:
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
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (-1, 1), (1, -1)]
        points_in_the_direction = [[] for _ in range(8)]

        sequence_lengths = []
        # Iterate over the directions
        for index, direction in enumerate(directions):
            # Get the starting position
            i, j = position

            # Initialize the sequence length
            seq_len = 0
            number_of_existing_sequence_points_used = 0

            while 0 <= i < matrix.shape[0] and 0 <= j < matrix.shape[1] and matrix[i][j] == 1:
                if self.is_location_in_any_sequence((i, j)):
                    number_of_existing_sequence_points_used += 1

                # if we are using more than 1 sequence point, we are extending in the same direction on an existing
                # sequence, so stop it
                if number_of_existing_sequence_points_used == 2:
                    seq_len -= 1
                    break
                seq_len += 1
                points_in_the_direction[index].append((i, j))
                i += direction[0]
                j += direction[1]

            # Update the maximum sequence length if needed
            sequence_lengths.append(seq_len)

        vertical_length = sequence_lengths[0] + sequence_lengths[1] - 1
        horizontal_length = sequence_lengths[2] + sequence_lengths[3] - 1
        top_left_to_bottom_right_length = sequence_lengths[4] + sequence_lengths[5] - 1
        bottom_left_to_top_right_length = sequence_lengths[6] + sequence_lengths[7] - 1

        if vertical_length >= 5:
            vertical_sequence_points = points_in_the_direction[0] + points_in_the_direction[1][1:]
            self.formed_sequences[self.current_player].append(vertical_sequence_points[:5])
        if horizontal_length >= 5:
            horizontal_sequence_points = points_in_the_direction[2] + points_in_the_direction[3][1:]
            self.formed_sequences[self.current_player].append(horizontal_sequence_points[:5])
        if top_left_to_bottom_right_length >= 5:
            top_left_to_bottom_right_points = points_in_the_direction[4] + points_in_the_direction[5][1:]
            self.formed_sequences[self.current_player].append(top_left_to_bottom_right_points[:5])
        if bottom_left_to_top_right_length >= 5:
            bottom_left_to_top_right_points = points_in_the_direction[6] + points_in_the_direction[7][1:]
            self.formed_sequences[self.current_player].append(bottom_left_to_top_right_points[:5])

        return max(vertical_length, horizontal_length, top_left_to_bottom_right_length, bottom_left_to_top_right_length)

    def give_player_card_at_hand_position(self, player, card, card_number_in_hand):
        """

        :param player:
        :param card:
        :param card_number_in_hand:
        """
        hand_positions = self.state['hand_positions']
        is_one_eyed_jack_dict = self.state['is_card_one_eyed_jack']
        player_hand_position = hand_positions[player]
        placeable_card_locations = self.get_valid_locations_for_card(card, player)
        new_card_positions = np.zeros((10,10))
        fill_2d_array_with_value(new_card_positions, 1, placeable_card_locations)
        player_hand_position[card_number_in_hand] = new_card_positions
        is_one_eyed_jack_dict[player][card_number_in_hand] = card == 49
