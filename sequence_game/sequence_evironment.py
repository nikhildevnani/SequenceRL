from typing import Optional, Union, List

import gym
import numpy as np
from gym.core import RenderFrame, ObsType, ActType
from gym.spaces import Box, Dict, MultiDiscrete
from gym.spaces import Tuple

from sequence_game.helper_functions import get_number_of_cards_for_players, get_card_positions_on_board, \
    generate_the_card_deck_and_index, get_one_indices_from_given_data, get_zero_indices_from_given_data, \
    fill_locations_with_ones, get_number_of_sequences_to_build, get_all_positions

CORNER_LOCATIONS = [(0, 0), (0, 9), (9, 0), (9, 9)]
ALL_POSITIONS = get_all_positions()


class SequenceEnvironment(gym.Env):

    def __init__(self, players, invalid_move_reward, sequence_length_reward_multiplier, final_sequence_reward):

        self.formed_sequences = None
        self.final_sequence_reward = final_sequence_reward
        self.sequence_length_reward_multiplier = sequence_length_reward_multiplier
        self.invalid_move_reward = invalid_move_reward
        self.state = None
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
        self.move_to_initial_state()

    def get_current_players_observation(self):
        """
        Gets the observation of the game from current player's POV

        :returns
        one giant matrix containing all the player's positions on the board and the players hand positions
        """
        players_hand = self.state["hand_positions"][self.current_player]
        total_observation = np.concatenate(self.state['player_board_positions'], players_hand)
        return total_observation

    def update_observation_for_regular_cards(self, position_placed):

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
        # remove from all other players card positions
        other_player_board_positions = self.state['other_player_board_positions']
        for player in range(self.players):
            other_player_board_positions[player] = 0

        # update the entire board
        entire_board_positions = self.state['board_occupied_locations']
        entire_board_positions[position_placed] = 0

    def step(self, action: ActType):
        # check if the action is a valid action for the current player
        card_played, row, col = action
        position_placed = (row, col)
        is_one_eyed_jack = self.state['is_card_one_eyed_jack'][self.current_player][card_played]
        if not self.is_action_valid(is_one_eyed_jack, position_placed):
            return self.get_current_players_observation(), self.invalid_move_reward, True, {
                'reason': 'player played an invalid move'}

        # update the player's hand, give them the next card
        next_card = self.get_next_card()
        entire_board_positions = self.state['board_occupied_locations']
        possible_locations = self.get_valid_locations_for_card(next_card, entire_board_positions,
                                                               self.state['other_player_board_positions'][
                                                                   self.current_player])
        card_positions = np.zeros((10, 10))
        card_positions[possible_locations] = 1
        players_hand = self.state['hand_positions'][self.current_player]
        players_hand[card_played] = card_positions

        length_factor = self.check_for_sequences_formed(position_placed)
        number_of_sequences_so_far = len(self.formed_sequences[self.current_player])
        reward = self.sequence_length_reward_multiplier ** length_factor + self.final_sequence_reward ** number_of_sequences_so_far

        # move to the next player
        self.current_player += 1
        self.current_player = self.current_player % self.players
        end = number_of_sequences_so_far == self.max_sequences_to_build
        return self.get_current_players_observation(), reward, end, {
            'reason': 'Max Sequences Formed' if end else 'Game continues'}

    def render(self) -> Optional[Union[RenderFrame, List[RenderFrame]]]:
        pass

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[dict] = None,
    ) -> Tuple[ObsType, dict]:
        self.move_to_initial_state()
        return self.state

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
        self.card_deck, self.card_action_index = generate_the_card_deck_and_index()
        self.card_positions_on_board = get_card_positions_on_board()
        self.player_cards = self.distribute_the_cards()

        # initializing an empty board
        player_board_positions = np.zeros((self.players, 10, 10))
        board_occupied_locations = np.zeros((10, 10))  # entire board is empty at the start
        self.state['board_occupied_locations'] = board_occupied_locations

        # everyone gets corners by default
        player_board_positions = fill_locations_with_ones(player_board_positions, CORNER_LOCATIONS)
        self.state['player_board_positions'] = player_board_positions

        # other's positions are empty for everyone initially
        others_positions = np.zeros((10, 10))
        self.state['other_player_board_positions'] = np.zeros((self.players, 10, 10))

        hand_positions = np.zeros((self.players, self.number_of_cards_per_player, 10, 10))
        for player, cards in self.player_cards.items():
            player_hand_position = hand_positions[player]
            for index, card in enumerate(cards):
                placeable_card_locations = self.get_valid_locations_for_card(card, board_occupied_locations,
                                                                             others_positions)
                # marking all the locations this card can be placed at
                player_hand_position[placeable_card_locations] = 1
        self.state['hand_positions'] = hand_positions

    def get_valid_locations_for_card(self, card_number, entire_board_occupied_locations, others_card_locations):
        """
        Looks at the current board positions and returns a list of positions where the card can be placed on the
        board, returns empty list if card cannot be placed anywhere
        """
        if card_number < 0:
            return []

        if card_number < 48:  # specific positions are possible for cards, except jacks
            card_locations_possible = self.card_positions_on_board[card_number]
        else:  # all locations are possible for jacks
            card_locations_possible = ALL_POSITIONS

        # filter positions based what can actually be filled
        if card_number == 49:  # one eyed jack can be placed only on already filled locations
            fillable_positions = get_one_indices_from_given_data(card_locations_possible, others_card_locations)
        else:
            fillable_positions = get_zero_indices_from_given_data(card_locations_possible,
                                                                  entire_board_occupied_locations)

        # remove corner locations since we can never place cards there
        corner_mask = np.isin(fillable_positions, CORNER_LOCATIONS)
        fillable_positions = fillable_positions[~corner_mask]
        return fillable_positions

    def is_action_valid(self, is_one_eyed_jack, position_placed):
        """
        Returns True if the given action is valid else False
        """
        if position_placed in CORNER_LOCATIONS:
            return False

        if position_placed in self.sequence_locations:
            return False

        board_occupied_locations = self.state['board_occupied_locations']
        if is_one_eyed_jack:
            # return if it is an empty location
            return board_occupied_locations[position_placed] == 0
        else:
            # check if there is someone else's card at the position
            current_player_locations = self.state['player_board_positions'][self.current_player]
            other_locations = board_occupied_locations - current_player_locations
            return other_locations[position_placed] == 1

    def get_next_card(self):
        try:
            next_card = next(self.card_deck)
            return next_card
        except:
            return -1

    def check_for_sequences_formed(self, position):
        """
        Get the maximum length of a sequence of 1 that can be formed given a matrix containing 1 and 0,
        a position where the new 1 is to be placed, a list of 2d indices which cannot be included in the sequence.
        The sequence can be horizontal or vertical or diagonal.

        Parameters:
            matrix (numpy.ndarray): The input matrix.
            position (tuple): A tuple representing the position where the new 1 is to be placed.
            sequences (list(list(tuple))): a list of existing sequences

        Returns:
            int: The maximum length of a sequence of 1 that can be formed in each direction multiplied

        """
        matrix = self.state['player_board_positions'][self.current_player]
        sequences = self.formed_sequences[self.current_player]
        # Define the directions to search in
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (-1, 1), (1, -1)]
        points_in_the_direction = [[] for _ in range(8)]

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

            while 0 <= i < matrix.shape[0] and 0 <= j < matrix.shape[1] and matrix[i][j] == 1:
                if (i, j) in sequences:
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
            self.formed_sequences[self.current_player].append(vertical_sequence_points)
        if horizontal_length >= 5:
            horizontal_sequence_points = points_in_the_direction[2] + points_in_the_direction[3][1:]
            self.formed_sequences[self.current_player].append(horizontal_sequence_points)
        if top_left_to_bottom_right_length >= 5:
            top_left_to_bottom_right_points = points_in_the_direction[4] + points_in_the_direction[5][1:]
            self.formed_sequences[self.current_player].append(top_left_to_bottom_right_points)
        if bottom_left_to_top_right_length >= 5:
            bottom_left_to_top_right_points = points_in_the_direction[6] + points_in_the_direction[7][1:]
            self.formed_sequences[self.current_player].append(bottom_left_to_top_right_points)

        return max(vertical_length, horizontal_length, top_left_to_bottom_right_length, bottom_left_to_top_right_length)
