import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from helper_functions import get_number_of_cards_for_players, get_a_valid_move
from replay_buffer import ReplayBuffer


class SequenceTwoPlayerQNetwork(nn.Module):
    def __init__(self, number_of_players):
        super(SequenceTwoPlayerQNetwork, self).__init__()
        self.number_of_players = number_of_players
        self.number_of_cards = get_number_of_cards_for_players(number_of_players)
        self.idx = torch.cartesian_prod(torch.arange(7), torch.arange(10), torch.arange(10))
        self.conv1 = nn.Conv2d(in_channels=9, out_channels=32, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(64 * 2 * 2, 128)
        self.fc2 = nn.Linear(128 + self.number_of_cards, 64)
        self.row_layer_1 = nn.Linear(64, 32)
        self.row_layer_2 = nn.Linear(32, 16)
        self.row_layer_3 = nn.Linear(16, 10)
        self.col_layer_1 = nn.Linear(64, 32)
        self.col_layer_2 = nn.Linear(32, 16)
        self.col_layer_3 = nn.Linear(16, 10)
        self.hand_layer_1 = nn.Linear(64, 32)
        self.hand_layer_2 = nn.Linear(32, 16)
        self.hand_layer_3 = nn.Linear(16, self.number_of_cards)

    def forward(self, board_and_hand_positions, is_card_one_eyed_jack):
        num_inputs = is_card_one_eyed_jack.shape[0]
        # Apply the convolutional layers
        board_and_hand_positions = F.relu(self.conv1(board_and_hand_positions))
        board_and_hand_positions = self.pool(F.relu(self.conv2(board_and_hand_positions)))
        # Flatten the output from the convolutional layers
        board_and_hand_positions = board_and_hand_positions.view(-1, 64 * 2 * 2)
        # Apply the linear layers
        board_and_hand_positions = F.relu(self.fc1(board_and_hand_positions))
        board_and_hand_positions = torch.cat([board_and_hand_positions, is_card_one_eyed_jack], dim=1)
        board_and_hand_positions = F.relu(self.fc2(board_and_hand_positions))

        row_layer_1_output = self.row_layer_1(board_and_hand_positions)
        row_layer_2_output = self.row_layer_2(row_layer_1_output)
        row_layer_3_output = self.row_layer_3(row_layer_2_output)

        col_layer_1_output = self.col_layer_1(board_and_hand_positions)
        col_layer_2_output = self.col_layer_2(col_layer_1_output)
        col_layer_3_output = self.col_layer_3(col_layer_2_output)

        hand_layer_1_output = self.hand_layer_1(board_and_hand_positions)
        hand_layer_2_output = self.hand_layer_2(hand_layer_1_output)
        hand_layer_3_output = self.hand_layer_3(hand_layer_2_output)

        result = torch.zeros(num_inputs, self.idx.shape[0])

        for i in range(self.idx.shape[0]):
            result[:, i] = hand_layer_3_output[:, self.idx[i, 0]] * row_layer_3_output[:,
                                                                    self.idx[i, 1]] * col_layer_3_output[:,
                                                                                      self.idx[i, 2]]
        return result


class DQNAgent:
    def __init__(self, number_of_players, lr=0.001, gamma=0.99, epsilon=0.1, batch_size=10):
        self.q_network = SequenceTwoPlayerQNetwork(number_of_players)
        self.target_q_network = SequenceTwoPlayerQNetwork(number_of_players)
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.loss_function = nn.MSELoss()
        self.gamma = gamma
        self.epsilon = epsilon
        self.replay_buffer = ReplayBuffer(100)
        self.batch_size = batch_size

    # Function to select an action based on epsilon-greedy exploration
    def select_action(self, state):
        player_board_positions, player_hand_positions, is_card_one_eyed_jack = state
        if torch.rand(1) < self.epsilon:
            with torch.no_grad():
                return get_a_valid_move(player_hand_positions)
        else:
            board_and_hand_positions = torch.cat([player_board_positions, player_hand_positions], dim=0)
            board_and_hand_positions = board_and_hand_positions.unsqueeze(0)
            is_card_one_eyed_jack = is_card_one_eyed_jack.unsqueeze(0)
            with torch.no_grad():
                q_values = self.q_network(board_and_hand_positions, is_card_one_eyed_jack)
                best_action = self.filter_invalid_actions(q_values, player_hand_positions)
                return best_action // 100, (best_action // 10) % 10, best_action % 10

    # Function to update the Q-network based on the Bellman equation
    def update_q_network(self):

        if len(self.replay_buffer) < self.batch_size:
            return
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        actions = torch.IntTensor([self.get_index_for_action(x) for x in actions])
        actions = actions.type(torch.int64)
        states_0, states_1, states_2 = states
        next_states_0, next_states_1, next_states_2 = next_states
        states_0 = torch.cat([states_0, states_1], dim=1)
        next_states_0 = torch.cat([next_states_0, next_states_1], dim=1)
        q_values = self.q_network(states_0, states_2).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_q_network(next_states_0, next_states_2).max(1)[0].detach()
        targets = rewards + self.gamma * next_q_values * ~dones
        targets = targets.unsqueeze(1)

        loss = F.mse_loss(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def get_index_for_action(self, action):
        return (action[0] * 100 + action[1] * 10 + action[2]).item()

    def update_target_q_network(self):
        self.target_q_network.load_state_dict(self.q_network.state_dict())

    def store_transition(self, state, action, reward, next_state, done):
        transition = ((state[0], state[1], state[2]), torch.tensor(action), torch.tensor(reward),
                      (next_state[0], next_state[1], next_state[2]), torch.tensor(done))
        self.replay_buffer.push(transition)

    def write_model_to_disk(self, player_number):
        base_dir = './models'
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        torch.save({
            'model_state_dict': self.target_q_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': 10
        }, self.get_model_path(player_number))

    def get_model_path(self, player_number):
        base_dir = './models'
        model_path = os.path.join(base_dir, f'model_player{player_number}.pth')
        return model_path

    def read_model_from_disk(self, player_number):
        torch.load(self.get_model_path(player_number))

    def filter_invalid_actions(self, q_values, player_hand_positions):
        mask = player_hand_positions.flatten() == 1
        max_indices = torch.argmax(q_values.masked_fill(~mask, float('-inf')))
        return max_indices.item()
