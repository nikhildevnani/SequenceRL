import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsummary


class SequenceTwoPlayerAgent(nn.Module):
    def __init__(self):
        super(SequenceTwoPlayerAgent, self).__init__()
        self.idx = torch.cartesian_prod(torch.arange(7), torch.arange(10), torch.arange(10))
        self.conv1 = nn.Conv2d(in_channels=9, out_channels=32, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.pool = nn.MaxPool2d(kernel_size=2)
        # Define the linear layers
        self.fc1 = nn.Linear(64 * 2 * 2, 128)
        self.fc2 = nn.Linear(128 + 7, 64)
        self.row_layer_1 = nn.Linear(64, 32)
        self.row_layer_2 = nn.Linear(32, 16)
        self.row_layer_3 = nn.Linear(16, 10)
        self.col_layer_1 = nn.Linear(64, 32)
        self.col_layer_2 = nn.Linear(32, 16)
        self.col_layer_3 = nn.Linear(16, 10)
        self.hand_layer_1 = nn.Linear(64, 32)
        self.hand_layer_2 = nn.Linear(32, 16)
        self.hand_layer_3 = nn.Linear(16, 7)

    def forward(self, board_and_hand_positions, is_card_one_eyed_jack):
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

        result = torch.zeros(2, self.idx.shape[0])

        for i in range(self.idx.shape[0]):
            result[:, i] = hand_layer_3_output[:, self.idx[i, 0]] * row_layer_3_output[:,
                                                                    self.idx[i, 1]] * col_layer_3_output[:,
                                                                                      self.idx[i, 2]]
        return result


model = SequenceTwoPlayerAgent()
torchsummary.summary(model, [(9, 10, 10), (7,)])
