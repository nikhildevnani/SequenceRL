import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MyNet(nn.Module):
    def __init__(self, batch_size=None):
        super(MyNet, self).__init__()
        self.batch_size = batch_size
        # Define the convolutional layers
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

    def forward(self, x, b):
        # Apply the convolutional layers
        print(x.shape, b.shape)
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        # Flatten the output from the convolutional layers
        x = x.view(-1, 64 * 2 * 2)
        # Apply the linear layers
        x = F.relu(self.fc1(x))
        x = torch.cat([x, b], dim=1)
        x = F.relu(self.fc2(x))
        print(x.shape)
        row_layer_1_output = self.row_layer_1(x)
        row_layer_2_output = self.row_layer_2(row_layer_1_output)
        row_layer_3_output = self.row_layer_3(row_layer_2_output)

        col_layer_1_output = self.col_layer_1(x)
        col_layer_2_output = self.col_layer_2(col_layer_1_output)
        col_layer_3_output = self.col_layer_3(col_layer_2_output)

        hand_layer_1_output = self.hand_layer_1(x)
        hand_layer_2_output = self.hand_layer_2(hand_layer_1_output)
        hand_layer_3_output = self.hand_layer_3(hand_layer_2_output)

        return row_layer_3_output, col_layer_3_output, hand_layer_3_output


sample_data_jacks = torch.Tensor(np.random.randint(2, size=(2, 9, 10, 10)))
sample_data_hand = torch.Tensor(np.random.randint(2, size=(2, 7)))
# print(sample_data_jacks, sample_data_hand)

model = MyNet()
x1, x2, x3 = model.forward(sample_data_jacks, sample_data_hand)

print(x1.shape, x2.shape, x3.shape)
