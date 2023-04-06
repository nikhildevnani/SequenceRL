import numpy as np
import torch

from my_agent import SequenceTwoPlayerQNetwork

model = SequenceTwoPlayerQNetwork(2)
sample_data_hand = torch.Tensor(np.random.randint(2, size=(2, 9, 10, 10)))
sample_data_jacks = torch.Tensor(np.random.randint(2, size=(2, 7)))

sample_output = model(sample_data_hand, sample_data_jacks)
