import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []

    def push(self, transition):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append(transition)

    def sample(self, batch_size):
        batch = np.random.choice(len(self.buffer), batch_size, replace=False)
        states0, states1, states2, actions, rewards, next_states0, next_states1, next_states2, dones = [], [], [], [], [], [], [], [], []
        for index in batch:
            state, action, reward, next_state, done = self.buffer[index]
            states0.append(state[0])
            states1.append(state[1])
            states2.append(state[2])
            actions.append(action)
            rewards.append(reward)
            next_states0.append(next_state[0])
            next_states1.append(next_state[1])
            next_states2.append(next_state[2])
            dones.append(done)
        states0 = torch.stack(states0, dim=0)
        states1 = torch.stack(states1, dim=0)
        states2 = torch.stack(states2, dim=0)
        actions = torch.stack(actions)
        rewards = torch.stack(rewards)
        next_states0 = torch.stack(next_states0, dim=0)
        next_states1 = torch.stack(next_states1, dim=0)
        next_states2 = torch.stack(next_states2, dim=0)
        dones = torch.stack(dones)
        states = (states0, states1, states2)
        next_states = (next_states0, next_states1, next_states2)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)
