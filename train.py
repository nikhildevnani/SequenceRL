import time

import torch

from my_agent import SequenceAgent
from sequence_game.sequence_evironment import SequenceEnvironment

torch.set_default_dtype(torch.float64)

# Initialize the DRL agents and other variables
num_episodes = 10
lr = 0.01  # Learning rate for the optimizer
gamma = 0.99  # Discount factor for the Bellman equation
epsilon = 0.1  # Epsilon for epsilon-greedy exploration
number_of_players = 2
update_target_frequency = 20  # updates the target network after these many steps
batch_size = 32
agents = [SequenceAgent(number_of_players, lr=lr, gamma=gamma, epsilon=epsilon, batch_size=2) for _ in
          range(number_of_players)]
env = SequenceEnvironment(number_of_players, -1000, 5, 1000)


def train(num_episodes, max_steps_per_episode=100, update_target_freq=10):
    start_time = time.time()
    train_start_time = start_time
    for episode in range(num_episodes):
        state = env.reset()
        total_reward_for_agent = [0] * len(agents)
        done = False
        for step in range(max_steps_per_episode):
            for index, agent in enumerate(agents):
                action = agent.select_action(state)
                next_state, reward, done, _ = env.step(action)
                total_reward_for_agent[index] += reward
                agent.store_transition(state, action, reward, next_state, done)
                state = next_state
                agent.update_q_network()
                if step % update_target_freq == 0:
                    agent.update_target_q_network()

                if done:
                    break
            if done:
                break
        if episode % 100 == 0:
            end_time = time.time() - start_time
            start_time = time.time()
            print(
                "Episode: {}, Total Steps: {}, Total Reward: {}, Total Time: {}, Time Taken: {}".format(episode + 1,
                                                                                                        step + 1,
                                                                                                        total_reward_for_agent,
                                                                                                        start_time - train_start_time,
                                                                                                        end_time))

        if episode % 500 == 499:  # write model after every 500 episodes to disk

            for index, agent in enumerate(agents):
                agent.write_model_to_disk(index)


train(20000, 100, 2)
