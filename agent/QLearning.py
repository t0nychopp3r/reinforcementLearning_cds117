import numpy as np
from collections import deque
import random

class QLearningAgent:
    def __init__(self, env, learning_rate=0.1, discount_factor=0.9, exploration_prob=1.0,
                 epsilon_decay=0.995, replay_buffer_size=1000, batch_size=32, num_bins=10):
        self.env = env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_prob = exploration_prob
        self.epsilon_decay = epsilon_decay
        self.replay_buffer_size = replay_buffer_size
        self.batch_size = batch_size
        self.num_bins = num_bins

        self.state_space_size = np.prod(env.observation_space.shape)
        self.action_space_size = env.action_space.n

        #initialize Q-table with zeros
        self.q_table = np.zeros((self.state_space_size, self.action_space_size))

        #initialize replay buffer
        self.replay_buffer = deque(maxlen=replay_buffer_size)
    
    def discretize_state(self, state):
        state_array = state[0] if isinstance(state, tuple) else state

        discretized_state = []
        for feature_values in state_array:
            bin_edges = np.linspace(np.min(feature_values), np.max(feature_values), self.num_bins + 1)
            discretized_feature = np.digitize(feature_values, bin_edges)
            discretized_state.append(discretized_feature)
        return discretized_state

    def select_action(self, state):
        #epsilon-greedy policy
        if np.random.rand() < self.exploration_prob:
            #explore
            return self.env.action_space.sample()
        else:
            #exploit
            state_discrete = self.discretize_state(state)
            return np.argmax(self.q_table[state_discrete])

    def update_q_table(self, state, action, reward, next_state):
        state_discrete = self.discretize_state(state)
        next_state_discrete = self.discretize_state(next_state)

        #Q-learning update rule
        #print("action:", action)
        #print the whole q table for debugging
        #print("q_table:", self.q_table)

        current_q_value = self.q_table[state_discrete, action]
        max_next_q_value = np.max(self.q_table[next_state_discrete])
        new_q_value = (1 - self.learning_rate) * current_q_value + \
                      self.learning_rate * (reward + self.discount_factor * max_next_q_value)

        self.q_table[state_discrete, action] = new_q_value

        #add the experience to the replay buffer
        self.replay_buffer.append((state, action, reward, next_state))

        #decay exploration probability
        self.exploration_prob = self.exploration_prob * self.epsilon_decay
        #ensure exploration probability is not less than 0.1
        self.exploration_prob = max(0.1, self.exploration_prob * self.epsilon_decay)
    def sample_from_replay_buffer(self):
        # Sample a batch of experiences from the replay buffer
        return random.sample(self.replay_buffer, self.batch_size)
'''
    def train(self, num_episodes=1000):
        for episode in range(num_episodes):
            state = self.env.reset()

            total_reward = 0
            done = False

            while not done:
                action = self.select_action(state)
                next_state, reward, done, _, _ = self.env.step(action)

                self.update_q_table(state, action, reward, next_state)

                state = next_state
                total_reward += reward

            # Decay exploration probability
            self.exploration_prob *= self.epsilon_decay

            print(f"Episode {episode + 1}, Total Reward: {total_reward}")
'''
