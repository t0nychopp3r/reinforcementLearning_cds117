import numpy as np
from collections import deque
import random

class QLearningAgent:
    def __init__(self, env, learning_rate=0.1, discount_factor=0.9, exploration_prob=1.0,
                 epsilon_decay=0.995, replay_buffer_size=1000, batch_size=32, q_table=None):
        self.env = env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_prob = exploration_prob
        self.epsilon_decay = epsilon_decay
        self.replay_buffer_size = replay_buffer_size
        self.batch_size = batch_size
        self.grid_size = env.grid_size

        self.state_space_size = np.prod(env.observation_space.shape)
        self.action_space_size = env.action_space.n

        #initialize Q-table with zeros or use the one provided
        #Q table is flattened, means there are grid_size * grid_size arrays of action_space_size length
        if q_table is not None:
            self.q_table = q_table
        else:
            self.q_table = np.zeros((self.state_space_size, self.action_space_size))

        #initialize replay buffer
        self.replay_buffer = deque(maxlen=replay_buffer_size)

    def select_action(self, state):
        #epsilon-greedy policy
        if np.random.rand() < self.exploration_prob:
            #explore
            #print("exploring")
            return self.env.action_space.sample()
        else:
            #exploit
            #print("exploiting")
            return np.argmax(self.q_table[np.argmax(state[0]==1)])

    def update_q_table(self, state, action, reward, next_state):
        # Q-learning update rule
        #print("action:", action)
        # print the whole q table for debugging
        #print("q_table:", self.q_table)
        #print("state:", state)
        
        agent_position = state[1]['agent_position']
        flat_index = np.ravel_multi_index(agent_position, (self.grid_size, self.grid_size))

        #use the agent's position as an index in the Q-table
        current_q_value = self.q_table.flat[flat_index * self.action_space_size + action]

        #extract agent's position from the next state
        next_agent_position = next_state[1]['agent_position']
        next_flat_index = np.ravel_multi_index(next_agent_position, (self.grid_size, self.grid_size))
        next_q_values = self.q_table.flat[next_flat_index * self.action_space_size:
                                          (next_flat_index + 1) * self.action_space_size]
        max_next_q_value = np.max(next_q_values)

        # Q-learning update rule
        '''
        new_q_value = (1 - self.learning_rate) * current_q_value + \
                      self.learning_rate * (reward + self.discount_factor * max_next_q_value)
        '''

        new_q_value = current_q_value + self.learning_rate * (reward + self.discount_factor * max_next_q_value - current_q_value)
              
        # Update the Q-value in the Q-table
        self.q_table.flat[flat_index * self.action_space_size + action] = new_q_value

        # add the experience to the replay buffer
        self.replay_buffer.append((state, action, reward, next_state))
    
        # decay exploration probability
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
