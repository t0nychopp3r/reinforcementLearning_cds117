import gym
from gym import spaces
import numpy as np
import pygame

#Create Gym Environment for Resource Manager
#The environment is a 2D grid with 4 possible actions: up, down, left, right
#The agent can move in any direction but cannot move outside the grid

class ResourceManagerEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}


    def __init__(self, grid_size=10, render_mode=None):

        #initialize the reward
        self.total_reward = 0

        #Define Grid Size
        self.grid_size = grid_size
        self.window_size = 500

        #Action Space:
        #0: Right, 1: up, 2: left, 3: down

        self.action_space = spaces.Discrete(4)

        #Map the action to the corresponding movement
        self.action_to_direction = {
            0: np.array([1, 0]), #right
            1: np.array([0, 1]), #up
            2: np.array([-1, 0]), #left
            3: np.array([0, -1]), #down
        }

        #Observation Space:
        #The observation space is a 2D grid with the agent's position marked as 1
        #and the rest of the grid marked as 0

        self.observation_space = spaces.Box(low=0, high=1, shape=(grid_size, grid_size), dtype=np.float32)
        self.reset()

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window = None
        self.clock = None



    #Needed for Environment Reset
    def reset(self, seed=None):
        super().reset(seed=seed)

        #Choose the agent's initial position at random
        self.agent_position = self.np_random.integers(low=0, high=self.grid_size, size=(2,))

        #Set the target position at random until it is different from the agent's position
        self.target_position = self.agent_position
        while np.all(self.target_position == self.agent_position):
            self.target_position = self.np_random.integers(low=0, high=self.grid_size, size=(2,))

        self.total_reward = 0

        observation = self.get_obs()
        info = self.get_info()

        if self.render_mode == "human":
            self.render_frame()

        return observation, info
    
    def get_obs(self):
        #Initialize observation
        observation = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)

        #Mark the agent's position
        observation[tuple(self.agent_position)] = 1
        return observation
    
    def get_info(self):
        #Initialize info
        info = {
            'agent_position': self.agent_position,
            'target_position': self.target_position,
            'total_reward': self.total_reward
        }
        return info
    
    def step(self, action):

        # ***** Move around the grid *****

        #store the agent's position before taking a step
        original_position = np.copy(self.agent_position)
        #make sure the action is valid
        if action not in [0, 1, 2, 3]:
            raise ValueError(f"Invalid action {action}. Action should be in the range [0, 1, 2, 3].")
        
        #choose a direction
        direction = self.action_to_direction[action]
        #Move the agent in that direction
        self.agent_position = np.clip(
            self.agent_position + direction,
            0,
            self.grid_size - 1
        )
        #check if the agent's position has changed
        position_changed = not np.all(self.agent_position == original_position)

        #define when done, use terminated as term as it is excpected in gym
        terminated = np.all(self.agent_position == self.target_position)

        # ***** Reward Function *****

        #calculate Manhatten distance between agent and target
        distance_to_target = np.abs(self.agent_position[0] - self.target_position[0]) + np.abs(self.agent_position[1] - self.target_position[1])


        if terminated:
            reward = 10  #the agent has reached the target
        elif position_changed:
            reward = -1  #the agent has taken a step
        else:
            #idea: implement already a reward if the agent did not move
            reward = -10  #the agent didn't move, so give a -10 reward
        
        #give reward to the agent if it is close to the target
        if distance_to_target == 1:
            reward += 2  # +2 reward when immediately around the target
        elif distance_to_target <= 4:
            reward += 1  # +1 reward when within 16 fields around the target

        self.total_reward += reward

        #stop the episode when reward is -100 or less
        if self.total_reward < -200:
            terminated = True
            self.reset()

        state = self.get_obs()
        info = self.get_info()

        if self.render_mode == "human":
            self.render_frame()

        #set truncated to false as it is not needed
        truncated = False
        done = truncated or terminated 


        return state, reward, done, truncated, info
    
    def render(self):
            if self.render_mode == "rgb_array":
                return self.render_frame()
    
    def render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()
        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.grid_size
        )  # The size of a single grid square in pixels
        # First we draw the target
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self.target_position,
                (pix_square_size, pix_square_size),
            ),
        )
        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self.agent_position + 0.5) * pix_square_size,
            pix_square_size / 3,
        )
        # Finally, add some gridlines
        for x in range(self.grid_size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )
        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
        
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
        self.window = None