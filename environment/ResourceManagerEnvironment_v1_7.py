import gym
from gym import spaces
import numpy as np
import pygame
import os

#add render mode for images

class ResourceManagerEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 15}


    def __init__(self, grid_size=20, render_mode=None, initial_water=100, num_water_resources=10, window_size=300):

        #initialize the reward
        self.total_reward = 0

        #Define Grid Size
        self.grid_size = grid_size
        #calculate window size based on grid size - check if needed
        self.window_size = int(window_size + 50 * (grid_size - 10))
        self.text_height = 100

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

        self.observation_space = spaces.Box(low=0, high=1, shape=(grid_size, grid_size), dtype=np.float32)
        #add water as resource
        self.initial_water = initial_water
        self.max_water_resources = self.initial_water * 2
        self.num_water_resources = num_water_resources

        #reset function will set positions only during the first call
        self.initial_reset_done = False

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window = None
        self.clock = None

    def reset(self, seed=None):
        super().reset(seed=seed)

        # ***** static option *****
        if not self.initial_reset_done:
            self.agent_position = self.np_random.integers(low=0, high=self.grid_size, size=(2,))
            self.initial_agent_position = np.copy(self.agent_position)

            self.water_positions = []
            for _ in range(self.num_water_resources):
                water_position = self.np_random.integers(low=0, high=self.grid_size, size=(2,))
                while np.any(water_position == self.agent_position) or any(np.all(water_position == pos) for pos in self.water_positions):
                    water_position = self.np_random.integers(low=0, high=self.grid_size, size=(2,))
                self.water_positions.append(np.copy(water_position))

            self.initial_water_positions = [np.copy(pos) for pos in self.water_positions]

            self.initial_reset_done = True

        self.agent_position = np.copy(self.initial_agent_position)
        self.water_positions = [np.copy(pos) for pos in self.initial_water_positions]


        #initialize reward and water
        self.total_reward = 0
        self.water_resource = self.initial_water

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

        #add water to the observation
        observation += self.water_resource / self.initial_water

        #alternative, mark it with W
        #for water_pos in self.water_positions:
        #    observation[tuple(water_pos)] = "W"

        return observation
    
    def get_info(self):
        #Initialize info
        info = {
            'agent_position': self.agent_position,
            'total_reward': self.total_reward,
            'water_resource': self.water_resource
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
        #move the agent in that direction
        self.agent_position = np.clip(
            self.agent_position + direction,
            0,
            self.grid_size - 1
        )
        #check if the agent's position has changed
        position_changed = not np.all(self.agent_position == original_position)

        # ***** Termination Logic *****
        #define when done, use terminated as term as it is excpected in gym
        terminated = self.water_resource <= 0
        if terminated:
            reward = -20
            terminated = True

        # ***** Reward Function  *****
        #calculate Manhattan distance between agent and water resources
        distances_to_water = [np.abs(self.agent_position[0] - water_pos[0]) + np.abs(self.agent_position[1] - water_pos[1]) for water_pos in self.water_positions]
        min_distance_to_water = min(distances_to_water)

        #Implementation of water depletion
        water_depletion_penalty = 0.5 
        if position_changed:
            #deduct water resource when the agent moves
            self.water_resource -= water_depletion_penalty

        if position_changed:
            #the agent has taken a step
            reward = -1  
        else:
            #idea: the agent didn't move, so give a -10 reward
            #should be higher than water depletion penalty
            reward = -5  
        
        #give reward to the agent if it is close to any water resource
        if min_distance_to_water == 1:
            reward += 4  #+4 reward when immediately around any water resource
        elif min_distance_to_water <= 4:
            reward += 2  #+2 reward when within 16 fields around any water resource

        # ***** Water Logic *****

        #check if agent is on a water resource, give reward when not over max water resource
        is_on_water = any(np.all(self.agent_position == pos) for pos in self.water_positions)
        refill_value = 5
        if is_on_water and self.water_resource < self.max_water_resources:
            #refill 5 units of water
            self.water_resource += refill_value
            reward += 20
        if is_on_water and self.water_resource + refill_value > self.max_water_resources:
            #give penalty when over max water resource
            reward -= 10
        

        self.total_reward += reward

        #stop the episode when reward is specific or less
        if self.total_reward < -1000:
            terminated = True

        state = self.get_obs()
        info = self.get_info()

        if self.render_mode == "human":
            self.render_frame()

        #set truncated to false as it is not needed
        truncated = False
        done = truncated or terminated 


        return state, reward, done, truncated, info
    
    def render(self):
            if self.render_mode == "human":
                return self.render_frame()

    def render_frame(self):
            if self.window is None and self.render_mode == "human":
                pygame.init()
                pygame.display.init()
                self.window = pygame.display.set_mode((self.window_size, self.window_size + self.text_height))
            if self.clock is None and self.render_mode == "human":
                self.clock = pygame.time.Clock()
            canvas = pygame.Surface((self.window_size, self.window_size + self.text_height))
            canvas.fill((255, 255, 255))
            pix_square_size = self.window_size / self.grid_size

            # ***** Load Images *****
            #load the images
            water_image = pygame.image.load(os.path.join("images", "water.jpg"))
            agent_image = pygame.image.load(os.path.join("images", "player_neutral.jpg"))
    
            #resize images to match the square size
            water_image = pygame.transform.scale(water_image, (int(pix_square_size), int(pix_square_size)))
            agent_image = pygame.transform.scale(agent_image, (int(pix_square_size), int(pix_square_size)))
            #draw water cells (blue squares)
            for water_pos in self.water_positions:
                canvas.blit(water_image, (water_pos * pix_square_size))
            #draw the agent
            canvas.blit(agent_image, (self.agent_position * pix_square_size))

            #draw gridlines
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

            #add text to display water resource and total reward
            font = pygame.font.Font(None, 36)
            water_text = font.render(f"Water: {self.water_resource}", True, (0, 0, 0))
            reward_text = font.render(f"Reward: {self.total_reward}", True, (0, 0, 0))

            #draw text on the canvas
            text_total_height = water_text.get_height() + reward_text.get_height() + 10
            vertical_position = (self.window_size + self.text_height - text_total_height)

            canvas.blit(water_text, (10, vertical_position))
            canvas.blit(reward_text, (10, vertical_position + water_text.get_height() + 10))


            if self.render_mode == "human":
                # Copy drawings from `canvas` to the visible window
                self.window.blit(canvas, canvas.get_rect())
                pygame.event.pump()
                pygame.display.update()
                # Ensure human-rendering occurs at the predefined framerate
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