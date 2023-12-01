import numpy as np


#Separate Node Class
class Node:
    def __init__(self, state, parent=None, action=None):
        #state of the environment
        self.state = state
        #parent node
        self.parent = parent
        #action that led to this node
        self.children = []
        #number of times this node was visited
        self.visits = 0
        #total reward of this node
        self.value = 0.0
        #water resource of this node
        self.water_resource = 0.0
        #action that led to this node
        self.action = action

class MonteCarloTreeSearch:
    def __init__(self, env, iterations=1000, simulation_steps=10):
        #load the environment
        self.env = env
        #create a root node to start the tree
        self.root = Node(state=env.reset())
        #number of times to run the simulation
        self.iterations = iterations
        #number of steps to run the simulation
        self.simulation_steps = simulation_steps
    
    # Select Function: Select the node with the highest UCB1 value
    # UCB1: Algorithm for Multi-Armed Bandit Problem 
    def select(self, node):
        while node.children:
            valid_children = [child for child in node.children if self.env.action_space.contains(child.action)]
            if valid_children:
                # UCB1 algorithm --> https://www.turing.com/kb/guide-on-upper-confidence-bound-algorithm-in-reinforced-learning
                node = max(valid_children, key=lambda child: (child.value / (child.visits + 1e-4)) + self.exploration_bonus(node, child))
            else:
                # If no valid child node is found, return the current node
                print("No valid children found!")
                return node
            #print(f"Selected action: {node.action}, Visits: {node.visits}, Value: {node.value}, Exploration Bonus: {self.exploration_bonus(node, node)}")
        return node


    #exploration bonus: calculate an exploration bonus based on the water resources
    def exploration_bonus(self, parent, child):
        exploration_weight = 0.1 / (child.visits + 1) #weight parameter to priotize exploring
        #water resource bonus term based on the water resource of the child node
        water_bonus = child.water_resource * exploration_weight
        #UCB1 bonus term for visitation count exploration
        visit_bonus = np.sqrt(np.log(parent.visits + 1) / (child.visits + 1e-4))
        return water_bonus + visit_bonus
    
    #Expand Function: Expand the tree by adding a new node
    def expand(self, node):
        actions = np.arange(self.env.action_space.n)
        new_node = None
        for action in actions:
            try:
                #unpack the results of the step function
                state,  reward, terminated, truncated, info = self.env.step(action)
                new_state = self.env.get_obs()
                #create a new node, with the new state, parent node, and action
                new_node = Node(state=new_state, parent=node, action=action)
                #add the new node to the parent node
                node.children.append(new_node)
                #set water resource for the new node
                new_node.water_resource = self.env.water_resource
            except Exception as e:
                print(f"Error in expand function for action {action}: {e}")
        return new_node

    #Simulate Function: Simulate the environment for a given number of steps
    def simulate(self, node):
        total_reward = 0
        for _ in range(self.simulation_steps):
            action = self.env.action_space.sample()
            #needed for troubleshooting
            #result = self.env.step(action)
            #print(result)
            state, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated:
                self.env.reset()
                break
        return total_reward
    
    #Get Best Action Function: Get the best action from the root node
    def get_best_action(self):
        if not self.root.children:
            return np.random.randint(4)  #if no children, return a random action

        # Select the action with the highest value
        best_action = max(self.root.children, key=lambda child: child.value).action
        return best_action

    #Backpropagate Function: Update the value and visit count of each node in the path
    def backpropagate(self, node, reward):
        while node is not None:
            node.visits += 1
            node.value += reward
            node = node.parent

    #Search Function: Run the Monte Carlo Tree Search algorithm
    def search(self, action=None):
        for _ in range(self.iterations):
            selected_node = self.select(self.root)
            new_node = self.expand(selected_node)
            simulation_result = self.simulate(new_node)
            #state, reward, done = self.env.step(action)  # Ignore info during expansion
            #new_state = self.env.get_obs()  # Assuming get_obs provides the new state
            #new_node = Node(state=new_state, parent=selected_node, action=action)
            self.backpropagate(new_node, simulation_result)
        best_action = self.get_best_action()
        return best_action

