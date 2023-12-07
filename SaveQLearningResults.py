import os
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt


def save_results(name, initial_observation, final_q_table, episode_rewards, episode_water_resources, epsiode_food_resources,
                 learning_rate, discount_factor, exploration_prob, epsilon_decay, replay_buffer_size, batch_size):

    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")
    folder_name = f"Results_{dt_string}_{name}"
    os.mkdir(folder_name)
    #save the observation in a text file
    # Save the observation in a text file
    with open(f"{folder_name}/observation.txt", "w") as text_file:
        for element in initial_observation:
            np_element = np.array(element)
            text_file.write(np.array2string(np_element, separator=', ') + '\n')


    #save the q-table in a text file
    np.savetxt(f"{folder_name}/q_table.txt", final_q_table, fmt="%s")

    #Subfolder for the images
    os.mkdir(f"{folder_name}/Plots")
    #save the plot of the total rewards
    plt.figure(figsize=(15,5))
    plt.plot(episode_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Learning Progress')
    #plot tendency curve
    plt.plot(np.convolve(episode_rewards, np.ones((10,))/10, mode='valid'))
    #plot average
    plt.plot(np.ones(len(episode_rewards))*np.mean(episode_rewards))
    plt.plot(episode_water_resources)
    plt.plot(epsiode_food_resources)
    plt.legend(['Episode Reward', 'Tendency', 'Average', 'Water'])
    plt.savefig(f"{folder_name}/Plots/learning_progress.png")

    # Subfolder for Hyperparameters
    os.mkdir(f"{folder_name}/Hyperparameters")
    #save the hyperparameters in a text file

    with open(f"{folder_name}/Hyperparameters/hyperparameters.txt", "w") as text_file:
        text_file.write(f"learning_rate={learning_rate}\n")
        text_file.write(f"discount_factor={discount_factor}\n")
        text_file.write(f"exploration_prob={exploration_prob}\n")
        text_file.write(f"epsilon_decay={epsilon_decay}\n")
        text_file.write(f"replay_buffer_size={replay_buffer_size}\n")
        text_file.write(f"batch_size={batch_size}\n")

    print("Save Complete")    
    return
