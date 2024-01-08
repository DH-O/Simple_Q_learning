# code source from https://3months.tistory.com/173

import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
import env.simple_gym as simple_gym
import util.utils as utils

discount = 0.99
alpha = 0.5
num_episodes = 150
reward_list = [] #  Total reward per episode
episode_length_list = [] # Total steps per episode
X_size = 10
Y_size = 10

env = simple_gym.TwoDimArrayMap(X_size, Y_size, 4)
env.SimpleAntMazation()

# Q Table reset
Q = np.zeros([env.observation_space_dim, env.action_space_dim])
V = np.zeros(env.observation_space_dim)

for episode in range(num_episodes):
    state = env.reset()
    rAll = 0
    done = False
    episode_length = 0
    while not done:
        action = np.argmax(Q[state, :] + np.random.randn(1, env.action_space_dim) / (episode + 1))  # E-greedy with decaying randomness
        
        new_state, reward, done = env.step(action)
        
        Q[state, action] = Q[state, action] + alpha * (reward + discount * np.max(Q[new_state, :]) - Q[state, action])
        V[state] = np.max(Q[state, :])
        
        rAll += reward
        state = new_state
        episode_length += 1
    reward_list.append(rAll)    # It should be just "1"
    episode_length_list.append(episode_length)

OneLineToCell_V = utils.OneLineToCell(X_size, Y_size)
V_table = OneLineToCell_V.FillGridByOneLineArray(V)

plt.plot(range(len(episode_length_list)), episode_length_list, color="blue")
plt.title("Episode Length per Episode")
plt.xlabel("Episode")
plt.ylabel("Episode Length")

now = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
path = f'./results/result_{now}'
if not os.path.isdir(path):
    os.makedirs(path)
plt.savefig(f'{path}/episode_length.png')
np.savetxt(f'{path}/SimpleMaze_table.txt', env.maze, fmt='%d')
np.savetxt(f'{path}/V_table.txt', V_table, fmt='%.3f')