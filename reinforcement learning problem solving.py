import numpy as np
import gym
import random

# Create the Taxi environment
env = gym.make("Taxi-v3")

# Initialize the Q-table
q_table = np.zeros((env.observation_space.n, env.action_space.n))

# Hyperparameters
alpha = 0.1  # Learning rate
gamma = 0.6  # Discount factor
epsilon = 0.1  # Exploration rate
num_episodes = 10000  # Number of episodes

# Training the agent
for i in range(num_episodes):
    state = env.reset()
    done = False
    while not done:
        # Choose action using epsilon-greedy policy
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # Explore action space
        else:
            action = np.argmax(q_table[state])  # Exploit learned values

        # Take action and observe the result
        next_state, reward, done, _ = env.step(action)

        # Update Q-value using the Bellman equation
        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[state, action] = new_value

        # Transition to the next state
        state = next_state

# Evaluate the agent's performance
total_epochs, total_penalties = 0, 0
episodes = 100

for _ in range(episodes):
    state = env.reset()
    epochs, penalties, reward = 0, 0, 0
    done = False

    while not done:
        action = np.argmax(q_table[state])  # Choose the best action
        state, reward, done, _ = env.step(action)

        if reward == -10:  # Penalty for illegal actions
            penalties += 1
        epochs += 1

    total_penalties += penalties
    total_epochs += epochs

print(f"Results after {episodes} episodes:")
print(f"Average timesteps per episode: {total_epochs / episodes}")
print(f"Average penalties per episode: {total_penalties / episodes}")
