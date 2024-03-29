# basic-Reinforcement-Learning- Maze QLearn

# Problem Statement

This Python script demonstrates Q-learning, a reinforcement learning technique, to solve a maze problem with a treasure located at the bottom-right corner.

# Description

The script uses a Q-learning algorithm to find an optimal policy for navigating through a maze to reach the treasure. The maze is represented as a grid with specified dimensions. The agent can move in four directions: up, down, left, and right. Each cell in the maze has a reward value, with the treasure cell having a reward of 100. The Q-values represent the expected cumulative reward for taking a specific action in a particular state.

# Parameters

- `NUM_ROWS`: Number of rows in the maze.
- `NUM_COLS`: Number of columns in the maze.
- `ACTION_UP`, `ACTION_DOWN`, `ACTION_LEFT`, `ACTION_RIGHT`: Constants representing actions.
- `LEARNING_RATE`: The rate at which the Q-values are updated during learning.
- `DISCOUNT_FACTOR`: Discount factor for future rewards.
- `NUM_EPISODES`: Number of episodes for training.
- `EPSILON`: Exploration probability for epsilon-greedy strategy.

# Output

The script prints the learned Q-values after training.


# Run
python qlearn.py
