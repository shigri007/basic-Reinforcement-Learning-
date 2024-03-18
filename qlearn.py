import numpy as np

# Maze dimensions
NUM_ROWS = 5
NUM_COLS = 5

# Actions
ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3

# Initialize maze with rewards (0 for regular cells, 100 for treasure cell)
maze = np.zeros((NUM_ROWS, NUM_COLS))
maze[NUM_ROWS - 1, NUM_COLS - 1] = 100  # Treasure location

# Initialize Q-values
q_values = np.zeros((NUM_ROWS, NUM_COLS, 4))  # 4 actions for each cell

# Learning parameters
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.9
NUM_EPISODES = 1000

# Exploration parameters
EPSILON = 0.1  # Exploration probability

# Helper function to choose action using epsilon-greedy strategy
def choose_action(state):
    if np.random.uniform(0, 1) < EPSILON:
        return np.random.choice(4)  # Random action
    else:
        return np.argmax(q_values[state[0], state[1]])

# Q-learning algorithm
for _ in range(NUM_EPISODES):
    state = [0, 0]  # Start at top-left corner
    while state != [NUM_ROWS - 1, NUM_COLS - 1]:  # Continue until treasure is reached
        action = choose_action(state)
        next_state = state.copy()

        if action == ACTION_UP:
            next_state[0] = max(0, state[0] - 1)
        elif action == ACTION_DOWN:
            next_state[0] = min(NUM_ROWS - 1, state[0] + 1)
        elif action == ACTION_LEFT:
            next_state[1] = max(0, state[1] - 1)
        elif action == ACTION_RIGHT:
            next_state[1] = min(NUM_COLS - 1, state[1] + 1)

        # Update Q-value using Bellman equation
        q_values[state[0], state[1], action] += LEARNING_RATE * (
                maze[next_state[0], next_state[1]] + DISCOUNT_FACTOR * np.max(q_values[next_state[0], next_state[1]]) -
                q_values[state[0], state[1], action])
        state = next_state

# Print learned Q-values
print("Learned Q-values:")
print(q_values)
