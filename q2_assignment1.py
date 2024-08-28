import numpy as np
import matplotlib.pyplot as plt

# grid world dimensions
grid_size = 9 # 9x9

# rewards matrix
rewards = np.zeros((grid_size, grid_size))
rewards[8, 8] = 1  # Goal position with reward +1

# tunnel locations
tunnel_in = (2, 2)
tunnel_out = (6, 6)

# blockages
blockages = [(3, 1), (3, 2), (3, 3), (2, 3), (1, 3), 
             (5, 5), (5, 6), (5, 7), (5, 8), (6, 5), 
             (7, 5), (8, 5)]

# actions
actions = ['up', 'down', 'left', 'right']
action_vectors = {
    'up': (-1, 0),
    'down': (1, 0),
    'left': (0, -1),
    'right': (0, 1)
}

#  Value Iteration function
def value_iteration(grid_size, rewards, gamma=0.9, theta=1e-6):
    V = np.zeros((grid_size, grid_size))
    policy = np.zeros((grid_size, grid_size), dtype=int)

    while True:
        delta = 0
        for i in range(grid_size):
            for j in range(grid_size):
                if (i, j) in blockages:
                    continue
                
                old_v = V[i, j]
                q_values = []

                for action in actions:
                    next_i = i + action_vectors[action][0]
                    next_j = j + action_vectors[action][1]

                    # Tunnel logic
                    if (i, j) == tunnel_in:
                        next_i, next_j = tunnel_out

                    if 0 <= next_i < grid_size and 0 <= next_j < grid_size and (next_i, next_j) not in blockages:
                        q_values.append(rewards[next_i, next_j] + gamma * V[next_i, next_j])
                    else:
                        q_values.append(rewards[i, j] + gamma * V[i, j])

                V[i, j] = max(q_values)
                policy[i, j] = np.argmax(q_values)
                delta = max(delta, abs(old_v - V[i, j]))

        if delta < theta:
            break

    return V, policy

# Policy Iteration function
def policy_iteration(grid_size, rewards, gamma=0.9):
    V = np.zeros((grid_size, grid_size))
    policy = np.random.choice(len(actions), size=(grid_size, grid_size))

    def policy_evaluation(policy, V, gamma=0.9, theta=1e-6):
        while True:
            delta = 0
            for i in range(grid_size):
                for j in range(grid_size):
                    if (i, j) in blockages:
                        continue
                    
                    old_v = V[i, j]
                    action = actions[policy[i, j]]
                    next_i = i + action_vectors[action][0]
                    next_j = j + action_vectors[action][1]

                    # Tunnel logic
                    if (i, j) == tunnel_in:
                        next_i, next_j = tunnel_out

                    if 0 <= next_i < grid_size and 0 <= next_j < grid_size and (next_i, next_j) not in blockages:
                        V[i, j] = rewards[next_i, next_j] + gamma * V[next_i, next_j]
                    else:
                        V[i, j] = rewards[i, j] + gamma * V[i, j]

                    delta = max(delta, abs(old_v - V[i, j]))

            if delta < theta:
                break

        return V

    while True:
        policy_stable = True
        V = policy_evaluation(policy, V, gamma)

        for i in range(grid_size):
            for j in range(grid_size):
                if (i, j) in blockages:
                    continue
                
                old_action = policy[i, j]
                q_values = []

                for action in actions:
                    next_i = i + action_vectors[action][0]
                    next_j = j + action_vectors[action][1]

                    # Tunnel logic meaning that when the robot enters into the tunnel it directly comes out of the tunnel 
                    if (i, j) == tunnel_in:
                        next_i, next_j = tunnel_out

                    if 0 <= next_i < grid_size and 0 <= next_j < grid_size and (next_i, next_j) not in blockages:
                        q_values.append(rewards[next_i, next_j] + gamma * V[next_i, next_j])
                    else:
                        q_values.append(rewards[i, j] + gamma * V[i, j])

                policy[i, j] = np.argmax(q_values)

                if old_action != policy[i, j]:
                    policy_stable = False

        if policy_stable:
            break

    return V, policy

# Value Iteration
V_vi, policy_vi = value_iteration(grid_size, rewards)

# Policy Iteration
V_pi, policy_pi = policy_iteration(grid_size, rewards)

# Visualization Function
def plot_policy(policy, title):
    action_dict = {
        0: (-1, 0),  # up
        1: (1, 0),   # down
        2: (0, -1),  # left
        3: (0, 1)    # right
    }
    
    X, Y = np.meshgrid(np.arange(grid_size), np.arange(grid_size))
    U = np.zeros_like(X, dtype=float)
    V = np.zeros_like(Y, dtype=float)

    for i in range(grid_size):
        for j in range(grid_size):
            if (i, j) in blockages:
                continue
            U[i, j], V[i, j] = action_dict[policy[i, j]]

    plt.figure(figsize=(7, 7))
    plt.quiver(Y, X, V, U)
    plt.title(title)
    # plt.xlim(-0.5, grid_size - 0.5)
    # plt.ylim(-0.5, grid_size - 0.5)
    # plt.gca().invert_yaxis()
    # plt.show()
    plt.gca().set_aspect('equal')
    # plt.gca().invert_yaxis()
    plt.grid()

    # Annotate the key locations
    plt.text(0, 0, 'Start', ha='center', va='center', color='blue', fontsize=12)
    plt.text(tunnel_in[1], tunnel_in[0], 'IN', ha='center', va='center', color='red', fontsize=12)
    plt.text(tunnel_out[1], tunnel_out[0], 'OUT', ha='center', va='center', color='green', fontsize=12)
    plt.text(8, 8, 'Goal', ha='center', va='center', color='purple', fontsize=12)
    plt.show()


# Plot the optimal policies
plot_policy(policy_vi, "Optimal Policy - Value Iteration")
plot_policy(policy_pi, "Optimal Policy - Policy Iteration")
