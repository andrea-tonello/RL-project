import matplotlib.pyplot as plt
import numpy as np
from warehouse_env import Actions

def plot_qvalues_slice(agent, env, fixed_context, save_path=None):
    """
    For each action, plots Q-values for a selected active crate (4 plots total).
    The selection is specified in fixed_context, a state dictionary with
    - the active_crate id and
    - the positions of the other crates

    Works with every agent that implements a "get_q_values" method that returns 
    four q-values (one per action), given a state.
    """
    rows, cols = env.rows, env.cols
    action_names = ["UP", "DOWN", "LEFT", "RIGHT"]
    active_id = fixed_context['active_id']
    obstacle_positions = list(fixed_context['fixed_positions'].values())
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(f"Per-action Q-Values (active crate: A{active_id+1})", fontsize=16)

    # For each action:
    for action_idx, action_name in enumerate(action_names):

        # Grid to be filled with q-values
        grid_values = np.zeros((rows, cols))

        for r in range(rows):
            for c in range(cols):
                
                # Skip tiles with obstacles
                if (r, c) in obstacle_positions:
                    grid_values[r, c] = np.nan
                    continue

                # Create an appropriate state (according to the input) 
                current_positions = [None] * env.num_crates
                current_positions[active_id] = (r, c)
                for crate_id, pos in fixed_context['fixed_positions'].items():
                    current_positions[crate_id] = pos
                
                state = {
                    'active_crate_id': active_id,
                    'crate_positions': np.array(current_positions, dtype=np.int32)
                }

                # Get the q-value for such state, for the specified action only
                q_value = agent.get_q_values(state)[action_idx]
                grid_values[r, c] = q_value
        
        # Heatmap for the action
        ax = axes[action_idx]
        im = ax.imshow(grid_values, cmap='plasma', interpolation='nearest')
        ax.set_title(f"Action: {action_name}")
        fig.colorbar(im, ax=ax)

        # Goal and obstacles markers
        goal_pos = env.goal_positions[active_id]
        ax.scatter(goal_pos[1], goal_pos[0], marker='*', color='gold', s=200, edgecolor='black')
        for pos in obstacle_positions:
            ax.scatter(pos[1], pos[0], marker='s', color='gray', s=150, edgecolor='black')
            
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)

    plt.show()


def plot_policy_slice(agent, env, fixed_context, save_path=None):
    """    
    Plots the optimal policy map for a selected active crate. 
    The selection is specified in fixed_context, a state dictionary with
    - the active_crate id and
    - the positions of the other crates

    Works with every agent that implements a "get_q_values" method that returns 
    four q-values (one per action), given a state
    """
    rows, cols = env.rows, env.cols
    active_id = fixed_context['active_id']
    obstacle_positions = list(fixed_context['fixed_positions'].values())

    max_q_values = np.full((rows, cols), -np.inf)
    best_actions = np.zeros((rows, cols), dtype=int)

    for r in range(rows):
        for c in range(cols):

            # Skip tiles with obstacles
            if (r, c) in obstacle_positions:
                continue

            # Create an appropriate state (according to the input) 
            current_positions = [None] * env.num_crates
            current_positions[active_id] = (r, c)
            for crate_id, pos in fixed_context['fixed_positions'].items():
                current_positions[crate_id] = pos
            
            state = {
                'active_crate_id': active_id,
                'crate_positions': np.array(current_positions, dtype=np.int32)
            }
            
            # Get the q-values for such state
            q_vals = agent.get_q_values(state)
            
            # With given q_values, find the best action accordingly
            if q_vals is not None and np.any(q_vals):
                 max_q_values[r, c] = np.max(q_vals)
                 best_actions[r, c] = np.argmax(q_vals)


    fig, ax = plt.subplots(figsize=(6, 4))

    ax.set_title(f"Optimal policy for crate A{active_id+1} ({agent.__class__.__name__})")
    ax.set_xticks(np.arange(cols))
    ax.set_yticks(np.arange(rows))
    ax.invert_yaxis()   # Invert y-axis to match gridworld indexing

    # Goal and obstacles markers
    goal_pos = env.goal_positions[active_id]
    ax.scatter(goal_pos[1], goal_pos[0], marker="*", color="gold", s=400, edgecolor="black", zorder=3, label=f"Crate A{active_id+1} goal")
    
    cmap = plt.get_cmap("tab10")
    for crate_id, pos in fixed_context['fixed_positions'].items():
        color = cmap(crate_id % 10)  # Loop crate colors if >10 crates
        ax.scatter(pos[1], pos[0], marker='s', color=color, s=200, edgecolor="black", zorder=3, label=f'Frozen crate A{crate_id+1}')

    # Actions dictionary to set arrow length
    action_arrows = {
        Actions.UP: (0, -0.4),
        Actions.DOWN: (0, 0.4),
        Actions.LEFT: (-0.4, 0),
        Actions.RIGHT: (0.4, 0)
    }

    # Draw an arrow on each tile according to the best possible action
    for r in range(rows):
        for c in range(cols):

            if max_q_values[r, c] > -np.inf: # only if the q-value is valid
                action = best_actions[r, c]
                dx, dy = action_arrows[action]
                ax.quiver(c, r, dx, dy, angles='xy', scale_units='xy', scale=1, color='black', headwidth=5, width=0.005)

    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)

    plt.show()


def plot_steps_rewards(n_episodes, trajectory, display_mode="reward", save_path=None):
    """    
    Plots either steps (display_mode="steps") or rewards (display_mode="reward") over episodes.
    Makes two plots: a standard one, and a rolling average one with a 50-episodes window.
    """   

    if display_mode in ["reward", "steps"]:
        if display_mode == "reward":
            y_label = "Reward"
            color = "orange"
        else:
            y_label = "Steps"
            color = "blue"

    fig, axs = plt.subplots(1, 2, figsize=(13.5,5))

    axs[0].plot(np.arange(0, n_episodes), trajectory, color=color)
    axs[0].set_xlabel("Episodes")
    axs[0].set_ylabel(y_label)
    axs[0].grid(True)

    window = 50
    rolling_avg = np.convolve(trajectory, np.ones(window)/window, mode='valid')
    axs[1].plot(rolling_avg, color=color)
    axs[1].set_xlabel("Episodes")
    axs[1].set_ylabel("Rolling Average " + y_label)
    axs[1].grid(True)

    fig.suptitle(y_label + " trajectory over episodes")
    
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)

    plt.show()


def plot_success_rate(test_indexes, success_rate_trajectory, save_path=None):
    """
    Plots success rate over episodes (at every tested index, e.g. every 2000 episodes).
    """

    plt.figure(figsize=(8, 4))
    plt.plot(test_indexes, success_rate_trajectory, marker='o')
    plt.title("Success rate over episodes")
    plt.xlabel("Test episodes (1 every 2000)")
    plt.ylabel("Success Rate (%)")
    plt.grid(True, alpha=0.5)
    plt.ylim(0, 105)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)

    plt.show()