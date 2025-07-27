import matplotlib.pyplot as plt
import numpy as np
from td_control_agent import state_to_key
from collections import defaultdict

def plot_qvalues_slice(agent, env, fixed_context, save_path=None):
    """
    Visualizza i Q-values per ogni singola azione su 4 grafici separati,
    per una specifica "fetta" dello spazio degli stati.
    """
    rows, cols = env.rows, env.cols
    active_id = fixed_context['active_id']
    action_names = ["UP", "DOWN", "LEFT", "RIGHT"]
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(f"Per-action Q-Values (active crate: A{active_id+1})", fontsize=16)

    obstacle_positions = list(fixed_context['fixed_positions'].values())

    for action_idx, action_name in enumerate(action_names):
        # Griglia per i valori di questa specifica azione
        grid_values = np.zeros((rows, cols))

        for r in range(rows):
            for c in range(cols):
                # Se la cella (r,c) è un ostacolo, la saltiamo
                if (r, c) in obstacle_positions:
                    grid_values[r, c] = np.nan # Usiamo NaN per le celle inaccessibili
                    continue

                # Costruisci lo stato completo in modo robusto
                current_positions = [None] * env.num_crates
                current_positions[active_id] = (r, c)
                for crate_id, pos in fixed_context['fixed_positions'].items():
                    current_positions[crate_id] = pos
                
                state = {
                    'active_crate_id': active_id,
                    'crate_positions': np.array(current_positions, dtype=np.int32)
                }
                s_key = state_to_key(state)

                # Estrai il Q-value solo per l'azione corrente
                q_value = agent.Qvalues.get(s_key, np.zeros(4))[action_idx]
                grid_values[r, c] = q_value
        
        # Plotta la heatmap per l'azione corrente
        ax = axes[action_idx]
        im = ax.imshow(grid_values, cmap='plasma', interpolation='nearest')
        ax.set_title(f"Action: {action_name}")
        fig.colorbar(im, ax=ax)

        # Aggiungi marker per goal e ostacoli per contesto
        goal_pos = env.goal_positions[active_id]
        ax.scatter(goal_pos[1], goal_pos[0], marker='*', color='gold', s=200, edgecolor='black', label='Goal')
        for pos in obstacle_positions:
            ax.scatter(pos[1], pos[0], marker='s', color='gray', s=150, edgecolor='black', label='Obstacle')
            
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)

    plt.show()


def plot_policy_slice(agent, env, fixed_context, save_path=None):
    """
    Funzione universale per plottare la mappa della policy per qualsiasi agente
    che implementi il metodo get_q_values(state).

    Args:
        agent: Un'istanza di SARSA_Agent o DQNAgent.
        env: Un'istanza di WarehouseEnv.
        fixed_context (dict): Dizionario che definisce lo scenario da visualizzare.
                              Es: {'active_id': 0, 'fixed_positions': {1: (r,c), ...}}
    """
    # --- 1. Setup Iniziale ---
    rows, cols = env.rows, env.cols
    active_id = fixed_context['active_id']
    
    max_q_values = np.full((rows, cols), -np.inf)
    best_actions = np.zeros((rows, cols), dtype=int)
    
    # Definisci le posizioni degli ostacoli fissi per questo scenario
    obstacle_positions = list(fixed_context['fixed_positions'].values())

    for r in range(rows):
        for c in range(cols):

            current_positions = [None] * env.num_crates
            
            # 1. Imposta la posizione della cassa ATTIVA che stiamo analizzando (r, c)
            current_positions[active_id] = (r, c)

            # 2. Controlla se la posizione (r, c) è già occupata da un ostacolo fisso
            is_obstacle = False
            for crate_id, pos in fixed_context['fixed_positions'].items():
                if (r, c) == pos:
                    is_obstacle = True
                    break # Inutile continuare, la cella è occupata
            
            if is_obstacle:
                continue

            # 3. Se la cella è libera, popola le posizioni delle altre casse fisse
            for crate_id, pos in fixed_context['fixed_positions'].items():
                current_positions[crate_id] = pos
            
            state = {
                'active_crate_id': active_id,
                'crate_positions': np.array(current_positions, dtype=np.int32)
            }
            
            q_vals = agent.get_q_values(state)
            
            # Memorizza l'azione migliore e il suo valore
            if q_vals is not None and np.any(q_vals):
                 max_q_values[r,c] = np.max(q_vals)
                 best_actions[r, c] = np.argmax(q_vals)

    # --- 3. Plotting con Matplotlib ---
    fig, ax = plt.subplots(figsize=(6, 4))

    ax.set_title(f"Optimal policy for crate A{active_id+1} ({agent.__class__.__name__})")
    ax.set_xticks(np.arange(cols))
    ax.set_yticks(np.arange(rows))
    ax.invert_yaxis()   # Inverte l'asse y per matchare imshow

    # Disegna i marcatori per il goal e gli ostacoli
    goal_pos = env.goal_positions[active_id]
    ax.scatter(goal_pos[1], goal_pos[0], marker="*", color="gold", s=400, edgecolor="black", zorder=3, label=f"Crate A{active_id+1} goal")
    
    cmap = plt.get_cmap("tab10")
    for crate_id, pos in fixed_context['fixed_positions'].items():
        color = cmap(crate_id % 10)  # Loop colors if >10 crates
        ax.scatter(pos[1], pos[0], marker='s', color=color, s=200, edgecolor="black", zorder=3, label=f'Frozen crate A{crate_id+1}')

    # Dizionario per mappare le azioni alle direzioni delle frecce
    action_arrows = {
        0: (0, -0.4),  # UP (dy=-1)
        1: (0, 0.4),   # DOWN (dy=+1)
        2: (-0.4, 0),  # LEFT (dx=-1)
        3: (0.4, 0)    # RIGHT (dx=+1)
    }

    # Disegna una freccia per ogni cella in base all'azione migliore
    for r in range(rows):
        for c in range(cols):
            # Disegna la freccia solo se abbiamo un Q-value valido per quella cella
            if max_q_values[r,c] > -np.inf:
                action = best_actions[r, c]
                dx, dy = action_arrows[action]
                ax.quiver(c, r, dx, dy, angles='xy', scale_units='xy', scale=1, color='black', headwidth=5, width=0.005)

    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)

    plt.show()


def plot_steps_rewards(n_episodes, trajectory, display_mode="reward", save_path=None):

    if display_mode in ["reward", "steps"]:
        if display_mode == "reward":
            y_label = "Reward"
            color = "orange"
        else:
            y_label = "Steps"
            color = "blue"

    fig, axs = plt.subplots(1, 2, figsize=(13.5,5))

    axs[0].plot(np.arange(50, n_episodes), trajectory[50:], color=color)
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


def plot_deltaQ(n_episodes, avg_dQ, max_dQ, interval=200, save_path=None):

    fig, axs = plt.subplots(1, 2, figsize=(15,5))

    axs[0].scatter(np.arange(0, n_episodes, interval), avg_dQ[::interval], marker='o', color="green")
    axs[0].set_xlabel("Episodes")
    axs[0].set_ylabel("Average deltaQ")
    axs[0].grid(True)

    axs[1].scatter(np.arange(0, n_episodes, interval), max_dQ[::interval], marker='o', color="red")
    axs[1].set_xlabel("Episodes")
    axs[1].set_ylabel("Maximum deltaQ")
    axs[1].grid(True)

    fig.suptitle("DeltaQ behavior over episodes")
    
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)

    plt.show()





"""def plot_policy_slice(agent, env, fixed_context, save_path=None):
    rows, cols = env.rows, env.cols
    active_id = fixed_context['active_id']
    
    max_q_values = np.full((rows, cols), -np.inf)
    best_actions = np.zeros((rows, cols), dtype=int)

    for r in range(rows):
        for c in range(cols):

            current_positions = [None] * env.num_crates
            
            # 1. Imposta la posizione della cassa ATTIVA che stiamo analizzando (r, c)
            current_positions[active_id] = (r, c)

            # 2. Controlla se la posizione (r, c) è già occupata da un ostacolo fisso
            is_obstacle = False
            for crate_id, pos in fixed_context['fixed_positions'].items():
                if (r, c) == pos:
                    is_obstacle = True
                    break # Inutile continuare, la cella è occupata
            
            if is_obstacle:
                continue

            # 3. Se la cella è libera, popola le posizioni delle altre casse fisse
            for crate_id, pos in fixed_context['fixed_positions'].items():
                current_positions[crate_id] = pos

            state = {
                'active_crate_id': active_id,
                'crate_positions': np.array(current_positions)
            }
            s_key = state_to_key(state)

            if s_key in agent.Qvalues:
                q_vals = agent.Qvalues[s_key]
                if np.any(q_vals): # Assicurati che i q-values non siano tutti zero
                    max_q_values[r, c] = np.max(q_vals)
                    best_actions[r, c] = np.argmax(q_vals)


    fig, ax = plt.subplots(figsize=(6, 4))

    ax.set_title(f'Optimal policy for crate A{active_id+1}')
    ax.set_xticks(np.arange(cols))
    ax.set_yticks(np.arange(rows))
    ax.invert_yaxis() # Inverte l'asse y per matchare imshow
    
    goal_pos = env.goal_positions[active_id]
    ax.scatter(goal_pos[1], goal_pos[0], marker='*', color='gold', s=400, zorder=3, label=f'Crate A{active_id+1} goal')

    cmap = plt.get_cmap("tab10")
    for crate_id, pos in fixed_context['fixed_positions'].items():
        color = cmap(crate_id % 10)  # Loop colors if >10 crates
        ax.scatter(pos[1], pos[0], marker='s', color=color, s=200, zorder=3, label=f'Frozen crate A{crate_id+1}')

    action_arrows = {
        0: (0, -0.4),  # UP
        1: (0, 0.4),   # DOWN
        2: (-0.4, 0),  # LEFT
        3: (0.4, 0)    # RIGHT
    }
    
    for r in range(rows):
        for c in range(cols):
            if max_q_values[r,c] > -np.inf:
                action = best_actions[r, c]
                dx, dy = action_arrows[action]
                ax.quiver(c, r, dx, dy, angles='xy', scale_units='xy', scale=1, color='black', headwidth=4)

    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)

    plt.show()"""