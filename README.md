# RL project

This project implements and compares TD Control and Function Approximation agents (Q-Learning, SARSA, and Deep Q-Networks) on a custom warehouse environment. The goal is to sort crates onto their designated goal positions in a grid-based warehouse, avoiding obstacles and optimizing for reward (and speed).

## Structure

- [`warehouse_env.py`](warehouse_env.py): defines the `WarehouseEnv` gymnasium environment with state representation, step logic, reward structure, and rendering.
- [`agents.py`](agents.py): implements RL agents.
  - `TDControl_Agent`: Tabular Q-Learning or SARSA agent (togglable).
  - `DQN_Agent`: Deep Q-Network agent.
  - Utility functions for state conversion, evaluation, and agent visualization.
- [`plot_utils.py`](plot_utils.py): visualization utilities for plotting:
  - Q-values and policy maps for agents.
  - Reward and steps trajectories.
  - Success rate over training.
- [`main.ipynb`](main.ipynb): main notebook for running experiments, training agents, and visualizing results.
- `results/`: Contains saved plots and experiment results for different agent configurations.

## Environment

- Grid-based warehouse with configurable rows, columns, and number of crates.
- Crates must be moved to goal positions in column-major order.
- Obstacles include walls, frozen crates (already sorted), and inactive crates.
- Rewards:
  - +20 for successfully sorting a crate.
  - -1 for valid movement.
  - -10 for invalid moves (collisions).

## RL Agents

### TD Control Agent(s)
- Toggle between Q-Learning and SARSA updates.
- Uses a defaultdict for Q-table storage due to large state space.

### DQN Agent
- Uses a convolutional neural network to estimate Q-values.
- Experience replay buffer + target network implementation

## Visualization

- Policy and Q-value heatmaps for selected crate contexts.
- Reward and steps trajectories over episodes.
- Success rate plots for agent evaluation.

## Usage

1. **Setup**: install dependencies (see below).
2. **Run Experiments** in [`main.ipynb`](main.ipynb).
3. **View Results**: plots and metrics are saved in the `results/` directory.

## Dependencies

- Python 3.11+
- `numpy`
- `matplotlib`
- `torch` (ROCm 6.3)
- `gymnasium`
- `IPython` (for notebook display utilities)

Install with:

```sh
pip install numpy matplotlib torch gymnasium ipython
```

Torch for AMD ROCm was used:
```sh
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.3
```

---
