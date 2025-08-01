import numpy as np
import gymnasium as gym
import random
import copy
from enum import IntEnum

class Actions(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

class WarehouseEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 1}

    def __init__(self, rows, cols, num_crates, reward=20):
        super().__init__()

        self.rows = rows
        self.cols = cols
        self.num_crates = num_crates
        self.reward = reward

        # Set the goals in column-major order
        self.goal_positions = self._fill_column_major(self.rows, self.num_crates)

        # State: active crates + crate positions
        self.observation_space = gym.spaces.Dict({
            "active_crate_id": gym.spaces.Discrete(self.num_crates),
            "crate_positions": gym.spaces.Box(
                low=0,
                high=max(self.rows, self.cols),
                shape=(self.num_crates, 2),
                dtype=np.int32
            )
        })

        # 4 actions: up, down, left, right
        self.action_space = gym.spaces.Discrete(len(Actions))

        self.reset()


    def __len__(self):
        """
        returns the theoretical state space size: C x ( (NxM)! / (NxM-C)! )
        """
        import math

        C = self.num_crates
        tiles = self.rows * self.cols

        return C * math.perm(tiles, C)


    def _fill_column_major(self, num_rows, num_elements):
        """
        Returns the indices, as a list, of the elements that are going to be filled in column-major order
        a, b, c
        d, e, f   ->   num_elements = 2   ->   returns [[0,0], [1,0]] (a and d)
        g, h, i 
        """
        active_cols = (num_elements // num_rows) if num_elements % num_rows == 0 else (num_elements // num_rows + 1)
        positions = [[i, j] for j in range(active_cols) for i in range(num_rows)]
        positions = positions[:num_elements]
        return positions


    def _get_obs(self):
        """
        The state representation to be used by the agent
        """
        active_id = self.current_crate_index
    
        if active_id >= self.num_crates:
            active_id = self.num_crates - 1
            
        return {
            'active_crate_id': np.int32(active_id),
            'crate_positions': np.array(self.crates, dtype=np.int32)
        }


    def reset(self, seed=None):
        super().reset(seed=seed)

        self.frozen_crates = []
        self.remaining_goals = copy.deepcopy(self.goal_positions)
        self.current_crate_index = 0

    # Candidate spawn points for crates (avoid goal positions)
        spawn_candidates = [
            [i, j] for i in range(self.rows) for j in range(self.cols)
            if [i, j] not in self.goal_positions
        ]
        all_crates = random.sample(spawn_candidates, self.num_crates) # pick spawns at random

    # Assign closest crate to each goal (goal -> crate), in column-major goal order
        assigned_crates = set()     # just to keep track of what we have already assigned
        crate_assignments = [None] * self.num_crates

        for i_g, goal in enumerate(self.goal_positions):
            min_dist = float('inf')
            chosen_idx = -1

            for i_c, crate in enumerate(all_crates):
                if i_c in assigned_crates:
                    continue    # skip if the crate was already assigned

                dist = abs(crate[0] - goal[0]) + abs(crate[1] - goal[1])    # Manhattan distance
                if dist < min_dist:
                    min_dist = dist
                    chosen_idx = i_c

            assigned_crates.add(chosen_idx)
            crate_assignments[i_g] = all_crates[chosen_idx]

        self.crates = crate_assignments
        self.current_crate_goal = self.goal_positions[0]

        return self._get_obs(), {}
        

    def step(self, action):
        #print(f"Current crate goal: {self.current_crate_goal}")
        terminated = False
        truncated = False

        crate_pos = self.crates[self.current_crate_index]
        x, y = crate_pos

        move  = {
            Actions.UP: (-1, 0), 
            Actions.DOWN: (1, 0), 
            Actions.LEFT: (0, -1), 
            Actions.RIGHT: (0, 1)
        }[action]
        
        new_x, new_y = x + move[0], y + move[1]

    # Collision checks

        # Two types of obstacles aside from walls:
        ## 1. Crates that have already reached a goal and have become frozen
        ## 2. Crates that have not reached a goal yet and are NOT currently being controlled,
        ##    i.e. the ones that have been spawned and are stationary, waiting for their turn
        obstacles = self.frozen_crates + [
            pos for i, pos in enumerate(self.crates) if i != self.current_crate_index
        ]

        if (
        # Wall collisions
            not (0 <= new_x < self.rows and 0 <= new_y < self.cols) or
        # Obstacle collisions 
            [new_x, new_y] in obstacles
        ):
            return self._get_obs(), -10, False, False, {}
        
    # Valid move:
    
        # Update active crate location
        self.crates[self.current_crate_index] = [new_x, new_y]

        # If we land on the correct goal:
        if [new_x, new_y] == self.current_crate_goal:

            self.frozen_crates.append([new_x, new_y])
            self.remaining_goals.remove(self.current_crate_goal)
            self.current_crate_index += 1

            # If we finished all the crates: terminate, else continue with the next crate
            if self.current_crate_index >= self.num_crates:
                terminated = True
                return self._get_obs(), self.reward, terminated, truncated, {}
            
            else:
                self.current_crate_goal = self.goal_positions[self.current_crate_index]
                return self._get_obs(), self.reward, terminated, truncated, {}

        # else we just made a movement-only step (no goal, no obstacles)
        return self._get_obs(), -1, terminated, truncated, {}


    def render(self):
        grid = np.full((self.rows, self.cols), ' . ', dtype='<U3')  # allow multi-char labels

        for r, c in self.goal_positions:
            if [r, c] == self.current_crate_goal:
                grid[r, c] = '[G]'
            else:
                grid[r, c] = ' G '

        for r, c in self.frozen_crates:
            grid[r, c] = ' X '

        for i, (r, c) in enumerate(self.crates):
            if i == self.current_crate_index:
                grid[r, c] = f" A{i+1}"
            elif [r, c] not in self.frozen_crates:
                grid[r, c] = f" a{i+1} "
                
        print("\n".join(" ".join(row) for row in grid))
        print()