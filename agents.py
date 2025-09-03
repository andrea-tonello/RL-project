import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from collections import defaultdict, namedtuple, deque
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done')) # for the replay buffer

import time
from IPython.display import clear_output

###### TD Control Agent-related ######

def state_to_key(state):
    """
    Convert the state dictionary into tuple of ints to allow Q-table indexing
    """
    active_id = int(state["active_crate_id"])
    crate_positions = tuple(tuple(pos) for pos in state["crate_positions"])
    return (active_id, crate_positions)

class TDControl_Agent():
    def __init__(self, action_size, gamma=1, lr=0.01):

        self.action_size = action_size        
        self.gamma = gamma
        self.alpha = lr
        
        # Q-table is a defaultdict since the theoretical state space size can be massive
        self.Qvalues = defaultdict(lambda: np.zeros(self.action_size))  # ["key"](0,0,0,0)
    
    def single_step_update_SARSA(self, s, a, r, new_s, new_a, done):
        """
        Uses a single step to update the values, using Temporal Difference for Q values.
        Employs the EXPERIENCED action in the new state <- Q(S_new, A_new).
        """
        s_key = state_to_key(s)
        new_s_key = state_to_key(new_s)

        if done:
            deltaQ = r - self.Qvalues[s_key][a]
        else:
            # deltaQ = R + gamma*Q(new_s, new_a) - Q(s,a)
            deltaQ = r + self.gamma * self.Qvalues[new_s_key][new_a] - self.Qvalues[s_key][a]
            
        self.Qvalues[s_key][a] += self.alpha * deltaQ
    
    def single_step_update_QL(self, s, a, r, new_s, done):
        """
        Uses a single step to update the values, using Temporal Difference for Q values.
        Uses the BEST (evaluated) action in the new state <- Q(S_new, A*) = max_A Q(S_new, A).
        """
        s_key = state_to_key(s)
        new_s_key = state_to_key(new_s)

        if done:
            deltaQ =  - self.Qvalues[s_key][a]
        else:
            # deltaQ = R + gamma*max_act Q(new_s, act) - Q(s,a)
            maxQ_over_actions = np.max(self.Qvalues[new_s_key])
            deltaQ = r + self.gamma * maxQ_over_actions - self.Qvalues[s_key][a]
            
        self.Qvalues[s_key][a] += self.alpha * deltaQ

    def get_action_epsilon_greedy(self, s, eps):
        """
        Choose an action using an epsilon-greedy policy wrt the current Q(s,a).
        """
        s_key = state_to_key(s)
        ran = np.random.rand()
        
        if (ran < eps):
            # Random action with prob. epsilon
            return np.random.choice(self.action_size) 
        
        else:
            q_vals = self.Qvalues[s_key]
            best_value = np.max(q_vals)     

            # There could be actions with equal best value:
            best_actions = np.where(q_vals == best_value)[0]

            return np.random.choice(best_actions)
    
    def get_q_values(self, state):
        """
        Get q-values for the given state, 0 if not visited.
        This method is required for plot purposes in plot_utils.
        """
        s_key = state_to_key(state)
        return self.Qvalues.get(s_key, np.zeros(self.action_size))

    def get_best_action(self, s):
        """
        Get the best possible action from the Q-table (no exploration).
        """
        s_key = state_to_key(s)
        q_vals = self.Qvalues[s_key]
        
        best_action = np.argmax(q_vals)
        return best_action


###### DQN Agent-related ######

def state_to_tensor(state, env):
    """
    Convert the state dictionary into a Pytorch tensor (1, C, H, W).
    """
    rows, cols, num_crates = env.rows, env.cols, env.num_crates
    
    # We have 4 channels in C: 0=active crate coords,  1=active goal coords,
    #                          2=queued crates coords, 3=frozen crate coords (i.e. finished)
    tensor = torch.zeros((4, rows, cols))
    
    # 0
    active_id = state['active_crate_id']
    active_pos = state['crate_positions'][active_id]
    tensor[0, active_pos[0], active_pos[1]] = 1
    
    # 1
    goal_pos = env.goal_positions[active_id]
    tensor[1, goal_pos[0], goal_pos[1]] = 1

    # 2 & 3
    for i in range(num_crates):
        if i == active_id:
            continue
        
        crate_pos = state['crate_positions'][i]
        # If the crate is frozen on its goal: channel 3
        if list(crate_pos) == env.goal_positions[i]:        # need to use "list" because crate_pos == [0 0] but we want [0,0]
             tensor[3, crate_pos[0], crate_pos[1]] = 1
        # Else, it is waiting for its turn: channel 2
        else:
             tensor[2, crate_pos[0], crate_pos[1]] = 1
            
    return tensor.unsqueeze(0) # Unsqueeze adds a batch dimension for the NN (1, C, H, W)

class ReplayBuffer():
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)    # deques are more efficient than lists if appending from the left

    def push(self, *args):
        # Saves a transition
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        # Samples a batch of transitions
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class QNetwork(nn.Module):
    def __init__(self, h, w, n_actions):
        super(QNetwork, self).__init__()
        # Input: 4 channels
        self.conv1 = nn.Conv2d(4, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        
        # Calculates the output dimension after a single convolution. The formula is:
        # floor[ (size - kernel + 2*padding) / stride ] + 1
        def conv2d_size_out(size, kernel_size=3, stride=1, padding=1):
            return (size - kernel_size + 2 * padding) // stride + 1
        
        convw = conv2d_size_out(conv2d_size_out(w))     # Two calls since we apply two convolutions
        convh = conv2d_size_out(conv2d_size_out(h))
        linear_input_size = convw * convh * 32
        
        self.fc1 = nn.Linear(linear_input_size, 256)
        self.fc2 = nn.Linear(256, n_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class DQN_Agent():
    def __init__(self, env, gamma, epsilon_start, epsilon_end, epsilon_decay, lr, buffer_size, batch_size, device):
        self.env = env
        self.n_actions = env.action_space.n
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        
        self.device = device

        h, w = env.rows, env.cols
        self.policy_net = QNetwork(h, w, self.n_actions).to(device)
        self.target_net = QNetwork(h, w, self.n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval() # Target network is for evaluation only

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_size)

    def select_action(self, state_tensor):
        """
        Select an action epsilon-greedily and perform epsilon decay
        """
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        if random.random() < self.epsilon:
            return torch.tensor([[random.randrange(self.n_actions)]], device=self.device, dtype=torch.long)
        else:
            with torch.no_grad():
                # .max(1) gets the maximum value by row
                # [0] is the actual value, [1] is the index of such value, i.e. our action.
                return self.policy_net(state_tensor).max(1)[1].view(1, 1)

    def learn(self):
        """
        Perform a learning step:

        `if` experience buffer < batch size: just explore

        `else`: sample transitions from the buffer, calculate q-values (predicted and target), compute bellmann,
        calculate loss, perform a gradient descent step.
        """
        if len(self.buffer) < self.batch_size:
            return # Do not learn if the replay buffer is not at least == to the batch size: just explore

        transitions = self.buffer.sample(self.batch_size)
        batch = Transition(*zip(*transitions))      # Transition( (s1, s2), (a1, a2), ... )

        # To tensor
        state_batch = torch.cat(batch.state).to(self.device)
        action_batch = torch.cat(batch.action).to(self.device)
        reward_batch = torch.cat(batch.reward).to(self.device)
        next_state_batch = torch.cat(batch.next_state).to(self.device)
        done_batch = torch.cat(batch.done).to(self.device)

        # Predicted and target values:
        q_values = self.policy_net(state_batch).gather(1, action_batch)
        next_q_values = self.target_net(next_state_batch).max(1)[0].detach()
        
        # Compute the expected Q values with the usual formula
        # If done=True, the value is 0
        expected_q_values =  reward_batch + (self.gamma * next_q_values * (1 - done_batch))

        # Loss and gradient descent
        loss = F.smooth_l1_loss(q_values, expected_q_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def update_target_net(self):
        """
        Copy active network weights to target network
        """
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def get_q_values(self, state):
        """
        Get q-values from the network for the given state.
        This method is required for plot purposes in plot_utils.
        """
        with torch.no_grad():
            state_tensor = state_to_tensor(state, self.env).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.cpu().numpy()[0]
        
    def get_best_action(self, state):
        """
        Get best action from the network for the given state
        """
        with torch.no_grad():
            state_tensor = state_to_tensor(state, self.env).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.max(1)[1].item()


###### Auxiliary functions to test performance of both agents ######

def evaluate_success_rate(agent, env, num_test_episodes=100, max_steps_per_episode=100):
    """
    Evaluate the agent in greedy mode (exploitation only), computing the success rate
    """
    successful_episodes = 0
    
    for _ in range(num_test_episodes):
        s, _ = env.reset()
        done = False
        eval_steps = 0
        
        while not done and eval_steps < max_steps_per_episode:

            # Choose the best action according to the learned policy
            action = agent.get_best_action(s)
            
            s, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            eval_steps += 1
            
        if terminated:
            # The episode is successful iff it terminates
            successful_episodes += 1
            
    return (successful_episodes / num_test_episodes) * 100.0


def watch_agent_perform(agent, env):
    """
    Run a single episode with a trained agent and render each step.
    """
    s, _ = env.reset(seed=123)
    done = False
    episode_reward = 0
    steps = 0
    max_steps = 100

    while not done and steps < max_steps:

        clear_output(wait=True)
        env.render()
        
        # Choose the best action according to the learned policy
        action = agent.get_best_action(s)
        
        s, r, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        episode_reward += r
        steps += 1
        
        time.sleep(0.5)

    clear_output(wait=True)
    env.render()
    
    if terminated:
        print(f"\nSuccess. The agent sorted the crates in {steps} steps.")
    else:
        print(f"\nFailure. The agent wasn't able to sort the crates in {max_steps} steps.")
        
    print(f"\nTotal reward: {episode_reward}")