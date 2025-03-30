import gym
import numpy as np
import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from utils import Pos_Emb

class MessengerEnv(gym.Env):
    """
    A simple grid world navigation task.
    The agent starts at a random location on a grid and has to reach a fixed target
    (the “message destination”). Actions: 0=up, 1=down, 2=left, 3=right.
    The observation is a vector: [agent_x, agent_y, target_x, target_y].
    Optionally, more complicated tasks (e.g., obstacles) can be added.
    """

    def __init__(self, grid_size=50, max_steps=500):
        super(MessengerEnv, self).__init__()
        self.grid_size = grid_size
        self.max_steps = max_steps

        # Action space: 4 discrete actions.
        self.action_space = gym.spaces.Discrete(4)
        # Observation space: agent_x, agent_y, target_x, target_y (all normalized between 0 and 1)
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32)
        self.emb = Pos_Emb()
        self.reset()


    def reset(self, iftest = None):
        # Random start for the agent and a fixed target at bottom right corner.
        self.agent_pos = np.array([random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1)])
        #self.target_pos = np.array([random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1)])
        #self.agent_pos = np.round(self.emb(self.agent_pos).mean(0))
        #self.target_pos = np.round(self.emb(self.target_pos).mean(0))
        self.target_pos = np.array([self.grid_size - 1, self.grid_size - 1])
        if iftest == 0:
            self.agent_pos=np.array([3,4])
            self.target_pos = np.array([self.grid_size - 1, self.grid_size - 1])
            self.next_star = self.target_pos
        if iftest == 1:
            self.agent_pos = self.next_star
            self.target_pos = np.array([random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1)])
        self.steps = 0
        return self._get_obs()


    def _get_obs(self):
        # Normalize positions by grid size so that values are in [0, 1]
        return np.concatenate([self.agent_pos, self.target_pos]).astype(np.float32) / (self.grid_size - 1)


    def step(self, action):
        self.steps += 1

        dis = np.linalg.norm(self.target_pos - self.agent_pos)

        # Update agent position
        if action == 0 and self.agent_pos[1] < self.grid_size - 1:  # up
            self.agent_pos[1] += 1
        elif action == 1 and self.agent_pos[1] > 0:  # down
            self.agent_pos[1] -= 1
        elif action == 2 and self.agent_pos[0] > 0:  # left
            self.agent_pos[0] -= 1
        elif action == 3 and self.agent_pos[0] < self.grid_size - 1:  # right
            self.agent_pos[0] += 1

        # Compute reward: +10 if reached target; else, negative step penalty.
        done = False
        if np.array_equal(self.agent_pos, self.target_pos):
            reward = 10.0
            done = True
        else:
            reward = -0.1

        if np.any(self.agent_pos == 0) or np.any(self.agent_pos == self.grid_size):
            reward -= 5

        if self.steps >= self.max_steps:
            done = True

        dis_next = np.linalg.norm(self.target_pos - self.agent_pos)
        if dis_next < dis:
            reward += 0.2

        return self._get_obs(), reward, done, {}




class LSTMPolicy(nn.Module):
    def __init__(self, input_dim, hidden_dim, lstm_hidden_dim, output_dim):
        """
        input_dim: Dimension of observation (here 4)
        hidden_dim: Hidden size of the first fully connected layer.
        lstm_hidden_dim: Hidden dimension for the LSTM.
        output_dim: Number of actions.
        """
        super(LSTMPolicy, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        # LSTM layer expects input as (seq_len, batch, input_size)
        self.lstm = nn.LSTM(hidden_dim, lstm_hidden_dim, batch_first=True)
        # Actor head: outputs action logits
        self.actor = nn.Linear(lstm_hidden_dim, output_dim)
        # Critic head: outputs state value
        self.critic = nn.Linear(lstm_hidden_dim, 1)


    def forward(self, x, lstm_state):
        """
        x: Input tensor shape (batch, input_dim).
           For sequence rollout, you can unsqueeze to (batch, seq_len=1, input_dim)
        lstm_state: Tuple (h, c) of LSTM hidden states.
        Returns: action logits, state value and new LSTM states.
        """
        x = self.relu(self.fc1(x))
        # Add sequence dimension: (batch, seq_len=1, hidden_dim)
        x = x.unsqueeze(1)
        lstm_out, lstm_state = self.lstm(x, lstm_state)  # lstm_out shape: (batch, 1, lstm_hidden_dim)
        lstm_out = lstm_out.squeeze(1)  # (batch, lstm_hidden_dim)
        action_logits = self.actor(lstm_out)
        state_value = self.critic(lstm_out)

        return action_logits, state_value, lstm_state


    def init_lstm_state(self, batch_size, device):
        # Initialize h and c states with zeros.
        h = torch.zeros(1, batch_size, self.lstm.hidden_size).to(device)
        c = torch.zeros(1, batch_size, self.lstm.hidden_size).to(device)
        return (h, c)



class PPOAgent(nn.Module):
    def __init__(self, env, device, hidden_dim=128, lstm_hidden_dim=128,
                 lr=3e-4, gamma=0.99, clip_eps=0.2, update_epochs=4, batch_size=32):
        super(PPOAgent, self).__init__()
        self.env = env
        self.device = device
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.update_epochs = update_epochs
        self.batch_size = batch_size

        self.input_dim = env.observation_space.shape[0]
        self.output_dim = env.action_space.n

        self.model = LSTMPolicy(self.input_dim, hidden_dim, lstm_hidden_dim, self.output_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)


    def select_action(self, obs, lstm_state):
        # Convert observation to tensor.
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)  # (1, input_dim)
        with torch.no_grad():
            logits, value, new_lstm_state = self.model(obs_tensor, lstm_state)
            probs = torch.softmax(logits, dim=-1)
            m = torch.distributions.Categorical(probs)
            action = m.sample()
            log_prob = m.log_prob(action)
        return action.item(), log_prob.item(), value.item(), new_lstm_state


    def compute_returns(self, rewards, dones, last_value):
        """
    Compute discounted returns for each timestep.
    rewards, dones: lists for one trajectory.
    last_value: estimated value for the last state.
    """
        returns = []
        R = last_value
        for r, done in zip(reversed(rewards), reversed(dones)):
            if done:
                R = 0.0
            R = r + self.gamma * R
            returns.insert(0, R)
        return returns


    def ppo_update(self, trajectories):
        """
    trajectories: dictionary with keys :
       'obs': list of observations (torch tensor)
       'actions': list of actions (torch tensor)
       'log_probs': list of log probabilities (torch tensor)
       'returns': list of returns (torch tensor)
       'values': list of state values (torch tensor)
    We perform multiple optimization epochs over the batch.
    """
        obs = torch.stack(trajectories['obs']).to(self.device)
        actions = torch.tensor(trajectories['actions']).to(self.device)
        old_log_probs = torch.tensor(trajectories['log_probs']).to(self.device)
        returns = torch.tensor(trajectories['returns']).to(self.device)
        values = torch.tensor(trajectories['values']).to(self.device)
        advantages = returns - values

        dataset = torch.utils.data.TensorDataset(obs, actions, old_log_probs, returns, advantages)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Reset the LSTM state for mini-batch updates. (For simplicity, we process one-step transitions.)
        for epoch in range(self.update_epochs):
            for batch in loader:
                batch_obs, batch_actions, batch_old_log_probs, batch_returns, batch_advantages = batch
                # For each batch, create new LSTM state with batch size.
                lstm_state = self.model.init_lstm_state(batch_obs.size(0), self.device)

                logits, state_values, _ = self.model(batch_obs, lstm_state)
                dist = torch.distributions.Categorical(logits=logits)
                new_log_probs = dist.log_prob(batch_actions)
                ratio = torch.exp(new_log_probs - batch_old_log_probs)

                # PPO surrogate loss
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = nn.MSELoss()(state_values.squeeze(-1), batch_returns)

                loss = actor_loss + 0.5 * critic_loss  # 0.5 is a coefficient for critic loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()


    def train(self, num_episodes=1000, max_steps=50, rollout_size=256):
        """
        Train the agent with PPO.
        We collect trajectories of length rollout_size (could span multiple episodes).
        """
        episode_rewards = []
        trajectories = {'obs': [], 'actions': [], 'log_probs': [], 'rewards': [], 'dones': [], 'values': []}
        total_steps = 0

        # Initialize LSTM state for rollout with batch_size=1.
        lstm_state = self.model.init_lstm_state(batch_size=1, device=self.device)
        obs = self.env.reset()
        episode_reward = 0
        for episode in range(num_episodes):
            for step in range(max_steps):
                total_steps += 1
                action, log_prob, value, lstm_state = self.select_action(obs, lstm_state)
                next_obs, reward, done, _ = self.env.step(action)

                trajectories['obs'].append(torch.FloatTensor(obs))
                trajectories['actions'].append(action)
                trajectories['log_probs'].append(log_prob)
                trajectories['rewards'].append(reward)
                trajectories['dones'].append(done)
                trajectories['values'].append(value)

                obs = next_obs
                episode_reward += reward

                # When trajectory length reaches rollout_size or episode done, then update.
                if len(trajectories['obs']) >= rollout_size or done:
                    # If episode continues, get last value estimate; else 0.
                    if not done:
                        # require new lstm state prediction for the last observation.
                        _, _, last_value, _ = self.select_action(obs, lstm_state)
                    else:
                        last_value = 0.0

                    returns = self.compute_returns(trajectories['rewards'], trajectories['dones'], last_value)
                    trajectories['returns'] = returns
                    # PPO update with collected episodes/rollout
                    self.ppo_update(trajectories)

                    # Clear trajectories
                    trajectories = {'obs': [], 'actions': [], 'log_probs': [], 'rewards': [], 'dones': [], 'values': []}

                    # If episode done, reset environment and LSTM state.
                    if done:
                        obs = self.env.reset()
                        lstm_state = self.model.init_lstm_state(batch_size=1, device=self.device)
                        episode_rewards.append(episode_reward)
                        print("Episode {} Reward: {:.2f}".format(episode + 1, episode_reward))
                        episode_reward = 0
                        break
            # End of for each episode
        return episode_rewards


    def test(self, times, v_steps=60):
        Traj = []
        start_end = []
        for i in range(times):
            obs = self.env.reset(iftest= i)
            start_end.append(np.stack((self.env.agent_pos.copy(), self.env.target_pos.copy())))
            done = False
            traj = [self.env.agent_pos.copy()]
            lstm_state = self.model.init_lstm_state(batch_size=1, device=self.device)
            total_reward = 0
            while not done:
                action, _, _, lstm_state = self.select_action(obs, lstm_state)
                obs, reward, done, _ = self.env.step(action)
                total_reward += reward
                traj.append(self.env.agent_pos.copy())
            Traj.append(np.array(traj))
            print("Test {} Episode Reward: {:.2f}".format(times, total_reward))

        traj = np.concatenate(Traj)
        start_end = np.concatenate(start_end)
        first_t = np.where(np.all(traj == start_end[1], axis=1))[0]
        sub = len(traj)// v_steps
        for s in range(sub):
            # Plot the trajectory on the grid.
            grid_size = self.env.grid_size
            plt.figure(figsize=(5, 5))
            plt.xlim(-0.5, grid_size - 0.5)
            plt.ylim(-0.5, grid_size - 0.5)
            plt.title("Test Trajectory")
            # Plot grid lines.
            """   
            for i in range(grid_size):
               plt.axhline(i - 0.5, color='gray', linestyle=':', linewidth=0.5)
               plt.axvline(i - 0.5, color='gray', linestyle=':', linewidth=0.5)
            """

            # Plot start, trajectory and target.
            if s == sub-1:
                plt.plot(traj[:, 0], traj[:, 1], marker='o', color='blue')
            else:
                plt.plot(traj[:s * v_steps, 0], traj[:s * v_steps, 1], marker='o', color='blue')
            #plt.scatter(start_end[-1, 0], start_end[-1, 1], marker='*', s=200, color='black', label='Target')
            plt.scatter(start_end[1, 0], start_end[1, 1], marker='*', s=200, color='red', label='Target')
            plt.scatter(start_end[0, 0], start_end[0, 0], marker='s', s=100, color='green', label='Start')
            plt.legend()
            plt.gca().invert_yaxis()  # to match grid coordinates
            plt.savefig(fname=f'./results/tra/tra_{s}.jpg', dpi=500, format='jpg', bbox_inches='tight')
            plt.show()



# Set device (change use_gpu=True to use GPU if available).
use_gpu = torch.cuda.is_available()
device = torch.device("cuda" if use_gpu else "cpu")
print("Using device:", device)
# Create the environment and agent.


env = MessengerEnv(grid_size=50, max_steps=200)

agent = PPOAgent(env, device, hidden_dim=128, lstm_hidden_dim=128, lr=1e-4, gamma=0.99, clip_eps=0.2, update_epochs=8, batch_size=128)
#agent = torch.load("results/agent.pth")
#agent.test(1,v_steps=8)

# Train the agent.
num_episodes = 300  # you can increase this number for better performance.
print("Beginning training...")
rewards = agent.train(num_episodes=num_episodes, max_steps=200, rollout_size=256)
torch.save(agent, "results/agent.pth")
# Visualize training rewards.
plt.figure()
plt.plot(rewards)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Training Rewards over Episodes")
plt.savefig(fname='./results/rewards.jpg', dpi=500, format='jpg', bbox_inches='tight')
plt.show()

# Test the trained model.
print("Testing the trained model...")
agent.test(1,v_steps=10)