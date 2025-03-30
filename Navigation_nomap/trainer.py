import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from collections import deque
from environment import CourierEnv
from agent import CourierAgent


class PPOTrainer:
    def __init__(self, args, dataset):
        self.args = args
        self.env = CourierEnv(dataset, args.num_landmarks, args.max_steps)
        self.agent = CourierAgent().to(args.device)
        self.device = args.device
        self.optimizer = torch.optim.Adam(self.agent.parameters(), lr=args.learning_rate)

        # Hyperparameters
        self.gamma = args.gamma
        self.gae_lambda = args.gae_lambda
        self.clip_epsilon = args.clip_epsilon
        self.ent_coef = args.ent_coef
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.num_episodes = args.num_episodes

    def collect_trajectories(self, num_episodes=10):
        batch_states = []
        batch_actions = []
        batch_rewards = []
        batch_log_probs = []
        batch_values = []
        batch_dones = []

        for _ in range(num_episodes):
            state = self.env.reset()
            episode_states = []
            episode_actions = []
            episode_rewards = []
            episode_log_probs = []
            episode_values = []
            episode_dones = []

            # LSTM hidden state management
            lstm_hidden = (
                torch.zeros(1, 1, self.agent.lstm.hidden_size).to(self.device),
                torch.zeros(1, 1, self.agent.lstm.hidden_size).to(self.device)
            )

            done = False
            while not done:
                # Prepare inputs
                view_tensor = torch.stack([state['view']]).to(self.device)
                landmark_tensor = torch.stack([state['landmark_encoding']]).to(self.device)
                last_action_tensor = torch.tensor([state['last_action']],
                                                  dtype=torch.long).to(self.device)
                last_reward_tensor = torch.tensor([state['last_reward']],
                                                  dtype=torch.float).to(self.device)

                # Get policy and value
                with torch.no_grad():
                    value, policy_logits, lstm_hidden = self.agent(
                        view_tensor,
                        landmark_tensor,
                        last_action_tensor,
                        last_reward_tensor,
                        lstm_hidden
                    )
                    policy_dist = Categorical(logits=policy_logits)
                    action = policy_dist.sample()
                    log_prob = policy_dist.log_prob(action)

                # Execute action
                next_state, reward, done, _ = self.env.step(action.item())

                # Store transition
                episode_states.append(state)
                episode_actions.append(action.item())
                episode_rewards.append(reward)
                episode_log_probs.append(log_prob.item())
                episode_values.append(value.item())
                episode_dones.append(done)

                state = next_state
                lstm_hidden = (lstm_hidden[0].detach(), lstm_hidden[1].detach())

            # Store episode data
            batch_states.extend(episode_states)
            batch_actions.extend(episode_actions)
            batch_rewards.extend(episode_rewards)
            batch_log_probs.extend(episode_log_probs)
            batch_values.extend(episode_values)
            batch_dones.extend(episode_dones)

        # Convert to tensors
        old_log_probs = torch.tensor(batch_log_probs, dtype=torch.float).to(self.device)
        old_values = torch.tensor(batch_values, dtype=torch.float).to(self.device)
        rewards = torch.tensor(batch_rewards, dtype=torch.float).to(self.device)
        dones = torch.tensor(batch_dones, dtype=torch.float).to(self.device)

        # Calculate advantages and returns
        advantages = self.compute_gae(rewards, old_values, dones)
        returns = advantages + old_values

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Prepare full dataset
        dataset = {
            'states': batch_states,
            'actions': torch.tensor(batch_actions, dtype=torch.long).to(self.device),
            'old_log_probs': old_log_probs,
            'advantages': advantages,
            'returns': returns
        }

        return dataset

    def compute_gae(self, rewards, values, dones):
        advantages = torch.zeros_like(rewards).to(self.device)
        last_advantage = 0

        # Reverse loop through timesteps
        for t in reversed(range(len(rewards))):
            next_value = values[t + 1] if t < len(rewards) - 1 else 0
            next_non_terminal = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            advantages[t] = delta + self.gamma * self.gae_lambda * next_non_terminal * last_advantage
            last_advantage = advantages[t]

        return advantages

    def update(self, dataset):
        # Create indices for mini-batch sampling
        indices = torch.randperm(len(dataset['states']))

        # Split into mini-batches
        for start in range(0, len(indices), self.batch_size):
            end = start + self.batch_size
            batch_indices = indices[start:end]

            # Get batch data
            batch_states = [dataset['states'][i] for i in batch_indices]
            batch_actions = dataset['actions'][batch_indices]
            batch_old_log_probs = dataset['old_log_probs'][batch_indices]
            batch_advantages = dataset['advantages'][batch_indices]
            batch_returns = dataset['returns'][batch_indices]

            # Prepare model inputs
            views = torch.stack([s['view'] for s in batch_states]).to(self.device)
            landmarks = torch.stack([s['landmark_encoding'] for s in batch_states]).to(self.device)
            last_actions = torch.tensor([s['last_action'] for s in batch_states],
                                        dtype=torch.long).to(self.device)
            last_rewards = torch.tensor([s['last_reward'] for s in batch_states],
                                        dtype=torch.float).to(self.device)

            # Forward pass
            values, policy_logits, _ = self.agent(
                views, landmarks, last_actions, last_rewards
            )
            values = values.squeeze()

            # Calculate policy loss
            policy_dist = Categorical(logits=policy_logits)
            new_log_probs = policy_dist.log_prob(batch_actions)
            entropy = policy_dist.entropy().mean()

            ratio = (new_log_probs - batch_old_log_probs).exp()
            surr1 = ratio * batch_advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            value_loss = F.mse_loss(values, batch_returns)

            # Total loss
            loss = policy_loss + 0.5 * value_loss - self.ent_coef * entropy

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.agent.parameters(), 0.5)
            self.optimizer.step()

    def train(self, total_timesteps=100000):
        timesteps = 0
        while timesteps < total_timesteps:
            # Collect data
            dataset = self.collect_trajectories(self.num_episodes)
            timesteps += len(dataset['states'])

            # Optimize for multiple epochs
            for _ in range(self.epochs):
                self.update(dataset)