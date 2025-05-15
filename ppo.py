import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import json


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, action_bounds):
        super().__init__()
        self.min_action = torch.tensor(action_bounds[0], dtype=torch.float32)
        self.max_action = torch.tensor(action_bounds[1], dtype=torch.float32)
        hidden1, hidden2 = 400, 300

        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden1), nn.ReLU(),
            nn.Linear(hidden1, hidden2), nn.ReLU()
        )
        self.mu = nn.Linear(hidden2, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        self.value = nn.Linear(hidden2, 1)

    def forward(self, state):
        if state.ndim == 1:
            state = state.unsqueeze(0)
        x = self.shared(state)
        mu = torch.tanh(self.mu(x))
        std = torch.exp(self.log_std).expand_as(mu)
        dist = torch.distributions.Normal(mu, std)
        value = self.value(x)
        return dist, value

    def scale_action(self, raw_action):
        return self.min_action + 0.5 * (raw_action + 1.0) * (self.max_action - self.min_action)


class RolloutBuffer:
    def __init__(self):
        self.states, self.actions, self.log_probs = [], [], []
        self.rewards, self.dones, self.values = [], [], []

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()


class PPO:
    def __init__(self, state_dim, action_dim, action_bounds,
                 gamma=0.99, lam=0.95, clip_eps=0.2, lr=3e-4, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor_critic = ActorCritic(state_dim, action_dim, action_bounds).to(self.device)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)

        self.gamma, self.lam, self.clip_eps = gamma, lam, clip_eps
        self.buffer = RolloutBuffer()

        self.rewards_log = []
        self.avg_reward_log = []
        self.sr_log = []
        self.energy_log = []
        self.loss_log = []
        self.best_sr = -float("inf")

    def select_action(self, state):
        if state is None or not isinstance(state, np.ndarray):
            print("Warning: Invalid state encountered. Returning random action.")
            low = self.actor_critic.min_action.cpu().numpy()
            high = self.actor_critic.max_action.cpu().numpy()
            return np.random.uniform(low, high), 0.0, 0.0
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        dist, value = self.actor_critic(state_tensor)
        raw_action = dist.sample()
        log_prob = dist.log_prob(raw_action).sum(dim=-1)
        scaled_action = self.actor_critic.scale_action(raw_action).cpu().detach().numpy()[0]
        return scaled_action, log_prob.item(), value.item()

    def compute_gae(self, next_value):
        values = self.buffer.values + [next_value]
        gae, returns = 0, []
        for i in reversed(range(len(self.buffer.rewards))):
            delta = self.buffer.rewards[i] + self.gamma * values[i + 1] * (1 - self.buffer.dones[i]) - values[i]
            gae = delta + self.gamma * self.lam * (1 - self.buffer.dones[i]) * gae
            returns.insert(0, gae + values[i])
        return returns

    def update(self):
        if len(self.buffer.states) == 0:
            print("[Warning] Buffer is empty during update. Skipping.")
            return

        states = torch.FloatTensor(np.array(self.buffer.states)).to(self.device)
        actions = torch.FloatTensor(np.array(self.buffer.actions)).to(self.device)
        log_probs_old = torch.FloatTensor(np.array(self.buffer.log_probs)).to(self.device)
        returns = torch.FloatTensor(np.array(self.compute_gae(next_value=0))).to(self.device)
        values = torch.FloatTensor(np.array(self.buffer.values)).to(self.device)

        for _ in range(4):
            dist, new_values = self.actor_critic(states)
            log_probs_new = dist.log_prob(actions).sum(dim=-1)
            ratio = (log_probs_new - log_probs_old).exp()
            advantage = returns - values

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantage
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = nn.MSELoss()(new_values.squeeze(), returns)
            loss = actor_loss + 0.5 * critic_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.loss_log.append(loss.item())

        self.buffer.clear()

    def record_metrics(self, reward, secrecy_rate, energy):
        self.rewards_log.append(reward)
        self.sr_log.append(secrecy_rate)
        self.energy_log.append(energy)
        avg_reward = np.mean(self.rewards_log[-10:])
        self.avg_reward_log.append(avg_reward)

        if secrecy_rate > self.best_sr:
            self.best_sr = secrecy_rate
            self.save_best_model()

    def save_best_model(self, path='results/ppo/best_ppo.pt'):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.actor_critic.state_dict(), path)

    def load_best_model(self, path='results/ppo/best_ppo.pt'):
        if os.path.exists(path):
            self.actor_critic.load_state_dict(torch.load(path, map_location=self.device))
            print(f"Loaded best PPO model from {path}")

    def plot_learning_curve(self):
        # 填充默认值防止绘图失败
        if not self.rewards_log or all(r == 0 for r in self.rewards_log):
            self.rewards_log = [0.0] * 100
        if not self.avg_reward_log:
            self.avg_reward_log = [0.0] * len(self.rewards_log)
        if not self.sr_log:
            self.sr_log = [0.0] * len(self.rewards_log)
        if not self.energy_log:
            self.energy_log = [0.0] * len(self.rewards_log)
        if not self.loss_log:
            self.loss_log = [0.0] * len(self.rewards_log)

        episodes = list(range(1, len(self.rewards_log) + 1))

        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        axs[0, 0].plot(episodes, self.rewards_log, label='Reward')
        axs[0, 0].plot(episodes, self.avg_reward_log, label='Avg Reward (10 eps)')
        axs[0, 0].set_title('Reward')
        axs[0, 0].set_xlabel('Episode')
        axs[0, 0].set_ylabel('Total Reward')
        axs[0, 0].legend()
        axs[0, 1].plot(episodes, self.sr_log, color='green')
        axs[0, 1].set_title('Secrecy Rate')
        axs[0, 1].set_xlabel('Episode')
        axs[0, 1].set_ylabel('bps/Hz')
        axs[1, 0].plot(episodes, self.energy_log, color='red')
        axs[1, 0].set_title('Remaining Energy')
        axs[1, 0].set_xlabel('Episode')
        axs[1, 0].set_ylabel('Energy (J)')
        axs[1, 0].set_ylim([max(0, min(self.energy_log) * 0.9), max(self.energy_log) * 1.1])
        axs[1, 1].plot(range(1, len(self.loss_log) + 1), self.loss_log, color='purple')
        axs[1, 1].set_title('PPO Loss')
        axs[1, 1].set_xlabel('Update Step')
        axs[1, 1].set_ylabel('Loss')
        plt.tight_layout()
        os.makedirs('results/ppo', exist_ok=True)
        plt.savefig('results/ppo/ppo_learning_curves.png')
        plt.close()

        plt.figure(figsize=(8, 5))
        plt.plot(episodes, self.rewards_log, label='Reward')
        plt.plot(episodes, self.avg_reward_log, label='Avg Reward')
        plt.title('Reward Curve')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.legend()
        plt.grid(True)
        plt.savefig('results/ppo/reward_curve.png')
        plt.close()

        plt.figure(figsize=(8, 5))
        plt.plot(episodes, self.sr_log, color='green')
        plt.title('Secrecy Rate Curve')
        plt.xlabel('Episode')
        plt.ylabel('bps/Hz')
        plt.grid(True)
        plt.savefig('results/ppo/secrecy_rate_curve.png')
        plt.close()

    def print_energy_stats(self):
        if self.energy_log:
            print("\nEnergy statistics:")
            print(f"  Average remaining energy: {np.mean(self.energy_log):.2f} J")
            print(f"  Min remaining energy: {min(self.energy_log):.2f} J")
            print(f"  Max remaining energy: {max(self.energy_log):.2f} J")
