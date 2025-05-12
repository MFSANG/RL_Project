import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, action_bounds):
        super().__init__()
        self.min_action, self.max_action = torch.tensor(action_bounds[0]), torch.tensor(action_bounds[1])
        hidden1, hidden2 = 400, 300

        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden1), nn.ReLU(),
            nn.Linear(hidden1, hidden2), nn.ReLU()
        )
        self.mu = nn.Linear(hidden2, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))  # learnable std
        self.value = nn.Linear(hidden2, 1)

    def forward(self, state):
        x = self.shared(state)
        mu = torch.tanh(self.mu(x))  # in [-1, 1]
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
        self.__init__()


class PPOAgent:
    def __init__(self, state_dim, action_dim, action_bounds,
                 gamma=0.99, lam=0.95, clip_eps=0.2, lr=3e-4, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor_critic = ActorCritic(state_dim, action_dim, action_bounds).to(self.device)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)
        self.gamma, self.lam, self.clip_eps = gamma, lam, clip_eps
        self.buffer = RolloutBuffer()

        self.rewards_log, self.sr_log, self.energy_log = [], [], []

    def select_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        dist, value = self.actor_critic(state_tensor)
        raw_action = dist.sample()
        log_prob = dist.log_prob(raw_action).sum(dim=-1)
        scaled_action = self.actor_critic.scale_action(raw_action).cpu().numpy()[0]
        return scaled_action, log_prob.item(), value.item()

    def compute_gae(self, next_value):
        values = self.buffer.values + [next_value]
        gae, returns = 0, []
        for i in reversed(range(len(self.buffer.rewards))):
            delta = self.buffer.rewards[i] + self.gamma * values[i+1] * (1 - self.buffer.dones[i]) - values[i]
            gae = delta + self.gamma * self.lam * (1 - self.buffer.dones[i]) * gae
            returns.insert(0, gae + values[i])
        return returns

    def update(self):
        states = torch.FloatTensor(self.buffer.states).to(self.device)
        actions = torch.FloatTensor(self.buffer.actions).to(self.device)
        log_probs_old = torch.FloatTensor(self.buffer.log_probs).to(self.device)
        returns = torch.FloatTensor(self.compute_gae(next_value=0)).to(self.device)
        values = torch.FloatTensor(self.buffer.values).to(self.device)

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

        self.buffer.clear()
