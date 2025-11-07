# common/models.py
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_dim, hidden=128, out_dim=64, act=nn.Tanh):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), act(),
            nn.Linear(hidden, hidden), act(),
            nn.Linear(hidden, out_dim), act(),
        )
    def forward(self, x): return self.net(x)

class ActorDiscrete(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden=128):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.head = nn.Linear(hidden, n_actions)

    def forward(self, x):
        z = self.body(x)
        logits = self.head(z)
        return logits

class Critic(nn.Module):
    """Used for PPO (local critic) with local obs."""
    def __init__(self, obs_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

class CentralCritic(nn.Module):
    """Used for MAPPO; input is global obs (concat of all agents' local obs)."""
    def __init__(self, global_obs_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(global_obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)
