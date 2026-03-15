"""A2C (Advantage Actor-Critic) Agent for 3D Pallet Packing.

Synchronous advantage actor-critic with action masking and n-step returns.
"""

import logging
import os
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

logger = logging.getLogger(__name__)
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "DEBUG"))


class A2CNetwork(nn.Module):
    """Actor-Critic network with separate heads."""

    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.actor = nn.Linear(hidden, n_actions)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, x, mask=None):
        feat = self.shared(x)
        logits = self.actor(feat)
        if mask is not None:
            logits = logits + (mask - 1) * 1e9
        value = self.critic(feat).squeeze(-1)
        return logits, value


class A2CAgent:
    """Advantage Actor-Critic with n-step returns and entropy regularization."""

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        lr: float = 7e-4,
        gamma: float = 0.99,
        n_steps: int = 5,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        hidden_dim: int = 256,
        device: str = "auto",
    ):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.gamma = gamma
        self.n_steps = n_steps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm

        self.model = A2CNetwork(obs_dim, n_actions, hidden_dim).to(self.device)
        self.optimizer = optim.RMSprop(self.model.parameters(), lr=lr, alpha=0.99, eps=1e-5)

        # N-step buffer
        self._states = []
        self._actions = []
        self._rewards = []
        self._dones = []
        self._masks = []

        self.step_count = 0
        self.update_count = 0

        logger.info(
            "A2C Agent: obs=%d, actions=%d, hidden=%d, n_steps=%d, device=%s",
            obs_dim, n_actions, hidden_dim, n_steps, self.device,
        )

    def select_action(self, state: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[int, float, float]:
        """Select action, return (action, log_prob, value)."""
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            mask_t = None
            if mask is not None:
                mask_t = torch.FloatTensor(mask).unsqueeze(0).to(self.device)

            logits, value = self.model(state_t, mask_t)
            dist = Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)

        return action.item(), log_prob.item(), value.item()

    def store(self, state, action, reward, done, mask):
        self._states.append(state)
        self._actions.append(action)
        self._rewards.append(reward)
        self._dones.append(done)
        self._masks.append(mask)
        self.step_count += 1

    def train_step(self, next_state: np.ndarray, next_mask: Optional[np.ndarray] = None) -> Optional[dict]:
        """Update after n steps collected."""
        if len(self._states) < self.n_steps:
            return None

        # Bootstrap value for last state
        with torch.no_grad():
            ns_t = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
            nm_t = None
            if next_mask is not None:
                nm_t = torch.FloatTensor(next_mask).unsqueeze(0).to(self.device)
            _, bootstrap_value = self.model(ns_t, nm_t)
            bootstrap_value = bootstrap_value.item()

        # Compute n-step returns
        n = len(self._states)
        returns = np.zeros(n, dtype=np.float32)
        R = bootstrap_value
        for t in reversed(range(n)):
            R = self._rewards[t] + self.gamma * R * (1 - self._dones[t])
            returns[t] = R

        states_t = torch.FloatTensor(np.array(self._states)).to(self.device)
        actions_t = torch.LongTensor(self._actions).to(self.device)
        returns_t = torch.FloatTensor(returns).to(self.device)
        masks_t = torch.FloatTensor(np.array(self._masks)).to(self.device)

        logits, values = self.model(states_t, masks_t)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions_t)
        entropy = dist.entropy().mean()

        advantages = returns_t - values
        pg_loss = -(log_probs * advantages.detach()).mean()
        vf_loss = advantages.pow(2).mean()
        loss = pg_loss + self.value_coef * vf_loss - self.entropy_coef * entropy

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()

        self.update_count += 1

        # Clear buffer
        self._states.clear()
        self._actions.clear()
        self._rewards.clear()
        self._dones.clear()
        self._masks.clear()

        metrics = {
            "loss": loss.item(),
            "pg_loss": pg_loss.item(),
            "vf_loss": vf_loss.item(),
            "entropy": entropy.item(),
        }
        return metrics

    def save(self, path: str):
        torch.save({
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step_count": self.step_count,
            "update_count": self.update_count,
        }, path)
        logger.info("Saved A2C model to %s", path)

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.step_count = checkpoint["step_count"]
        self.update_count = checkpoint["update_count"]
        logger.info("Loaded A2C model from %s", path)
