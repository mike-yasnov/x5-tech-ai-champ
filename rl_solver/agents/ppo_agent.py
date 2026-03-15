"""PPO Agent for 3D Pallet Packing.

Proximal Policy Optimization with action masking, GAE, and entropy bonus.
Actor-Critic architecture with shared feature extractor.
"""

import logging
import os
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

logger = logging.getLogger(__name__)
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "DEBUG"))


class ActorCritic(nn.Module):
    """Shared-backbone Actor-Critic for masked discrete actions."""

    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 512):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.actor = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, n_actions),
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x, mask=None):
        feat = self.shared(x)
        logits = self.actor(feat)
        if mask is not None:
            logits = logits + (mask - 1) * 1e9
        value = self.critic(feat).squeeze(-1)
        return logits, value

    def get_action_and_value(self, x, mask=None, action=None):
        logits, value = self.forward(x, mask)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy, value


class RolloutBuffer:
    """Stores trajectories for PPO update."""

    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
        self.masks = []

    def add(self, state, action, reward, done, log_prob, value, mask):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.masks.append(mask)

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.log_probs.clear()
        self.values.clear()
        self.masks.clear()

    def __len__(self):
        return len(self.states)


class PPOAgent:
    """PPO with GAE and action masking."""

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        n_epochs: int = 4,
        batch_size: int = 64,
        rollout_len: int = 2048,
        hidden_dim: int = 512,
        device: str = "auto",
    ):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.rollout_len = rollout_len

        self.model = ActorCritic(obs_dim, n_actions, hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, eps=1e-5)
        self.buffer = RolloutBuffer()
        self.step_count = 0
        self.update_count = 0

        logger.info(
            "PPO Agent: obs=%d, actions=%d, hidden=%d, device=%s, params=%dK",
            obs_dim, n_actions, hidden_dim, self.device,
            sum(p.numel() for p in self.model.parameters()) // 1000,
        )

    def select_action(self, state: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[int, float, float]:
        """Select action and return (action, log_prob, value)."""
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            mask_t = None
            if mask is not None:
                mask_t = torch.FloatTensor(mask).unsqueeze(0).to(self.device)

            action, log_prob, _, value = self.model.get_action_and_value(state_t, mask_t)

        return action.item(), log_prob.item(), value.item()

    def store(self, state, action, reward, done, log_prob, value, mask):
        self.buffer.add(state, action, reward, done, log_prob, value, mask)
        self.step_count += 1

    def _compute_gae(self, rewards, values, dones, last_value):
        """Compute Generalized Advantage Estimation."""
        n = len(rewards)
        advantages = np.zeros(n, dtype=np.float32)
        gae = 0.0

        for t in reversed(range(n)):
            if t == n - 1:
                next_value = last_value
            else:
                next_value = values[t + 1]

            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae

        returns = advantages + np.array(values, dtype=np.float32)
        return advantages, returns

    def train_step(self, last_value: float = 0.0) -> Optional[dict]:
        """PPO update from collected rollout."""
        if len(self.buffer) < self.batch_size:
            return None

        rewards = np.array(self.buffer.rewards)
        values = np.array(self.buffer.values)
        dones = np.array(self.buffer.dones)

        advantages, returns = self._compute_gae(rewards, values, dones, last_value)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        states = np.array(self.buffer.states)
        actions = np.array(self.buffer.actions)
        old_log_probs = np.array(self.buffer.log_probs)
        masks = np.array(self.buffer.masks)

        n = len(states)
        total_loss = 0.0
        total_pg_loss = 0.0
        total_vf_loss = 0.0
        total_ent = 0.0
        n_updates = 0

        for _ in range(self.n_epochs):
            indices = np.random.permutation(n)
            for start in range(0, n, self.batch_size):
                end = min(start + self.batch_size, n)
                idx = indices[start:end]

                s_t = torch.FloatTensor(states[idx]).to(self.device)
                a_t = torch.LongTensor(actions[idx]).to(self.device)
                old_lp_t = torch.FloatTensor(old_log_probs[idx]).to(self.device)
                adv_t = torch.FloatTensor(advantages[idx]).to(self.device)
                ret_t = torch.FloatTensor(returns[idx]).to(self.device)
                m_t = torch.FloatTensor(masks[idx]).to(self.device)

                _, new_log_prob, entropy, new_value = self.model.get_action_and_value(
                    s_t, m_t, a_t
                )

                # Policy loss (clipped)
                ratio = torch.exp(new_log_prob - old_lp_t)
                pg_loss1 = -adv_t * ratio
                pg_loss2 = -adv_t * torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                vf_loss = ((new_value - ret_t) ** 2).mean()

                # Entropy bonus
                ent_loss = -entropy.mean()

                loss = pg_loss + self.value_coef * vf_loss + self.entropy_coef * ent_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_loss += loss.item()
                total_pg_loss += pg_loss.item()
                total_vf_loss += vf_loss.item()
                total_ent += entropy.mean().item()
                n_updates += 1

        self.buffer.clear()
        self.update_count += 1

        metrics = {
            "loss": total_loss / max(n_updates, 1),
            "pg_loss": total_pg_loss / max(n_updates, 1),
            "vf_loss": total_vf_loss / max(n_updates, 1),
            "entropy": total_ent / max(n_updates, 1),
            "update": self.update_count,
        }
        logger.debug("PPO update #%d: %s", self.update_count, metrics)
        return metrics

    def save(self, path: str):
        torch.save({
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step_count": self.step_count,
            "update_count": self.update_count,
        }, path)
        logger.info("Saved PPO model to %s", path)

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.step_count = checkpoint["step_count"]
        self.update_count = checkpoint["update_count"]
        logger.info("Loaded PPO model from %s", path, )
