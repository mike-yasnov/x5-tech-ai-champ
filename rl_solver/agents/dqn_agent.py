"""DQN Agent for 3D Pallet Packing.

Uses Double DQN with Dueling architecture and Prioritized Experience Replay.
Action masking ensures only valid placements are selected.
"""

import logging
import os
import random
from collections import deque
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

logger = logging.getLogger(__name__)
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "DEBUG"))


class SumTree:
    """Sum tree for prioritized experience replay."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = [None] * capacity
        self.write_idx = 0
        self.size = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, priority, data):
        idx = self.write_idx + self.capacity - 1
        self.data[self.write_idx] = data
        self.update(idx, priority)
        self.write_idx = (self.write_idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def update(self, idx, priority):
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]


class PrioritizedReplayBuffer:
    """Prioritized Experience Replay buffer using SumTree."""

    def __init__(self, capacity: int, alpha: float = 0.6):
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.max_priority = 1.0

    def push(self, state, action, reward, next_state, done, mask, next_mask):
        priority = self.max_priority ** self.alpha
        self.tree.add(priority, (state, action, reward, next_state, done, mask, next_mask))

    def sample(self, batch_size: int, beta: float = 0.4):
        batch = []
        idxs = []
        priorities = []
        segment = self.tree.total() / batch_size

        for i in range(batch_size):
            s = random.uniform(segment * i, segment * (i + 1))
            idx, priority, data = self.tree.get(s)
            if data is None:
                # Fallback to random valid entry
                s = random.uniform(0, self.tree.total())
                idx, priority, data = self.tree.get(s)
            if data is not None:
                batch.append(data)
                idxs.append(idx)
                priorities.append(priority)

        if not batch:
            return None

        total = self.tree.total()
        probs = np.array(priorities) / (total + 1e-8)
        weights = (self.tree.size * probs + 1e-8) ** (-beta)
        weights /= weights.max()

        states, actions, rewards, next_states, dones, masks, next_masks = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32),
            np.array(masks),
            np.array(next_masks),
            np.array(weights, dtype=np.float32),
            idxs,
        )

    def update_priorities(self, idxs, td_errors):
        for idx, td_error in zip(idxs, td_errors):
            priority = (abs(td_error) + 1e-6) ** self.alpha
            self.max_priority = max(self.max_priority, priority)
            self.tree.update(idx, priority)

    def __len__(self):
        return self.tree.size


class DuelingDQN(nn.Module):
    """Dueling DQN architecture."""

    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 512):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        # Value stream
        self.value = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )
        # Advantage stream
        self.advantage = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, n_actions),
        )

    def forward(self, x):
        feat = self.feature(x)
        val = self.value(feat)
        adv = self.advantage(feat)
        # Dueling: Q = V + (A - mean(A))
        q = val + adv - adv.mean(dim=-1, keepdim=True)
        return q


class DQNAgent:
    """Double Dueling DQN with Prioritized Experience Replay and action masking."""

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        lr: float = 1e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: int = 50000,
        buffer_size: int = 100000,
        batch_size: int = 64,
        target_update_freq: int = 1000,
        hidden_dim: int = 512,
        device: str = "auto",
    ):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.policy_net = DuelingDQN(obs_dim, n_actions, hidden_dim).to(self.device)
        self.target_net = DuelingDQN(obs_dim, n_actions, hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.buffer = PrioritizedReplayBuffer(buffer_size)
        self.step_count = 0

        logger.info(
            "DQN Agent: obs=%d, actions=%d, hidden=%d, device=%s, params=%dK",
            obs_dim, n_actions, hidden_dim, self.device,
            sum(p.numel() for p in self.policy_net.parameters()) // 1000,
        )

    def select_action(self, state: np.ndarray, action_mask: Optional[np.ndarray] = None) -> int:
        """Epsilon-greedy action selection with masking."""
        self.epsilon = max(
            self.epsilon_end,
            self.epsilon - (1.0 - self.epsilon_end) / self.epsilon_decay,
        )
        self.step_count += 1

        if random.random() < self.epsilon:
            # Random valid action
            if action_mask is not None and action_mask.sum() > 0:
                valid_actions = np.where(action_mask > 0)[0]
                return int(np.random.choice(valid_actions))
            return random.randrange(self.n_actions)

        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_t).squeeze(0)

            if action_mask is not None:
                mask_t = torch.FloatTensor(action_mask).to(self.device)
                q_values = q_values + (mask_t - 1) * 1e9  # mask invalid to -inf

            return int(q_values.argmax().item())

    def store(self, state, action, reward, next_state, done, mask, next_mask):
        self.buffer.push(state, action, reward, next_state, done, mask, next_mask)

    def train_step(self) -> Optional[float]:
        """Single training step. Returns loss or None."""
        if len(self.buffer) < self.batch_size:
            return None

        beta = min(1.0, 0.4 + self.step_count * (1.0 - 0.4) / 100000)
        sample = self.buffer.sample(self.batch_size, beta)
        if sample is None:
            return None

        states, actions, rewards, next_states, dones, masks, next_masks, weights, idxs = sample

        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)
        next_masks_t = torch.FloatTensor(next_masks).to(self.device)
        weights_t = torch.FloatTensor(weights).to(self.device)

        # Current Q values
        q_values = self.policy_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Double DQN: use policy net to select action, target net to evaluate
        with torch.no_grad():
            next_q_policy = self.policy_net(next_states_t)
            next_q_policy = next_q_policy + (next_masks_t - 1) * 1e9
            next_actions = next_q_policy.argmax(dim=1)

            next_q_target = self.target_net(next_states_t)
            next_q = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target = rewards_t + self.gamma * next_q * (1 - dones_t)

        td_errors = (q_values - target).detach().cpu().numpy()
        loss = (weights_t * (q_values - target) ** 2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()

        # Update priorities
        self.buffer.update_priorities(idxs, td_errors)

        # Update target network
        if self.step_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            logger.debug("Updated target network at step %d", self.step_count)

        return loss.item()

    def save(self, path: str):
        torch.save({
            "policy_net": self.policy_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step_count": self.step_count,
            "epsilon": self.epsilon,
        }, path)
        logger.info("Saved DQN model to %s", path)

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.step_count = checkpoint["step_count"]
        self.epsilon = checkpoint["epsilon"]
        logger.info("Loaded DQN model from %s (step=%d)", path, self.step_count)
