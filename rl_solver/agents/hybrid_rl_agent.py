"""Hybrid RL Agent: RL learns box ordering, original solver does placement.

Instead of learning raw (x,y,z) placements (huge action space, hard to learn),
the RL agent learns to REORDER boxes. The actual packing uses the battle-tested
greedy EP packer from the original solver.

Architecture:
  - State: box features + pallet state summary
  - Action: which remaining box to place next (pointer network style)
  - Packing: greedy EP from original solver
  - Postprocessing: compact_downward + reorder_fragile + try_insert

This approach dramatically reduces action space complexity and leverages
the strong existing packer.
"""

import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "DEBUG"))


# ─── Pointer Network for ordering ──────────────────────────────────────

class PointerNetwork(nn.Module):
    """Pointer network that selects next item from remaining set.

    Takes global context + per-item embeddings, outputs selection probabilities.
    """

    def __init__(self, item_feat_dim: int = 8, context_dim: int = 16, hidden: int = 128):
        super().__init__()
        self.item_encoder = nn.Sequential(
            nn.Linear(item_feat_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        self.context_encoder = nn.Sequential(
            nn.Linear(context_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        # Attention mechanism
        self.query = nn.Linear(hidden, hidden)
        self.key = nn.Linear(hidden, hidden)
        self.scale = hidden ** 0.5

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, item_features: torch.Tensor, context: torch.Tensor,
                mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            item_features: (batch, n_items, item_feat_dim)
            context: (batch, context_dim)
            mask: (batch, n_items) - 1 for available, 0 for already placed

        Returns:
            logits: (batch, n_items)
            value: (batch,)
        """
        # Encode items
        item_emb = self.item_encoder(item_features)  # (B, N, H)
        ctx_emb = self.context_encoder(context)  # (B, H)

        # Attention: query from context, keys from items
        q = self.query(ctx_emb).unsqueeze(1)  # (B, 1, H)
        k = self.key(item_emb)  # (B, N, H)
        logits = (q * k).sum(-1) / self.scale  # (B, N)

        # Mask unavailable items
        logits = logits + (mask - 1) * 1e9

        # Value from context
        value = self.value_head(ctx_emb).squeeze(-1)  # (B,)

        return logits, value


# ─── Hybrid RL Environment ─────────────────────────────────────────────

class HybridPackingEnv:
    """Environment where RL chooses box order, solver does placement."""

    def __init__(self, scenarios: List[Dict], max_items: int = 200):
        self.scenarios = scenarios
        self.max_items = max_items  # fixed max for padding
        self._scenario_idx = 0

        # Will be set in reset
        self.pallet = None
        self.boxes: List[Dict] = []
        self.available_mask: np.ndarray = None
        self.order_chosen: List[int] = []
        self.n_items = 0

    def _expand_boxes(self, box_specs: List[Dict]) -> List[Dict]:
        expanded = []
        for spec in box_specs:
            for i in range(spec["quantity"]):
                expanded.append({**spec, "instance_index": i})
        return expanded

    def _get_item_features(self) -> np.ndarray:
        """8 features per box, padded to max_items."""
        feats = np.zeros((self.max_items, 8), dtype=np.float32)
        pL = self.pallet["length_mm"]
        pW = self.pallet["width_mm"]
        pH = self.pallet["max_height_mm"]
        pWt = self.pallet["max_weight_kg"]

        for i, b in enumerate(self.boxes):
            if i >= self.max_items:
                break
            vol = b["length_mm"] * b["width_mm"] * b["height_mm"]
            feats[i] = [
                b["length_mm"] / pL,
                b["width_mm"] / pW,
                b["height_mm"] / pH,
                b["weight_kg"] / max(pWt, 1),
                vol / (pL * pW * pH),
                float(b.get("fragile", False)),
                float(b.get("strict_upright", False)),
                float(b.get("stackable", True)),
            ]
        return feats

    def _get_context(self) -> np.ndarray:
        """16 context features about current state."""
        n_placed = len(self.order_chosen)
        n_remaining = self.n_items - n_placed

        # Weight of chosen items
        chosen_weight = sum(self.boxes[i]["weight_kg"] for i in self.order_chosen) if self.order_chosen else 0
        max_weight = self.pallet["max_weight_kg"]

        # Volume of chosen items
        chosen_vol = sum(
            self.boxes[i]["length_mm"] * self.boxes[i]["width_mm"] * self.boxes[i]["height_mm"]
            for i in self.order_chosen
        ) if self.order_chosen else 0
        total_vol = self.pallet["length_mm"] * self.pallet["width_mm"] * self.pallet["max_height_mm"]

        # Fragile/upright/nostack counts in remaining
        remaining_fragile = sum(
            1 for i in range(self.n_items)
            if self.available_mask[i] and self.boxes[i].get("fragile", False)
        )
        remaining_upright = sum(
            1 for i in range(self.n_items)
            if self.available_mask[i] and self.boxes[i].get("strict_upright", False)
        )
        remaining_nostack = sum(
            1 for i in range(self.n_items)
            if self.available_mask[i] and not self.boxes[i].get("stackable", True)
        )

        ctx = np.zeros(16, dtype=np.float32)
        ctx[0] = n_placed / max(self.n_items, 1)
        ctx[1] = n_remaining / max(self.n_items, 1)
        ctx[2] = chosen_weight / max(max_weight, 1)
        ctx[3] = chosen_vol / max(total_vol, 1)
        ctx[4] = self.pallet["length_mm"] / 1219  # normalized
        ctx[5] = self.pallet["width_mm"] / 1016
        ctx[6] = self.pallet["max_height_mm"] / 2000
        ctx[7] = max_weight / 1500
        ctx[8] = remaining_fragile / max(self.n_items, 1)
        ctx[9] = remaining_upright / max(self.n_items, 1)
        ctx[10] = remaining_nostack / max(self.n_items, 1)
        # Steps taken ratio
        ctx[11] = len(self.order_chosen) / max(self.n_items, 1)
        # Remaining weight budget
        ctx[12] = (max_weight - chosen_weight) / max(max_weight, 1)
        # Average remaining box volume
        if n_remaining > 0:
            avg_remaining_vol = sum(
                self.boxes[i]["length_mm"] * self.boxes[i]["width_mm"] * self.boxes[i]["height_mm"]
                for i in range(self.n_items) if self.available_mask[i]
            ) / n_remaining
            ctx[13] = avg_remaining_vol / max(total_vol, 1)
        ctx[14] = 0.0  # reserved
        ctx[15] = 0.0  # reserved
        return ctx

    def reset(self, scenario_idx: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Reset and return (item_features, context, mask)."""
        if scenario_idx is not None:
            self._scenario_idx = scenario_idx
        scenario = self.scenarios[self._scenario_idx % len(self.scenarios)]
        self._scenario_idx += 1

        self.pallet = scenario["pallet"]
        self.boxes = self._expand_boxes(scenario["boxes"])
        self.n_items = min(len(self.boxes), self.max_items)
        self.available_mask = np.zeros(self.max_items, dtype=np.float32)
        self.available_mask[:self.n_items] = 1.0
        self.order_chosen = []
        self._scenario = scenario

        return self._get_item_features(), self._get_context(), self.available_mask.copy()

    def step(self, action: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
        """Choose next box. Returns (item_features, context, mask, done)."""
        if action < self.n_items and self.available_mask[action] > 0:
            self.order_chosen.append(action)
            self.available_mask[action] = 0.0

        done = len(self.order_chosen) >= self.n_items or self.available_mask.sum() == 0
        return self._get_item_features(), self._get_context(), self.available_mask.copy(), done

    def evaluate_ordering(self) -> Tuple[Dict, float]:
        """Pack boxes in chosen order using original solver, apply postprocessing.

        Returns (response_dict, validator_score).
        """
        from solver.models import Box as SBox, Pallet as SPallet, Placement, Solution
        from solver.pallet_state import PalletState
        from solver.orientations import get_orientations
        from solver.scoring import score_placement
        from rl_solver.postprocess_wrapper import apply_postprocessing

        p = self.pallet
        pallet = SPallet(
            type_id=p.get("type_id", ""),
            length_mm=p["length_mm"], width_mm=p["width_mm"],
            max_height_mm=p["max_height_mm"], max_weight_kg=p["max_weight_kg"],
        )

        # Build boxes in chosen order
        ordered_boxes = [self.boxes[i] for i in self.order_chosen]

        # Greedy EP packing in the RL-chosen order
        state = PalletState(pallet)
        placements = []

        for box_dict in ordered_boxes:
            sku = box_dict["sku_id"]
            sbox = SBox(
                sku_id=sku, description=box_dict.get("description", ""),
                length_mm=box_dict["length_mm"], width_mm=box_dict["width_mm"],
                height_mm=box_dict["height_mm"], weight_kg=box_dict["weight_kg"],
                quantity=1,
                strict_upright=box_dict.get("strict_upright", False),
                fragile=box_dict.get("fragile", False),
                stackable=box_dict.get("stackable", True),
            )
            orientations = get_orientations(sbox)
            if not orientations:
                continue

            if state.current_weight + sbox.weight_kg > pallet.max_weight_kg:
                continue

            best_score = -1e9
            best_placement = None

            for ep in state.extreme_points:
                ex, ey, ez = ep
                for dx, dy, dz, rot_code in orientations:
                    if not state.can_place(dx, dy, dz, ex, ey, ez,
                                           sbox.weight_kg, sbox.fragile, sbox.stackable):
                        continue

                    sc = score_placement(
                        state, dx, dy, dz, ex, ey, ez,
                        sbox.weight_kg, sbox.fragile,
                        pallet.max_height_mm,
                    )
                    if sc > best_score:
                        best_score = sc
                        best_placement = (ex, ey, ez, dx, dy, dz, rot_code)

            if best_placement:
                ex, ey, ez, dx, dy, dz, rot_code = best_placement
                state.place(sku, dx, dy, dz, ex, ey, ez,
                            sbox.weight_kg, sbox.fragile, sbox.stackable)
                placements.append({
                    "sku_id": sku,
                    "instance_index": box_dict["instance_index"],
                    "position": {"x_mm": ex, "y_mm": ey, "z_mm": ez},
                    "dimensions_placed": {"length_mm": dx, "width_mm": dy, "height_mm": dz},
                    "rotation_code": rot_code,
                })

        # Build unplaced
        placed_skus = {}
        for p in placements:
            placed_skus[p["sku_id"]] = placed_skus.get(p["sku_id"], 0) + 1

        unplaced = []
        seen = set()
        for b in self.boxes:
            sid = b["sku_id"]
            if sid in seen:
                continue
            seen.add(sid)
            total_qty = sum(1 for x in self.boxes if x["sku_id"] == sid)
            placed_qty = placed_skus.get(sid, 0)
            if placed_qty < total_qty:
                unplaced.append({
                    "sku_id": sid,
                    "quantity_unplaced": total_qty - placed_qty,
                    "reason": "no_space",
                })

        raw_response = {
            "task_id": self._scenario.get("task_id", ""),
            "solver_version": "hybrid-rl-1.0.0",
            "solve_time_ms": 0,
            "placements": placements,
            "unplaced": unplaced,
        }

        # Apply postprocessing
        improved = apply_postprocessing(
            raw_response, self._scenario,
            do_compact=True, do_reorder_fragile=True, do_try_insert=True,
            time_budget_ms=3000,
        )

        # Validate
        from validator import evaluate_solution
        result = evaluate_solution(self._scenario, improved)
        score = result.get("final_score", 0.0) if result.get("valid") else 0.0

        return improved, score


# ─── PPO Training for Hybrid Agent ─────────────────────────────────────

class HybridPPOAgent:
    """PPO agent that learns box ordering via pointer network."""

    def __init__(
        self,
        item_feat_dim: int = 8,
        context_dim: int = 16,
        hidden: int = 128,
        lr: float = 3e-4,
        gamma: float = 0.99,
        clip_eps: float = 0.2,
        entropy_coef: float = 0.05,
        value_coef: float = 0.5,
        n_epochs: int = 4,
        device: str = "auto",
    ):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.gamma = gamma
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.n_epochs = n_epochs

        self.model = PointerNetwork(item_feat_dim, context_dim, hidden).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, eps=1e-5)
        self.update_count = 0

        logger.info("HybridPPO: hidden=%d, device=%s, params=%dK",
                     hidden, self.device,
                     sum(p.numel() for p in self.model.parameters()) // 1000)

    def select_action(self, item_feats: np.ndarray, context: np.ndarray,
                      mask: np.ndarray) -> Tuple[int, float, float]:
        """Select next box index."""
        with torch.no_grad():
            items_t = torch.FloatTensor(item_feats).unsqueeze(0).to(self.device)
            ctx_t = torch.FloatTensor(context).unsqueeze(0).to(self.device)
            mask_t = torch.FloatTensor(mask).unsqueeze(0).to(self.device)

            logits, value = self.model(items_t, ctx_t, mask_t)
            dist = Categorical(logits=logits.squeeze(0))
            action = dist.sample()
            log_prob = dist.log_prob(action)

        return action.item(), log_prob.item(), value.item()

    def train_on_episode(self, trajectory: List[dict], final_reward: float) -> dict:
        """REINFORCE-style update with baseline from value head."""
        if not trajectory:
            return {}

        # Assign discounted rewards (final_reward spread across steps)
        n = len(trajectory)
        rewards = np.zeros(n, dtype=np.float32)
        # Give reward proportional to step (later picks matter less)
        for i in range(n):
            rewards[i] = final_reward / n

        # Compute returns
        returns = np.zeros(n, dtype=np.float32)
        R = 0
        for t in reversed(range(n)):
            R = rewards[t] + self.gamma * R
            returns[t] = R

        # Collect tensors
        items_list = [t["items"] for t in trajectory]
        ctx_list = [t["context"] for t in trajectory]
        mask_list = [t["mask"] for t in trajectory]
        actions = [t["action"] for t in trajectory]
        old_log_probs = [t["log_prob"] for t in trajectory]

        items_t = torch.FloatTensor(np.array(items_list)).to(self.device)
        ctx_t = torch.FloatTensor(np.array(ctx_list)).to(self.device)
        mask_t = torch.FloatTensor(np.array(mask_list)).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        old_lp_t = torch.FloatTensor(old_log_probs).to(self.device)
        returns_t = torch.FloatTensor(returns).to(self.device)

        total_loss = 0
        for _ in range(self.n_epochs):
            logits, values = self.model(items_t, ctx_t, mask_t)
            dist = Categorical(logits=logits)
            new_log_probs = dist.log_prob(actions_t)
            entropy = dist.entropy().mean()

            advantages = returns_t - values.detach()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            ratio = torch.exp(new_log_probs - old_lp_t)
            pg1 = -advantages * ratio
            pg2 = -advantages * torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps)
            pg_loss = torch.max(pg1, pg2).mean()

            vf_loss = ((values - returns_t) ** 2).mean()
            loss = pg_loss + self.value_coef * vf_loss - self.entropy_coef * entropy

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()
            total_loss += loss.item()

        self.update_count += 1
        return {
            "loss": total_loss / self.n_epochs,
            "entropy": entropy.item(),
            "final_reward": final_reward,
        }

    def save(self, path: str):
        torch.save({
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "update_count": self.update_count,
        }, path)
        logger.info("Saved HybridPPO to %s", path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.update_count = ckpt["update_count"]
        logger.info("Loaded HybridPPO from %s", path)
