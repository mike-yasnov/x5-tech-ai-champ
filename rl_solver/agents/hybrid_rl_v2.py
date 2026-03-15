"""Hybrid RL Agent v2: improved architecture + multi-sample + LNS.

Improvements over v1:
  1. Multi-head attention pointer network with LayerNorm
  2. Richer item features (12 dims) with cross-item statistics
  3. Multi-sample evaluation: generate N orderings, pick best
  4. LNS refinement on top of best solution
  5. Curriculum: start with easy scenarios, increase difficulty
  6. Greedy baseline for REINFORCE variance reduction
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


# ─── Improved Pointer Network ──────────────────────────────────────────

class MultiHeadPointerNet(nn.Module):
    """Multi-head attention pointer network with residual connections."""

    def __init__(self, item_dim: int = 12, ctx_dim: int = 20,
                 hidden: int = 256, n_heads: int = 4, n_layers: int = 2):
        super().__init__()
        self.hidden = hidden
        self.n_heads = n_heads
        head_dim = hidden // n_heads

        # Item embedding
        self.item_embed = nn.Sequential(
            nn.Linear(item_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
        )

        # Context embedding
        self.ctx_embed = nn.Sequential(
            nn.Linear(ctx_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
        )

        # Self-attention layers for items (learn item-item relationships)
        self.self_attn_layers = nn.ModuleList()
        for _ in range(n_layers):
            self.self_attn_layers.append(nn.MultiheadAttention(hidden, n_heads, batch_first=True))

        self.self_attn_norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(n_layers)])
        self.self_attn_ffns = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden, hidden * 2), nn.ReLU(), nn.Linear(hidden * 2, hidden))
            for _ in range(n_layers)
        ])
        self.self_attn_ffn_norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(n_layers)])

        # Pointer attention (context queries items)
        self.pointer_query = nn.Linear(hidden, hidden)
        self.pointer_key = nn.Linear(hidden, hidden)
        self.scale = hidden ** 0.5
        self.clip_logits = 10.0  # tanh clipping for logit stability

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, item_features: torch.Tensor, context: torch.Tensor,
                mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            item_features: (B, N, item_dim)
            context: (B, ctx_dim)
            mask: (B, N)
        Returns:
            logits: (B, N), value: (B,)
        """
        # Embed
        h = self.item_embed(item_features)  # (B, N, H)
        ctx = self.ctx_embed(context)  # (B, H)

        # Self-attention with residual
        key_padding_mask = (mask == 0)  # True = ignore
        for attn, norm, ffn, ffn_norm in zip(
            self.self_attn_layers, self.self_attn_norms,
            self.self_attn_ffns, self.self_attn_ffn_norms
        ):
            h2, _ = attn(h, h, h, key_padding_mask=key_padding_mask)
            h = norm(h + h2)
            h = ffn_norm(h + ffn(h))

        # Pointer: context → item attention
        q = self.pointer_query(ctx).unsqueeze(1)  # (B, 1, H)
        k = self.pointer_key(h)  # (B, N, H)
        logits = (q * k).sum(-1) / self.scale  # (B, N)

        # Tanh clipping for stability
        logits = self.clip_logits * torch.tanh(logits)

        # Mask
        logits = logits + (mask - 1) * 1e9

        # Value: pool items + context
        item_pool = (h * mask.unsqueeze(-1)).sum(1) / (mask.sum(1, keepdim=True) + 1e-8)
        value = self.value_head(torch.cat([ctx, item_pool], dim=-1)).squeeze(-1)

        return logits, value


# ─── Environment v2 with richer features ────────────────────────────────

class HybridPackingEnvV2:
    """Environment with richer features and incremental packing feedback."""

    def __init__(self, scenarios: List[Dict], max_items: int = 200):
        self.scenarios = scenarios
        self.max_items = max_items
        self._scenario_idx = 0

        self.pallet = None
        self.boxes: List[Dict] = []
        self.available_mask: np.ndarray = None
        self.order_chosen: List[int] = []
        self.n_items = 0
        self._scenario = None

        # Incremental packing state for richer context
        self._packed_weight = 0.0
        self._packed_volume = 0
        self._packed_count = 0
        self._max_z_reached = 0

    def _expand_boxes(self, box_specs: List[Dict]) -> List[Dict]:
        expanded = []
        for spec in box_specs:
            for i in range(spec["quantity"]):
                expanded.append({**spec, "instance_index": i})
        return expanded

    def _get_item_features(self) -> np.ndarray:
        """12 features per box with cross-item statistics."""
        feats = np.zeros((self.max_items, 12), dtype=np.float32)
        pL = self.pallet["length_mm"]
        pW = self.pallet["width_mm"]
        pH = self.pallet["max_height_mm"]
        pWt = self.pallet["max_weight_kg"]
        pV = pL * pW * pH

        # Compute statistics over available items
        available_vols = []
        available_weights = []
        for i in range(self.n_items):
            if self.available_mask[i] > 0:
                b = self.boxes[i]
                available_vols.append(b["length_mm"] * b["width_mm"] * b["height_mm"])
                available_weights.append(b["weight_kg"])

        avg_vol = np.mean(available_vols) if available_vols else 0
        max_vol = max(available_vols) if available_vols else 0
        avg_wt = np.mean(available_weights) if available_weights else 0

        for i, b in enumerate(self.boxes):
            if i >= self.max_items:
                break
            vol = b["length_mm"] * b["width_mm"] * b["height_mm"]
            base_area = b["length_mm"] * b["width_mm"]
            feats[i] = [
                b["length_mm"] / pL,
                b["width_mm"] / pW,
                b["height_mm"] / pH,
                b["weight_kg"] / max(pWt, 1),
                vol / max(pV, 1),
                base_area / max(pL * pW, 1),
                float(b.get("fragile", False)),
                float(b.get("strict_upright", False)),
                float(b.get("stackable", True)),
                # Relative to available items
                vol / max(max_vol, 1) if max_vol > 0 else 0,
                b["weight_kg"] / max(avg_wt, 1) if avg_wt > 0 else 0,
                # Density (weight/volume)
                (b["weight_kg"] / (vol / 1e9)) / 1000 if vol > 0 else 0,
            ]
        return feats

    def _get_context(self) -> np.ndarray:
        """20 context features."""
        n_placed = len(self.order_chosen)
        n_remaining = self.n_items - n_placed
        max_weight = self.pallet["max_weight_kg"]
        pV = self.pallet["length_mm"] * self.pallet["width_mm"] * self.pallet["max_height_mm"]

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
        remaining_weight = sum(
            self.boxes[i]["weight_kg"] for i in range(self.n_items) if self.available_mask[i]
        )
        remaining_vol = sum(
            self.boxes[i]["length_mm"] * self.boxes[i]["width_mm"] * self.boxes[i]["height_mm"]
            for i in range(self.n_items) if self.available_mask[i]
        )

        ctx = np.zeros(20, dtype=np.float32)
        ctx[0] = n_placed / max(self.n_items, 1)
        ctx[1] = n_remaining / max(self.n_items, 1)
        ctx[2] = self._packed_weight / max(max_weight, 1)
        ctx[3] = self._packed_volume / max(pV, 1)
        ctx[4] = self.pallet["length_mm"] / 1219
        ctx[5] = self.pallet["width_mm"] / 1016
        ctx[6] = self.pallet["max_height_mm"] / 2000
        ctx[7] = max_weight / 1500
        ctx[8] = remaining_fragile / max(n_remaining, 1)
        ctx[9] = remaining_upright / max(n_remaining, 1)
        ctx[10] = remaining_nostack / max(n_remaining, 1)
        ctx[11] = (max_weight - self._packed_weight) / max(max_weight, 1)
        ctx[12] = remaining_weight / max(max_weight, 1)
        ctx[13] = remaining_vol / max(pV, 1)
        ctx[14] = self._packed_count / max(self.n_items, 1)
        ctx[15] = self._max_z_reached / max(self.pallet["max_height_mm"], 1)
        # Can we fit all remaining weight?
        ctx[16] = 1.0 if remaining_weight <= (max_weight - self._packed_weight) else 0.0
        # Fraction of constrained items remaining
        n_constrained = remaining_fragile + remaining_upright + remaining_nostack
        ctx[17] = n_constrained / max(n_remaining, 1)
        ctx[18] = 0.0
        ctx[19] = 0.0
        return ctx

    def reset(self, scenario_idx: Optional[int] = None):
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
        self._packed_weight = 0.0
        self._packed_volume = 0
        self._packed_count = 0
        self._max_z_reached = 0

        return self._get_item_features(), self._get_context(), self.available_mask.copy()

    def step(self, action: int):
        if action < self.n_items and self.available_mask[action] > 0:
            b = self.boxes[action]
            self.order_chosen.append(action)
            self.available_mask[action] = 0.0
            self._packed_weight += b["weight_kg"]
            self._packed_volume += b["length_mm"] * b["width_mm"] * b["height_mm"]
            self._packed_count += 1

        done = len(self.order_chosen) >= self.n_items or self.available_mask.sum() == 0
        return self._get_item_features(), self._get_context(), self.available_mask.copy(), done

    def evaluate_ordering(self, use_postprocess: bool = True) -> Tuple[Dict, float]:
        """Pack in chosen order using greedy EP + optional postprocessing."""
        from solver.models import Box as SBox, Pallet as SPallet
        from solver.pallet_state import PalletState
        from solver.orientations import get_orientations
        from solver.scoring import score_placement

        p = self.pallet
        pallet = SPallet(type_id=p.get("type_id", ""), length_mm=p["length_mm"],
                         width_mm=p["width_mm"], max_height_mm=p["max_height_mm"],
                         max_weight_kg=p["max_weight_kg"])

        ordered_boxes = [self.boxes[i] for i in self.order_chosen]
        state = PalletState(pallet)
        placements = []

        for box_dict in ordered_boxes:
            sku = box_dict["sku_id"]
            sbox = SBox(sku_id=sku, description=box_dict.get("description", ""),
                        length_mm=box_dict["length_mm"], width_mm=box_dict["width_mm"],
                        height_mm=box_dict["height_mm"], weight_kg=box_dict["weight_kg"],
                        quantity=1, strict_upright=box_dict.get("strict_upright", False),
                        fragile=box_dict.get("fragile", False),
                        stackable=box_dict.get("stackable", True))
            orientations = get_orientations(sbox)
            if not orientations or state.current_weight + sbox.weight_kg > pallet.max_weight_kg:
                continue

            best_score, best_pl = -1e9, None
            for ep in state.extreme_points:
                ex, ey, ez = ep
                for dx, dy, dz, rot_code in orientations:
                    if not state.can_place(dx, dy, dz, ex, ey, ez,
                                           sbox.weight_kg, sbox.fragile, sbox.stackable):
                        continue
                    sc = score_placement(state, dx, dy, dz, ex, ey, ez,
                                         sbox.weight_kg, sbox.fragile, pallet.max_height_mm)
                    if sc > best_score:
                        best_score = sc
                        best_pl = (ex, ey, ez, dx, dy, dz, rot_code)

            if best_pl:
                ex, ey, ez, dx, dy, dz, rot_code = best_pl
                state.place(sku, dx, dy, dz, ex, ey, ez,
                            sbox.weight_kg, sbox.fragile, sbox.stackable)
                placements.append({
                    "sku_id": sku, "instance_index": box_dict["instance_index"],
                    "position": {"x_mm": ex, "y_mm": ey, "z_mm": ez},
                    "dimensions_placed": {"length_mm": dx, "width_mm": dy, "height_mm": dz},
                    "rotation_code": rot_code,
                })

        # Unplaced
        placed_skus = {}
        for pl in placements:
            placed_skus[pl["sku_id"]] = placed_skus.get(pl["sku_id"], 0) + 1
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
                unplaced.append({"sku_id": sid, "quantity_unplaced": total_qty - placed_qty, "reason": "no_space"})

        response = {
            "task_id": self._scenario.get("task_id", ""),
            "solver_version": "hybrid-rl-v2",
            "solve_time_ms": 0,
            "placements": placements,
            "unplaced": unplaced,
        }

        if use_postprocess:
            from rl_solver.postprocess_wrapper import apply_postprocessing
            response = apply_postprocessing(response, self._scenario, time_budget_ms=3000)

        from validator import evaluate_solution
        result = evaluate_solution(self._scenario, response)
        score = result.get("final_score", 0.0) if result.get("valid") else 0.0
        return response, score


# ─── Agent v2 with multi-sample and greedy baseline ────────────────────

class HybridPPOAgentV2:
    """Improved PPO with multi-sample eval and greedy baseline."""

    def __init__(self, item_dim: int = 12, ctx_dim: int = 20, hidden: int = 256,
                 n_heads: int = 4, n_layers: int = 2, lr: float = 1e-4,
                 gamma: float = 0.99, clip_eps: float = 0.2,
                 entropy_coef: float = 0.02, value_coef: float = 0.5,
                 n_epochs: int = 4, device: str = "auto"):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.gamma = gamma
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.n_epochs = n_epochs

        self.model = MultiHeadPointerNet(item_dim, ctx_dim, hidden, n_heads, n_layers).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, eps=1e-5)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=1000, eta_min=1e-5)
        self.update_count = 0

        # Baseline: exponential moving average of rewards
        self.baseline = 0.0
        self.baseline_alpha = 0.05

        params = sum(p.numel() for p in self.model.parameters())
        logger.info("HybridPPO-v2: hidden=%d, heads=%d, layers=%d, device=%s, params=%dK",
                     hidden, n_heads, n_layers, self.device, params // 1000)

    def select_action(self, item_feats, context, mask, greedy=False):
        with torch.no_grad():
            items_t = torch.FloatTensor(item_feats).unsqueeze(0).to(self.device)
            ctx_t = torch.FloatTensor(context).unsqueeze(0).to(self.device)
            mask_t = torch.FloatTensor(mask).unsqueeze(0).to(self.device)

            logits, value = self.model(items_t, ctx_t, mask_t)
            if greedy:
                action = logits.squeeze(0).argmax()
                log_prob = torch.tensor(0.0)
            else:
                dist = Categorical(logits=logits.squeeze(0))
                action = dist.sample()
                log_prob = dist.log_prob(action)
        return action.item(), log_prob.item(), value.item()

    def generate_ordering(self, env: HybridPackingEnvV2, scenario_idx: int,
                          greedy: bool = False) -> Tuple[List[dict], float]:
        """Generate one complete ordering and return (trajectory, score)."""
        items, ctx, mask = env.reset(scenario_idx=scenario_idx)
        trajectory = []
        done = False

        while not done:
            action, log_prob, value = self.select_action(items, ctx, mask, greedy=greedy)
            trajectory.append({
                "items": items.copy(), "context": ctx.copy(), "mask": mask.copy(),
                "action": action, "log_prob": log_prob, "value": value,
            })
            items, ctx, mask, done = env.step(action)

        _, score = env.evaluate_ordering(use_postprocess=True)
        return trajectory, score

    def multi_sample_eval(self, env: HybridPackingEnvV2, scenario_idx: int,
                          n_samples: int = 8) -> Tuple[Dict, float]:
        """Generate N orderings, return the best one."""
        best_response = None
        best_score = -1.0

        for _ in range(n_samples):
            items, ctx, mask = env.reset(scenario_idx=scenario_idx)
            done = False
            while not done:
                action, _, _ = self.select_action(items, ctx, mask, greedy=False)
                items, ctx, mask, done = env.step(action)
            response, score = env.evaluate_ordering(use_postprocess=True)
            if score > best_score:
                best_score = score
                best_response = response

        # Also try greedy
        items, ctx, mask = env.reset(scenario_idx=scenario_idx)
        done = False
        while not done:
            action, _, _ = self.select_action(items, ctx, mask, greedy=True)
            items, ctx, mask, done = env.step(action)
        response, score = env.evaluate_ordering(use_postprocess=True)
        if score > best_score:
            best_score = score
            best_response = response

        return best_response, best_score

    def train_on_episode(self, trajectory: List[dict], reward: float) -> dict:
        if not trajectory:
            return {}

        # Update baseline
        self.baseline = self.baseline * (1 - self.baseline_alpha) + reward * self.baseline_alpha

        n = len(trajectory)
        # Reward shaping: spread reward, slight front-loading
        rewards = np.zeros(n, dtype=np.float32)
        for i in range(n):
            # Front-weighted: early decisions matter more
            weight = 1.0 + 0.5 * (1.0 - i / n)
            rewards[i] = reward * weight / n

        returns = np.zeros(n, dtype=np.float32)
        R = 0
        for t in reversed(range(n)):
            R = rewards[t] + self.gamma * R
            returns[t] = R

        items_t = torch.FloatTensor(np.array([t["items"] for t in trajectory])).to(self.device)
        ctx_t = torch.FloatTensor(np.array([t["context"] for t in trajectory])).to(self.device)
        mask_t = torch.FloatTensor(np.array([t["mask"] for t in trajectory])).to(self.device)
        actions_t = torch.LongTensor([t["action"] for t in trajectory]).to(self.device)
        old_lp_t = torch.FloatTensor([t["log_prob"] for t in trajectory]).to(self.device)
        returns_t = torch.FloatTensor(returns).to(self.device)

        total_loss = 0
        for _ in range(self.n_epochs):
            logits, values = self.model(items_t, ctx_t, mask_t)
            dist = Categorical(logits=logits)
            new_lp = dist.log_prob(actions_t)
            entropy = dist.entropy().mean()

            advantages = returns_t - values.detach()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            ratio = torch.exp(new_lp - old_lp_t)
            pg1 = -advantages * ratio
            pg2 = -advantages * torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps)
            pg_loss = torch.max(pg1, pg2).mean()
            vf_loss = ((values - returns_t) ** 2).mean()
            loss = pg_loss + self.value_coef * vf_loss - self.entropy_coef * entropy

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            total_loss += loss.item()

        self.scheduler.step()
        self.update_count += 1
        return {"loss": total_loss / self.n_epochs, "entropy": entropy.item(),
                "baseline": self.baseline, "lr": self.optimizer.param_groups[0]["lr"]}

    def save(self, path: str):
        torch.save({
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "update_count": self.update_count,
            "baseline": self.baseline,
        }, path)
        logger.info("Saved HybridPPO-v2 to %s", path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            self.scheduler.load_state_dict(ckpt["scheduler"])
        self.update_count = ckpt.get("update_count", 0)
        self.baseline = ckpt.get("baseline", 0.0)
        logger.info("Loaded HybridPPO-v2 from %s", path)
