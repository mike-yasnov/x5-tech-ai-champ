"""ML-based placement scorer using a small MLP (pure NumPy inference)."""

import logging
import os
import numpy as np
from typing import Optional, Tuple

from .pallet_state import PalletState, _overlap_area

logger = logging.getLogger(__name__)

# Feature dimension
N_FEATURES = 20

# Default model path (relative to this file)
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "ml_ranker.npz")


def extract_features(
    state: PalletState,
    dx: int, dy: int, dz: int,
    x: int, y: int, z: int,
    weight_kg: float,
    fragile: bool,
    stackable: bool,
    items_remaining: int,
    total_items: int,
    remaining_volume: int,
) -> np.ndarray:
    """Extract normalized features for a placement candidate.

    Returns a flat array of N_FEATURES floats.
    """
    pallet = state.pallet
    pL = pallet.length_mm
    pW = pallet.width_mm
    pH = pallet.max_height_mm
    pV = pL * pW * pH
    pMaxW = pallet.max_weight_kg

    x2, y2, z2 = x + dx, y + dy, z + dz

    # Box features (normalized by pallet dimensions)
    f_dx = dx / pL
    f_dy = dy / pW
    f_dz = dz / pH
    f_vol = (dx * dy * dz) / pV if pV > 0 else 0.0
    f_weight = weight_kg / pMaxW if pMaxW > 0 else 0.0

    # Position features
    f_x = x / pL
    f_y = y / pW
    f_z = z / pH

    # Height growth
    new_max_z = max(state.max_z, z2)
    f_height_growth = (new_max_z - state.max_z) / pH if pH > 0 else 0.0

    # Fill state
    placed_volume = sum(
        (b.x_max - b.x_min) * (b.y_max - b.y_min) * (b.z_max - b.z_min)
        for b in state.boxes
    )
    f_fill_ratio = placed_volume / pV if pV > 0 else 0.0
    f_weight_ratio = state.current_weight / pMaxW if pMaxW > 0 else 0.0

    # Remaining items info
    f_items_remaining = items_remaining / total_items if total_items > 0 else 0.0
    f_remaining_vol = remaining_volume / pV if pV > 0 else 0.0

    # Support area ratio (for items not on floor)
    if z == 0:
        f_support = 1.0
    else:
        base_area = dx * dy
        support_area = 0
        for box in state.boxes:
            if box.z_max == z:
                support_area += _overlap_area(
                    x, y, x2, y2,
                    box.x_min, box.y_min, box.x_max, box.y_max,
                )
        f_support = support_area / base_area if base_area > 0 else 0.0

    # Wall touches (0-4)
    wall_touches = 0
    if x == 0:
        wall_touches += 1
    if y == 0:
        wall_touches += 1
    if x2 == pL:
        wall_touches += 1
    if y2 == pW:
        wall_touches += 1
    f_wall_touches = wall_touches / 4.0

    # On floor
    f_on_floor = 1.0 if z == 0 else 0.0

    # Constraints
    f_fragile = 1.0 if fragile else 0.0
    f_stackable = 1.0 if stackable else 0.0

    # Max-z ratio
    f_max_z = state.max_z / pH if pH > 0 else 0.0

    features = np.array([
        f_dx, f_dy, f_dz, f_vol, f_weight,         # 0-4: box features
        f_x, f_y, f_z,                               # 5-7: position
        f_height_growth, f_fill_ratio, f_weight_ratio,  # 8-10: state
        f_items_remaining, f_remaining_vol,            # 11-12: remaining
        f_support, f_wall_touches, f_on_floor,        # 13-15: placement quality
        f_fragile, f_stackable,                        # 16-17: constraints
        f_max_z,                                       # 18: current state
        0.0,                                           # 19: reserved
    ], dtype=np.float32)

    return features


class MLPScorer:
    """Tiny MLP scorer for placement ranking. Pure NumPy inference."""

    def __init__(self, weights: Optional[list] = None):
        """Initialize with list of (weight_matrix, bias_vector) tuples."""
        self.layers = weights or []
        self._loaded = len(self.layers) > 0

    @classmethod
    def load(cls, path: str = MODEL_PATH) -> Optional["MLPScorer"]:
        """Load model weights from .npz file."""
        if not os.path.exists(path):
            logger.debug("[MLPScorer] model not found at %s", path)
            return None
        try:
            data = np.load(path)
            layers = []
            i = 0
            while f"w{i}" in data and f"b{i}" in data:
                layers.append((data[f"w{i}"], data[f"b{i}"]))
                i += 1
            scorer = cls(weights=layers)
            logger.info("[MLPScorer] loaded model with %d layers from %s", len(layers), path)
            return scorer
        except Exception as e:
            logger.warning("[MLPScorer] failed to load model: %s", e)
            return None

    def save(self, path: str = MODEL_PATH) -> None:
        """Save model weights to .npz file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        save_dict = {}
        for i, (w, b) in enumerate(self.layers):
            save_dict[f"w{i}"] = w
            save_dict[f"b{i}"] = b
        np.savez(path, **save_dict)
        logger.info("[MLPScorer] saved model to %s", path)

    def predict(self, features: np.ndarray) -> float:
        """Forward pass through MLP. Returns scalar score."""
        x = features
        for i, (w, b) in enumerate(self.layers):
            x = x @ w + b
            if i < len(self.layers) - 1:
                # LeakyReLU for hidden layers
                x = np.where(x > 0, x, 0.01 * x)
        return float(x[0]) if x.ndim == 1 else float(x)

    def predict_batch(self, features_batch: np.ndarray) -> np.ndarray:
        """Forward pass for batch of features. Returns array of scores."""
        x = features_batch
        for i, (w, b) in enumerate(self.layers):
            x = x @ w + b
            if i < len(self.layers) - 1:
                x = np.where(x > 0, x, 0.01 * x)
        return x.ravel()

    @property
    def is_loaded(self) -> bool:
        return self._loaded
