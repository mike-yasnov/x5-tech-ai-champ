from typing import List, Tuple


def get_orientations(
    length: int, width: int, height: int, strict_upright: bool
) -> List[Tuple[int, int, int, str]]:
    """Return list of (placed_l, placed_w, placed_h, rotation_code).

    If strict_upright is True, only rotations that keep the original height
    on the Z axis are allowed (rotate around Z only).
    Deduplicates orientations with identical (l, w, h) tuples.
    """
    # All 6 permutations mapping (L,W,H) -> (X-dim, Y-dim, Z-dim)
    all_rotations = [
        (length, width, height, "LWH"),
        (length, height, width, "LHW"),
        (width, length, height, "WLH"),
        (width, height, length, "WHL"),
        (height, length, width, "HLW"),
        (height, width, length, "HWL"),
    ]

    if strict_upright:
        # Only rotations where Z-dim == original height
        candidates = [(l, w, h, code) for l, w, h, code in all_rotations if h == height]
    else:
        candidates = all_rotations

    # Deduplicate by (l, w, h) — keep first occurrence
    seen = set()
    result = []
    for l, w, h, code in candidates:
        key = (l, w, h)
        if key not in seen:
            seen.add(key)
            result.append((l, w, h, code))
    return result
