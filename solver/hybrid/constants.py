ROTATION_CODES = ["LWH", "LHW", "WLH", "WHL", "HLW", "HWL"]

SUPPORT_THRESHOLD = 0.60
FRAGILE_WEIGHT_THRESHOLD = 2.0
FRAGILE_PENALTY_PER = 0.05
EPSILON = 1e-6

UNPLACED_REASONS = {
    "weight": "weight_limit_exceeded",
    "height": "height_limit_exceeded",
    "space": "no_space",
}
