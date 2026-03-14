# Heuristics Catalog

This document describes the heuristic families added for CI comparison.

The implementation now separates two decisions:

- `sort heuristic` - in what order box instances are considered
- `placement policy` - how the best feasible extreme-point candidate is chosen for the current box

That gives a large benchmark space while keeping the solver architecture stable.

## Placement Policies

All placement policies use the same candidate generator: every valid `extreme point x orientation` combination.

### `balanced`

- Baseline scoring from the original solver
- Mixes low height, wall/neighbor contact, fragility penalty, and gap filling
- Good default baseline for comparison

### `dbfl`

- Deepest-Bottom-Left-Fill style policy
- Strongly prefers low `z`, then low coordinates
- Usually builds compact low stacks quickly

### `max_support`

- Maximizes support ratio / support area under the candidate
- Useful when support constraints are binding or the load is fragile
- Often produces safer upper layers

### `max_contact`

- Maximizes contact with walls and neighboring boxes
- Encourages tighter, more interlocked structures
- Useful for stability-oriented comparisons

### `min_height`

- Minimizes resulting stack height and favors flatter top surfaces
- Similar in spirit to layer-building heuristics
- Useful when many boxes have comparable heights

### `center_stable`

- Rewards supported placements closer to pallet center
- Tries to reduce off-center heavy stacks
- Useful for weight-sensitive scenarios

## Sort Heuristics

## `volume_desc`

- Largest volume first
- Classic 3D bin packing baseline
- Good when large boxes dominate the geometry

## `weight_desc`

- Heaviest first
- Helps build heavy-bottom pallets early
- Useful when weight distribution matters more than pure fill

## `base_area_desc`

- Largest footprint first
- Promotes wide, stable lower layers
- Often strong with `max_support`

## `density_desc`

- Highest density first
- Puts compact heavy items lower
- Useful for center-of-gravity control

## `constrained_first`

- Prioritizes `strict_upright` and `stackable=False`
- Packs hard-to-place items before flexibility disappears
- Good constraint-oriented baseline

## `max_dim_desc`

- Longest dimension first
- Tries to place awkward long boxes before fragmentation grows
- Useful on mixed aspect-ratio instances

## `perimeter_desc`

- Largest footprint perimeter first
- Encourages edge and wall occupation early
- Often pairs well with `max_contact`

## `fragile_last`

- Pushes fragile items later
- Helps naturally move fragile cargo toward top layers
- Strong candidate for retail-like cases

## `stackable_first`

- Prioritizes boxes that can safely support other cargo
- Defers weak support surfaces
- Often pairs well with `max_support`

## `layer_height_desc`

- Groups boxes by similar heights, then by footprint
- Mimics layer-building ideas from palletizing literature and practice
- Useful for regular carton mixes

## `homogeneous`

- Groups identical SKUs together
- Approximates block stacking / mono-SKU columns
- Useful when requests contain repeated carton types

## `upright_first`

- Prioritizes orientation-constrained cargo
- Helps place liquids and other upright-only items before flexible ones

## `slenderness_desc`

- Prioritizes long or slender boxes
- Attempts to place shape-sensitive items early
- Useful for stress-testing residual-space behavior

## `weighted_volume`

- Combined score of volume and weight
- Trades off pure fill rate and base stability
- Good compromise baseline

## `smalls_last`

- Delays small items
- Lets large geometry define cavities first, then uses small items as fillers

## Strategy Space

The benchmark strategies are predefined in `solver/heuristics.py` as `STRATEGY_CONFIGS`.

Examples:

- `volume__balanced`
- `volume__max_contact`
- `area__max_support`
- `weight__fragile_last`
- `layer__min_height`
- `stackable__max_support`
- `homogeneous__max_support`
- `slender__min_height`

There are also randomized variants:

- `<sort_key>__mutated`

These keep the base ordering but perturb nearby positions in the sorted list. They are useful because greedy 3D packing is highly order-sensitive.

## Recommended First Candidates

If we later want to prune the search space, the first configs worth comparing are:

- `area__max_support`
- `weight__fragile_last`
- `layer__min_height`
- `stackable__max_support`
- `homogeneous__max_support`
- `weighted_volume__center`

## CI Usage

`benchmark.py` now defaults to running all strategy configs, so CI can compare the full heuristic catalog without extra flags.
