# Bicycle Trajectory Lane Coordinate Transform
## ETH Zürich IVT — Project Continuation README

---

## Project Overview

Transform GPS bicycle trajectories from global EPSG:2056 (x, y) coordinates to local road-aligned (s, d) coordinates for intersection analysis. Data comes from drone video at 12 Zürich locations. Three sites processed so far:

| Site file | Intersection |
|---|---|
| `maps_june_D1.py` | Langstrasse × Zollstrasse × Röntgenstrasse + Mattengasse T-junction |
| `maps_sep_D1H.py` | September campaign H location |
| `maps_sep_D2I.py` | September campaign I location |

---

## Repository Structure

All output files are in the working directory. Key files:

| File | Purpose |
|---|---|
| `tools_site_builder.py` | Phase 1–4 registry pipeline |
| `maps_june_D1.py` | Site definition for D1 (June) |
| `maps_sep_D1H.py` | Site definition for D1H (Sep) |
| `maps_sep_D2I.py` | Site definition for D2I (Sep) |
| `tools_lane_coords_V4.py` | Lane coordinate transform pipeline |
| `tools_map_visualization.py` | Folium registry map |
| `tools_plot_registry.py` | Matplotlib registry validation plots |
| `tools_plot_results.py` | Results plotting (trajectory map, debug panel, fleet summary) |

---

## Registry Architecture

### Three-layer structure

**`geometry_store`** — one entry per physical road axis:
```python
{
    'spline':       (tck, unew, cum_dist),  # B-spline
    'total_length': float,                   # L [m]
    'positive_dir': str,                     # e.g. 'EB', 'NB'
    's_stop':       float,                   # primary stop line [m]
    's_yield':      float,                   # primary yield line [m]
    's_change':     float,                   # 0.5*(s_stop+s_yield), primary handoff
    # Extra keys for T-junctions (e.g. Zollstr at Mattengasse):
    's_zollstr_east_stop':  float,
    's_zollstr_west_yield': float,
    # Intersection area polygons (computed from normal lines at s_change):
    'intersection_area_MainInt': Polygon,    # stored directly
    'intersection_area_MattInt': Polygon,
    # Pre-expanded versions (set by setup_registry):
    '__intersection_area_MainInt_expanded':  Polygon,
    '__intersection_area_MainInt_prepared':  PreparedGeometry,
    'x_offset': float, 'y_offset': float,   # local EPSG:2056 origin
}
```

**`segment_registry`** — one entry per directed segment:
```python
{
    'type':             'lane' | 'turn',
    'geometry_key':     str,
    'direction':        str,        # 'EB', 'NB', etc.
    'is_forward':       bool,       # direction == positive_dir
    'mode':             str,        # 'shared' | 'bike' | 'car'
    'bike_lane':        dict | None,
    'd_left':           float,      # validity [m] left of travel
    'd_right':          float,      # validity [m] right of travel
    'd_max':            float,
    'validity_polygon': Polygon,
    '_poly_expanded':   Polygon,    # set by setup_registry
    # Turn segments additionally have:
    'approach_seg':           str,
    'departure_seg':          str,
    'approach_s_change_key':  str,  # e.g. 's_change' or 's_zollstr_east_stop'
    'departure_s_change_key': str,
}
```

**`movement_registry`** — `{key: [(seg_key, role), …]}`

### Site builder pipeline (4 phases)

```python
geometry_store    = register_geometries(RAW_AXES, gdf_swisstopo, x_offset, y_offset)
segment_registry  = build_segment_registry(geometry_store, SEG_DEFS)
add_bike_lane_boundaries(segment_registry, geometry_store, gdf_bike_boundaries)
# Step 3b: intersection polygons (after Phase 2, before Phase 3)
geometry_store['intersection_area_MainInt'] = build_intersection_polygon(arm_defs, ...)
geometry_store['intersection_area_MattInt'] = build_intersection_polygon(arm_defs, ...)
turn_keys         = build_turns(geometry_store, segment_registry, TURN_DEFS)
movement_registry = build_movement_registry(geometry_store, segment_registry, MOVEMENT_DEFS)
```

### Intersection area polygons

Built from normal lines at `s_change` points per arm.
Width = `d_right(pos_seg)` LEFT + `d_right(opp_seg)` RIGHT of spline normal.

```python
# MainInt: 4 arms
arm_defs = [
    {'geom_key': 'Roentgenstr', 's_change_key': 's_change',
     'pos_seg_key': 'Roentgenstr_WB', 'opp_seg_key': 'Roentgenstr_EB'},
    {'geom_key': 'LangstrN',    's_change_key': 's_change', ...},
    {'geom_key': 'Zollstr',     's_change_key': 's_change', ...},
    {'geom_key': 'LangstrS',    's_change_key': 's_change', ...},
]
# MattInt: 3 arms (T-junction: 2 Zollstr boundaries + 1 Matteng)
arm_defs = [
    {'geom_key': 'Zollstr', 's_change_key': 's_zollstr_east_stop', ...},
    {'geom_key': 'Zollstr', 's_change_key': 's_zollstr_west_yield', ...},
    {'geom_key': 'Matteng', 's_change_key': 's_change', ...},
]
```

---

## Lane Coordinate Transform — V4 Algorithm

### One-time setup (call before vehicle loop)

```python
from tools_lane_coords_V4 import setup_registry, to_lane_coordinates
setup_registry(geometry_store, segment_registry)
# Builds: spline LUTs, _poly_expanded, __intersection_area_*_expanded/prepared
```

### Per-trajectory: `assign_segments`

Sequential chain loop, up to `max_chain_length` iterations. Each iteration:

**Step 1 — Polygon walk**
Walk fragment from index 0, find first continuous run inside `poly.buffer(tolerance)` → `(entry_idx, exit_idx)`.

Group assignment:
- **Group A**: `entry_idx == 0` AND (turn segment OR first point NOT in intersection box)
- **Group B**: `entry_idx > 0` OR (lane segment with first point inside intersection box)
- **Group C**: no polygon match → rejected

**Step 2 — Choose scoring group**
- Group A non-empty → score Group A, `is_fallback=False`
- Group A empty, Group B non-empty → score Group B, `is_fallback=True`
- Both empty: expand to `all_segs` **only at iteration 0**, else stop chain

**Step 3 — Score each candidate**
On `[entry_idx : exit_idx]` window:
1. Detect `is_reverse` from net s sign vs `is_forward`
2. Veto reversed turns (`score=inf`)
3. Lane segments: exclude points inside `intersection_area_*` from scoring
4. Hard lateral veto: `median_d` outside `[-(d_veto_right+1), +(d_veto_left+1)]`
   — with d_left/d_right **swapped** for reverse traversal
5. Score = `W_DIST·dist + w_head_eff·head + W_ARC·arc + W_PILE·pile + W_REVERSE`

**Weights:**
```python
W_DIST      = 1.0
W_HEAD      = 0.5   # lanes
W_HEAD_TURN = 1.5   # turns — heading is primary discriminator between turns
W_ARC       = 1.5
W_PILE      = 1.0   # turns only
W_REVERSE   = 0.5   # lane reverse traversal penalty
POOR_MATCH_THRESHOLD   = 3.0
FORCED_MATCH_THRESHOLD = 6.0
```

**Selection criterion (chain-structure aware):**
- Group A or turns or iteration 0: `(score, entry_idx)` — pure score
- After turn (departure lanes, `prev_role='turn'`): `(entry_idx, score)` — earliest entry wins

**Step 4 — Accept / reject**
- `score > FORCED_MATCH_THRESHOLD` → stop chain
- `score > POOR_MATCH_THRESHOLD` → `match_quality='poor'`
- else → `match_quality='good'`
- `is_fallback=True` → `match_quality='fallback'`

**Step 5 — Handoff at s_change**
`_sustained_crossing(s_boundary, k=3)`: requires k=3 consecutive points past boundary (prevents false firing from brief oscillations near stop line).

Returns `(handoff_local, s_change_key_fired)`:
1. Check primary `s_change` → `(idx, 's_change')`
2. If not reached, check `s_change_*` extra keys (T-junction boundaries) → `(idx, key)`
3. Nothing crossed → `(exit_idx, None)`

Special cases: turns and departure segments → `(len(s_win), None)`

**T-junction confirmation** (when `s_change_key_fired` is a secondary key):
- Find minor road departure segments (different `geometry_key` from through-road)
- Sample remaining fragment — check for points inside minor road polygon AND past `s_change` on minor road
- **Confirmed** → turn happened, proceed with MattInt turn candidates
- **Not confirmed** → straight through:
  - Re-run primary `s_change` crossing only
  - `s_change_key_fired = 's_change'` or `None`

**Step 6 — Role and next candidates**

Role by chain position (`_assign_role`):
- Iteration 0 → `_infer_role_from_registry` (looks up movement registry)
- After turn → `'departure'`
- After departure → `'approach'` (extended chain)

`_is_departure`: True when `prev_role == 'turn'` OR (iteration 0 AND registry-inferred role is `'departure'`)

`get_next_candidates(seg_key, role, ..., s_change_key_fired)`:
- Filters turn candidates by `approach_s_change_key == s_change_key_fired`
- Chain extension: departure also searched as approach (through-roads serving multiple intersections)
- Opposite-direction counterparts added for lane candidates

Reverse traversal: lookup via opposite-direction key + flipped role.

### Output columns

| Column | Type | Description |
|---|---|---|
| `movement_key` | str | e.g. `'LangstrN_SB_2_LangstrS_SB'` |
| `segment_id` | str | e.g. `'LangstrS_SB'` |
| `segment_type` | str | `'lane'` or `'turn'` |
| `segment_role` | str | `'approach'`, `'turn'`, `'departure'` |
| `match_quality` | str | `'good'`, `'poor'`, `'fallback'`, `'unmatched'` |
| `is_fallback` | bool | polygon matched mid-fragment |
| `is_reverse` | bool | cyclist against segment direction |
| `s_native` | float | arc-length along spline [m], 0→L, **invertible** |
| `d_native` | float | lateral offset, left of spline = + [m], **invertible** |
| `s` | float | directed s [m], increases in travel direction |
| `d` | float | lateral offset, left of travel = + [m] |
| `s_dot` | float | longitudinal speed [km/h] |
| `d_dot` | float | lateral speed [km/h] |
| `s_ddot` | float | longitudinal acceleration [m/s²] |
| `d_ddot` | float | lateral acceleration [m/s²] |
| `in_bike_lane` | float | 1.0 / 0.0 / NaN |
| `d_to_bike_boundary` | float | distance from bike lane inner boundary [m] |

`(s_native, d_native, segment_id)` is the **invertible** triple → recovers `(x, y)`.

### Usage

```python
import pickle
from tools_lane_coords_V4 import setup_registry, to_lane_coordinates

with open('../data/registry_....pkl', 'rb') as f:
    reg = pickle.load(f)

geometry_store    = reg['geometry_store']
segment_registry  = reg['segment_registry']
movement_registry = reg['movement_registry']
max_chain_length  = reg['max_chain_length']   # 5 for june_D1

setup_registry(geometry_store, segment_registry)

results = []
for veh_id, group in trajectory_df.groupby('veh_id'):
    df = group.copy().reset_index(drop=True)
    df = to_lane_coordinates(
        df, movement_registry, segment_registry, geometry_store,
        max_chain_length=max_chain_length,
        verbose=False,   # True for per-vehicle debug logging
    )
    results.append(df)

output_df = pd.concat(results, ignore_index=True)
```

---

## Plotting

```python
from tools_plot_results import (
    plot_trajectory_map,   # folium map, per vehicle
    plot_debug_panel,      # 2×2 matplotlib debug panel, per vehicle
    plot_lane_coords,      # time-series of s, d, s_dot, d_dot, s_ddot, d_ddot
    plot_fleet_summary,    # fleet-level diagnostics
)
from tools_plot_registry import plot_geometry_store, plot_segment_registry
from tools_map_visualization import create_registry_map

# Debug panel: (0,0) XY path, (0,1) s vs d, (1,0) speed, (1,1) acceleration
fig = plot_debug_panel(bike_df, geometry_store, segment_registry, time_col='time')

# Registry folium map (shows intersection polygons, validity polygons, etc.)
m = create_registry_map(geometry_store, segment_registry, movement_registry,
                         gdf_swisstopo, save_path='map.html')
```

---

## Key Design Decisions

### Coordinate systems
- **`s_native`**: raw arc-length 0→L along spline. Invertible.
- **`d_native`**: left of spline = positive. Invertible.
- **`s`** (directed): increases in travel direction. `s = s_native + offset(role, is_forward, s_change)`.
- **`d`** (travel): left of travel = positive. `d = d_native × sign(is_forward) × sign(is_reverse)`.

### T-junction handling (Zollstr × Mattengasse)
- Zollstr `extra_changes`: `s_zollstr_east_stop`, `s_zollstr_west_yield` stored individually (not midpoint).
- Turn defs use direction-specific `approach_s_change_key`:
  - `Zollstr_EB → Matteng_NB`: `approach_s_change_key='s_zollstr_east_stop'`
  - `Zollstr_WB → Matteng_NB`: `approach_s_change_key='s_zollstr_west_yield'`
- Straight-through detection: `_confirm_minor_road_entry` checks remaining fragment for points on Mattengasse past its `s_change`. If not confirmed → revert to primary `s_change` handoff.

### Intersection area polygons
- Used in scoring to exclude lane segment points inside the box.
- Used in Step 1 to demote lane segments from Group A when trajectory starts inside box.
- NOT used for turn segments (turns should claim intersection-starting trajectories).

### Reverse traversal
- Detected from net s progression sign within polygon window.
- `is_reverse=True` for lane: d and heading sign flipped; `W_REVERSE=0.5` penalty added.
- `is_reverse=True` for turn: always vetoed (`score=inf`) — physically meaningless.
- Hard lateral veto: `d_left`/`d_right` swapped for reverse traversal.

### Sustained crossing (`k=3`)
- `_find_handoff` requires 3 consecutive points past `s_change` before firing.
- Prevents false handoff from brief oscillations near stop line.

---

## Known Issues / TODO

- [ ] `POLYGON_TOLERANCE = 1.0m` — may need tuning per site. Tight GPS from drone should allow tightening to 0.5m.
- [ ] `POOR_MATCH_THRESHOLD = 3.0` — borderline scores (e.g. 3.102) could be relaxed slightly.
- [ ] Fleet summary: run `plot_fleet_summary(output_df)` to identify which vehicles need manual inspection after each batch.
- [ ] `max_chain_length=5` for june_D1 (has T-junction); `max_chain_length=3` sufficient for other sites.
- [ ] `s_decreasing` removed — use `s_dot < 0` downstream for backward motion detection.
- [ ] Speed units: `s_dot`, `d_dot` in **km/h** (same as `speed_ekf`). `s_ddot`, `d_ddot` in **m/s²**.

---

## File Dependencies

```
maps_june_D1.py
├── tools_site_builder.py
│   └── tools_coordinate_transform.py
│   └── tools_infrastructure_geometry.py
└── tools_plot_registry.py
└── tools_map_visualization.py

to_lane_coordinates()
└── tools_lane_coords_V4.py
    └── tools_coordinate_transform.py (project_point_full via local copy)

tools_plot_results.py  (standalone, imports from tools_lane_coords_V4 only for setup_registry)
```




# Session Changes README
## ETH Zürich IVT — Lane Coordinate Transform V4 Debugging Session

This document records all changes made to `tools_lane_coords_V4.py` and
`tools_site_builder.py` during this debugging session. Use it to continue
work in a new conversation.

---

## Files Modified

| File | Purpose |
|---|---|
| `tools_lane_coords_V4.py` | Lane coordinate transform pipeline |
| `tools_site_builder.py` | Registry builder (geometry, segments, turns, intersection polygons) |

---

## Changes to `tools_lane_coords_V4.py`

### 1. `_infer_role_from_registry` — Role flip for reverse traversal

**Problem:** When a cyclist traverses a segment in reverse, the registry-based
role is physically wrong (e.g. reverse on an approach segment = effectively
departure behaviour).

**Fix:** Added `is_reverse=False` parameter. When `is_reverse=True`, the
returned role is flipped: `approach → departure`, `departure → approach`.

```python
def _infer_role_from_registry(seg_key, segment_registry, movement_registry,
                               is_reverse=False):
    ...
    registry_role = 'departure' if roles == {'departure'} else 'approach'
    if is_reverse:
        return 'departure' if registry_role == 'approach' else 'approach'
    return registry_role
```

---

### 2. `_assign_role` — Thread `is_reverse` through

**Fix:** Added `is_reverse=False` parameter and passes it to
`_infer_role_from_registry`.

---

### 3. `_find_handoff` — Multiple fixes

#### 3a. Turn segments: last-valid-s handoff
**Problem:** Turns returned `len(s_arr)` (full window). But turn validity
polygons are convex hulls + buffer, making polygon exit unreliable. Points
beyond the turn spline end clamp to the endpoint.

**Fix:** For turns, find the last index where `s_native <= L - 0.5m`.
This is now used in both `assign_segments` and `to_lane_coordinates_forced`.

```python
if seg_type == 'turn':
    geom_L    = geometry_store[seg_key]['total_length']
    tolerance = 0.5
    valid     = np.where(s_arr <= geom_L - tolerance)[0]
    handoff   = int(valid[-1]) + 1 if len(valid) > 0 else len(s_arr)
    return handoff, None
```

#### 3b. `is_departure` skips only primary `s_change`, not secondary boundaries
**Problem:** `is_departure=True` returned `len(s_arr)` immediately, preventing
secondary T-junction boundaries (`s_zollstr_*`) from firing even for departure
segments that turn off at MattInt.

**Fix:** `is_departure=True` now skips Step 1 (primary `s_change`) but still
runs Step 2 (secondary boundaries) and Step 3 (polygon exit fallback).

#### 3c. Step 1 `idx > 0` guard
**Problem:** `_sustained_crossing` firing at index 0 (cyclist starts past
`s_change`) returned `handoff_local=0` → empty `seg_indices` → chain break.

**Fix:** Step 1 only returns if `idx > 0`. If `idx == 0`, falls through to
Step 2 (secondary boundaries) — secondary boundary may still fire later
(e.g. veh 81: starts past MainInt `s_change`, but `s_zollstr_east_stop` fires
mid-trajectory).

---

### 4. `_is_departure` override block (in `assign_segments`)

**Problem:** Three vehicles exposed different failure modes of `_is_departure`:

- **Veh 12** (`Roentgenstr_EB`, `is_reverse=True`): approach segment traversed
  in reverse → cyclist heading away, no handoff needed → `_is_departure=True`.
- **Veh 15** (`Zollstr_EB`, `is_reverse=True`, mixed registry role): role flip
  gives `_early_role='departure'` → `_is_departure=True` suppresses handoff,
  but cyclist genuinely crosses `s_change` mid-trajectory.
- **Veh 107** (`Roentgenstr_EB`, `is_reverse=False`): starts at `s_change`,
  Step 1 fires at 0, falls through to Step 2 (no secondary boundaries) →
  Step 3 returns `len(s_arr)` → full window → turn picked up next iteration.

**Fix:** Two-case override at `iteration == 0`:

```python
if iteration == 0:
    if best_reverse and _early_role == 'approach':
        # Case A: reverse on approach → heading away → true departure
        _is_departure = True
    elif _early_role == 'departure':
        # Case B: mixed segment role flip may be too aggressive —
        # trial run to check if cyclist genuinely crosses s_change
        _trial_handoff, _ = _find_handoff(..., is_departure=False)
        if _trial_handoff > 0:
            _is_departure = False
```

---

### 5. Straight-through revert respects `_is_departure`

**Problem:** When `_confirm_minor_road_entry` returns False (straight through
at T-junction), the revert code ran a bare loop on `s_change` without
respecting `_is_departure`. For departure segments (already past `s_change`),
this fired at index 0 → `handoff_local=0` → break.

**Fix:** If `_is_departure=True`, use `len(s_win)` directly. The bare loop
also got the `i > 0` guard for consistency.

---

### 6. `_point_in_intersection` — `use_expanded` parameter

**Fix:** Added `use_expanded=True` parameter. The Group A/B demotion check
now uses `use_expanded=False` (raw polygon) so cyclists at the boundary mouth
(within the 1m buffer only) are not incorrectly demoted.

---

### 7. Symmetric Group A/B demotion

**Problem:** Only lane segments were demoted to Group B when starting inside
the intersection. Turns could beat lane segments even when the trajectory
starts on the approach road.

**Fix:** Symmetric demotion using `use_expanded=False` intersection polygon:
- Lane segment + first point **inside** intersection → Group B
- Turn segment + first point **outside** intersection → Group B

```python
first_in_intersection = _point_in_intersection(
    fragment_xy[0], geometry_store, use_expanded=False)
if seg_type_i == 'lane' and first_in_intersection:
    group_b[seg_key] = (entry_idx, exit_idx)
elif seg_type_i == 'turn' and not first_in_intersection:
    group_b[seg_key] = (entry_idx, exit_idx)
else:
    group_a[seg_key] = (entry_idx, exit_idx)
```

---

### 8. `_confirm_minor_road_entry` — Two fixes

#### 8a. Condition order inverted
**Problem:** Polygon containment check (a) ran before spline projection (b).
Points in the MattInt intersection area lie between validity polygons and were
rejected by (a) before (b) was tested.

**Fix:** Run spline projection + `s_change` check first (primary discriminator),
polygon containment second (spatial sanity check using `_poly_expanded`).

#### 8b. `remaining_xy` uses full tail
**Problem:** `remaining_xy = fragment_xy[best_entry + handoff_local : best_exit]`
— capped at `best_exit` (Zollstr polygon exit). Points entering Matteng were
beyond `best_exit` and never seen by confirmation.

**Fix:** `remaining_xy = fragment_xy[best_entry + handoff_local :]`
— unbounded, covers full remaining trajectory.

---

### 9. `agent_mode` parameter

Added `agent_mode='bike'` (default) to both `assign_segments` and
`to_lane_coordinates`. Filters candidate segments by mode:
- `'bike'` → `mode in ('shared', 'bike')`
- `'vehicle'` → `mode in ('shared', 'car')`
- Turn segments always included regardless of mode.

---

### 10. `to_lane_coordinates_forced` — New function

Bypasses polygon walk and scoring. Takes an explicit ordered chain of segment
keys and computes lane coordinates for each segment in sequence.

```python
df = to_lane_coordinates_forced(
    bike_df,
    forced_chain=['LangstrS_NB', 'turn_LangstrS_NB_2_LangstrN_NB', 'LangstrN_NB'],
    segment_registry=segment_registry,
    geometry_store=geometry_store,
    movement_registry=movement_registry,
    verbose=True,
)
```

- Reuses `_find_handoff`, `_is_departure` logic, `transform_segment`.
- Calls `_polygon_walk` per segment to bound the window by polygon exit.
- Turn handoff uses last-valid-s (`s <= L - 0.5m`) via `_find_handoff`.
- `match_quality='forced'` for all matched rows.
- `derive_movement_key` called with minimal chain entries `{seg_key, role}`.

---

## Changes to `tools_site_builder.py`

### 1. `build_segment_registry` — Validity polygon clipped at `s_change`

**Problem:** Validity polygons covered the full spline `(0, L)`, allowing
cyclists to match segments far into the intersection area.

**Fix:** Clip at `s_change` using the longer-interval heuristic:
```python
if s_change is not None:
    if s_change >= L - s_change:   # (0, s_change) is longer
        s_start, s_end = 0.0, s_change
    else:                           # (s_change, L) is longer
        s_start, s_end = s_change, L
else:
    s_start, s_end = 0.0, L
```

**Note:** Requires rebuilding the registry pickle after this change.

---

### 2. `build_turns` — Store `approach_s_change_key` and `departure_s_change_key`

**Problem:** `app_key` and `dep_key` were computed from `TURN_DEFS` and used
to build turn geometry, but never stored in `segment_registry`. This caused
`_confirm_minor_road_entry` to always find `approach_s_change_key=None` →
`minor_segs` always empty → T-junction confirmation always failed.

**Fix:** Added to turn segment registry entry:
```python
segment_registry[turn_key] = {
    ...
    'approach_s_change_key':  app_key,
    'departure_s_change_key': dep_key,
    ...
}
```

---

### 3. `build_intersection_polygon` — Stop-line clipping

**Problem:** Intersection polygons (convex hull of arm normal-line endpoints)
bled past the stop lines of perpendicular roads.

**Fix:** After building the convex hull, clip against each arm's stop line
using a half-plane. Inward side determined by `approach_seg_key`:
- `approach_seg_key == pos_seg_key` → intersection at s=L → keep high-s side
- `approach_seg_key != pos_seg_key` → intersection at s=0 → keep low-s side

New required key in `arm_defs`:
```python
arm_defs = [
    {'geom_key': 'Roentgenstr', 's_change_key': 's_change',
     'pos_seg_key': 'Roentgenstr_WB', 'opp_seg_key': 'Roentgenstr_EB',
     'approach_seg_key': 'Roentgenstr_EB'},   # EB approaches → s=0 side
    ...
]
```

Arms without `approach_seg_key` are skipped (backward compatible).

The clipping boundary uses `s_change_key` (not `s_stop`) so T-junction arms
(`s_zollstr_east_stop`, `s_zollstr_west_yield`) are clipped correctly.

---

## Vehicles Debugged and Fixed

| Veh | Issue | Fix |
|---|---|---|
| 12 | `Roentgenstr_EB` reverse=True, `handoff_local=0`, empty seg_indices | `_is_departure` Case A: reverse on approach → true departure |
| 15 | `Zollstr_EB` reverse=True, handoff suppressed by role flip on mixed segment | `_is_departure` Case B: trial run detects genuine mid-trajectory crossing |
| 38 | `LangstrS_SB` reverse=True (pure departure, flip to approach), handoff not firing | `_infer_role_from_registry` role flip + `_is_departure=False` |
| 81 | `Zollstr_EB` forward, starts past `s_change`, T-junction not confirmed → `minor_segs` empty | `approach_s_change_key` stored in registry; `remaining_xy` unbounded; condition order in `_confirm_minor_road_entry` |
| 107 | `Roentgenstr_EB` forward, starts at `s_change`, `handoff_local=0` | Step 1 `idx > 0` guard → falls through to Step 3 → full window → turn picked at next iteration |
| 31 | `Zollstr_EB` departure after turn, straight-through revert fired at 0 | Straight-through revert respects `_is_departure` |
| 109 | Turn matched instead of `Roentgenstr_EB` reverse, first point near intersection boundary | Symmetric demotion with `use_expanded=False` |
| 126 | Wrong segment matched due to `d_right` too tight | Site tuning in `maps_june_D1.py` (not a code fix) |
| 134 | Turn beat lane segment at iteration 0 on approach road | Symmetric Group A/B demotion |

---

## Known Open Issues / TODO

- [ ] `POLYGON_TOLERANCE = 1.0m` — may need per-site tuning.
- [ ] `POOR_MATCH_THRESHOLD = 3.0` — borderline scores could be relaxed slightly.
- [ ] Veh 283: `Matteng_NB` reverse=True scored `inf` (lateral veto) — needs
      investigation of `d_left`/`d_right` values for Matteng segments.
- [ ] Registry pickle must be rebuilt after all `tools_site_builder.py` changes.
- [ ] `maps_june_D1.py` arm_defs need `approach_seg_key` added for stop-line
      clipping to take effect.
- [ ] `maps_sep_D1H.py` and `maps_sep_D2I.py` not yet updated with new arm_defs
      format or tested against the new pipeline.