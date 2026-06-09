# Lane Coordinate Transform — Development Plan
**Project**: Bicycle Trajectory Analysis — ETH Zürich IVT  
**Status**: Pre-implementation design document  
**Last updated**: April 2026

---

## Background & Goal

Convert smoothed bicycle/vehicle trajectories from XY:2056 (LV95/CH1903+) into
lane-aligned roadway coordinates `(s, d, ṡ, ḋ)`, where:

- `s` = arc-length along the reference segment centerline [m]
- `d` = signed lateral offset from the centerline [m] (positive = left of travel direction)
- `ṡ` = longitudinal speed [m/s]
- `ḋ` = lateral speed [m/s]

The transform runs on **RTS-smoothed EKF output** (already in LV95). Bike lane
membership is derived from `d` relative to a known car/bike boundary offset and
lane width.

---

## Core Design Decisions

### 1. Segmented representation (not homogeneous)

Each vehicle trajectory is decomposed into a named **sequence of segments**
rather than projected onto a single monolithic centerline. Example:

```
movement "S_2_E":
    Kasernenstr_NB  →  S_2_E_turn  →  Gessnerbrucke_EB
    (approach)          (turn)          (departure)
```

**Why**: lateral offset `d` only has physical meaning relative to a specific lane
geometry. A single `s` across the full movement loses interpretability at the
intersection; bike lane membership is undefined inside the intersection box.

### 2. Three segment types

| Type | Geometry | `d` meaningful | Bike lane |
|------|----------|----------------|-----------|
| `ApproachSegment` | B-spline | Yes | Possible |
| `TurnSegment` | Clothoid chain | Computed but flagged | Never |
| `DepartureSegment` | B-spline | Yes | Possible |

### 3. Bike lane from boundary + width

No bike lane centerline available. Instead, for each approach/departure segment:
- One car/bike boundary polyline (from swisstopo, hand-drawn)
- One measured bike lane width `w_bike` [m]
- Boundary side (which side of the boundary is the bike lane)

These are projected onto the segment centerline to produce `d_boundary(s)` — a
1D spline giving boundary offset as a function of arc-length — and then:

```
d_bike_center(s) = d_boundary(s) + sign * w_bike / 2
in_bike_lane     = |d − d_bike_center(s*)| < w_bike / 2 + tolerance
```

This handles boundaries that are not parallel to the centerline.

### 4. Corrected speed decomposition

Use heading-relative decomposition, not global angle projection:

```
Δψ(t) = ψ_vehicle(t) − ψ_lane(s*(t))
ṡ(t)  = v(t) · cos(Δψ(t))
ḋ(t)  = v(t) · sin(Δψ(t))
```

`ψ_lane(s*)` is derived from `arctan2(dy/dt, dx/dt)` of the spline at `t*`
via `splev(t*, tck, der=1)`. This is correct on curved roads; the current
global-angle approach is only accurate on straight segments.

### 5. Warm-started local projection

Replace the current `minimize_scalar(bounds=(0,1))` global search with:
1. **Coarse pass**: evaluate spline at ~50 uniform `t` values, find nearest
2. **Refine**: `minimize_scalar` restricted to `[t_prev − δ, t_prev + δ]`

Benefits: faster at 20 Hz, robust against snapping to the wrong segment at
approach/intersection boundary.

### 6. Bidirectionality

Some links (especially bike infrastructure) carry traffic in both directions on
a single physical geometry. A single spline is stored but two **directed views**
are instantiated:

```
BidirectionalSplineSegment(spline)
    .as_directed(forward=True)   →  BSplineSegment  (parameterization direction)
    .as_directed(forward=False)  →  BSplineSegment  (reversed: d negated, ψ flipped by π)
```

In the registry each directed entry shares a `geometry_key` but has its own
`direction` field and direction-aware `side` for bike lane attachment:

```python
"Veloweg_NB": {
    "geometry_key": "Veloweg",   # shared physical spline
    "direction": "forward",
    "bike_lane": { "side": -1, "w_bike": 1.8, ... }
},
"Veloweg_SB": {
    "geometry_key": "Veloweg",   # same spline object, no duplication
    "direction": "reverse",
    "bike_lane": { "side": +1, "w_bike": 1.8, ... }  # side flips with direction
}
```

When `direction="reverse"`:
- `s = S_total − s_projected`
- `d = −d_projected`
- `ψ_lane = ψ_projected + π`

### 7. U-turns — flag, don't handle specially

U-turns are rare but present. The lane coordinate transform applies **no special
handling**. It projects faithfully onto the assigned segment sequence and outputs
`s(t)` and `d(t)` as-is. On the departure leg of a U-turn, `s` will decrease
as `t` increases — this is correct output, not an error.

A per-timestep flag is added to the output:

```python
df['s_decreasing'] = df['s_dot'] < −0.5   # [m/s] threshold, tunable
```

**Critical**: the segment transition logic must not advance to the next segment
because `s` decreases. Transitions are triggered only by `s ≥ s_max` (forward
progress), never by the sign of `ṡ`.

**Post-processing** (Phase D, outside the transform) handles ID splitting:
1. Detect U-turn apex as `argmax(s(t))`
2. Split trajectory at apex into `{veh_id}_A` (approach) and `{veh_id}_B`
   (departure)
3. Maintain a cross-reference table `(original_id, half) → split_id` so the
   full maneuver can be reconstructed for maneuver-level analysis if needed

This keeps the transform dumb and the splitting logic in one place with access
to the full trajectory.

---

## What Needs to Be Built

### Phase A — Geometry preparation (manual + scripted, one-time per intersection)

This is the prerequisite for everything else. Currently geometry is expressed as
full-movement centerlines (e.g. `S_2_E`). It must be re-expressed as typed,
named segments.

**Step A1 — Trim approach/departure splines at intersection boundary**
Each approach spline must end at the stop-line; each departure spline must start
at the yield-line. Use `cut_line_at_stop` (already in `tools_coordinate_transform.py`)
with intersection boundary polygons or stop-line geometries.

Output: `Kasernenstr_NB`, `Gessnerbrucke_EB`, etc. — B-splines from road start
to stop-line only.

**Step A2 — Build turning movement clothoids**
For each turning movement (S→E, S→N, through, etc.), construct a G2 clothoid
chain from the approach stop-line to the departure yield-line using
`pyclothoids.SolveG2`. Store `(pieces, lengths, S_total)`.

The gap inference pipeline already has this machinery (`_build_gap_geometry`,
`_eval_clothoid` in `tools_kalman.py`) and can be reused directly.

For U-turns: `SolveG2` is numerically sensitive near π heading difference.
Prefer a semicircular arc or Hermite fallback for U-turn clothoids, consistent
with the existing fallback chain in `_build_gap_geometry`.

**Step A3 — Project bike lane boundaries onto segment centerlines**
For each approach/departure segment with a bike lane:
1. Project each boundary polyline point `b_i` onto the segment centerline
   → sparse `(s_i, d_boundary_i)` table
2. Fit a 1D spline `d_boundary(s)` over that table
3. Record `w_bike` (measured from swisstopo), `side` (+1 or −1), and
   `s_domain = (s_min, s_max)` of the boundary coverage

Outside `s_domain`: `in_bike_lane = NaN` (boundary marking ends before
stop-line and resumes after yield-line — normal at intersections).

**Step A4 — Build geometry registry**
Serialize the full segment graph per intersection/date into a `.pkl` or
structured dict. Schema:

```python
registry = {
    "S_2_E": [
        {
            "type": "approach",
            "key": "Kasernenstr_NB",
            "geometry_key": "Kasernenstr",   # for bidirectional sharing
            "direction": "forward",
            "spline": (tck, unew, cum_dist),
            "bike_lane": {
                "d_boundary_spline": ...,    # 1D spline d_boundary(s)
                "w_bike": 1.8,               # [m]
                "side": -1,                  # −1 = bike lane right of boundary
                "s_domain": (s_min, s_max)   # valid arc-length range
            }
        },
        {
            "type": "turn",
            "key": "S_2_E_turn",
            "clothoid": (pieces, lengths, S_total)
        },
        {
            "type": "departure",
            "key": "Gessnerbrucke_EB",
            "geometry_key": "Gessnerbrucke",
            "direction": "forward",
            "spline": (tck, unew, cum_dist),
            "bike_lane": None
        }
    ],
    ...
}
```

---

### Phase B — Core transform module (`tools_lane_coords.py`, new file)

**Step B1 — `BSplineSegment` class**
Wraps a fitted B-spline segment with an explicit direction. Responsibilities:
- `project(x, y, t_init=None)` → `(s, d, tangent, normal, t_star)` using
  warm-started local search; applies direction convention to `s`, `d`, `ψ_lane`
- `tangent_at(t)` → `(tx, ty)` via `splev(t, tck, der=1)`, flipped if reverse
- `in_bike_lane(s, d)` → `bool | NaN` using `d_boundary(s)` and `w_bike`;
  returns `NaN` outside `s_domain`
- `d_to_bike_boundary(s, d)` → signed distance to boundary [m]; `NaN` outside
  `s_domain`

**Step B2 — `ClothoidSegment` class**
Wraps a clothoid chain `(pieces, lengths, S_total)`. Responsibilities:
- `project(x, y, s_init=None)` → `(s, d, tangent, normal)` via coarse
  arc-length grid search then local refinement using `_eval_clothoid`
- `tangent_at(s)` → from `_eval_clothoid`
- `in_bike_lane` always returns `NaN`
- `d_to_bike_boundary` always returns `NaN`

**Step B3 — `to_lane_coordinates(filt_veh_df, segment_sequence)` function**
Main transform. For each timestep:
1. Determine active segment — advance when `s ≥ s_max` of current segment
   (never advance on decreasing `s` alone — U-turn safe)
2. Project `(x, y)` onto active segment, warm-started from previous `t*` or `s*`
3. Compute `Δψ = ψ_vehicle − ψ_lane(s*)`, then `ṡ = v·cos(Δψ)`, `ḋ = v·sin(Δψ)`
4. Do **not** clamp `ṡ` to non-negative — negative `ṡ` is valid for U-turn
   departure legs and reverse-directed bidirectional segments
5. Set `s_decreasing = (ṡ < −threshold)` flag
6. Query `in_bike_lane(s, d)` and `d_to_bike_boundary(s, d)`
7. Record all output columns

---

### Phase C — Refactor `main_coordinate_transform.py`

Replace the current monolithic per-row `convert_xy2056_to_roadway_coordinates`
loop with:

1. Look up movement key → segment sequence from registry
2. Call `to_lane_coordinates(bike_df, segment_sequence)`
3. Append results to output dataframe

The outer per-vehicle loop and file I/O remain unchanged.

---

### Phase D — U-turn post-processing (new script or function)

After the lane coordinate transform is complete, a separate post-processing step
handles U-turn ID splitting:

1. Identify trajectories with sustained `s_decreasing` after a clear apex
2. Detect apex as `argmax(s(t))`
3. Split into `{veh_id}_A` and `{veh_id}_B` at the apex timestep
4. Write cross-reference table: `original_id, half, split_id, apex_time`
5. Output a new dataframe with split IDs, otherwise identical schema

This step is intentionally separate from Phase C so the transform output is
always complete and unsplit — the original trajectory is preserved for
maneuver-level analysis if needed.

---

## Output Schema

Replaces / extends the current output columns:

| Column | Type | Notes |
|--------|------|-------|
| `segment_type` | str | `'approach'`, `'turn'`, `'departure'` |
| `segment_id` | str | e.g. `'Kasernenstr_NB'` |
| `s` | float | Arc-length from segment origin [m] |
| `d` | float | Signed lateral offset from segment centerline [m] |
| `s_dot` | float | Longitudinal speed [m/s]; negative valid for U-turns / reverse |
| `d_dot` | float | Lateral speed [m/s] |
| `d_to_bike_boundary` | float | Signed distance to car/bike boundary [m]; NaN in turn or outside s_domain |
| `in_bike_lane` | bool/NaN | NaN in turn or outside bike lane s_domain |
| `s_decreasing` | bool | True when ṡ < −threshold; U-turn flag for Phase D |

---

## What Stays Unchanged

- `fit_roadway_centerline_spline` — no changes needed
- `match_bicycle_to_centerline` / `_with_heading` — centerline selection works
- All EKF/Kalman pipeline (`tools_kalman.py`) — purely post-processing layer
- Outer loop structure and file I/O in `main_coordinate_transform.py`
- `cut_line_at_stop` — reused in Phase A1

---

## Known Issues in Current Code to Fix

| Issue | Location | Fix |
|-------|----------|-----|
| Global projection `minimize_scalar(0,1)` — slow and fragile near segment junctions | `project_point_onto_spline` | Warm-started local search (Step B1) |
| Speed decomposition uses global angle, not `Δψ` | `main_coordinate_transform.py` | Use `ψ_lane(s*)` from spline derivative (Step B3) |
| Fixed `d` threshold for bike lane — ignores non-parallel boundaries | `main_coordinate_transform.py` | `d_boundary(s)` spline (Step A3) |
| No segment typing — `d` and `in_bike_lane` computed inside intersection | `main_coordinate_transform.py` | Segment transition logic (Step B3) |
| No handling of bidirectional links — `d` sign undefined for opposing direction | `main_coordinate_transform.py` | Directed segment views (Steps B1, A4) |

---

## Implementation Order

```
A1  Trim approach/departure splines at stop-lines
A2  Build turning movement clothoids
A3  Project bike lane boundaries → d_boundary(s) splines
A4  Build and serialize geometry registry (includes direction + geometry_key fields)

B1  BSplineSegment class (with direction support)
B2  ClothoidSegment class
B3  to_lane_coordinates() function (U-turn safe, s_decreasing flag)

C   Refactor main_coordinate_transform.py

D   U-turn post-processing: apex detection, ID splitting, cross-reference table
```

Steps A1–A4 must precede B and C. Steps B1, B2, B3 can be developed and
unit-tested in parallel once the registry schema (A4) is fixed. Phase D can
be written any time after B3 is complete.

---

## Open Questions (to resolve before or during implementation)

1. **How many movements per intersection?** Determines manual effort for A1–A4.
   A four-arm signalized intersection yields ~12 movements (3 destinations × 4
   origins) plus U-turns.
2. **Are turning movement clothoids already stored** from the gap inference
   pipeline, or must they be rebuilt? If rebuilt, `_build_gap_geometry` /
   `_eval_clothoid` from `tools_kalman.py` apply directly.
3. **Intersection boundary polygons**: are stop-lines / yield-lines already
   digitized, or must they be drawn in swisstopo for A1?
4. **Sign convention for `d`**: confirm positive = left of spline parameterization
   direction, fixed to geometry not vehicle heading. Flips automatically for
   reverse-directed segments.
5. **Tolerance for bike lane membership**: `w_bike/2 + tolerance` — suggested
   0.3–0.5 m to account for residual trajectory noise after RTS smoothing.
6. **U-turn threshold for `s_decreasing` flag**: suggested −0.5 m/s for `ṡ`;
   should be robust to small projection noise near `ṡ ≈ 0` at low speed.
7. **How many U-turns are in the dataset?** Determines whether Phase D warrants
   a full automated pipeline or a simpler manual review step.