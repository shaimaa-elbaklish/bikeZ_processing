# Lane Coordinate Transform — Project README
**Project**: Bicycle & Vehicle Trajectory Analysis  
**Institution**: ETH Zürich, IVT — Institute for Transportation Planning and Systems  
**Last updated**: April 2026  
**Status**: Phase A complete, Phase B in progress (matching all-NaN bug being debugged)

---

## 1. Project Overview

Convert smoothed EKF/RTS trajectory data from XY:2056 (LV95/CH1903+, EPSG:2056) into
lane-aligned roadway coordinates `(s, d, ṡ, ḋ)`:

- `s` = arc-length along segment centerline [m], resets to 0 at each segment entry
- `d` = signed lateral offset from centerline [m] (positive = left of travel direction)
- `ṡ` = longitudinal speed [m/s]
- `ḋ` = lateral speed [m/s]

Secondary outputs: `in_bike_lane`, `d_to_bike_boundary`, `s_decreasing` (U-turn flag),
`s_ddot`, `d_ddot`.

**Dataset**: 12 intersections in Zürich, Switzerland. Dates: June 2025, September 2025.
Sampling rate: 20 Hz.

---

## 2. Coordinate System & Offsets

All splines are stored in **local coordinates** (LV95 minus bounding box offset):

```python
x_local = x_LV95 - X_2056_offset    # X_2056_offset = XY_2056_Bounds[0][0]
y_local = y_LV95 - Y_2056_offset    # Y_2056_offset = XY_2056_Bounds[1][0]
```

This is essential for numerical stability of `splprep` with large LV95 values (~1e6).
The offsets are stored inside `geometry_store`:

```python
geometry_store['x_offset'] = X_2056_offset
geometry_store['y_offset'] = Y_2056_offset
```

Trajectory data uses full LV95 in `x_act_ekf`, `y_act_ekf`. Always subtract offset
before projecting onto splines.

---

## 3. Repository Structure

```
project/
├── maps_June_D1.py                  # Phase A geometry preparation (per intersection)
├── main_coordinate_transform.py     # Phase C — main transform loop
├── tools_lane_coords.py             # Phase B — matching + transform functions
├── tools_infrastructure_geometry.py # Phase A helpers (spline fitting, registry building)
├── tools_coordinate_transform.py    # Core spline projection functions
├── tools_osmnx.py                   # OSMnx edge merging + spline fitting
├── tools_plotting.py                # Debug plots (geometry_store, turn splines)
├── tools_map_visualization.py       # Folium map helpers
├── tools_kalman.py                  # EKF + RTS smoother (upstream pipeline)
├── tools_gap_inference.py           # Clothoid gap reconstruction (upstream)
├── _constants.py                    # BikeZ_Config — all dataset constants
│
├── data/
│   ├── registry_{date}_{intersection}_{code}.pkl   # Phase A output (per location)
│   └── centerlines_splines_{date}_{intersection}.pkl  # legacy (pre-Phase A)
│
└── maps/
    ├── from_swisstopo/
    │   └── June_D1.kml              # Hand-drawn geometry (boundaries, stop/yield lines)
    └── registry_map_{date}_{intersection}.html  # Folium validation map
```

---

## 4. Three-Layer Registry Structure

Everything is serialized to `registry_{date}_{intersection}_{code}.pkl`:

```python
registry = {
    'metadata':         {...},
    'geometry_store':   geometry_store,
    'segment_registry': segment_registry,
    'movement_registry': movement_registry,
}
```

### 4a. `geometry_store`

Physical geometry objects, one per road. Keys: `'x_offset'`, `'y_offset'`, and one
entry per road name.

```python
geometry_store = {
    'x_offset': float,    # LV95 X offset for numerical stability
    'y_offset': float,    # LV95 Y offset for numerical stability
    'Roentgenstr': {
        'spline':       (tck, unew, cum_dist),  # scipy B-spline in LOCAL coords
        'line_wgs84':   Shapely LineString,      # raw OSMnx edge in WGS84
        'positive_dir': 'WB',                   # cardinal dir of increasing t
        'total_length': float,                  # arc-length [m]
        's_stop':       float,                  # stop-line arc-length [m], native
        's_yield':      float,                  # yield-line arc-length [m], native
    },
    # ... LangstrN, LangstrS, Zollstr
    # Turn entries (added in Phase A2):
    'turn_Roentgenstr_EB_2_LangstrN_NB': {
        'spline':       (tck, unew, cum_dist),
        'total_length': float,
        'positive_dir': None,   # turns have no cardinal direction
        's_stop':       None,
        's_yield':      None,
        'line_wgs84':   None,
        'method':       'clothoid' or 'hermite',
    },
}
```

**Key design**: `s_stop` and `s_yield` are in **native spline arc-length** (no ordering
assumed — sometimes `s_stop > s_yield` depending on spline parameterization direction).

### 4b. `segment_registry`

Directed typed segments. One entry per traversable unit.

```python
segment_registry = {
    'Roentgenstr_EB': {
        'type':             'lane',           # 'lane' or 'turn'
        'geometry_key':     'Roentgenstr',
        'direction':        'EB',
        'is_forward':       False,            # EB != positive_dir('WB')
        'mode':             'shared',         # 'car', 'bike', 'shared'
        'approach_native':  (s_yield, L),     # native arc-length (reverse)
        'departure_native': (0.0, s_stop),    # native arc-length (reverse)
        'bike_lane': {
            'w_bike':            1.6,         # [m]
            'd_boundary_spline': interp1d,    # d_boundary(s), scipy interp1d
            's_domain':          (s_min, s_max),
            'side':              +1,          # derived from sign of projected d
        }
    },
    # Turn segments:
    'turn_Roentgenstr_EB_2_LangstrN_NB': {
        'type':             'turn',
        'geometry_key':     'turn_Roentgenstr_EB_2_LangstrN_NB',
        'approach_seg':     'Roentgenstr_EB',
        'departure_seg':    'LangstrN_NB',
        'is_forward':       True,
        'mode':             'shared',
        'approach_native':  (0.0, total_length),
        'departure_native': (0.0, total_length),
        'bike_lane':        None,
    },
}
```

**`approach_native` / `departure_native` logic** (no swap — uses `s_stop` directly):
```python
if is_forward:
    approach_native  = (0.0,    s_stop)
    departure_native = (s_stop, L)
else:
    approach_native  = (s_stop, L)
    departure_native = (0.0,    s_stop)
```

### 4c. `movement_registry`

Named sequences of `(segment_key, role)` tuples.

```python
movement_registry = {
    'Roentgenstr_EB_2_LangstrN_NB': [
        ('Roentgenstr_EB',                    'approach'),
        ('turn_Roentgenstr_EB_2_LangstrN_NB', 'turn'),
        ('LangstrN_NB',                       'departure'),
    ],
    # ... 12 movements total for this intersection
}
```

---

## 5. Phase A — Geometry Preparation (per intersection, one-time)

**Script**: `maps_{date}_{location}.py`  
**Output**: `data/registry_{date}_{intersection}_{code}.pkl`

### Steps

| Step | Description | Key function |
|------|-------------|--------------|
| A1 | Fit OSMnx centerline splines (local coords) | `fit_spline_from_osmnx()` |
| A1 | Split Langstrasse N/S at intersection | `cut_line_at_stop()` |
| A1 | Compute s_stop, s_yield from KML stop/yield lines | `get_s_domain()` |
| A2 | Build turning movement splines via `connect_lines_g2` | `build_all_turns()` |
| A3 | Project bike lane boundaries → `d_boundary(s)` | `add_bike_lane_boundaries()` |
| A4 | Build segment_registry, movement_registry | `build_segment_registry()`, `build_movement_registry()` |
| A4 | Serialize to .pkl | `serialize_registry()` |

### KML layers (drawn in swisstopo per intersection)

| Description pattern | Type |
|---------------------|------|
| `{street}_{EB/WB/NB/SB}` | Car/bike lane boundary |
| `{street}_Stop` | Stop-line at intersection entry |
| `{street}_Yield` | Yield-line at intersection exit |
| `Intersection_Area` | Intersection box polygon |

**Naming convention** (must be consistent — no typos):
```
LangstrN_NB, LangstrN_SB, LangstrN_Stop, LangstrN_Yield
LangstrS_NB, LangstrS_SB, LangstrS_Stop, LangstrS_Yield
Zollstr_EB,  Zollstr_WB,  Zollstr_Stop,  Zollstr_Yield
Roentgenstr_EB, Roentgenstr_WB, Roentgenstr_Stop, Roentgenstr_Yield
Intersection_Area
```

### What to draw in swisstopo (per intersection)

1. **Bike lane boundaries** — polyline along car/bike lane boundary
2. **Stop-lines** — short line at intersection entry (near edge of crosswalk)
3. **Yield-lines** — short line at intersection exit (far edge of crosswalk)
4. **Intersection box** — polygon enclosing the full junction area
5. **Road centerlines** — from OSMnx (not hand-drawn, but clip to bbox)

**No need to draw**: turning movement paths (built algorithmically), bike lane
centerlines (derived from boundary + width).

### POSITIVE_DIR (must be set correctly per intersection)

Determined by inspecting OSMnx edge start/end points:

```python
POSITIVE_DIR = {
    'Roentgenstr': 'WB',   # spline starts east, ends west
    'Zollstr':     'EB',   # spline starts near intersection, ends east
    'LangstrN':    'NB',   # spline starts south, ends north
    'LangstrS':    'NB',   # same
}
```

Validate with the positive direction plot (red o = start, black x = end).

### Bike lane info (per intersection, per directed segment)

```python
BIKE_LANE_INFO = {
    'Roentgenstr_EB': {'w_bike': 1.6},
    'Roentgenstr_WB': {'w_bike': 1.6},
    'Zollstr_EB':     None,              # no dedicated bike lane
    'Zollstr_WB':     {'w_bike': 2.0},  # entire lane is bike
    'LangstrN_NB':    None,
    'LangstrN_SB':    None,
    'LangstrS_NB':    {'w_bike': 2.75},
    'LangstrS_SB':    {'w_bike': 2.75},
}
MODE = {
    'Zollstr_WB': 'bike',     # entire lane dedicated
    # all others: 'shared'
}
```

---

## 6. Phase B — Lane Coordinate Transform (`tools_lane_coords.py`)

### Key functions

| Function | Description |
|----------|-------------|
| `project_point_warm()` | Warm-started local spline projection (2-phase: coarse + refine) |
| `project_point_full()` | Full projection returning `(t*, tangent, normal, s, d)` |
| `score_segment()` | Score trajectory vs segment: proximity + heading + overlap |
| `match_segment()` | Match trajectory fragment to best segment from candidate list |
| `get_next_candidates()` | Registry lookup: given segment+role → possible next segments |
| `derive_movement_key()` | Build movement key from chain (exact or partial) |
| `find_handoff_index()` | First trajectory index where s crosses s_stop |
| `assign_segments()` | Full sequential chaining: matching phase |
| `transform_segment()` | Project + compute s,d,ṡ,ḋ for one segment |
| `to_lane_coordinates()` | Main entry point: matching + transformation for one vehicle |

### Sequential segment chaining algorithm

```
1. Candidates = all LANE segments (pass 1)
2. Match fragment → best segment + role + score
3. If score > threshold: retry with ALL segments including turns (pass 2)
4. If still poor: mark as 'unmatched', stop
5. Find handoff index (where s crosses s_stop)
6. Append to chain; remaining = trajectory after handoff
7. Next candidates = get_next_candidates(seg, role, movement_registry)
8. Repeat from step 2 with remaining trajectory
9. Cap at MAX_CHAIN_LENGTH iterations
```

### Matching score

```python
# Only in-domain points contribute to proximity + heading scores
# Score is inversely weighted by overlap fraction
score = (W_DIST * mean_abs_d + W_HEAD * heading_rmse) / overlap
```

- `W_DIST = 1.0`, `W_HEAD = 0.5`
- `MIN_OVERLAP_PTS = 5` (0.25s at 20 Hz)
- `POOR_MATCH_THRESHOLD = 2.0`

### Speed/acceleration decomposition

Uses velocity vector dot products (correct, reused from existing pipeline):
```python
vx = speed_ekf * cos(angle_ekf)
vy = speed_ekf * sin(angle_ekf)
s_dot = dot([vx, vy], tangent)
d_dot = dot([vx, vy], normal)
```

Tangent and normal are **flipped** for reverse segments (`is_forward=False`).

### U-turn flag

```python
s_decreasing = s_dot < -0.5   # [m/s] threshold
```

Post-processing (Phase D): detect apex as `argmax(s(t))`, split trajectory into
`{veh_id}_A` and `{veh_id}_B`.

### Bike lane membership

```python
d_bnd  = d_boundary_spline(s_native)
d_far  = d_bnd + side * w_bike
in_bike_lane = (min(d_bnd, d_far) - 0.4) <= d <= (max(d_bnd, d_far) + 0.4)
# Returns NaN if s outside bike lane s_domain
# Returns True always if mode == 'bike'
```

---

## 7. Phase C — Main Transform Loop (`main_coordinate_transform.py`)

```python
# Load registry
registry = pickle.load(open(f'data/registry_{date}_{intersection}_{code}.pkl', 'rb'))
geometry_store    = registry['geometry_store']
segment_registry  = registry['segment_registry']
movement_registry = registry['movement_registry']

# Loop
for veh_id in tqdm(unique_ids):
    veh_df = df[df['veh_id'] == veh_id].copy()
    veh_df = to_lane_coordinates(
        veh_df, movement_registry,
        segment_registry, geometry_store,
        max_chain_length=3
    )
    mod_df = pd.concat([mod_df, veh_df])
```

### Output columns added to dataframe

| Column | Type | Notes |
|--------|------|-------|
| `movement_key` | str | e.g. `'LangstrS_NB_2_Roentgenstr_WB'` or `'LangstrS_NB_2_unknown'` |
| `segment_id` | str | e.g. `'LangstrS_NB'` |
| `segment_type` | str | `'lane'` or `'turn'` |
| `segment_role` | str | `'approach'`, `'turn'`, `'departure'` |
| `s` | float | Directed arc-length from segment entry [m] |
| `d` | float | Signed lateral offset [m] |
| `s_dot` | float | Longitudinal speed [m/s] |
| `d_dot` | float | Lateral speed [m/s] |
| `s_ddot` | float | Longitudinal acceleration [m/s²] |
| `d_ddot` | float | Lateral acceleration [m/s²] |
| `in_bike_lane` | bool/NaN | NaN outside bike lane s_domain or in turn |
| `d_to_bike_boundary` | float | Signed distance to car/bike boundary [m] |
| `s_decreasing` | bool | U-turn flag: `s_dot < -0.5 m/s` |

---

## 8. Phase D — U-turn Post-processing (not yet implemented)

```python
# Detect U-turns: trajectories with sustained s_decreasing after an apex
# Split at argmax(s(t)) → {veh_id}_A (approach) and {veh_id}_B (departure)
# Save cross-reference table: original_id, half, split_id, apex_time
```

---

## 9. Known Issues & Open Items (as of April 2026)

| Issue | Status | Location |
|-------|--------|----------|
| All output columns NaN — matching phase returning empty chain | **ACTIVE BUG** | `tools_lane_coords.py` / `assign_segments()` |
| Debug plot `plot_lane_coord_debug()` written, not yet tested | Pending | `tools_plotting.py` |
| Clothoid spatial shift for Vehicle 14 (gap inference) | Deferred | `tools_kalman.py` |
| `match_bicycle_to_centerline_with_heading` still in codebase | Legacy | `tools_coordinate_transform.py` |

### Debugging the all-NaN issue

Run this diagnostic before calling `to_lane_coordinates`:

```python
x_offset = geometry_store['x_offset']
y_offset = geometry_store['y_offset']

# Check trajectory vs spline coordinate ranges
print(f"traj x∈[{veh_df['x_act_ekf'].min():.1f}, {veh_df['x_act_ekf'].max():.1f}]")
print(f"traj y∈[{veh_df['y_act_ekf'].min():.1f}, {veh_df['y_act_ekf'].max():.1f}]")
for geom_key, entry in geometry_store.items():
    if geom_key in {'x_offset', 'y_offset'} or entry.get('positive_dir') is None:
        continue
    tck, unew, _ = entry['spline']
    x_loc, y_loc = splev(unew, tck)
    print(f"{geom_key}: x∈[{x_loc.min()+x_offset:.1f}, {x_loc.max()+x_offset:.1f}]")
```

Likely cause: trajectory coordinates in local coords but comparison done in full LV95,
or vice versa. Check that `x_act_ekf = x_ekf + X_2056_offset` (full LV95) and that
`pt_local = xy - [x_offset, y_offset]` is applied in `score_segment`.

---

## 10. Checklist for a New Intersection

For each of the 12 intersections, repeat the following:

### Swisstopo drawing (one-time manual step)
- [ ] Draw bike lane boundaries (`{street}_{direction}`) for each arm with a bike lane
- [ ] Draw stop-lines (`{street}_Stop`) — one per arm at near edge of crosswalk
- [ ] Draw yield-lines (`{street}_Yield`) — one per arm at far edge of crosswalk
- [ ] Draw intersection box polygon (`Intersection_Area`)
- [ ] Export as KML to `maps/from_swisstopo/{date}_{location}.kml`
- [ ] Verify naming consistency (no typos in Description field)

### Python setup (per intersection)
- [ ] Copy `maps_June_D1.py` → `maps_{date}_{location}.py`
- [ ] Update street names for OSMnx extraction
- [ ] Set `POSITIVE_DIR` for each road (validate with start/end plot)
- [ ] Set `BIKE_LANE_INFO` and `MODE` per directed segment
- [ ] Set `TURNING_MOVEMENTS` (N×(approach, departure) pairs)
- [ ] Set `MOVEMENTS` for `movement_registry`
- [ ] Run Phase A, validate with `plot_geometry_store()` and folium map
- [ ] Serialize registry to `data/registry_{date}_{location}_{code}.pkl`

### Validation checks
- [ ] `plot_geometry_store`: approach/departure domains correct for each direction
- [ ] Stop/yield crossing dots sit on intersection boundary (not mid-road)
- [ ] Bike lane boundaries on correct side, `side` sign correct
- [ ] Turn splines start/end at stop/yield lines, stay inside intersection polygon
- [ ] Folium map: toggle each movement layer, verify full path is physically correct
- [ ] `plot_lane_coord_debug`: trajectory matches expected segment, s increases monotonically

---

## 11. Key Design Decisions (rationale summary)

| Decision | Rationale |
|----------|-----------|
| Segmented not homogeneous representation | `d` only meaningful relative to specific lane geometry |
| `s_stop` as sole boundary (not `s_yield`) | Gap between stop/yield is ~7m — unlikely to be observed alone |
| No `direction` field in registry — inferred from segment key suffix | Avoids redundancy; `_EB/_WB/_NB/_SB` suffix is canonical |
| `side` derived from sign of projected `d`, not hardcoded | Robust to geometry changes; eliminates manual error |
| Splines in local coords (offset subtracted) | `splprep` fails with large LV95 values (~1e6) due to floating-point precision |
| Turn segments use B-splines (not clothoid objects) | Unified representation; `_eval_clothoid` not needed |
| U-turns flagged by `s_decreasing`, split in post-processing | Keeps transform dumb; splitting requires full trajectory context |
| Sequential chaining (not global movement match) | Robust to partial trajectories of any length |
| Inversely weighted matching score: `(dist + head) / overlap` | Prevents short segments winning with locally perfect but partial match |

---

## 12. Dependencies

```python
# Core
numpy, pandas, scipy, matplotlib

# Geospatial
geopandas, pyproj, shapely, folium, osmnx

# Splines / clothoids
scipy.interpolate.splprep, splev, interp1d
pyclothoids (SolveG2)

# Internal
tools_kalman.py          # _eval_clothoid (used in gap inference, not turns)
tools_coordinate_transform.py  # convert_xy2056_to_roadway_coordinates
                               # convert_roadway_to_xy2056_coordinates
                               # fit_roadway_centerline_spline
                               # connect_lines_g2
                               # cut_line_at_stop
```

---

## 13. Context for Resuming Conversation

When starting a new conversation, share this README and the following files:
- `tools_lane_coords.py` — Phase B (current active development)
- `tools_infrastructure_geometry.py` — Phase A helpers
- `tools_coordinate_transform.py` — core projection functions
- `maps_{date}_{location}.py` — the specific intersection being worked on
- The active `.pkl` registry file path

**Current active task**: Debug why `assign_segments()` returns an empty chain
(all output columns NaN). The debug plot `plot_lane_coord_debug()` is the primary
diagnostic tool. Suspected cause: coordinate space mismatch between trajectory
points and splines in `score_segment()`.