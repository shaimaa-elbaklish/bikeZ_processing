"""
TITLE OF PAPAER
-------------------------------------------
Authors:        Shaimaa El-Baklish
Organization:   ETH Zürich, Switzerland, IVT - Institute for Transportation Planning and Systems
Development:    2025-2026
Submitted to:   JOURNAL
-------------------------------------------
"""

# #############################################################################
# IMPORTS
# #############################################################################
import sys
import shapely

import numpy as np
import pandas as pd

from dataclasses import dataclass
from scipy.interpolate import splev

from tools_utils import _get_logger
from tools_utils import w_bike_at

# #############################################################################
# CONSTANTS
# #############################################################################

# --- breakpoint proposal -----------------------------------------------------
MERGE_M            = 0.5     # merge candidates within this arc distance
HYST_M             = 0.5     # suppress re-crossing of the same gate within
HEAD_WIN_M         = 5.0     # sliding window for heading-change detection
HEAD_THRESH        = np.pi / 4 # [rad] heading change over the window
DENSE_M            = 2.0     # fallback: densify breakpoints every DENSE_M

# --- run structure -----------------------------------------------------------
V_STOP             = 0.5     # [km/h] below this a node is excluded from the
                             #        run-level heading statistic
MIN_RUN_M          = 2.0     # runs shorter than this are structurally forbidden
W_MOV_MIN          = 1.0     # [m] of moving arc needed before heading counts

# --- emission scales ---------------------------------------------------------
SIGMA_D_LANE       = 3.0
SIGMA_D_TURN       = 7.0
SIGMA_PSI          = np.pi / 4
SIGMA_CLAMP        = 1.0     # metres of overshoot past the spline end
SIGMA_BND          = 10.0    # metres of shortfall before s_change
CONTRAFLOW_ENTRY_PRIOR = {'bike': 12.0, 'vehicle': 60.0}
REVERSE_TURN_ENTRY_PRIOR = {'bike': 4.0, 'vehicle': 35.0}

# --- emission caps (outlier mixture: beyond this, no further information) ----
C_D                = 9.0
C_OUT              = 1.0     # binary: cost charged per node outside the corridor
C_PSI              = 4.0
C_CLAMP            = 9.0
C_BND              = 4.0
 
# --- emission weights --------------------------------------------------------
W_LAT              = 1.0
W_COR              = 3.0     # multiplies the ARC FRACTION spent outside
W_PSI              = 2.0
W_CLAMP            = 2.0
W_PRG              = 6.0

# --- transition --------------------------------------------------------------
TAU_PER_RUN        = 3.0     # regularises against inventing breakpoints
LAM_CONT           = 4.0     # turn must be entered near s = 0
LAM_BND            = 3.0     # lane exit should happen near s_change

# Transition class priors, per mode. Read as: "metres-of-evidence units the
# alternative hypothesis must save before this manoeuvre is believed."
TAU_BY_MODE = {
    'bike': {
        'legal':     2.0,
        'revturn':   8.0,    # registered turn maneuver ridden BACKWARDS
        'cross':     8.0,    # corridor changes, direction does not
        'flip':     15.0,    # direction flips, corridor does not (U-turn in place)
        'uturn':    25.0,    # both change
        'irregular': 40.0,   # unregistered link-to-link (crosswalk, corner cut)
        'off':      30.0,
        'blocked':  np.inf,
    },
    'vehicle': {
        'legal':     2.0,
        'revturn':   10.0,    # a car does not circulate a roundabout clockwise
        'cross':     20.0,
        'flip':      60.0,
        'uturn':     25.0,
        'irregular': 90.0,
        'off':       30.0,
        'blocked':   np.inf,
    },
}

# --- match quality thresholds (cost per metre) -------------------------------
Q_GOOD             = 3.5
Q_POOR             = 8.0
OFF_COST_PER_M     = 15.0    # flat per-metre emission for the OFF state
COST_PER_M_MAX     = 6.0     # above this, trigger the dense-breakpoint fallback

# --- misc --------------------------------------------------------------------
PROJ_MARGIN_M      = 3.0     # corridor buffer defining the projection mask
GATE_EXCLUDE_KEYS  = {'s_stop', 's_yield'}   # behavioural, not topological
 
OPP_DIR = {'EB': 'WB', 'WB': 'EB', 'NB': 'SB', 'SB': 'NB',
           'NE': 'SW', 'SW': 'NE'}
 
_ALLOWED_MODES = {
    'bike':    {'shared', 'bike'},
    'vehicle': {'shared', 'car'},
}
LUT_RESOLUTION      = 500      # spline LUT resolution
BIKE_LANE_TOLERANCE = 0.2      # [m] d tolerance inside bike lane boundary
CLAMP_TOL           = 0.001    # t values within this of 0/1 are clamped

# #############################################################################
# DATACLASSES
# #############################################################################
@dataclass(frozen=True)
class Gate:
    """
    An oriented iso-s cross-section on a centerline.
 
    No geometry is stored: a crossing is detected as a sign change of
    (proj_s - s_value), bounded laterally by d_max and guarded against
    clamped projections.
    """
    key: str                  # 'Zollstr:s_change'
    geom_key: str
    s_key: str                # 's_change', 's_zollstr_west_yield', ...
    s_value: float
    d_max: float              # lateral bound, derived from corridor widths
    periodic: bool = False    # closed centerline: s wraps at total_length
    total_length: float = 0.0


@dataclass(frozen=True)
class State:
    """
    (corridor, travel direction). Two states per lane corridor; one per turn.
 
    A '_rev' state shares the parent's polygon, spline, projection column and
    lane bands — it differs only in travel direction, which flips the sign of
    d and rotates the lane heading by pi.
    """
    key:            str
    corridor:       str      # segment_registry key that owns polygon / d bounds
    geom_key:       str      # geometry_store key that owns the spline
    geom_col:       int      # column into Proj per-geometry arrays
    poly_col:       int      # column into Proj.poly_dist (per corridor)
    travel_forward: bool
    type:           str      # 'lane' | 'turn' | 'off'
    d_sign:         int
    psi_offset:     float
    entry_prior: float = 0.0
    d_center:    float = 0.0
    d_half:      float = 3.0
    road:        str   = ''      # _road_base(geom_key) — shared by both directions
    contraflow:  bool  = False   # True for the mirrored (_rev) state
    periodic:    bool  = False   # closed corridor: s wraps at total_length
    total_length: float = 0.0


@dataclass(frozen=True)
class Crossing:
    gate_key: str
    sign:     int      # +1 if s increasing at the crossing, -1 otherwise
    node:     int      # snapped node index
    frac:     float    # exact sub-node position, kept for reporting
 
 
@dataclass
class Traj:
    """Per-trajectory arrays. Node index == positional row index in bike_df."""
    xy:          np.ndarray   # (N, 2)
    psi:         np.ndarray   # (N,)   [rad]
    speed:       np.ndarray   # (N,)   [km/h] — as stored upstream of main
    t_sec:       np.ndarray   # (N,)   [s]
    ds:          np.ndarray   # (N,)   arc weight per node [m]
    arc:         np.ndarray   # (N,)   cumulative arc length [m]
    node_to_row: np.ndarray   # (N,)   positional index into bike_df
 
    @property
    def n(self) -> int:
        return len(self.xy)
 
 
@dataclass
class Proj:
    """
    Projection arrays. Per-geometry columns are shared by the forward and
    reverse states of the same corridor; d_sign / psi_offset are applied at
    read time, never here.
    """
    s:         np.ndarray   # (N, G)
    d:         np.ndarray   # (N, G)  native (spline-left positive)
    psi_tan:   np.ndarray   # (N, G)  spline tangent heading
    t:         np.ndarray   # (N, G)
    overshoot: np.ndarray   # (N, G)  metres past a clamped endpoint, along endpoint tangent
    inside:    np.ndarray   # (N, P)  bool — exact corridor containment
    near:      np.ndarray   # (N, P)  bool — containment in buffered corridor
    valid:     np.ndarray   # (N, G)  bool — projection actually computed
 
 
@dataclass
class Pre:
    """Prefix sums, shape (N+1, S). Makes runcost O(1) per span."""
    ds:     np.ndarray
    lat:    np.ndarray
    cor:    np.ndarray
    clamp:  np.ndarray
    ang2:   np.ndarray
    ds_mov: np.ndarray
    fv:     np.ndarray   # (N+1, S) first valid index >= i, else N
    lv:     np.ndarray   # (N,   S) last  valid index <= i, else -1
 
 
@dataclass
class Run:
    a:           int
    b:           int          # half-open [a, b)
    state:       State
    cost:        float
    cost_per_m:  float
    trans_class: str
    role:        str = 'approach'

# #############################################################################
# 1. SITE SETUP
# #############################################################################
def _road_base(geom_key):
    """
    Road identity shared by both directions.

    Geometry keys are sometimes undirected ('BaslerstrE', hosting both the
    WB and EB corridors) and sometimes directed ('KasernenstrN_NB', where
    each direction has its own spline and geometry key == segment key).
    Stripping a trailing direction token collapses both conventions onto one
    key, so 'is this the same road' works either way.
    """
    base, _, tail = geom_key.rpartition('_')
    return base if base and tail in OPP_DIR else geom_key


def build_state_space(geometry_store, segment_registry, mode='bike',
                      allow_reverse_turns=False):
    """
    Two states per lane corridor, one per turn, plus OFF.
 
    The mode filter is PHYSICAL accessibility only — a bollarded cycle path is
    absent from the vehicle state space. Legality is never a filter; it lives
    in the transition priors so that illegal manoeuvres are decoded and
    classified rather than discarded.
    
    allow_reverse_turns: Mint a '_rev' state for turn corridors too. Off by default.
    """
    allowed = _ALLOWED_MODES.get(mode, {'shared', 'bike', 'car'})
 
    geom_keys, poly_keys = [], []
 
    def _gcol(k):
        if k not in geom_keys:
            geom_keys.append(k)
        return geom_keys.index(k)
 
    def _pcol(k):
        if k not in poly_keys:
            poly_keys.append(k)
        return poly_keys.index(k)
 
    states = []
    for seg_key, ent in segment_registry.items():
        if seg_key in {'x_offset', 'y_offset'} or not isinstance(ent, dict):
            continue
        stype = ent.get('type')
        if stype not in ('lane', 'turn', 'ring'):
            continue
        if stype in ('lane', 'ring') and ent.get('mode', 'shared') not in allowed:
            continue
 
        gkey = ent['geometry_key']
        geo  = geometry_store[gkey]
        per  = bool(geo.get('periodic'))
        Lg   = float(geo.get('total_length', 0.0))
        gc, pc = _gcol(gkey), _pcol(seg_key)
        fwd = bool(ent['is_forward'])
        road = gkey if stype == 'turn' else _road_base(gkey)
        
        dl = ent.get('d_left', SIGMA_D_LANE if stype in ('lane', 'ring') else SIGMA_D_TURN)
        dr = ent.get('d_right', SIGMA_D_LANE if stype in ('lane', 'ring') else SIGMA_D_TURN)
        states.append(State(key=seg_key, corridor=seg_key, geom_key=gkey,
                            geom_col=gc, poly_col=pc, travel_forward=fwd,
                            type=stype,
                            d_sign=(1 if fwd else -1),
                            psi_offset=(0.0 if fwd else np.pi),
                            d_center=(dl - dr) / 2.0,
                            d_half=max((dl + dr) / 2.0, 1.0),
                            road=road, contraflow=False,
                            periodic=per, total_length=Lg))
        
        rev_types = ('lane', 'ring') + (('turn',) if allow_reverse_turns else ())
        if stype in rev_types:
            # Mirrored state: same corridor, opposite travel direction.
            # This is what makes "riding EB on the WB sidewalk" expressible.
            prior = (REVERSE_TURN_ENTRY_PRIOR[mode] if stype == 'turn'
                     else CONTRAFLOW_ENTRY_PRIOR[mode])
            states.append(State(key=seg_key + '_rev', corridor=seg_key,
                                geom_key=gkey, geom_col=gc, poly_col=pc,
                                travel_forward=(not fwd), type=stype,
                                d_sign=(-1 if fwd else 1),
                                psi_offset=(np.pi if fwd else 0.0),
                                d_center=(dl - dr) / 2.0,
                                d_half=max((dl + dr) / 2.0, 1.0),
                                road=road, contraflow=True,
                                periodic=per, total_length=Lg,
                                entry_prior=prior))
 
    off = State(key='OFF', corridor='OFF', geom_key='OFF', geom_col=-1,
                poly_col=-1, travel_forward=True, type='off',
                d_sign=1, psi_offset=0.0)
    states.append(off)
    
    geom_to_polycols = {}
    for st in states:
        if st.type == 'off':
            continue
        geom_to_polycols.setdefault(st.geom_key, [])
        if st.poly_col not in geom_to_polycols[st.geom_key]:
            geom_to_polycols[st.geom_key].append(st.poly_col)
    
    return states, geom_keys, poly_keys, geom_to_polycols


def build_transition_table(states, segment_registry, movement_registry):
    """
    Classify every ordered state pair once.
 
        legal      registered movement adjacency
        revturn    registered movement adjacency, ridden backwards
        cross      corridor changes, travel direction does not
        flip       travel direction flips, corridor does not (U-turn in place)
        uturn      both change (across the carriageway, reversing)
        irregular  different geometries, corridors touch, not registered
        blocked    no geometric adjacency
 
    Successors of 'X_rev' are the successors of the OPPOSITE-direction
    segment, since that is the direction of travel.
 
    Returns
    -------
    tau_class : (S, S) array of str
    """
    S = len(states)
    idx = {st.key: i for i, st in enumerate(states)}
 
    # --- registered adjacency on segment keys --------------------------------
    succ = {}
    pred = {}
    for sequence in movement_registry.values():
        for i in range(len(sequence) - 1):
            a, b = sequence[i][0], sequence[i + 1][0]
            succ.setdefault(a, set()).add(b)
            pred.setdefault(b, set()).add(a)
    
    # (road, direction) -> segment key. Works for both geometry conventions.
    by_road_dir = {}
    for k, e in segment_registry.items():
        if not isinstance(e, dict) or e.get('type') != 'lane':
            continue
        rk = (_road_base(e['geometry_key']), e.get('direction'))
        if rk in by_road_dir:
            raise ValueError(f"road/direction collision: {rk} maps to both "
                             f"'{by_road_dir[rk]}' and '{k}'")
        by_road_dir[rk] = k
    
    def _legal_targets(st):
        """
        {successor key: 'legal' | 'revturn'} for one state.
        """
        if st.type == 'off':
            return {}
    
        ent     = segment_registry[st.corridor]
        opp     = OPP_DIR.get(ent.get('direction'))
        opp_key = by_road_dir.get((st.road, opp)) if opp else None
    
        if st.key.endswith('_rev'):
            fwd_key, back_key = opp_key, st.corridor
        else:
            fwd_key, back_key = st.corridor, opp_key
    
        out = {k: 'legal' for k in succ.get(fwd_key, ())} if fwd_key else {}
        if back_key:
            for p in pred.get(back_key, ()):
                out.setdefault(p + '_rev', 'revturn')
        return out
    
    def _relative_class(b, legal_keys):
        """
        Class of b measured against the nearest LEGAL successor rather than
        against a. Necessary whenever a is a turn: a turn's geom_key is its
        own spline, so a departure on the same street but the other corridor
        matches none of the a-vs-b tests and wrongly falls through to
        'irregular'. Compared against the legal departure it is an ordinary
        carriageway cross.
        """
        for lk in legal_keys:
            i = idx.get(lk)
            if i is None:
                continue
            l = states[i]
            if l.road != b.road:
                continue
            if l.corridor == b.corridor:
                return 'flip' if l.travel_forward != b.travel_forward else None
            return 'cross' if l.travel_forward == b.travel_forward else 'uturn'
        return None
 
    # --- coarse geometric adjacency ------------------------------------------
    raw_poly = {st.corridor: segment_registry[st.corridor].get('validity_polygon')
                for st in states if st.type != 'off'}
 
    def _shares_boundary(a, b):
        pa, pb = raw_poly.get(a.corridor), raw_poly.get(b.corridor)
        if pa is None or pb is None or pa.is_empty or pb.is_empty:
            return False
        # Coarse on purpose: emission does the real work. A transition can
        # only be placed where both states are cheap, i.e. where the corridors
        # actually overlap.
        return pa.intersects(pb) or pa.distance(pb) < 5.0
 
    tau_class = np.empty((S, S), dtype=object)
    tau_class[:] = 'blocked'
 
    for i, a in enumerate(states):
        legal = _legal_targets(a)
        for j, b in enumerate(states):
            if i == j:
                tau_class[i, j] = 'stay'
                continue
            if a.type == 'off' or b.type == 'off':
                tau_class[i, j] = 'off'
                continue

            if b.key in legal:
                tau_class[i, j] = legal[b.key]          # 'legal' or 'revturn'
            elif (rc := _relative_class(b, legal)) is not None:
                tau_class[i, j] = rc
            elif a.corridor == b.corridor:
                tau_class[i, j] = 'flip'
            elif a.road == b.road:
                tau_class[i, j] = ('uturn'
                                   if a.travel_forward != b.travel_forward
                                   else 'cross')
            elif _shares_boundary(a, b):
                tau_class[i, j] = 'irregular'
 
    return tau_class, idx


def build_gates(geometry_store, segment_registry, extra_gate_defs=None):
    """
    Collect one Gate per topological boundary on every lane geometry.
 
    Sources
    -------
    - every geometry_store key starting with 's_', excluding GATE_EXCLUDE_KEYS
      ('s_stop' / 's_yield' are behavioural boundaries — crossing a stop line
      does not change which corridor you are in, so a gate there would only
      propose breakpoints the DP always rejects).
    - extra_gate_defs: optional [(geom_key, s_key, s_value)] for mid-block
      gates on long links. These exist purely so that a mid-block U-turn
      produces a crisp signed double-crossing rather than relying on the
      heading detector alone.
 
    d_max is derived from the widest lane corridor on the geometry, so the
    gate automatically spans sidewalks without any width configuration.
    """
    gates = []
 
    def _d_max_for(geom_key):
        widths = [max(e.get('d_left', 0.0), e.get('d_right', 0.0))
                  for e in segment_registry.values()
                  if isinstance(e, dict)
                  and e.get('type') in ('lane', 'ring')
                  and e.get('geometry_key') == geom_key]
        return max(widths) if widths else 20.0
 
    for geom_key, geo in geometry_store.items():
        if geom_key.startswith('__') or geom_key.startswith('intersection_area'):
            continue
        if not isinstance(geo, dict) or 'spline' not in geo:
            continue
        d_max = _d_max_for(geom_key)
        for s_key, s_val in geo.items():
            if not s_key.startswith('s_') or s_key in GATE_EXCLUDE_KEYS:
                continue
            if s_val is None or not np.isscalar(s_val):
                continue
            gates.append(Gate(key=f'{geom_key}:{s_key}', geom_key=geom_key,
                              s_key=s_key, s_value=float(s_val), d_max=d_max,
                              periodic=bool(geo.get('periodic')),
                              total_length=float(geo.get('total_length', 0.0))))
 
    for geom_key, s_key, s_val in (extra_gate_defs or []):
        geo_x = geometry_store.get(geom_key, {})
        gates.append(Gate(key=f'{geom_key}:{s_key}', geom_key=geom_key,
                          s_key=s_key, s_value=float(s_val),
                          d_max=_d_max_for(geom_key),
                          periodic=bool(geo_x.get('periodic')),
                          total_length=float(geo_x.get('total_length', 0.0))))
 
    return gates


def build_buffer_cache(poly_keys, segment_registry, margin=PROJ_MARGIN_M):
    """
    Buffered corridor polygons, keyed by corridor name.

    Not keyed on id(poly): CPython recycles id values after garbage
    collection, so reloading the registry in a live kernel can hand a new
    polygon the address of a freed one and return the wrong geometry from
    cache. Built once per site in setup_site and carried in `site`, so the
    lifetime is explicit rather than process-global.

    buffer() returns a NEW geometry, so it needs its own prepare().
    """
    cache = {}
    for ckey in poly_keys:
        poly = segment_registry.get(ckey, {}).get('validity_polygon')
        if poly is None or poly.is_empty:
            continue
        shapely.prepare(poly)
        buf = poly.buffer(margin)
        shapely.prepare(buf)
        cache[ckey] = buf
    return cache

# #############################################################################
# 2. PRE-PROCESSING
# #############################################################################
def build_spline_lut(tck, n=LUT_RESOLUTION):
    t_lut     = np.linspace(0, 1, n)
    xs, ys    = splev(t_lut, tck)
    return t_lut, np.column_stack([xs, ys])


def build_registry_luts(geometry_store):
    """
    Precompute spline LUTs for all geometry entries.
    Stores result in geometry_store[geom_key]['lut']. Idempotent.
    """
    skip = {'x_offset', 'y_offset'}
    for geom_key, geom in geometry_store.items():
        if geom_key in skip or not isinstance(geom, dict):
            continue
        if 'spline' not in geom:
            continue
        if 'lut' not in geom:
            geom['lut'] = build_spline_lut(geom['spline'][0])
    return geometry_store


def prepare_trajectory(bike_df, time_col='t'):
    """
    Build per-node arrays plus the arc-length weight ds.
 
    ds[i] is half the distance to each neighbour. Weighting every cost by ds
    is what makes the constants frame-rate invariant: a stopped cyclist
    contributes almost no weight instead of hundreds of zero-information
    points at 25 Hz, and the same TAU values work on 10 fps, 25 fps and
    subsampled data without rescaling.
 
    Note: no resampling is performed. Node index == positional row index in
    bike_df, so decoded runs map straight back onto the original rows.
    """
    xy = bike_df[['x_ekf', 'y_ekf']].to_numpy(dtype=float)
    n = len(xy)
 
    step = np.zeros(n)
    if n > 1:
        seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        step[:-1] += 0.5 * seg
        step[1:] += 0.5 * seg
    
    if n > 1:
        arc = np.concatenate(
            [[0.0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
        )
    else: 
        arc = np.zeros(1)
 
    t_sec = (bike_df[time_col].to_numpy(dtype=float)
             if time_col in bike_df.columns else np.arange(n, dtype=float))
 
    return Traj(xy=xy,
                psi=bike_df['angle_ekf'].to_numpy(dtype=float),
                speed=bike_df['speed_ekf'].to_numpy(dtype=float),
                t_sec=t_sec,
                ds=step,
                arc=arc,
                node_to_row=np.arange(n))


def corridor_membership(traj, poly_keys, segment_registry, buffers):
    """
    Vectorized point-in-corridor test — the single geometric prefilter.
 
    Returns two boolean (N, P) arrays from the same polygons:
 
      inside : exact containment. Drives the corridor cost term and the
               active-state filter.
      near   : containment in the corridor buffered by `margin`. Drives the
               projection mask.
 
    The two levels matter. Runs legitimately extend a little past a corridor
    edge — a cyclist clipping a corner, riding the kerb line, or swerving —
    and the DP needs s, d and psi_tan for those nodes in order to price the
    run at all. Masking projection on EXACT containment would reintroduce
    V4's hard polygon gate; the buffer keeps the mask generous while the
    cost stays exact.
 
    Replaces the old poly_dist entirely. Graded behaviour is not lost: a
    binary indicator, ds-weighted and averaged over a span in runcost,
    integrates to the FRACTION OF THE RUN'S ARC LENGTH spent outside the
    corridor — continuous, bounded in [0, 1] and frame-rate invariant.
    """
    x, y = np.ascontiguousarray(traj.xy[:, 0]), np.ascontiguousarray(traj.xy[:, 1])
    P = len(poly_keys)
    inside = np.zeros((traj.n, P), dtype=bool)
    near = np.zeros((traj.n, P), dtype=bool)
 
    for pi, ckey in enumerate(poly_keys):
        poly = segment_registry.get(ckey, {}).get('validity_polygon')
        if poly is None or poly.is_empty:
            continue
        buf = buffers.get(ckey)
        inside[:, pi] = _contains_xy(poly, x, y)
        near[:, pi] = _contains_xy(buf, x, y)
 
    return inside, near
 
 
def _contains_xy(poly, x, y):
    """
    Vectorized point-in-polygon (shapely >= 2.0). Loops in C and constructs no
    Point objects.
 
    NOTE: `contains` is boundary-EXCLUSIVE. A node lying exactly on a corridor
    edge counts as outside.
    """
    shapely.prepare(poly)          # no-op if already prepared
    return shapely.contains_xy(poly, x, y)


def project_all(traj, geom_keys, geometry_store, poly_keys,
                segment_registry, geom_to_polycols, buffers):
    """
    Project every point onto every geometry, masked by corridor membership.

    The mask is the union of `near` over the corridors that this geometry
    hosts. That is exact geometry rather than a bounding box, so a cyclist on
    a parallel street is never projected at all.

    Batched: all masked nodes for a geometry are refined together, so splev
    is called ~15 times per geometry instead of ~12 times per node. Warm
    starting is dropped in favour of a per-node LUT seed, which is a better
    initial guess anyway and removes the mask-gap bookkeeping entirely.

    `overshoot` is what makes the t=0 / t=1 failure detectable. When t clamps,
    d is measured against the endpoint normal, capturing only the
    perpendicular component and silently discarding tangential overshoot — so
    a cyclist 50 m past the end of a spline reports d ~ 0 and looks like a
    perfect match. overshoot recovers that discarded distance.
    """
    n, G = traj.n, len(geom_keys)
    s = np.full((n, G), np.nan)
    d = np.full((n, G), np.nan)
    psi_tan = np.full((n, G), np.nan)
    t_arr = np.full((n, G), np.nan)
    over = np.full((n, G), np.nan)
    valid = np.zeros((n, G), dtype=bool)

    inside, near = corridor_membership(traj, poly_keys, segment_registry,
                                       buffers)

    for gi, gkey in enumerate(geom_keys):
        geo = geometry_store.get(gkey)
        periodic = bool(geo.get('periodic'))
        if geo is None or 'spline' not in geo:
            continue
        cols = geom_to_polycols.get(gkey, [])
        if not cols:
            continue
        idxs = np.flatnonzero(near[:, cols].any(axis=1))
        if idxs.size == 0:
            continue

        tck, unew, cum = geo['spline']
        pts = traj.xy[idxs]
        px, py = pts[:, 0], pts[:, 1]

        # --- coarse seed from the LUT, chunked to bound memory -------------
        lut = geo.get('lut')
        if lut is None:
            lut = geo['lut'] = build_spline_lut(tck)
        t_lut, xy_lut = lut
        t_v = np.empty(idxs.size)
        CH = 4096                          # (CH, LUT_RESOLUTION, 2) floats
        for c0 in range(0, idxs.size, CH):
            blk = pts[c0:c0 + CH]
            diff = xy_lut[None, :, :] - blk[:, None, :]
            t_v[c0:c0 + CH] = t_lut[np.argmin((diff * diff).sum(2), axis=1)]

        # --- batched Newton -------------------------------------------------
        for _ in range(5):
            xp, yp = splev(t_v, tck, der=0)
            xp1, yp1 = splev(t_v, tck, der=1)
            xp2, yp2 = splev(t_v, tck, der=2)
            ex, ey = xp - px, yp - py
            f1 = ex * xp1 + ey * yp1
            f2 = xp1 ** 2 + yp1 ** 2 + ex * xp2 + ey * yp2
            safe = np.abs(f2) > 1e-12
            step = np.where(safe, -f1 / np.where(safe, f2, 1.0), 0.0)
            step = np.clip(step, -0.1, 0.1)
            if periodic:
                # Closed curve: wrap instead of clamping. Clamping stalls
                # Newton against a boundary that does not exist, so a point
                # whose true foot straddles t=0 never converges.
                t_v = np.mod(t_v + step, 1.0)
            else:
                t_v = np.clip(t_v + step, 0.0, 1.0)
            if np.max(np.abs(step)) < 1e-7:
                break

        # --- outputs ---------------------------------------------------------
        xp, yp = splev(t_v, tck, der=0)
        dx, dy = splev(t_v, tck, der=1)
        nrm = np.hypot(dx, dy)
        nrm = np.where(nrm > 1e-12, nrm, 1.0)
        tx, ty = dx / nrm, dy / nrm

        s[idxs, gi] = np.interp(t_v, unew, cum)
        d[idxs, gi] = (px - xp) * (-ty) + (py - yp) * tx   # left of spline
        psi_tan[idxs, gi] = np.arctan2(ty, tx)
        t_arr[idxs, gi] = t_v
        over[idxs, gi] = (np.zeros(idxs.size) if periodic
                          else _overshoot_vec(pts, t_v, _spline_endpoints(tck)))
        valid[idxs, gi] = True

    return Proj(s=s, d=d, psi_tan=psi_tan, t=t_arr, overshoot=over,
                inside=inside, near=near, valid=valid)


def _spline_endpoints(tck):
    """[(p_start, tangent_start), (p_end, tangent_end)] — unit tangents."""
    out = []
    for tv in (0.0, 1.0):
        x, y = splev(tv, tck, der=0)
        dx, dy = splev(tv, tck, der=1)
        nrm = float(np.hypot(dx, dy)) or 1.0
        out.append((np.array([float(x), float(y)]),
                    np.array([float(dx) / nrm, float(dy) / nrm])))
    return out


def _overshoot_vec(pts, t_v, ends):
    """Metres past a clamped spline endpoint, along the endpoint tangent."""
    (p0, tg0), (p1, tg1) = ends
    out = np.zeros(len(t_v))
    lo = t_v <= CLAMP_TOL
    if lo.any():
        out[lo] = np.maximum(0.0, (pts[lo] - p0) @ (-tg0))
    hi = t_v >= 1.0 - CLAMP_TOL
    if hi.any():
        out[hi] = np.maximum(0.0, (pts[hi] - p1) @ tg1)
    return out



# #############################################################################
# 3. BREAKPOINTS
# #############################################################################
# The set B is a SUPERSET filter. The DP may ignore any candidate, so a false
# proposal is free; what it cannot do is recover a breakpoint never offered.
# Recall matters, precision does not — propose generously.

def detect_crossings(traj, proj, gates, geom_keys):
    """
    Gate crossing = sign change of (proj_s - s_value) between consecutive
    nodes, bounded laterally by d_max and guarded against clamped projections.
 
    Two guards are essential:
      - |d| <= d_max: a cyclist on a parallel street still projects onto this
        geometry and would otherwise register a spurious crossing.
      - t strictly interior: a point past the spline end has s pinned at 0 or
        L, and a pinned value on the wrong side of s_change fabricates a
        crossing when the cyclist re-enters.
    """
    gcol = {k: i for i, k in enumerate(geom_keys)}
    out = []
 
    for g in gates:
        c = gcol.get(g.geom_key)
        if c is None:
            continue
        s, d, t = proj.s[:, c], proj.d[:, c], proj.t[:, c]
        ok = np.isfinite(s) & (np.abs(d) <= g.d_max)
        if not g.periodic:
            # A point past a spline end has s pinned at 0 or L.
            # A closed curve has no ends, and Newton wraps rather than clamping.
            ok &= (t > CLAMP_TOL) & (t < 1.0 - CLAMP_TOL)

        L = g.total_length
        last_arc = -np.inf
        for k in range(len(s) - 1):
            if not (ok[k] and ok[k + 1]):
                continue
            if g.periodic:
                # Circular offsets from the gate.
                f0 = (s[k]     - g.s_value + L / 2.0) % L - L / 2.0
                f1 = (s[k + 1] - g.s_value + L / 2.0) % L - L / 2.0
                if abs(f0 - f1) > L / 2.0:
                    continue                      # wrapped between samples
                ds_sign = 1 if f1 > f0 else -1
            else:
                f0, f1 = s[k] - g.s_value, s[k + 1] - g.s_value
                ds_sign = 1 if s[k + 1] > s[k] else -1

            if f0 * f1 > 0 or f0 == f1:
                continue
            frac = float(f0 / (f0 - f1))
            node = k if frac < 0.5 else k + 1
            if traj.arc[node] - last_arc < HYST_M:
                continue
            out.append(Crossing(gate_key=g.key, sign=ds_sign,
                                node=int(node), frac=frac))
            last_arc = traj.arc[node]
 
    return sorted(out, key=lambda c: c.node)
 
 
def polygon_transitions(traj, proj, poly_keys):
    """
    Every corridor polygon entry AND exit — not just the first continuous run.
 
    Proposing every transition means a cyclist who dips out of a corridor 
    and back (swerve around a parked car, EKF wobble) yields four candidates 
    instead of a truncated window; the DP declines three of them.
 
    This is also the detector that covers manoeuvres crossing no gate: the
    sidewalk excursion, the mid-block corridor change.
    """
    bps = []
    for pi in range(proj.inside.shape[1]):
        flips = np.flatnonzero(np.diff(proj.inside[:, pi].astype(np.int8)) != 0)
        bps.extend(int(f) + 1 for f in flips)
    return bps


def _wrap_pi(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi

 
def heading_peaks(traj, win_m=HEAD_WIN_M, thresh=HEAD_THRESH):
    """
    Sustained heading change over an ARC-LENGTH window, so the detector is
    frame-rate invariant and immune to the heading noise of a stopped cyclist.
 
    Catches turn entry/exit independently of geometry — important where the
    ridden path diverges from the registered spline (corner cutting).
 
    It also covers IN-PLACE reversals and two-stage turns, which is not
    obvious. The window is defined by searchsorted on cumulative arc, not by
    sample count, so during a stationary period — where `arc` is flat — `lo`
    lands win_m BEFORE the stop began and `hi` lands win_m AFTER it ended.
    The window straddles the whole pivot, and both endpoints are sampled from
    moving nodes, so the headings are clean. A cyclist who turns around while
    stationary produces change ~ pi and fires normally. No separate stop
    detector is needed.
 
    On a stationary plateau `change` is constant, so the peak fires at the
    LAST node of the plateau. That is fine: stationary nodes carry near-zero
    ds weight, contribute nothing to runcost either way, and all sit at
    effectively the same arc position.
    """
    n = traj.n
    if n < 3:
        return []
    lo = np.searchsorted(traj.arc, traj.arc - win_m, side='left')
    hi = np.searchsorted(traj.arc, traj.arc + win_m, side='right') - 1
    change = np.zeros(n)
    for i in range(n):
        a, b = lo[i], min(hi[i], n - 1)
        if b <= a:
            continue
        change[i] = abs(_wrap_pi(traj.psi[b] - traj.psi[a]))
 
    bps = []
    for i in range(1, n - 1):
        if change[i] >= thresh and change[i] >= change[i - 1] \
                and change[i] > change[i + 1]:
            bps.append(i)
    return bps
 
 
def propose_breakpoints(traj, proj, gates, geom_keys, poly_keys,
                        dense_m=None):
    """
    Merge all detectors into a sorted candidate set, keeping provenance.
 
    Priority on merge: gate > corridor > heading. Provenance is worth
    storing — when a match looks wrong later, knowing a breakpoint came from a
    heading peak rather than a gate tells you immediately where to look.
    """
    cand = []
    for c in detect_crossings(traj, proj, gates, geom_keys):
        cand.append((c.node, f'gate:{c.gate_key}'))
    for b in polygon_transitions(traj, proj, poly_keys):
        cand.append((b, 'corridor'))
    for b in heading_peaks(traj):
        cand.append((b, 'heading'))
    if dense_m:
        step = dense_m
        target = step
        for i in range(traj.n):
            if traj.arc[i] >= target:
                cand.append((i, 'dense'))
                target += step
 
    cand.append((0, 'endpoint'))
    cand.append((traj.n, 'endpoint'))
 
    order = {'gate': 0, 'corridor': 1, 'heading': 2,
             'dense': 3, 'endpoint': 4}
    cand = sorted(set(cand), key=lambda x: (x[0], order[x[1].split(':', 1)[0]]))
 
    merged, prov = [], []
    for node, src in cand:
        node = int(np.clip(node, 0, traj.n))
        if merged:
            a_prev = traj.arc[min(merged[-1], traj.n - 1)]
            a_cur = traj.arc[min(node, traj.n - 1)]
            if node != traj.n and abs(a_cur - a_prev) < MERGE_M:
                continue
        merged.append(node)
        prov.append(src)
 
    if merged[0] != 0:
        merged.insert(0, 0); prov.insert(0, 'endpoint')
    if merged[-1] != traj.n:
        merged.append(traj.n); prov.append('endpoint')
 
    return np.array(merged, dtype=int), prov

# #############################################################################
# 4. COST
# #############################################################################
def build_prefix_sums(traj, proj, states):
    """
    Cumulative ds-weighted sums per state, so runcost is O(1) per span.
    
    lat: how far off the centerline
    cor: inside or outside of the corridor
    clamp: how far past the end of the spline
    ang2: how wrong the heading is
 
    Note lat / cor / clamp are direction-independent (|d| and distances are
    unsigned), so a forward/reverse pair could share those columns. Kept
    per-state here for clarity; S is small.
    """
    n, S = traj.n, len(states)
    z = lambda: np.zeros((n + 1, S))
    ds, lat, cor, clamp, ang2, ds_mov = z(), z(), z(), z(), z(), z()
 
    moving = traj.speed > V_STOP
    
    fv = np.full((n + 1, S), n, dtype=int)
    lv = np.full((n, S), -1, dtype=int)
    idx = np.arange(n)
 
    for k, st in enumerate(states):
        if st.type == 'off':
            ds[1:, k] = np.cumsum(traj.ds)
            lat[1:, k] = np.cumsum(traj.ds * OFF_COST_PER_M)
            continue
 
        gc, pc = st.geom_col, st.poly_col
 
        psi_lane = proj.psi_tan[:, gc] + st.psi_offset
        ok = proj.valid[:, gc]
        
        # First/last valid node lookups. runcost previously called
        # np.flatnonzero over the span — O(N) inside an O(1) function.
        fv[:n, k] = np.minimum.accumulate(np.where(ok, idx, n)[::-1])[::-1]
        lv[:, k] = np.maximum.accumulate(np.where(ok, idx, -1))
 
        d_nat = proj.d[:, gc]
        e_lat = np.where(ok, np.clip(((d_nat - st.d_center) / st.d_half) ** 2, 0, C_D), C_D)
        # Binary containment. ds-weighted and averaged over a span in runcost,
        # this integrates to the fraction of the run's arc spent outside.
        e_cor = np.where(proj.inside[:, pc], 0.0, C_OUT)
        e_clm = np.where(ok, np.clip((np.nan_to_num(proj.overshoot[:, gc])
                                      / SIGMA_CLAMP) ** 2, 0, C_CLAMP), C_CLAMP)
        ang = np.where(ok, _wrap_pi(traj.psi - np.nan_to_num(psi_lane)), 0.0)
        a2 = np.clip((ang / SIGMA_PSI) ** 2, 0, C_PSI)
 
        w = traj.ds
        wm = np.where(moving & ok, w, 0.0)
        ds[1:, k] = np.cumsum(w)
        lat[1:, k] = np.cumsum(w * e_lat)
        cor[1:, k] = np.cumsum(w * e_cor)
        clamp[1:, k] = np.cumsum(w * e_clm)
        ang2[1:, k] = np.cumsum(wm * a2)
        ds_mov[1:, k] = np.cumsum(wm)
 
    return Pre(ds=ds, lat=lat, cor=cor, clamp=clamp, ang2=ang2, ds_mov=ds_mov, fv=fv, lv=lv)


def build_tau_tables(B, states, tau_class, tau_vals, proj, geometry_store):
    """
    tau(bi, j, k) = base[j, k] + cont[bi, k] + bnd[bi, j]

    tau is additively decomposable — the class prior depends only on the state
    pair, turn continuity only on the destination, boundary shortfall only on
    the origin — so it tabulates instead of being called S^2 |B|^2 / 2 times.
    """
    S, nb = len(states), len(B)
    nodes = np.minimum(B, proj.s.shape[0] - 1)

    base = np.full((S, S), np.inf)
    for j in range(S):
        for k in range(S):
            v = tau_vals.get(tau_class[j, k], np.inf)
            if np.isfinite(v):
                base[j, k] = v + TAU_PER_RUN + states[k].entry_prior
    np.fill_diagonal(base, 0.0)

    cont = np.zeros((nb, S))
    bnd = np.zeros((nb, S))
    for k, st in enumerate(states):
        if st.type == 'turn':
            L = max(geometry_store[st.geom_key].get('total_length', 1.0), 1e-6)
            s_in = np.nan_to_num(proj.s[nodes, st.geom_col])
            # cont[:, k] = LAM_CONT * np.clip((s_in / L) ** 2, 0, 1)
            # A turn is entered at the end travel STARTS from: s = 0 forward,
            # s = L reversed. Measuring from 0 unconditionally charges every
            # reversed turn the full LAM_CONT, which is not a preference for
            # forward turns — it is a bug that makes them unreachable.
            frac = s_in / L if st.travel_forward else (L - s_in) / L
            cont[:, k] = LAM_CONT * np.clip(frac ** 2, 0, 1)
        elif st.type == 'lane':
            sc = geometry_store.get(st.geom_key, {}).get('s_change')
            if sc is not None:
                s_out = np.nan_to_num(proj.s[nodes, st.geom_col])
                short = (sc - s_out) if st.travel_forward else (s_out - sc)
                bnd[:, k] = LAM_BND * np.clip(
                    (np.maximum(short, 0.0) / SIGMA_BND) ** 2, 0, C_BND)
    return base, cont, bnd


def build_runcost_cache(B, states, pre, traj, proj):
    """
    rc[ai, bi, k] — runcost for every candidate span and state, vectorized
    over (ai, bi). Identical arithmetic to runcost(); kept as a cache so
    backtracking reads the same numbers the DP minimised over.
    """
    nb, S, n = len(B), len(states), traj.n
    Ba = np.minimum(B, n - 1)                    # span start; B[-1] == n
    Bm1 = np.minimum(np.maximum(B - 1, 0), n - 1)  # span end, inclusive
    arc = traj.arc[Bm1][None, :] - traj.arc[Ba][:, None]      # (ai, bi)
    too_short = arc < MIN_RUN_M

    rc = np.full((nb, nb, S), np.inf)

    def span(pfx, k):
        v = pfx[B, k]
        return v[None, :] - v[:, None]

    for k, st in enumerate(states):
        W = span(pre.ds, k)
        good = (W > 0) & ~too_short
        if not good.any():
            continue
        Wsafe = np.where(good, W, 1.0)

        lat = span(pre.lat, k) / Wsafe
        if st.type == 'off':
            rc[:, :, k] = np.where(good, W * lat, np.inf)
            continue

        cor = span(pre.cor, k) / Wsafe
        clm = span(pre.clamp, k) / Wsafe

        Wm = span(pre.ds_mov, k)
        head = np.where(Wm > W_MOV_MIN, span(pre.ang2, k) / np.where(Wm > 0, Wm, 1.0), 0.0)

        i0 = pre.fv[Ba, k][:, None]               # (ai, 1) first valid >= a
        i1 = pre.lv[Bm1, k][None, :]              # (1, bi) last  valid <= b-1
        okp = (i0 < B[None, :]) & (i1 >= Ba[:, None]) & (i0 < i1)
        i0c, i1c = np.clip(i0, 0, n - 1), np.clip(i1, 0, n - 1)
        sc = proj.s[:, st.geom_col]
        sp = np.maximum(traj.arc[i1c] - traj.arc[i0c], 1e-6)
        if st.periodic:
            # s wraps
            L = st.total_length
            fwd = np.mod(sc[i1c] - sc[i0c], L)
            bwd = L - fwd
            ds_net = np.where(np.abs(fwd - sp) <= np.abs(bwd - sp), fwd, -bwd)
            if not st.travel_forward:
                ds_net = -ds_net
        else:
            ds_net = (sc[i1c] - sc[i0c]) * (1 if st.travel_forward else -1)
        prog = np.where(okp, np.clip(1.0 - ds_net / sp, 0.0, 1.0), 1.0)

        tot = W * (W_LAT * lat + W_COR * cor + W_CLAMP * clm +
                   W_PSI * head + W_PRG * prog)
        rc[:, :, k] = np.where(good, tot, np.inf)

    return rc
 
 
def runcost(a, b, k, states, pre, traj, proj):
    """
    Cost of explaining span [a, b) with state k, as a unit.
 
    Two terms here are RUN-LEVEL and have no per-point equivalent:
 
      heading  — RMSE over the whole run, restricted to moving nodes. Robust
                 where a per-point term collapses (slow, stopped, queueing).
      progress — s must advance monotonically in the travel direction.
 
    The corridor term is binary per node, but `cor` below is the ds-weighted
    MEAN over the span — i.e. the fraction of the run's arc length spent
    outside the corridor. Grading is recovered at run level, which is where
    it belongs now that every cost is a run statistic.
 
    Multiplying the per-metre mean by W keeps run costs additive, so different
    segmentations of the same trajectory are directly comparable.
    """
    st = states[k]
    arc = traj.arc[min(b - 1, traj.n - 1)] - traj.arc[a]
    if arc < MIN_RUN_M:
        return np.inf                       # structural anti-flicker
 
    W = pre.ds[b, k] - pre.ds[a, k]
    if W <= 0:
        return np.inf
 
    lat = (pre.lat[b, k] - pre.lat[a, k]) / W
    if st.type == 'off':
        return W * lat
 
    cor = (pre.cor[b, k] - pre.cor[a, k]) / W
    clm = (pre.clamp[b, k] - pre.clamp[a, k]) / W
 
    Wm = pre.ds_mov[b, k] - pre.ds_mov[a, k]
    head = ((pre.ang2[b, k] - pre.ang2[a, k]) / Wm) if Wm > W_MOV_MIN else 0.0
 
    # First and last VALID node in the span, not the span endpoints. Run
    # boundaries are exactly where nodes are most likely to be unprojected,
    # so reading s at a and b-1 lets a single marginal endpoint force
    # prog = 1.0 (maximum penalty) on a run whose interior progresses fine.
    v = np.flatnonzero(proj.valid[a:b, st.geom_col])
    if len(v) >= 2:
        s_a = proj.s[a + v[0], st.geom_col]
        s_b = proj.s[a + v[-1], st.geom_col]
        span = max(traj.arc[a + v[-1]] - traj.arc[a + v[0]], 1e-6)
        if st.periodic:
            # s is wrapped for rings.
            L = st.total_length
            fwd = (s_b - s_a) % L          # arc travelled going CCW
            bwd = L - fwd                  # arc travelled going CW
            ds_net = fwd if abs(fwd - span) <= abs(bwd - span) else -bwd
            if not st.travel_forward:
                ds_net = -ds_net
        else:
            ds_net = (s_b - s_a) * (1 if st.travel_forward else -1)
        prog = float(np.clip(1.0 - ds_net / span, 0.0, 1.0))
    else:
        prog = 1.0
 
    return W * (W_LAT * lat + W_COR * cor + W_CLAMP * clm +
                W_PSI * head + W_PRG * prog)
 
 
def tau(node, j, k, states, tau_class, tau_vals, proj, geometry_store):
    """
    Transition prior. Read as: metres-of-evidence the alternative must save
    before this manoeuvre is believed.
 
    TAU_PER_RUN regularises against inventing breakpoints. LAM_CONT replaces
    V4's pile_term; LAM_BND replaces _sustained_crossing — soft, and with no
    sample-count parameter, so it behaves identically at any frame rate.
    """
    if j == k:
        return 0.0
 
    cls = tau_class[j, k]
    base = tau_vals.get(cls, np.inf)
    if not np.isfinite(base):
        return np.inf
 
    a, b = states[j], states[k]
    t = base + TAU_PER_RUN + b.entry_prior
 
    if b.type == 'turn':
        L = geometry_store[b.geom_key].get('total_length', 1.0)
        s_in = proj.s[min(node, proj.s.shape[0] - 1), b.geom_col]
        # if np.isfinite(s_in):
        #     t += LAM_CONT * float(np.clip((s_in / max(L, 1e-6)) ** 2, 0, 1))
        if np.isfinite(s_in):
            frac = s_in / max(L, 1e-6)
            if not b.travel_forward:
                frac = 1.0 - frac
            t += LAM_CONT * float(np.clip(frac ** 2, 0, 1))
 
    if a.type == 'lane':
        s_change = geometry_store.get(a.geom_key, {}).get('s_change')
        s_out = proj.s[min(node, proj.s.shape[0] - 1), a.geom_col]
        if s_change is not None and np.isfinite(s_out):
            short = (s_change - s_out) if a.travel_forward else (s_out - s_change)
            short = max(0.0, float(short))
            t += LAM_BND * float(np.clip((short / SIGMA_BND) ** 2, 0, C_BND))
 
    return t


def _emission_cost_per_m(runs, traj):
    """
    Mean runcost per metre over MAPPED runs only.

    The fallback trigger asks whether the best available segmentation was
    still expensive — a proxy for 'the true breakpoint was never proposed'.
    OFF runs are not evidence of that: a cyclist genuinely off the map is
    correctly explained at OFF_COST_PER_M, and densifying B cannot improve
    it. Transition costs are excluded for the same reason — a legitimate
    U-turn pays its tau once, which says nothing about breakpoint recall.
    """
    cost = arc = 0.0
    for r in runs:
        if r.state.type == 'off':
            continue
        cost += r.cost
        arc += max(traj.arc[min(r.b - 1, traj.n - 1)] - traj.arc[r.a], 0.0)
    return (cost / arc) if arc > 1.0 else None

# #############################################################################
# 5. DP
# #############################################################################
def segmental_dp(B, states, pre, traj, proj, tau_class, tau_vals,
                 geometry_store):
    """
    C[bi, k] = min over ai < bi, j != k of
                  C[ai, j] + rc[ai, bi, k] + tau(ai, j, k)

    Exhaustive over the candidate set, so there is no greedy commitment.
    tau and runcost are tabulated up front, so the inner loop over origin
    states is a single numpy reduction rather than S Python calls.
    """
    nb, S = len(B), len(states)
    base, cont, bnd = build_tau_tables(B, states, tau_class, tau_vals,
                                       proj, geometry_store)
    rc = build_runcost_cache(B, states, pre, traj, proj)

    C = np.full((nb, S), np.inf)
    back = np.full((nb, S, 2), -1, dtype=int)
    C[0, :] = 0.0        # no run has occurred yet; entry_prior belongs to the
                         # FIRST RUN's state, applied in the ai == 0 branch below

    for bi in range(1, nb):
        for k in range(S):
            best, bj, ba = C[bi, k], -1, -1
            for ai in range(bi):
                r = rc[ai, bi, k]
                if not np.isfinite(r):
                    continue
                if ai == 0:
                    # Free entry: no transition, but the run's own state prior
                    # still applies — otherwise a trajectory that starts
                    # already contraflow never pays for it, since no transition
                    # into the mirrored state ever fires.
                    tot = C[0, :] + r + states[k].entry_prior
                else:
                    tot = (C[ai, :] + base[:, k] + bnd[ai, :]
                           + cont[ai, k] + r)
                    tot[k] = np.inf
                j = int(np.argmin(tot))
                if tot[j] < best:
                    best, bj, ba = tot[j], j, ai
            if bj >= 0:
                C[bi, k] = best
                back[bi, k] = (ba, bj)

    k_end = int(np.argmin(C[-1, :]))
    if not np.isfinite(C[-1, k_end]):
        return [], np.inf

    runs, bi, k = [], nb - 1, k_end
    while bi > 0:
        ai, j = back[bi, k]
        if ai < 0:
            break
        a, b = int(B[ai]), int(B[bi])
        cost = float(rc[ai, bi, k])
        arc = max(traj.arc[min(b - 1, traj.n - 1)] - traj.arc[a], 1e-6)
        runs.append(Run(a=a, b=b, state=states[k], cost=cost,
                        cost_per_m=cost / arc,
                        trans_class=(tau_class[j, k] if ai != 0 else 'start')))
        bi, k = ai, int(j)

    runs.reverse()
    return runs, float(C[-1, k_end])


def segmental_dp_old(B, states, pre, traj, proj, tau_class, tau_vals,
                 geometry_store, segment_registry, verbose=False, log=None):
    """
    C[bi, k] = min over ai < bi, j != k of
                  C[ai, j] + runcost(B[ai], B[bi], k) + tau(B[ai], j, k)
 
    Exhaustive over the candidate set, so there is no greedy commitment.
 
    Loop order matters: runcost depends on (ai, bi, k) only, so it is hoisted
    out of the j loop.
    """
    nb, S = len(B), len(states)
    C = np.full((nb, S), np.inf)
    back = np.full((nb, S, 2), -1, dtype=int)
 
    for k, st in enumerate(states):
        C[0, k] = st.entry_prior
 
    for bi in range(1, nb):
        for k in range(S):
            for ai in range(bi):
                rc = runcost(B[ai], B[bi], k, states, pre, traj, proj)
                if not np.isfinite(rc):
                    continue
                for j in range(S):
                    if not np.isfinite(C[ai, j]):
                        continue
                    if j == k and ai != 0:
                        continue              # runs are maximal by construction
                    tv = tau(B[ai], j, k, states, tau_class, tau_vals,
                             proj, geometry_store) if ai != 0 else 0.0
                    if not np.isfinite(tv):
                        continue
                    v = C[ai, j] + rc + tv
                    if v < C[bi, k]:
                        C[bi, k] = v
                        back[bi, k] = (ai, j)
 
    k_end = int(np.argmin(C[-1, :]))
    if not np.isfinite(C[-1, k_end]):
        return [], np.inf
 
    runs, bi, k = [], nb - 1, k_end
    while bi > 0:
        ai, j = back[bi, k]
        if ai < 0:
            break
        a, b = int(B[ai]), int(B[bi])
        cost = runcost(a, b, k, states, pre, traj, proj)
        arc = max(traj.arc[min(b - 1, traj.n - 1)] - traj.arc[a], 1e-6)
        runs.append(Run(a=a, b=b, state=states[k], cost=cost,
                        cost_per_m=cost / arc,
                        trans_class=(tau_class[j, k] if ai != 0 else 'start')))
        bi, k = ai, int(j)
 
    runs.reverse()
    return runs, float(C[-1, k_end])

# #############################################################################
# 6. DECODE
# #############################################################################
def assign_roles(runs):
    """
    Role from position relative to the first turn. No inference from s-domain,
    no _infer_role_from_registry: the run sequence already carries it.
    """
    seen_turn = False
    for r in runs:
        if r.state.type == 'turn':
            r.role = 'turn'
            seen_turn = True
        elif r.state.type == 'ring':
            r.role = 'ring'
            seen_turn = True          # the ring is past the entry turn
        elif r.state.type == 'off':
            r.role = 'off'
        else:
            r.role = 'departure' if seen_turn else 'approach'
    return runs


def _s_role(run, traj, proj, geometry_store):
    """
    'approach' | 'departure' | None, from the ds-weighted fraction of the run
    lying past s_change. None for turns (no s_change).

    Returns a straddle flag when the run spans s_change: the DP's handoff is
    soft, so this is possible, and it means no single offset is right for the
    whole run. Rare, but silent if unflagged.
    """
    st = run.state
    if st.type == 'turn':
        return None, False
    sc = geometry_store.get(st.geom_key, {}).get('s_change')
    if sc is None:
        return None, False

    gc = st.geom_col
    ok = proj.valid[run.a:run.b, gc]
    if not ok.any():
        return None, False

    w = traj.ds[run.a:run.b][ok]
    s = proj.s[run.a:run.b, gc][ok]
    W = w.sum()
    if W <= 0:
        return None, False

    frac_after = float((w * (s >= sc)).sum() / W)
    # Which side of s_change is the departure depends on travel direction:
    # for a state travelling in decreasing native s, the high-s side is the
    # APPROACH.
    is_departure = (frac_after >= 0.5) == st.travel_forward
    return ('departure' if is_departure else 'approach',
            0.1 < frac_after < 0.9)

 
def runs_to_chain(runs, traj, proj, geometry_store, is_fallback=False):
    """Convert decoded runs into chain dicts."""
    chain = []
    for r in runs:
        if r.state.type == 'off':
            continue
        idx = traj.node_to_row[r.a:r.b]
        if len(idx) == 0:
            continue
        if r.cost_per_m <= Q_GOOD:
            quality = 'good'
        elif r.cost_per_m <= Q_POOR:
            quality = 'poor'
        else:
            quality = 'weak'
        
        # Geometric role: is this run before or after s_change on its own
        # geometry. Distinct from r.role, which is the MOVEMENT role and is
        # relative to the first turn — a trajectory entering after the
        # intersection has no turn, so every run reads 'approach' there.
        s_role, straddle = _s_role(r, traj, proj, geometry_store)
        
        chain.append({
            'seg_key':              r.state.corridor,
            'geom_col':             r.state.geom_col,
            'role':                 r.role,       # movement role, for reporting
            's_role':               s_role,       # geometric role, drives directed s
            'straddles':            straddle,
            'df_indices':           list(idx),
            's_arr':                None,
            'score':                r.cost_per_m,
            'match_quality':        quality,
            'is_reverse_traversal': r.state.contraflow,
            'is_fallback':          is_fallback,
            's_change_key_fired':   None,
            'trans_class':          r.trans_class,
        })
    return chain
 
 
def classify_movement(runs, movement_registry):
    """
    Movement key from the DECODED RUN SEQUENCE — no gates involved.
 
    Gates propose breakpoints; they do not classify. Once the DP has decoded
    an ordered run sequence, mapping it to a movement key is a direct lookup,
    and the DP's free entry/exit already handles trajectories that start or
    end mid-scene better than prefix/suffix matching on a gate signature would.
 
    'partial' requires a UNIQUE contiguous match. V4's derive_movement_key
    used a subset test and returned whichever movement came first in dict
    order, so a short chain silently picked one of several candidates — a
    quiet corruption of the movement statistics even when the segment
    assignment was correct. Here ambiguity is reported, not resolved.
 
    Anomalies are RESULTS, not errors: contraflow, sidewalk riding, mid-block
    U-turns and cut-throughs all decode successfully and are labelled by the
    transition classes that fired.
 
    Returns
    -------
    (movement_key, kind) with kind in
    {'registered', 'partial', 'ambiguous', 'anomalous'}
    """
    seq = tuple(r.state.corridor for r in runs if r.state.type != 'off')
    rev = any(r.state.contraflow for r in runs if r.state.type != 'off')
    irregular = sorted({r.trans_class for r in runs
                        if r.trans_class in ('flip', 'cross', 'uturn', 'irregular')})
    off_run = any(r.state.type == 'off' for r in runs)
 
    if seq and not irregular and not off_run:
        exact = [m for m, s in movement_registry.items()
                 if tuple(x[0] for x in s) == seq]
        if len(exact) == 1:
            return exact[0], 'registered'
 
        hits = [m for m, s in movement_registry.items()
                if _is_contiguous_sub(seq, tuple(x[0] for x in s))]
        if len(hits) == 1:
            return hits[0], 'partial'
        if len(hits) > 1:
            return 'ambiguous', 'ambiguous'
 
    label = '+'.join(irregular) if irregular else 'unregistered'
    if off_run:
        label += '+off_excursion'
    if rev and 'flip' not in irregular:
        # Already riding against the nominal direction on entry, so no flip
        # transition ever fired — the run's travel_forward is the only evidence.
        label += '+contraflow'
    return f'anomalous:{label}', 'anomalous'
 
 
def _is_contiguous_sub(sub, full):
    """True if `sub` is a strictly shorter contiguous subsequence of `full`."""
    n = len(sub)
    return 0 < n < len(full) and any(full[i:i + n] == sub
                                     for i in range(len(full) - n + 1))

# #############################################################################
# 7. TRANSFORMATION UTILITIES
# #############################################################################
def project_point_warm(point_local, tck, t_init=None, delta=0.1, lut=None):
    """
    Project a 2D point onto a spline with warm-started Newton refinement.

    Phase 1 — Coarse search (skipped if t_init provided):
        If lut given: O(M) numpy nearest-neighbour on precomputed LUT.
        Else: vectorised splev on 200 uniform t values.
    Phase 2 — Newton refinement (4 iterations max).

    Returns
    -------
    t_star        : float
    closest_point : (2,) array
    """
    point = np.asarray(point_local, dtype=float)
    px, py = point[0], point[1]

    if t_init is None:
        if lut is not None:
            t_lut, xy_lut = lut
            diffs  = xy_lut - point
            t_init = float(t_lut[np.argmin((diffs * diffs).sum(1))])
        else:
            t_c    = np.linspace(0, 1, 200)
            xs, ys = splev(t_c, tck)
            t_init = float(t_c[np.argmin((xs - px)**2 + (ys - py)**2)])

    t = float(t_init)
    for _ in range(4):
        xp,  yp  = splev(t, tck, der=0)
        xp1, yp1 = splev(t, tck, der=1)
        xp2, yp2 = splev(t, tck, der=2)
        ex = float(xp) - px;  ey = float(yp) - py
        f1 = ex * float(xp1) + ey * float(yp1)
        f2 = float(xp1)**2 + float(yp1)**2 + ex * float(xp2) + ey * float(yp2)
        if abs(f2) < 1e-12:
            break
        step = max(-delta, min(delta, -f1 / f2))
        t    = max(0.0, min(1.0, t + step))
        if abs(step) < 1e-7:
            break

    xp, yp = splev(t, tck, der=0)
    return t, np.array([float(xp), float(yp)])


def project_point_full(point_local, tck, unew, cum_dist, t_init=None, lut=None):
    """
    Full projection: closest point + s, d, tangent, normal.

    Returns
    -------
    t_star, tangent (unit), normal (unit), s [m], d [m]
    d positive = left of spline direction (native, before is_forward correction).
    """
    t_star, closest = project_point_warm(point_local, tck, t_init, lut=lut)
    s          = float(np.interp(t_star, unew, cum_dist))
    dx, dy     = splev(t_star, tck, der=1)
    tang       = np.array([dx, dy])
    tang      /= np.linalg.norm(tang)
    norm       = np.array([-tang[1], tang[0]])
    d          = float(np.dot(np.asarray(point_local) - closest, norm))
    return t_star, tang, norm, s, d


def transform_segment(bike_df, seg_key, df_indices, is_reverse,
                       segment_registry, geometry_store,
                       proj=None, geom_col=None, s_role=None):
    """
    Project trajectory points for one segment and compute lane coordinate
    outputs.

    Velocity and acceleration are decomposed onto tangent/normal via dot
    products — no finite differences.

    Parameters
    ----------
    bike_df    : DataFrame — full trajectory (original indices)
    seg_key    : str
    df_indices : list of int — row indices belonging to this segment
    is_reverse : bool
    proj / geom_col : optional. When given, the projection computed during
        matching is reused instead of recomputed.

    Returns
    -------
    result : dict of column_name → array of length len(df_indices)
    """
    entry           = segment_registry[seg_key]
    geom_key        = entry['geometry_key']
    tck, unew, cum_dist = geometry_store[geom_key]['spline']
    is_forward      = entry['is_forward']
    bike_lane       = entry.get('bike_lane')

    xy     = bike_df.iloc[df_indices][['x_ekf', 'y_ekf']].to_numpy()
    speed  = bike_df.iloc[df_indices]['speed_ekf'].to_numpy()
    psi    = bike_df.iloc[df_indices]['angle_ekf'].to_numpy()
    accel  = bike_df.iloc[df_indices]['a_ekf'].to_numpy()
    n      = len(df_indices)
    
    if proj is not None and geom_col is not None:
        rows     = np.asarray(df_indices, dtype=int)
        gc       = geom_col
        s_native = proj.s[rows, gc].copy()
        d_nat    = proj.d[rows, gc].copy()
        psi_t    = proj.psi_tan[rows, gc].copy()

        # Backfill any node the projection mask skipped. A run can extend a
        # little past the buffered corridor at its boundary; those nodes are
        # rare, so projecting just them is far cheaper than redoing the span.
        missing = ~proj.valid[rows, gc]
        if missing.any():
            lut = geometry_store[geom_key].get('lut')
            for i in np.flatnonzero(missing):
                _t, tang, _nrm, s_i, d_i = project_point_full(
                    xy[i], tck, unew, cum_dist, t_init=None, lut=lut)
                s_native[i] = s_i
                d_nat[i]    = d_i
                psi_t[i]    = np.arctan2(tang[1], tang[0])

        tx, ty = np.cos(psi_t), np.sin(psi_t)
    else:
        # ── Batched Newton projection ─────────────────────────────────────────
        lut = geometry_store[geom_key].get('lut')
        if lut is not None:
            t_lut, xy_lut = lut
            diff       = xy_lut[np.newaxis, :, :] - xy[:, np.newaxis, :]
            t_init_arr = t_lut[np.argmin((diff * diff).sum(axis=2), axis=1)]
        else:
            t_c        = np.linspace(0, 1, 200)
            xs_c, ys_c = splev(t_c, tck)
            pts_c      = np.stack([xs_c, ys_c], axis=1)
            diff       = pts_c[np.newaxis] - xy[:, np.newaxis, :]
            t_init_arr = t_c[np.argmin((diff * diff).sum(2), axis=1)]
    
        delta = 0.1
        t_v   = t_init_arr.copy()
        px, py = xy[:, 0], xy[:, 1]
        for _ in range(5):
            xp,  yp  = splev(t_v, tck, der=0)
            xp1, yp1 = splev(t_v, tck, der=1)
            xp2, yp2 = splev(t_v, tck, der=2)
            ex   = xp - px;  ey = yp - py
            f1   = ex * xp1 + ey * yp1
            f2   = xp1**2 + yp1**2 + ex * xp2 + ey * yp2
            safe = np.abs(f2) > 1e-12
            step = np.where(safe, -f1 / np.where(safe, f2, 1.0), 0.0)
            t_v  = np.clip(t_v + np.clip(step, -delta, delta), 0.0, 1.0)
            if np.max(np.abs(step)) < 1e-7:
                break
    
        xp_f,  yp_f  = splev(t_v, tck, der=0)
        xp1_f, yp1_f = splev(t_v, tck, der=1)
        s_native = np.interp(t_v, unew, cum_dist)
    
        tang_len  = np.sqrt(xp1_f**2 + yp1_f**2)
        tang_len  = np.where(tang_len > 1e-12, tang_len, 1.0)
        tx = xp1_f / tang_len;  ty = yp1_f / tang_len
        ex_f = px - xp_f;  ey_f = py - yp_f
        d_nat = ex_f * (-ty) + ey_f * tx
    
    tang_arr  = np.stack([tx, ty], axis=1)    # (N, 2) — native spline direction
    norm_arr  = np.stack([-ty, tx], axis=1)   # (N, 2) — left of spline

    # ── Apply is_forward, then is_reverse ─────────────────────────────────
    # is_forward=False: negate to put in travel-direction coords
    # is_reverse=True:  negate again (double flip = cancel)
    # Net effect: flip iff is_forward XOR is_reverse
    if not is_forward:
        tang_arr = -tang_arr
        norm_arr = -norm_arr
        d_arr    = -d_nat
    else:
        d_arr    = d_nat

    if is_reverse:
        tang_arr = -tang_arr
        norm_arr = -norm_arr
        d_arr    = -d_arr

    # ── Directed s ────────────────────────────────────────────────────────
    s_directed = compute_directed_s(
        s_native, seg_key, segment_registry, geometry_store, is_reverse, 
        s_role=s_role
    )

    # ── Velocity decomposition ────────────────────────────────────────────
    vx    = speed * np.cos(psi)
    vy    = speed * np.sin(psi)
    s_dot = tang_arr[:, 0] * vx + tang_arr[:, 1] * vy
    d_dot = norm_arr[:, 0] * vx + norm_arr[:, 1] * vy

    # ── Acceleration decomposition ────────────────────────────────────────
    ax     = accel * np.cos(psi)
    ay     = accel * np.sin(psi)
    s_ddot = tang_arr[:, 0] * ax + tang_arr[:, 1] * ay
    d_ddot = norm_arr[:, 0] * ax + norm_arr[:, 1] * ay

    # ── Bike lane membership ──────────────────────────────────────────────
    in_bike_lane       = np.full(n, np.nan)
    d_to_bike_boundary = np.full(n, np.nan)

    if bike_lane is not None and 'd_boundary_spline' in bike_lane:
        d_bnd_spl        = bike_lane['d_boundary_spline']
        # w_bike           = bike_lane['w_bike']
        side             = bike_lane['side']
        s_bl_min, s_bl_max = bike_lane['s_domain']

        for i in range(n):
            s_i = float(s_native[i])
            if not (s_bl_min <= s_i <= s_bl_max):
                continue
            # d_boundary_spline is in native spline coords
            d_i_native = d_arr[i] if is_forward else -d_arr[i]
            if is_reverse:
                d_i_native = -d_i_native

            d_bnd = float(d_bnd_spl(s_i))
            d_far = d_bnd + side * w_bike_at(bike_lane, s_i)
            d_lo  = min(d_bnd, d_far) - BIKE_LANE_TOLERANCE
            d_hi  = max(d_bnd, d_far) + BIKE_LANE_TOLERANCE

            in_bike_lane[i]       = bool(d_lo <= d_i_native <= d_hi)
            d_to_bike_boundary[i] = d_i_native - d_bnd

    return {
        's_native':            s_native,
        'd_native':            d_nat,
        's':                   s_directed,
        'd':                   d_arr,
        's_dot':               s_dot,
        'd_dot':               d_dot,
        's_ddot':              s_ddot,
        'd_ddot':              d_ddot,
        'in_bike_lane':        in_bike_lane,
        'd_to_bike_boundary':  d_to_bike_boundary,
    }


def compute_directed_s(s_native, seg_key, segment_registry, geometry_store,
                       is_reverse, s_role=None):
    """
    Convert native arc-length s to directed s — always increases in
    travel direction from 0 at segment entry.

    Uses s_change from geometry_store[geom_key]['s_change'].
    
    s_role : 'approach' | 'departure' | None. When given, it decides the
        offset directly. The old np.mean(s_native) test is a fallback: with
        soft DP handoffs a run can straddle s_change, and then the mean picks
        one branch for the whole run.

    Cases (normal traversal):
        forward  approach : s_directed = s_native              (0 at entry end)
        forward  departure: s_directed = s_native - s_change   (0 at s_change)
        reverse  approach : s_directed = L - s_native          (0 at L end)
        reverse  departure: s_directed = s_change - s_native   (0 at s_change)

    Reverse traversal flips: a forward segment traversed in reverse
    is treated as if is_forward were flipped.
    """
    entry      = segment_registry[seg_key]
    geom_key   = entry['geometry_key']
    is_forward = entry['is_forward']
    L          = geometry_store[geom_key]['total_length']
    s_change   = geometry_store[geom_key].get('s_change')

    s_native = np.asarray(s_native, dtype=float)

    if s_change is None:
        if geometry_store[geom_key].get('periodic'):
            # RING: s is the canonical WRAPPED position on the ring.
            # Do NOT flip it here.
            return s_native
        # TURN: an open path with s from 0 at its start. Traversed backwards
        # it runs L -> 0, so without this flip s DECREASES along travel and
        # cumulative_s doubles back on itself.
        return (L - s_native) if (is_forward ^ is_reverse) is False else s_native

    # Effective direction: flip is_forward when actually traversing in reverse
    eff_forward = is_forward ^ is_reverse   # XOR: True if effectively forward
    
    if s_role is None:
        s_role = ('departure'
                  if (float(np.mean(s_native)) >= s_change) == eff_forward
                  else 'approach')
    if eff_forward:
        return s_native - s_change if s_role == 'departure' else s_native
    return (s_change - s_native) if s_role == 'departure' else (L - s_native)

    # s_mean = float(np.mean(s_native))
    # if eff_forward:
    #     # Approach: s_mean < s_change → s_directed = s_native
    #     # Departure: s_mean >= s_change → s_directed = s_native - s_change
    #     return s_native if s_mean < s_change else s_native - s_change
    # else:
    #     # Approach: s_mean >= s_change → s_directed = L - s_native
    #     # Departure: s_mean < s_change → s_directed = s_change - s_native
    #     return (L - s_native) if s_mean >= s_change else (s_change - s_native)


# #############################################################################
# 8. ENTRY POINTS
# #############################################################################
def setup_site(geometry_store, segment_registry, movement_registry,
               mode='bike', extra_gate_defs=None,
               allow_reverse_turns=False):
    """
    One-time setup. Call once after loading the registry .pkl, before the
    vehicle loop, and pass the result into assign_segments as `site`.
    """
    build_registry_luts(geometry_store)
 
    states, geom_keys, poly_keys, geom_to_polycols = build_state_space(
        geometry_store, segment_registry, mode=mode,
        allow_reverse_turns=allow_reverse_turns,
    )
    tau_class, _idx = build_transition_table(
        states, segment_registry, movement_registry
    )
    gates = build_gates(geometry_store, segment_registry, extra_gate_defs)
 
    return {
        'states':     states,
        'geom_keys':  geom_keys,
        'poly_keys':  poly_keys,
        'geom_to_polycols': geom_to_polycols,
        'gates':      gates,
        'tau_class':  tau_class,
        'tau_vals':   TAU_BY_MODE.get(mode, TAU_BY_MODE['bike']),
        'mode':       mode,
        'buffers': build_buffer_cache(poly_keys, segment_registry),
    }


def assign_segments(traj, proj, site, movement_registry, segment_registry,
                    geometry_store, agent_mode='bike',
                    verbose=False, log=None):
    """
    DP-based segment chaining for one vehicle trajectory.

    Parameters
    ----------
    traj
    proj
    site              : the cached setup produced by setup_site()
    movement_registry : dict
    segment_registry  : dict
    geometry_store    : dict
    agent_mode        : str — 'bike' or 'vehicle'. Filters candidates to
                        segments accessible by that agent type:
                        'bike'    → mode in ('shared', 'bike')
                        'vehicle' → mode in ('shared', 'car')

    Returns
    -------
    chain        : list of dicts
    movement_key : str
    """
    if traj.n < 3:
        return [], 'unmatched'
    # --- active states -------------------------------------------------------
    # Exact containment, so this prunes harder than a bounding box: a corridor
    # the cyclist never actually enters is dropped from the DP entirely. OFF is
    # always kept — it is what explains genuinely unmapped excursions.
    all_states = site['states']
    active = [k for k, st in enumerate(all_states)
              if st.type == 'off' or proj.inside[:, st.poly_col].any()]
    if len(active) <= 1:
        return [], 'unmatched'
 
    states = [all_states[k] for k in active]
    tau_class = site['tau_class'][np.ix_(active, active)]
 
    B, prov = propose_breakpoints(traj, proj, site['gates'],
                                  site['geom_keys'], site['poly_keys'])
    pre = build_prefix_sums(traj, proj, states)
 
    # runs, total = segmental_dp_old(B, states, pre, traj, proj,
    #                            tau_class, site['tau_vals'],
    #                            geometry_store, segment_registry,
    #                            verbose=verbose, log=log)
    runs, total = segmental_dp(B, states, pre, traj, proj,
                               tau_class, site['tau_vals'],
                               geometry_store)
    
    # --- fallback: the true breakpoint may not have been proposed ------------
    cpm = _emission_cost_per_m(runs, traj) if runs else None
    used_fallback = False
    if not runs or (cpm is not None and cpm > COST_PER_M_MAX):
        if verbose and log is not None:
            log.debug(f"dense-breakpoint fallback fired "
                      f"(mapped cost/m={cpm if cpm is None else round(cpm, 2)})")
        used_fallback = True
        B, prov = propose_breakpoints(traj, proj, site['gates'],
                                      site['geom_keys'], site['poly_keys'],
                                      dense_m=DENSE_M)
        # runs2, total2 = segmental_dp_old(B, states, pre, traj, proj,
        #                              tau_class, site['tau_vals'],
        #                              geometry_store, segment_registry,
        #                              verbose=verbose, log=log)
        runs2, total2 = segmental_dp(B, states, pre, traj, proj,
                                     tau_class, site['tau_vals'],
                                     geometry_store)
        if runs2 and (total2 < total or not runs):
            runs, total = runs2, total2
 
    if not runs:
        return [], 'unmatched'

    runs = assign_roles(runs)
    chain = runs_to_chain(runs, traj, proj, geometry_store, is_fallback=used_fallback)
    mov_key, kind = classify_movement(runs, movement_registry)
 
    if verbose and log is not None and kind != 'registered':
        # Gate crossings are not used for classification, but they are the
        # most readable description of what the cyclist did — worth logging
        # whenever the decode is not a plain registered movement.
        sig = tuple((c.gate_key, c.sign)
                    for c in detect_crossings(traj, proj, site['gates'],
                                              site['geom_keys']))
        log.debug(f"{kind}: {mov_key} | gates={sig}")
 
    return chain, mov_key


def to_lane_coordinates(bike_df, movement_registry,
                         segment_registry, geometry_store,
                         agent_mode='bike', site=None,
                         verbose=False, log=None):
    """
    Full lane coordinate transform pipeline for one vehicle/bike.

    Adds all lane coordinate columns to bike_df. Unmatched rows get NaN.

    Required bike_df columns:
        x_ekf, y_ekf    — position in local EPSG:2056 [m]
        speed_ekf       — speed [km/h]
        angle_ekf       — heading [rad]
        a_ekf           — scalar acceleration [m/s²]
        veh_id          — vehicle identifier (for logging)

    Added columns:
        movement_key, segment_id, segment_type, segment_role
        match_quality      — 'good' | 'poor' | 'fallback' | 'unmatched'
        is_fallback        — bool
        is_reverse         — bool
        s, d               — lane coordinates [m]
        s_dot, d_dot       — velocity components, same units as speed_ekf [km/h]
        s_ddot, d_ddot     — acceleration components [m/s²]
        in_bike_lane       — bool or NaN
        d_to_bike_boundary — float or NaN

    Parameters
    ----------
    bike_df           : DataFrame
    movement_registry : dict
    segment_registry  : dict
    geometry_store    : dict
    agent_mode        : str — 'bike' or 'vehicle'. Controls which segments
                        are considered as candidates:
                        'bike'    → mode in ('shared', 'bike')
                        'vehicle' → mode in ('shared', 'car')

    Returns
    -------
    bike_df : DataFrame with added columns
    """
    veh_id = bike_df['veh_id'].iloc[0]
    if log is None:
        log = _get_logger(debug=verbose)
    log.debug(f'=== to_lane_coordinates | veh={veh_id} ===')

    # Initialise output columns
    bike_df = bike_df.copy()
    for col in ('s_native', 'd_native', 's', 'd', 's_dot', 'd_dot',
                's_ddot', 'd_ddot', 'd_to_bike_boundary'):
        bike_df[col] = np.nan
    bike_df['movement_key'] = 'unmatched'
    bike_df['segment_id'] = None
    bike_df['segment_type'] = None
    bike_df['segment_role'] = None
    bike_df['match_quality'] = 'unmatched'
    bike_df['in_bike_lane'] = np.nan
    bike_df['is_fallback'] = False
    bike_df['is_reverse'] = False
    
    if site is None:
        site = setup_site(geometry_store, segment_registry, movement_registry,
                          mode=agent_mode)
    traj = prepare_trajectory(bike_df)
    if traj.n < 3:
        return bike_df
    
    proj = project_all(traj, site['geom_keys'], geometry_store,
                       site['poly_keys'], segment_registry,
                       site['geom_to_polycols'], site['buffers'])
    
    chain, movement_key = assign_segments(
        traj, proj, site, movement_registry, segment_registry, geometry_store,
        agent_mode=agent_mode, verbose=verbose, log=log
    )
 
    if not chain:
        return bike_df
 
    for seg_entry in chain:
        seg_key = seg_entry['seg_key']
        df_indices = seg_entry['df_indices']
        is_reverse = seg_entry['is_reverse_traversal']
        entry = segment_registry[seg_key]
 
        result = transform_segment(bike_df, seg_key, df_indices, is_reverse,
                                   segment_registry, geometry_store,
                                   proj=proj, geom_col=seg_entry['geom_col'],
                                   s_role=seg_entry['s_role'])
 
        idx = bike_df.index[df_indices]
        for col, vals in result.items():
            bike_df.loc[idx, col] = vals
 
        bike_df.loc[idx, 'movement_key'] = movement_key
        bike_df.loc[idx, 'segment_id'] = seg_key
        bike_df.loc[idx, 'segment_type'] = entry['type']
        bike_df.loc[idx, 'segment_role'] = seg_entry['role']
        bike_df.loc[idx, 'match_quality'] = seg_entry['match_quality']
        bike_df.loc[idx, 'is_fallback'] = seg_entry['is_fallback']
        bike_df.loc[idx, 'is_reverse'] = is_reverse
 
    return bike_df

# #############################################################################
# 9. OTHER ENTRY POINTS
#         TRAVEL-DIRECTED (s, d) RECONSTRUCTION + CUMULATIVE S
#         CAR LANE MEMBERSHIP
# #############################################################################
def _unwrap_ring_s(s, L):
    """
    Wrapped ring s -> continuous, along one time-ordered run.

    Only ever applied to the copy used for cumulative_s. The canonical
    's' column stays wrapped, because s must be a property of POSITION ON
    THE GEOMETRY: two cyclists at the same physical point on the ring have
    to report the same s regardless of how many laps they had done. That
    is what makes queue position, conflict points and headway comparable
    across riders. cumulative_s is trajectory-dependent by definition, so
    lap accumulation belongs there and nowhere else.
    """
    L = float(L)
    if L <= 0:
        return np.asarray(s, dtype=float)
    theta = np.asarray(s, dtype=float) * (2.0 * np.pi / L)
    return np.unwrap(theta) * (L / (2.0 * np.pi))


def _stitch_continuous_s(s_raw, seg_ids, xy):
    """
    Offset-stitch per-segment directed 's' into one continuous axis,
    preserving the local shape of 's' (no abs()/odometer conversion).
 
    Adjacent matched segments (no unmatched rows between them) are
    re-anchored so the new segment's first valid 's' picks up exactly
    where the previous segment's last valid 's' left off — zero added
    distance, since the handoff row represents the same physical point.
 
    Runs separated by unmatched rows (segment_id null / not found, or a
    matched run with no finite 's' at all) are bridged with the
    straight-line xy distance between the last valid point before the
    gap and the first valid point after it.
 
    NaN-safe: works whether unmatched segment_id is Python None (in
    memory) or NaN (after a CSV round-trip).
 
    Parameters
    ----------
    s_raw   : (n,) array — per-row directed 's' (NaN where unmatched)
    seg_ids : (n,) array-like — per-row segment_id (None/NaN allowed)
    xy      : (n,2) array — per-row x_ekf, y_ekf
 
    Returns
    -------
    s_stitched : (n,) array
    """
    n = len(s_raw)
    s_stitched = np.full(n, np.nan)
 
    # Contiguous runs of the same segment_id, NaN-safe.
    seg_filled = pd.Series(seg_ids).fillna('__UNMATCHED__').to_numpy()
    run_id = np.zeros(n, dtype=int)
    for i in range(1, n):
        run_id[i] = run_id[i - 1] + (0 if seg_filled[i] == seg_filled[i - 1] else 1)
 
    offset        = 0.0
    prev_s_last   = np.nan
    prev_last_idx = None
 
    for r in np.unique(run_id):
        idx = np.where(run_id == r)[0]
 
        if pd.isna(seg_ids[idx[0]]):
            continue  # unmatched run — leave NaN, bridge on next matched run
 
        s_win = s_raw[idx]
        valid = np.isfinite(s_win)
        if not valid.any():
            continue  # matched but no finite 's' — treat like a gap too
 
        first_local = int(np.argmax(valid))
        first_idx   = idx[first_local]
 
        if np.isfinite(prev_s_last):
            if prev_last_idx is not None and first_idx == prev_last_idx + 1:
                gap_dist = 0.0  # immediately adjacent — pure re-anchoring
            else:
                gap_dist = float(np.hypot(xy[first_idx, 0] - xy[prev_last_idx, 0],
                                           xy[first_idx, 1] - xy[prev_last_idx, 1]))
            offset = (prev_s_last + gap_dist) - s_win[first_local]
        # else: first matched run in the trajectory — offset stays 0.0
 
        s_stitched[idx] = np.where(valid, s_win + offset, np.nan)
 
        valid_idx     = idx[valid]
        prev_s_last   = s_stitched[valid_idx[-1]]
        prev_last_idx = valid_idx[-1]
 
    return s_stitched


def compute_travel_directed_s_d(bike_df, segment_registry, geometry_store):
    """
    Recompute 's', 'd' and 'cumulative_s' for one trajectory from its
    reduced columns ('segment_id', 'segment_role', 'is_reverse', 's_native',
    'd_native', 'x_ekf', 'y_ekf').

    's' / 'd'
    ---------
    Reconstructed exactly as in transform_segment:
        eff_forward = is_forward XOR is_reverse
        d = d_native if eff_forward else -d_native
        s = compute_directed_s(..., s_role=segment_role)

    Passing segment_role is the V5 change. V4 had to fall back on
    np.mean(s_native) >= s_change to guess approach vs departure, which
    picks one branch for a whole run — wrong whenever a soft DP handoff
    lets a run straddle s_change. V5 records the role the DP decided, so
    the guess is unnecessary.

    Ring segments have no s_change, so compute_directed_s returns s_native
    unchanged: the canonical WRAPPED position on the ring, which is what
    keeps s comparable between riders. Note this is deliberately
    independent of is_reverse — a clockwise rider at a given point reports
    the same s as a counter-clockwise one. Direction lives in 'is_reverse'
    and in the sign of 'd'.

    'cumulative_s'
    --------------
    's' re-anchored to be continuous across segment boundaries. Ring runs
    are unwrapped FIRST, so a seam crossing does not inject a spurious
    -L jump mid-run, and a second lap accumulates instead of resetting.

    Rows with no segment match are left NaN.

    Returns
    -------
    df : copy of bike_df with 's', 'd', 'cumulative_s' added/overwritten.
    """
    df = bike_df.copy().reset_index(drop=True)
    n  = len(df)

    if n == 0:
        df['s'] = df['d'] = df['cumulative_s'] = pd.Series(dtype=float)
        return df

    s_directed = np.full(n, np.nan)
    d_directed = np.full(n, np.nan)
    s_for_stitch = np.full(n, np.nan)     # ring runs unwrapped

    has_role = 'segment_role' in df.columns

    # A DP run is (segment, direction). Grouping on segment_id alone merges a
    # state with its own '_rev' twin, and is_reverse is then read from the
    # first node and applied to both halves — flipping d and mis-directing s
    # for the second one.
    seg = df['segment_id'].to_numpy(dtype=object)
    rev = df['is_reverse'].to_numpy()
    run_lbl = np.array(
        [None if (s is None or s != s) else f'{s}|{int(bool(r))}'
         for s, r in zip(seg, rev)], dtype=object)

    lbl_filled = pd.Series(run_lbl).fillna('__UNMATCHED__')
    run_id = (lbl_filled != lbl_filled.shift()).cumsum().to_numpy()

    for r in np.unique(run_id):
        idx = np.where(run_id == r)[0]
        seg_key = df['segment_id'].iloc[idx[0]]

        if pd.isna(seg_key) or seg_key not in segment_registry:
            continue                       # unmatched run — leave NaN

        entry    = segment_registry[seg_key]
        geo      = geometry_store[entry['geometry_key']]
        is_fwd   = entry['is_forward']
        is_rev   = bool(df['is_reverse'].iloc[idx[0]])

        s_native = df['s_native'].to_numpy(dtype=float)[idx]
        d_native = df['d_native'].to_numpy(dtype=float)[idx]

        role = df['segment_role'].iloc[idx[0]] if has_role else None
        if role not in ('approach', 'departure'):
            role = None                    # 'turn'/'ring'/None -> let it infer

        s_dir = compute_directed_s(s_native, seg_key, segment_registry,
                                   geometry_store, is_rev, s_role=role)
        d_dir = d_native if (is_fwd ^ is_rev) else -d_native

        s_directed[idx] = s_dir
        d_directed[idx] = d_dir

        if geo.get('periodic'):
            L = float(geo['total_length'])
            jumps = np.abs(np.diff(s_dir))
            if jumps.size and jumps.max() > 0.5 * L:
                print("[WARN]: "
                    f"compute_travel_directed_s_d: run on '{seg_key}' has a "
                    f"{jumps.max():.1f} m step against L={L:.1f} m — "
                    f"cumulative_s may be off by a lap."
                )
            s_uw = _unwrap_ring_s(s_dir, L)
            if is_rev:
                s_uw = -s_uw
            s_for_stitch[idx] = s_uw
        else:
            s_for_stitch[idx] = s_dir

    df['s'] = s_directed
    df['d'] = d_directed

    xy = df[['x_ekf', 'y_ekf']].to_numpy(dtype=float)
    df['cumulative_s'] = _stitch_continuous_s(
        s_for_stitch, run_lbl, xy
    )

    return df


def add_car_lane_membership(df, segment_registry, tol=0.15, use_polygon=False):
    """
    Adds a 'car_lane_idx' column: the car lane index a row's point falls
    into, based on segment_registry[segment_id]['car_lane_d_bnd'].

    Uses d_native (native spline lateral offset — the same frame
    car_lane_d_bnd is defined in), NOT d (travel-direction-relative).

    Only assigns a lane where:
      - segment_type == 'lane'  (turns have no car_lane_d_bnd)
      - in_bike_lane is 0 or NaN  (not inside the bike lane)
      - the segment has a car_lane_d_bnd entry
      - d_native falls within one of the (d_lb - tol, d_ub + tol) bins

    Parameters
    ----------
    df : DataFrame — needs 'segment_id', 'segment_type', 'd_native',
         'in_bike_lane' columns (schema above)
    segment_registry : dict — 'lane' entries may carry
         'car_lane_d_bnd' = {lane_idx: (d_lb, d_ub), ...}
    tol : float — lateral tolerance [m] expanding each lane's bounds
         outward, to absorb GPS/matching noise near lane dividers.
         Set to 0.0 for exact bounds. Overlapping expanded bins (tol
         wide enough to bridge adjacent lanes) resolve to whichever
         lane_idx is iterated last — see note below.

    Returns
    -------
    df : copy of input with new 'car_lane_idx' column (Int64, <NA>
         where not applicable / no bin matched)
    """
    df = df.copy()
    df['car_lane_idx'] = pd.array([pd.NA] * len(df), dtype='Int64')

    not_in_bike = df['in_bike_lane'].isna() | (df['in_bike_lane'] == 0)
    eligible    = not_in_bike & (df['segment_type'] == 'lane')

    for seg_id in df.loc[eligible, 'segment_id'].unique():
        car_lane_d_bnd = segment_registry.get(seg_id, {}).get('car_lane_d_bnd')
        if not car_lane_d_bnd:
            continue

        seg_mask = eligible & (df['segment_id'] == seg_id)

        for lane_idx, lane_val in car_lane_d_bnd.items():
            # Unwrap {'d_bounds':..., 'polygon':...} dict form, if present.
            if isinstance(lane_val, dict):
                d_lb, d_ub = lane_val.get('d_bounds', (None, None))
                polygon    = lane_val.get('polygon')
            else:
                d_lb, d_ub = lane_val
                polygon    = None
                
            if d_lb is None or d_ub is None:
                continue   # unresolved bounds (e.g. polygon-only stub)

            is_function = callable(d_lb) or callable(d_ub)
            
            # ── Case (b) + polygon check ────────────────────────────────
            if use_polygon and polygon is not None and not polygon.is_empty:
                import shapely.vectorized
                
                poly_buf = polygon.buffer(tol) if tol else polygon
                x_arr = df.loc[seg_mask, 'x_ekf'].to_numpy()
                y_arr = df.loc[seg_mask, 'y_ekf'].to_numpy()
                inside = shapely.vectorized.contains(poly_buf, x_arr, y_arr)

                lane_mask = seg_mask.copy()
                lane_mask.loc[seg_mask] = inside

            # ── Case (a)/(b) via d-band ──────────────────────────────────
            else:
                if is_function:
                    s_arr = df.loc[seg_mask, 's_native'].to_numpy()
                    d_lb_arr = d_lb(s_arr) if callable(d_lb) else np.full_like(s_arr, d_lb)
                    d_ub_arr = d_ub(s_arr) if callable(d_ub) else np.full_like(s_arr, d_ub)
                    d_arr    = df.loc[seg_mask, 'd_native'].to_numpy()
                    inside   = (d_arr >= d_lb_arr - tol) & (d_arr < d_ub_arr + tol)

                    lane_mask = seg_mask.copy()
                    lane_mask.loc[seg_mask] = inside
                else:
                    lane_mask = seg_mask & df['d_native'].between(
                        d_lb - tol, d_ub + tol, inclusive='left'
                    )
            
            df.loc[lane_mask, 'car_lane_idx'] = lane_idx

    n_assigned = df['car_lane_idx'].notna().sum()
    print(f"car lane membership: {n_assigned}/{eligible.sum()} eligible rows assigned "
          f"({len(df)} total rows, tol={tol} m, use_polygon={use_polygon})")

    return df

