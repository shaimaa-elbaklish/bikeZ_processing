"""
tools_lane_coords_V4.py
-----------------------
Phase B: Lane coordinate transform pipeline — Version 4.

Key changes vs V3
-----------------
1. Polygon-based fragment extraction (replaces approach/departure domains):
   For each candidate segment the polygon walk finds [entry_idx : exit_idx] —
   the first continuous run of fragment points inside poly.buffer(tolerance).
   This is used for scoring (only those points) AND defines which rows
   belong to this segment in the output.

2. Preference ordering — entry_idx == 0 preferred:
   Group A (entry_idx == 0): polygon matches from start of fragment.
   Group B (entry_idx  > 0): polygon only matches mid-fragment.
   Group C: no points inside polygon.
   Scoring uses Group A if any exist, else Group B (flagged as fallback).
   If only Group C: all outputs NaN, match_quality='unmatched'.

3. Handoff at s_change:
   Within the polygon window [entry_idx : exit_idx], the handoff fires at
   the first point where s crosses s_change in the travel direction.
   If no crossing found, handoff = exit_idx (polygon exit is fallback).
   For turn segments: handoff = exit_idx (turns have no s_change).

4. Reverse traversal:
   Detected from net s progression sign within the polygon window.
   Forward segment (is_forward=True) + s decreasing net → is_reverse.
   Reverse segment (is_forward=False) + s increasing net → is_reverse.
   When reverse: d sign flipped, psi_lane flipped by π, s_directed mirrored,
   s_change handoff fires in opposite s direction.

5. Role assigned by chain position (not inferred from s-domain):
   Iteration 0 with lane candidates → 'approach'
   Turn segment matched → 'turn'
   Lane after turn → 'departure'
   Lane after departure (extended chain) → 'approach' for next sub-movement

6. approach_native / departure_native removed entirely from all logic.
   compute_directed_s uses s_change from geometry_store[geom_key]['s_change'].

7. pile_term removed for lane segments (no domain reference point).
   Kept for turn segments (pile near s=0 still meaningful).

Authors : ETH Zürich IVT
"""

# =============================================================================
# IMPORTS
# =============================================================================
import logging
import numpy as np
import pandas as pd

from scipy.interpolate import splev
from shapely.geometry import Point

from _logger import Logger


# =============================================================================
# LOGGER
# =============================================================================
def _get_logger(debug: bool) -> logging.Logger:
    logger = logging.getLogger(__name__)
    level  = logging.DEBUG if debug else logging.WARNING
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter('[%(levelname)s] %(name)s — %(message)s'))
        logger.addHandler(h)
    logger.setLevel(level)
    return logger


# =============================================================================
# CONSTANTS
# =============================================================================
MIN_OVERLAP_PTS         = 3        # minimum points in polygon window to score
SIGMA_DIST_M            = 3.0      # [m] lateral distance normalizer — lane
SIGMA_DIST_TURN_M       = 7.0      # [m] lateral distance normalizer — turn
SIGMA_HEAD_RAD          = np.pi / 4   # [rad] heading error normalizer (45°)
W_DIST                  = 1.0      # weight: lateral proximity
W_HEAD                  = 0.5      # weight: heading alignment
W_ARC                   = 1.5      # weight: arc-length consistency
W_PILE                  = 1.0      # weight: pileup (turn only)
W_HEAD_TURN             = 1.5      # heading weight for turn segments
                                   # higher than lane (W_HEAD=0.5) since
                                   # heading is the primary discriminator
                                   # between turns sharing the same entry
MIN_TRAVEL_ARC_M        = 5.0      # [m] minimum arc to activate arc/pile terms
CLAMP_TOL               = 0.001    # t values within this of 0/1 are clamped
POOR_MATCH_THRESHOLD    = 4.0      # score above this → 'poor'
FORCED_MATCH_THRESHOLD  = 8.0      # score above this → reject
HEADING_RELIABLE_SPEED  = 0.5      # [km/h] below this speed, heading ignored
POLYGON_TOLERANCE       = 1.0      # [m] poly.buffer() for containment test
BIKE_LANE_TOLERANCE     = 0.2      # [m] d tolerance inside bike lane boundary
W_REVERSE               = 1.5      # additive score penalty for reverse lane traversal
                                   # breaks ties on shared centerlines (fwd vs rev)
LUT_RESOLUTION          = 500      # spline LUT resolution


# =============================================================================
# WARM-STARTED SPLINE PROJECTION
# =============================================================================
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


# =============================================================================
# LUT PRECOMPUTATION
# =============================================================================
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


# =============================================================================
# ONE-TIME SETUP
# =============================================================================

def setup_registry(geometry_store, segment_registry,
                   polygon_tolerance=POLYGON_TOLERANCE):
    """
    One-time setup — call once after loading the registry .pkl,
    before the vehicle processing loop.

    Steps
    -----
    1. Spline LUTs
       Precomputes nearest-neighbour lookup tables for every geometry entry
       (stored as geometry_store[key]['lut']). Speeds up the coarse phase
       of project_point_warm.

    2. Validity polygon expansion
       Buffers each segment's validity_polygon by polygon_tolerance and
       caches the result as entry['_poly_expanded'].
       Used by _polygon_walk — avoids repeated .buffer() calls per vehicle.

    3. Intersection area expansion
       Buffers each intersection_area_* polygon by polygon_tolerance and
       caches as geometry_store['__intersection_area_*_expanded'].
       Used by score_segment — avoids repeated .buffer() calls per vehicle.

    Parameters
    ----------
    geometry_store    : dict
    segment_registry  : dict
    polygon_tolerance : float — buffer margin [m], default POLYGON_TOLERANCE
    """
    # Step 1: spline LUTs
    build_registry_luts(geometry_store)

    # Step 2: pre-expand validity polygons
    n_poly = 0
    for entry in segment_registry.values():
        poly = entry.get('validity_polygon')
        if poly is not None and not poly.is_empty:
            entry['_poly_expanded'] = poly.buffer(polygon_tolerance)
            n_poly += 1

    # Step 3: pre-expand and pre-prepare intersection area polygons
    # Stores both the expanded polygon and its shapely.prepared version
    # for fast repeated contains() calls in score_segment.
    from shapely.prepared import prep as _prep_setup
    int_area_keys = [k for k in geometry_store
                     if k.startswith('intersection_area')
                     and not k.startswith('__')]
    for ia_key in int_area_keys:
        expanded = geometry_store[ia_key].buffer(polygon_tolerance)
        geometry_store[f'__{ia_key}_expanded']  = expanded
        geometry_store[f'__{ia_key}_prepared']  = _prep_setup(expanded)

    n_geo = sum(
        1 for k, v in geometry_store.items()
        if k not in {'x_offset', 'y_offset'}
        and not k.startswith('__')
        and not k.startswith('intersection_area')
        and isinstance(v, dict)
    )
    print(
        f"Registry setup complete: "
        f"{n_geo} geometry entries, "
        f"{len(segment_registry)} segments, "
        f"{n_poly} validity polygon(s) expanded, "
        f"{len(int_area_keys)} intersection area(s) expanded"
    )


def _point_in_intersection(xy_point, geometry_store, use_expanded=True):
    """
    Return True if xy_point (local EPSG:2056) is inside any
    intersection_area_* polygon. Uses pre-prepared polygons from
    setup_registry when available.

    use_expanded : bool — if True (default), use the buffered/prepared
        polygon. If False, use the raw (unexpanded) polygon — useful for
        demotion checks where only clearly-interior points should be demoted.
    """
    pt = Point(float(xy_point[0]), float(xy_point[1]))
    for k in geometry_store:
        if not k.startswith('intersection_area') or k.startswith('__'):
            continue
        if use_expanded:
            prep_key = f'__{k}_prepared'
            poly = geometry_store.get(prep_key, geometry_store[k])
        else:
            poly = geometry_store[k]
        if poly.contains(pt):
            return True
    return False


# =============================================================================
# POLYGON WALK — core of V4 Layer 0
# =============================================================================
def _polygon_walk(fragment_xy, seg_key, segment_registry,
                  tolerance=POLYGON_TOLERANCE):
    """
    Find the first continuous run of fragment points inside the segment's
    validity polygon (expanded by tolerance).

    Walks the full fragment from index 0. Records entry_idx (first point
    inside) and exit_idx (first point outside after entry, exclusive).

    Returns
    -------
    entry_idx : int or None — first index inside polygon. None if no match.
    exit_idx  : int         — exclusive end of the run. len(fragment_xy)
                              if all remaining points are inside.
    """
    poly = segment_registry[seg_key].get('validity_polygon')
    if poly is None or poly.is_empty:
        # No polygon — claim the full fragment
        return 0, len(fragment_xy)

    # Use pre-expanded polygon from setup_registry when available
    expanded = segment_registry[seg_key].get(
        '_poly_expanded', poly.buffer(tolerance)
    )
    entry_idx = None
    exit_idx  = len(fragment_xy)

    for i in range(len(fragment_xy)):
        pt      = Point(float(fragment_xy[i, 0]), float(fragment_xy[i, 1]))
        inside  = expanded.contains(pt)

        if entry_idx is None:
            if inside:
                entry_idx = i
        else:
            if not inside:
                exit_idx = i
                break

    return entry_idx, exit_idx


# =============================================================================
# REVERSE TRAVERSAL DETECTION
# =============================================================================
def _detect_reverse(s_arr, is_forward):
    """
    Detect if a cyclist is traversing a segment in reverse.

    For a forward segment (is_forward=True), normal travel increases s.
    If s net decreases → reverse traversal.
    For a reverse segment (is_forward=False), normal travel decreases s.
    If s net increases → reverse traversal.

    Uses net s displacement (robust to oscillation near stop line).

    Returns
    -------
    is_reverse : bool
    """
    if len(s_arr) < 2:
        return False
    s_net = float(s_arr[-1] - s_arr[0])
    if is_forward:
        return s_net < 0.0
    else:
        return s_net > 0.0


# =============================================================================
# HANDOFF DETECTION
# =============================================================================
def _find_handoff(s_arr, seg_key, segment_registry, geometry_store,
                  is_reverse, exit_idx, is_departure=False):
    """
    Find the handoff index within the polygon window.

    Turn segments
        → len(s_arr): full window belongs to the turn.

    Departure lane segments (is_departure=True)
        → len(s_arr): cyclist has already crossed the relevant boundary
          (came out of a turn, or reverse traversal of an approach segment).
          The polygon exit is the natural end of the window.

    Approach lane segments (is_departure=False)
        Search for the first sustained boundary crossing in travel direction:

        Step 1 — primary s_change:
            If fires at idx > 0 → return (idx, 's_change').
            If fires at idx == 0 → cyclist started past s_change; fall
            through to Step 2 (a secondary boundary may still fire later).

        Step 2 — secondary boundaries (s_zollstr_*, etc.):
            Return earliest that fires, regardless of whether Step 1 fired
            at 0 or not.

        Step 3 — nothing crossed → len(s_arr) (polygon exit as fallback).

    Parameters
    ----------
    s_arr        : (N,) array — projected s values for the polygon window
    seg_key      : str
    is_reverse   : bool
    exit_idx     : int — polygon exit index (unused; kept for API compat)
    is_departure : bool — True when segment is a departure in the chain

    Returns
    -------
    (handoff_local, s_change_key_fired)
        handoff_local      : int — index within s_arr
        s_change_key_fired : str | None — which boundary key fired
    """
    entry      = segment_registry[seg_key]
    seg_type   = entry['type']
    is_forward = entry['is_forward']

    if seg_type == 'turn':
        # Hand off at the last index where s_native is still within the
        # turn spline domain. Beyond L the projection clamps to the endpoint
        # (convex hull polygon + buffer expansion make the polygon boundary
        # unreliable), so we use s = L as the geometric turn exit.
        geom_L    = geometry_store[seg_key]['total_length']
        tolerance = 0.5   # metres — avoid clamped plateau at spline end
        valid     = np.where(s_arr <= geom_L - tolerance)[0]
        handoff   = int(valid[-1]) + 1 if len(valid) > 0 else len(s_arr)
        return handoff, None

    geom_key = entry['geometry_key']
    geo      = geometry_store[geom_key]
    s_change = geo.get('s_change')

    if s_change is None:
        return len(s_arr), None

    # Travel direction: determines which side of a boundary is "past".
    # Forward non-reverse / reverse non-forward → s increases in travel dir.
    moving_increasing_s = (is_forward and not is_reverse) or \
                          (not is_forward and is_reverse)

    def _sustained_crossing(s_boundary, k=10):
        """
        Return the first index where s has crossed s_boundary AND stayed
        past it for k=10 consecutive points (~0.4 s at 25 Hz). Prevents
        false handoff triggers from brief oscillations near the stop line.
        Returns None if no sustained crossing found.
        """
        count = 0
        for i, s_i in enumerate(s_arr):
            past = (s_i >= s_boundary) if moving_increasing_s \
                   else (s_i <= s_boundary)
            if past:
                count += 1
                if count >= k:
                    return max(0, i - k + 1)
            else:
                count = 0
        return None

    # ── Step 1: primary s_change ──────────────────────────────────────────
    # Skipped for departure segments — cyclist has already crossed s_change
    # coming out of a turn. Only return if crossing is mid-trajectory
    # (idx > 0); idx == 0 means cyclist started past s_change, fall through.
    if not is_departure:
        idx = _sustained_crossing(s_change)
        if idx is not None and idx > 0:
            return idx, 's_change'

    # ── Step 2: secondary boundaries (T-junction stop/yield lines) ───────
    # Always checked — departure segments may still turn off at a T-junction
    # (e.g. Zollstr_EB departure → Matteng_NB via s_zollstr_west_yield).
    extra_keys = [k for k in geo
                  if k.startswith('s_') and
                  k not in ('s_stop', 's_yield', 's_change')]
    first_idx, first_key = None, None
    for k in extra_keys:
        idx_e = _sustained_crossing(geo[k])
        if idx_e is not None:
            if first_idx is None or idx_e < first_idx:
                first_idx = idx_e
                first_key = k
    if first_idx is not None:
        return first_idx, first_key

    # ── Step 3: no boundary crossed — polygon exit ────────────────────────
    return len(s_arr), None


def _confirm_minor_road_entry(remaining_xy, s_change_key_fired,
                               best_seg_key, segment_registry,
                               geometry_store, movement_registry,
                               n_sample=8):
    """
    Confirm that a cyclist on a through-road actually turned onto the minor
    road at a T-junction, rather than going straight through.

    Called when s_change_key_fired is a secondary boundary (not 's_change'),
    indicating the cyclist crossed a T-junction boundary on the through-road.

    Logic
    -----
    1. Find all movements where best_seg_key appears as approach with
       approach_s_change_key == s_change_key_fired.
    2. Collect the departure segments of those movements (the minor road segs).
    3. For each minor road segment, project a sample of remaining_xy onto
       its spline. Check if any projected point is:
       (a) inside the segment validity polygon, AND
       (b) on the departure side of s_change on the minor road
           (i.e. s_native past s_change, away from junction).
    4. Return True if any minor road segment is confirmed, False otherwise.

    Parameters
    ----------
    remaining_xy        : (N, 2) array — fragment points after handoff
    s_change_key_fired  : str — secondary boundary key that fired
    best_seg_key        : str — the through-road segment key
    segment_registry    : dict
    geometry_store      : dict
    movement_registry   : dict
    n_sample            : int — points to sample from remaining_xy

    Returns
    -------
    bool
    """
    if len(remaining_xy) < 2:
        return False

    # Step 1: find departure (minor road) segments for this through-road + boundary
    # Only segments on a DIFFERENT geometry than the through-road are minor roads.
    # e.g. Zollstr_WB is excluded (same road as Zollstr_EB); Matteng_NB included.
    through_geom = segment_registry[best_seg_key]['geometry_key']
    minor_segs   = set()
    for sequence in movement_registry.values():
        # Find movements: best_seg_key as approach → turn → departure
        for i, (s_key, s_role) in enumerate(sequence):
            if s_key != best_seg_key or s_role != 'approach':
                continue
            if i + 1 >= len(sequence):
                continue
            turn_key = sequence[i + 1][0]
            turn_entry = segment_registry.get(turn_key)
            if turn_entry is None or turn_entry['type'] != 'turn':
                continue
            if turn_entry.get('approach_s_change_key') != s_change_key_fired:
                continue
            # Found a matching turn — get departure segment
            if i + 2 < len(sequence):
                dep_seg       = sequence[i + 2][0]
                dep_seg_entry = segment_registry.get(dep_seg)
                if dep_seg_entry is None:
                    continue
                # Exclude same-road departures (through-road continuing)
                if dep_seg_entry['geometry_key'] == through_geom:
                    continue
                minor_segs.add(dep_seg)
    # print(minor_segs)
    if not minor_segs:
        return False

    # Sample points from remaining fragment
    n   = len(remaining_xy)
    idx = np.round(np.linspace(0, n - 1, min(n, n_sample))).astype(int)
    pts = remaining_xy[idx]

    # Step 2: check each minor road segment
    for seg_key in minor_segs:
        entry    = segment_registry.get(seg_key)
        if entry is None:
            continue
        poly     = entry.get('_poly_expanded',
                             entry.get('validity_polygon'))
        geom_key = entry['geometry_key']
        geo      = geometry_store.get(geom_key, {})
        if 'spline' not in geo:
            continue
        tck, unew, cum_dist = geo['spline']
        s_change_minor      = geo.get('s_change')
        is_forward_minor    = entry['is_forward']
        lut                 = geo.get('lut')

        for pt in pts:
            pt_geom = Point(float(pt[0]), float(pt[1]))

            # (b) project onto minor road spline — primary discriminator.
            # Check s_native is on the departure side of s_change
            # (cyclist has entered the minor road past the stop line).
            # Done first because points in the MattInt intersection area
            # lie between validity polygons and would be rejected by (a)
            # before (b) is ever tested.
            _, _, _, s_nat, _ = project_point_full(
                pt, tck, unew, cum_dist, lut=lut
            )

            if s_change_minor is None:
                pass   # no boundary info — fall through to polygon check
            elif is_forward_minor:
                if s_nat <= s_change_minor:
                    continue   # not yet past stop line on minor road
            else:
                if s_nat >= s_change_minor:
                    continue   # not yet past stop line on minor road

            # (a) loose spatial guard: point must be inside the expanded
            # validity polygon (_poly_expanded, pre-buffered by setup_registry).
            # The buffer already covers intersection-area points at the
            # junction mouth, so no additional buffering is needed here.
            if poly is not None and not poly.is_empty:
                if not poly.contains(pt_geom):
                    continue

            return True

    return False


# =============================================================================
# SEGMENT SCORING
# =============================================================================
def score_segment(xy_window, psi_window, speed_window,
                  seg_key, segment_registry, geometry_store,
                  is_reverse, verbose=False, log=None):
    """
    Score how well a trajectory window matches a directed segment.
    Lower score = better match. Returns np.inf if vetoed.

    The window is already [entry_idx : exit_idx] from _polygon_walk.
    All window points are presumed spatially inside the polygon.

    Scoring terms
    -------------
    dist_term : median |d| / SIGMA_DIST_M
                d is in travel-direction coords.
    head_term : RMSE heading error for moving points / SIGMA_HEAD_RAD
    arc_term  : 1 - clip(s_spread / window_arc, 0, 1)
                → 0 for correct longitudinal match
                → 1 for perpendicular crossing
    pile_term : turn only — how far min(s) is from s=0 / total_length

    Hard veto: median_d outside [-d_right, +d_left] → np.inf.

    Reverse traversal
    -----------------
    When is_reverse=True: d and psi_lane are sign-flipped so that scoring
    is always in travel-direction coordinates.

    Parameters
    ----------
    xy_window, psi_window, speed_window : arrays for the polygon window
    seg_key    : str
    is_reverse : bool — detected before calling this function
    verbose    : bool

    Returns
    -------
    score : float
    """
    entry      = segment_registry[seg_key]
    geom_key   = entry['geometry_key']
    tck, unew, cum_dist = geometry_store[geom_key]['spline']
    is_forward = entry['is_forward']
    seg_type   = entry['type']
    d_left     = entry.get('d_left',  entry.get('d_max', 30.0))
    d_right    = entry.get('d_right', entry.get('d_max', 30.0))
    lut        = geometry_store[geom_key].get('lut')

    N = len(xy_window)
    if N < MIN_OVERLAP_PTS:
        return np.inf

    # ── Veto reversed turns ───────────────────────────────────────────────
    # Turn splines are fitted in a specific travel direction. A reversed
    # turn means the wrong turn was selected (the opposing turn's convex
    # hull overlaps this one). Always veto — reverse traversal of a turn
    # is physically meaningless.
    if seg_type == 'turn' and is_reverse:
        return np.inf

    # ── Exclude intersection-area points for lane segments ────────────────
    # Points inside any intersection_area_* polygon belong to the box,
    # not to the approach/departure lane. Mask them out before scoring.
    # If too few points remain outside all boxes, veto the candidate.
    #
    # Uses pre-expanded polygons (setup_registry stores them under
    # '__intersection_area_*_expanded'). Falls back to raw polygon if
    # setup_registry was not called.
    #
    # Vectorised: all points checked against each box in one loop;
    # shapely.prepared.prep() accelerates repeated contains() calls.
    if seg_type == 'lane':
        # Collect pre-prepared intersection area polygons.
        # setup_registry stores them as '__intersection_area_*_prepared'.
        # Falls back to building prepared polygon on the fly if not cached.
        ia_prepared = []
        for k in geometry_store:
            if k.startswith('intersection_area') and not k.startswith('__'):
                prep_key = f'__{k}_prepared'
                if prep_key in geometry_store:
                    ia_prepared.append(geometry_store[prep_key])
                else:
                    from shapely.prepared import prep as _prep
                    ia_prepared.append(_prep(
                        geometry_store.get(f'__{k}_expanded',
                                           geometry_store[k])
                    ))

        if ia_prepared:
            outside_box = np.ones(N, dtype=bool)
            for ia_prep in ia_prepared:
                for i in np.where(outside_box)[0]:
                    if ia_prep.contains(
                        Point(float(xy_window[i, 0]),
                              float(xy_window[i, 1]))
                    ):
                        outside_box[i] = False

            n_outside = int(outside_box.sum())
            if n_outside < MIN_OVERLAP_PTS:
                return np.inf
            # Restrict window to outside-box points
            xy_window    = xy_window[outside_box]
            psi_window   = psi_window[outside_box]
            speed_window = speed_window[outside_box]
            N            = n_outside

    # ── Project all window points ─────────────────────────────────────────
    s_arr    = np.zeros(N)
    d_arr    = np.zeros(N)
    t_arr    = np.zeros(N)
    psi_lane = np.zeros(N)
    t_prev   = None

    for i, pt in enumerate(xy_window):
        t_star, tang, _, s_i, d_i = project_point_full(
            pt, tck, unew, cum_dist, t_init=t_prev, lut=lut
        )
        s_arr[i]  = s_i
        t_arr[i]  = t_star
        psi_raw   = float(np.arctan2(tang[1], tang[0]))

        # Apply is_forward, then reverse correction
        if is_forward:
            d_arr[i]    = d_i
            psi_lane[i] = psi_raw
        else:
            d_arr[i]    = -d_i
            psi_lane[i] = psi_raw + np.pi

        if is_reverse:
            d_arr[i]    = -d_arr[i]
            psi_lane[i] = psi_lane[i] + np.pi

        t_prev = t_star

    # ── Clamping filter ───────────────────────────────────────────────────
    not_clamped = (t_arr > CLAMP_TOL) & (t_arr < 1.0 - CLAMP_TOL)
    n_valid     = int(not_clamped.sum())
    if n_valid < MIN_OVERLAP_PTS:
        return np.inf

    # ── Hard lateral veto ─────────────────────────────────────────────────
    # For reverse traversal, the cyclist's left/right are physically swapped
    # relative to the segment definition — d_left and d_right are exchanged.
    median_d    = float(np.median(d_arr[not_clamped]))
    d_veto_left  = d_right if is_reverse else d_left
    d_veto_right = d_left  if is_reverse else d_right
    if median_d > d_veto_left + 1.0 or median_d < -d_veto_right - 1.0:
        if verbose and log is not None:
            log.debug(f"  [{seg_key}] HARD VETO median_d={median_d:.2f}m "
                      f"outside [-{d_veto_right:.1f}, +{d_veto_left:.1f}]"
                      f"{' [reverse]' if is_reverse else ''}")
        return np.inf

    # ── Term 1: lateral proximity ─────────────────────────────────────────
    sigma  = SIGMA_DIST_TURN_M if seg_type == 'turn' else SIGMA_DIST_M
    dist_term = float(np.median(np.abs(d_arr[not_clamped]))) / sigma

    # ── Term 2: heading alignment ─────────────────────────────────────────
    # Turn segments use W_HEAD_TURN > W_HEAD — heading is the primary
    # signal distinguishing turns sharing the same approach entry point.
    w_head_eff = W_HEAD_TURN if seg_type == 'turn' else W_HEAD
    ang_diff   = np.abs(
        (psi_window[not_clamped] - psi_lane[not_clamped] + np.pi)
        % (2 * np.pi) - np.pi
    )
    speed_mask = speed_window[not_clamped] > HEADING_RELIABLE_SPEED
    if speed_mask.any():
        head_term = float(np.sqrt(np.mean(ang_diff[speed_mask] ** 2))) / SIGMA_HEAD_RAD
        head_term *= float(speed_mask.mean())
    else:
        head_term = 0.0

    # ── Term 3: arc-length consistency ────────────────────────────────────
    xy_valid   = xy_window[not_clamped]
    window_arc = float(np.sum(np.linalg.norm(np.diff(xy_valid, axis=0), axis=1))) \
                 if n_valid > 1 else 0.0
    if window_arc > MIN_TRAVEL_ARC_M:
        s_valid   = s_arr[not_clamped]
        s_spread  = float(s_valid.max() - s_valid.min())
        arc_term  = 1.0 - float(np.clip(s_spread / window_arc, 0.0, 1.0))
    else:
        arc_term  = 0.0

    # ── Term 4: pile-up (turn segments only) ──────────────────────────────
    if seg_type == 'turn' and window_arc > MIN_TRAVEL_ARC_M:
        L_turn    = float(geometry_store[geom_key]['total_length'])
        s_valid   = s_arr[not_clamped]
        pile_term = float(np.clip(s_valid.min() / max(L_turn, 1e-6), 0.0, 1.0))
    else:
        pile_term = 0.0

    # Reverse traversal penalty for lane segments.
    # Breaks ties on shared centerlines where fwd and rev
    # segments project identically (same spline, flipped signs).
    # Not applied to turns — reversed turns are already vetoed.
    reverse_penalty = W_REVERSE if (seg_type == 'lane' and is_reverse) else 0.0

    score = (W_DIST   * dist_term +
             w_head_eff * head_term +
             W_ARC    * arc_term  +
             W_PILE   * pile_term +
             reverse_penalty)

    if verbose and log is not None:
        log.debug(f"  [{seg_key}] reverse={is_reverse}  n={N}  "
                  f"median_d={median_d:.2f}m")
        log.debug(f"    dist={W_DIST*dist_term:.3f}  "
                  f"head={w_head_eff*head_term:.3f}(w={w_head_eff})  "
                  f"arc={W_ARC*arc_term:.3f}  pile={W_PILE*pile_term:.3f}  "
                  f"rev_pen={reverse_penalty:.3f}  → score={score:.3f}")

    return score


# =============================================================================
# REGISTRY HELPERS
# =============================================================================
OPP_DIR = {'EB': 'WB', 'WB': 'EB', 'NB': 'SB', 'SB': 'NB',
           'NE': 'SW', 'SW': 'NE'}


def get_next_candidates(seg_key, role, movement_registry, segment_registry,
                        s_change_key_fired=None):
    """
    Return the set of possible next segment keys from movement_registry.

    s_change_key_fired: when provided, filters turn candidates by their
    stored approach_s_change_key — ensuring that firing at a secondary
    junction boundary (e.g. 's_zollstr_west_yield') only returns turn
    candidates for that junction, not the main intersection turns.
    's_change' → MainInt turns only.
    's_zollstr_west_yield' → MattInt turns only.
    None → no filtering.
    """
    next_cands = set()

    def _matches(turn_key):
        if s_change_key_fired is None:
            return True
        e = segment_registry.get(turn_key)
        if e is None or e['type'] != 'turn':
            return True
        return e.get('approach_s_change_key', 's_change') == s_change_key_fired

    # Primary lookup
    for sequence in movement_registry.values():
        for i, (s_key, s_role) in enumerate(sequence):
            if s_key == seg_key and s_role == role and i + 1 < len(sequence):
                cand = sequence[i + 1][0]
                if _matches(cand):
                    next_cands.add(cand)

    # Chain extension: departure → also search as approach
    if role == 'departure':
        for sequence in movement_registry.values():
            for i, (s_key, s_role) in enumerate(sequence):
                if s_key == seg_key and s_role == 'approach'                         and i + 1 < len(sequence):
                    cand = sequence[i + 1][0]
                    if _matches(cand):
                        next_cands.add(cand)

    # Opposite-direction counterpart for lane candidates
    for cand_key in list(next_cands):
        entry = segment_registry.get(cand_key)
        if entry is None or entry['type'] != 'lane':
            continue
        opp_dir = OPP_DIR.get(entry['direction'])
        if opp_dir is None:
            continue
        opp_key = f"{entry['geometry_key']}_{opp_dir}"
        if opp_key in segment_registry and opp_key != cand_key:
            next_cands.add(opp_key)

    return next_cands


def derive_movement_key(chain, movement_registry):
    """
    Derive movement key from segment chain.
    Returns exact key if found, else builds partial key.
    """
    seg_keys = [e['seg_key'] for e in chain]

    for mov_key, sequence in movement_registry.items():
        mov_segs = [s for s, _ in sequence]
        if all(s in mov_segs for s in seg_keys):
            return mov_key

    approach_seg  = next(
        (e['seg_key'] for e in chain if e['role'] == 'approach'), 'unknown')
    departure_seg = next(
        (e['seg_key'] for e in chain if e['role'] == 'departure'), 'unknown')
    return f'{approach_seg}_2_{departure_seg}'


# =============================================================================
# ROLE ASSIGNMENT
# =============================================================================
def _infer_role_from_registry(seg_key, segment_registry, movement_registry,
                               is_reverse=False):
    """
    Infer the role of a segment at chain start (iteration 0) from the
    movement registry rather than from s-domain position.

    Logic:
        Turn segment             → always 'turn'
        Appears only as 'departure' in all movements → 'departure'
        All other cases          → 'approach'

    The preference for 'approach' when a segment plays multiple roles
    (e.g. LangstrS_NB: approach in MainInt, never departure) is safe
    because get_next_candidates handles chain extension transparently.

    When is_reverse=True the effective role is flipped: a cyclist
    traversing an approach segment in reverse is heading away from the
    intersection (departure behaviour), and vice-versa.
    """
    if segment_registry[seg_key]['type'] == 'turn':
        return 'turn'

    roles = set()
    for sequence in movement_registry.values():
        for s_key, s_role in sequence:
            if s_key == seg_key:
                roles.add(s_role)

    registry_role = 'departure' if roles == {'departure'} else 'approach'

    if is_reverse:
        return 'departure' if registry_role == 'approach' else 'approach'
    return registry_role


def _assign_role(iteration, seg_type, prev_role,
                 seg_key=None, segment_registry=None, movement_registry=None,
                 is_reverse=False):
    """
    Assign role from chain position.

    At iteration 0: use _infer_role_from_registry when seg_key and
    registries are provided (handles trajectories starting mid-turn or
    mid-departure). Falls back to 'approach' if not provided.

    Rules for iteration > 0:
        seg_type == 'turn'      → 'turn'
        after 'turn', lane      → 'departure'
        after 'departure', lane → 'approach' (extended chain, new sub-movement)
    """
    if seg_type == 'turn':
        return 'turn'
    if iteration == 0:
        if seg_key is not None and segment_registry is not None \
                and movement_registry is not None:
            return _infer_role_from_registry(
                seg_key, segment_registry, movement_registry,
                is_reverse=is_reverse,
            )
        return 'approach'
    if prev_role == 'turn':
        return 'departure'
    if prev_role == 'departure':
        return 'approach'    # extended chain — new sub-movement
    return 'approach'        # fallback


# =============================================================================
# COORDINATE TRANSFORM
# =============================================================================
def compute_directed_s(s_native, seg_key, segment_registry, geometry_store,
                       is_reverse):
    """
    Convert native arc-length s to directed s — always increases in
    travel direction from 0 at segment entry.

    Uses s_change from geometry_store[geom_key]['s_change'].

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
        # Turn segment — s always starts at 0
        return s_native

    # Effective direction: flip is_forward when actually traversing in reverse
    eff_forward = is_forward ^ is_reverse   # XOR: True if effectively forward

    s_mean = float(np.mean(s_native))
    if eff_forward:
        # Approach: s_mean < s_change → s_directed = s_native
        # Departure: s_mean >= s_change → s_directed = s_native - s_change
        return s_native if s_mean < s_change else s_native - s_change
    else:
        # Approach: s_mean >= s_change → s_directed = L - s_native
        # Departure: s_mean < s_change → s_directed = s_change - s_native
        return (L - s_native) if s_mean >= s_change else (s_change - s_native)


def transform_segment(bike_df, seg_key, df_indices, is_reverse,
                       segment_registry, geometry_store):
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

    Returns
    -------
    result : dict of column_name → array of length len(df_indices)
    """
    entry           = segment_registry[seg_key]
    geom_key        = entry['geometry_key']
    tck, unew, cum_dist = geometry_store[geom_key]['spline']
    is_forward      = entry['is_forward']
    seg_type        = entry['type']
    bike_lane       = entry.get('bike_lane')

    xy     = bike_df.iloc[df_indices][['x_ekf', 'y_ekf']].to_numpy()
    speed  = bike_df.iloc[df_indices]['speed_ekf'].to_numpy()
    psi    = bike_df.iloc[df_indices]['angle_ekf'].to_numpy()
    accel  = bike_df.iloc[df_indices]['a_ekf'].to_numpy()
    n      = len(df_indices)

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
    tang_arr  = np.stack([tx, ty], axis=1)    # (N, 2) — spline native
    norm_arr  = np.stack([-ty, tx], axis=1)   # (N, 2) — left of spline

    ex_f = px - xp_f;  ey_f = py - yp_f
    d_nat = ex_f * norm_arr[:, 0] + ey_f * norm_arr[:, 1]

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
        s_native, seg_key, segment_registry, geometry_store, is_reverse
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
        w_bike           = bike_lane['w_bike']
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
            d_far = d_bnd + side * w_bike
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


# =============================================================================
# MAIN MATCHING PHASE
# =============================================================================
def assign_segments(bike_df, movement_registry,
                    segment_registry, geometry_store,
                    max_chain_length=5,
                    agent_mode='bike',
                    verbose=False, log=None):
    """
    Sequential segment chaining for one vehicle trajectory.

    V4 matching logic — per iteration:

    Step 1 — Polygon walk for all candidates.
        For each candidate: walk fragment → (entry_idx, exit_idx).
        Group A: entry_idx == 0  (polygon claims from fragment start).
        Group B: entry_idx  > 0  (polygon claims mid-fragment).
        Group C: no match        (rejected).

    Step 2 — Choose scoring group.
        Group A non-empty → score Group A, is_fallback=False.
        Group A empty, Group B non-empty → score Group B, is_fallback=True.
        Both empty → match_quality='unmatched', stop chain.

    Step 3 — Score each candidate on its [entry_idx : exit_idx] window.
        Detect is_reverse from s progression sign.
        score_segment() returns np.inf if hard-vetoed.

    Step 4 — Accept best scoring candidate.
        score > FORCED_MATCH_THRESHOLD → unmatched, stop.
        score > POOR_MATCH_THRESHOLD   → match_quality='poor'.
        else                           → match_quality='good'.

    Step 5 — Handoff at s_change.
        Within [entry_idx : exit_idx], find first index where s crosses
        s_change in travel direction. Fallback: exit_idx.
        Next fragment starts at global index (entry_idx + handoff_local).

    Step 6 — Get next candidates from movement_registry.

    Reverse traversal
    -----------------
    When is_reverse=True for a lane segment, the lookup for next candidates
    uses the opposite-direction key and flipped role so the movement registry
    returns the correct turning candidates.

    Parameters
    ----------
    bike_df           : DataFrame
    movement_registry : dict
    segment_registry  : dict
    geometry_store    : dict
    max_chain_length  : int
    agent_mode        : str — 'bike' or 'vehicle'. Filters candidates to
                        segments accessible by that agent type:
                        'bike'    → mode in ('shared', 'bike')
                        'vehicle' → mode in ('shared', 'car')

    Returns
    -------
    chain        : list of dicts
    movement_key : str
    """
    # Mode filter: bikes use shared+bike lanes; vehicles use shared+car lanes.
    _ALLOWED_MODES = {
        'bike':    {'shared', 'bike'},
        'vehicle': {'shared', 'car'},
    }
    allowed_modes = _ALLOWED_MODES.get(agent_mode, {'shared', 'bike', 'car'})

    all_segs = [
        k for k in segment_registry
        if k not in {'x_offset', 'y_offset'}
        and (
            segment_registry[k].get('type') == 'turn'   # turns always included
            or segment_registry[k].get('mode', 'shared') in allowed_modes
        )
    ]

    chain              = []
    candidates         = all_segs   # include turns from the start
    iteration          = 0
    prev_role          = None
    # Global index into bike_df for the current fragment start
    frag_global_start  = 0

    while frag_global_start < len(bike_df) and iteration < max_chain_length:

        fragment    = bike_df.iloc[frag_global_start:].reset_index(drop=False)
        fragment_xy = fragment[['x_ekf', 'y_ekf']].to_numpy()
        fragment_psi   = fragment['angle_ekf'].to_numpy()
        fragment_speed = fragment['speed_ekf'].to_numpy()

        if len(fragment_xy) < MIN_OVERLAP_PTS:
            break

        # ── Step 1: polygon walk for all candidates ───────────────────────
        group_a = {}   # entry_idx == 0 → {seg_key: (entry_idx, exit_idx)}
        group_b = {}   # entry_idx  > 0

        for seg_key in candidates:
            entry_idx, exit_idx = _polygon_walk(
                fragment_xy, seg_key, segment_registry
            )
            if entry_idx is None:
                continue    # Group C — no match
            window_len = exit_idx - entry_idx
            if window_len < MIN_OVERLAP_PTS:
                continue    # too short to score

            if entry_idx == 0:
                seg_type_i = segment_registry[seg_key]['type']
                first_in_intersection = _point_in_intersection(
                    fragment_xy[0], geometry_store, use_expanded=False)

                if seg_type_i == 'lane' and first_in_intersection:
                    # Lane demoted: trajectory starts inside intersection
                    # → turns should claim it.
                    group_b[seg_key] = (entry_idx, exit_idx)
                elif seg_type_i == 'turn' and not first_in_intersection:
                    # Turn demoted: trajectory starts on approach road
                    # → lane segment should claim it first.
                    group_b[seg_key] = (entry_idx, exit_idx)
                else:
                    group_a[seg_key] = (entry_idx, exit_idx)
            else:
                group_b[seg_key] = (entry_idx, exit_idx)

        # ── Step 2: choose scoring group ─────────────────────────────────
        if group_a:
            scoring_group = group_a
            is_fallback   = False
        elif group_b:
            scoring_group = group_b
            is_fallback   = True
            if verbose and log is not None:
                log.debug(f"[iter {iteration}] fallback — no polygon matched "
                          f"from start, using mid-fragment matches")
        else:
            # All candidates failed.
            # Only expand to all segments at iteration 0 (trajectory starts
            # somewhere unregistered). For mid-chain iterations, stop — the
            # cyclist has exited the scene or the chain is complete.
            if candidates != all_segs and iteration == 0:
                if verbose and log is not None:
                    log.debug(f"[iter {iteration}] no polygon match — "
                              f"expanding to all segments")
                candidates = all_segs
                continue
            if verbose and log is not None:
                log.debug(f"[iter {iteration}] unmatched — no polygon "
                          f"match, stopping chain")
            break

        # ── Step 3: score each candidate in the group ────────────────────
        # Sentinel: best_entry initialised to len(fragment_xy) so any real
        # candidate (entry_idx < len) beats it in the tuple comparison.
        best_seg_key = None
        best_score   = np.inf
        best_entry   = len(fragment_xy)
        best_exit    = len(fragment_xy)
        best_reverse = False

        for seg_key, (entry_idx, exit_idx) in scoring_group.items():
            xy_w    = fragment_xy[entry_idx:exit_idx]
            psi_w   = fragment_psi[entry_idx:exit_idx]
            speed_w = fragment_speed[entry_idx:exit_idx]

            # Detect reverse from s progression
            # Project a sample to get s values
            entry      = segment_registry[seg_key]
            geom_key   = entry['geometry_key']
            tck, unew, cum_dist = geometry_store[geom_key]['spline']
            lut        = geometry_store[geom_key].get('lut')
            is_forward = entry['is_forward']

            # Sample a few s values for reverse detection
            n_sample  = min(len(xy_w), 10)
            idx_s     = np.round(np.linspace(0, len(xy_w)-1, n_sample)).astype(int)
            s_sample  = np.array([
                project_point_full(xy_w[i], tck, unew, cum_dist, lut=lut)[3]
                for i in idx_s
            ])
            is_reverse = _detect_reverse(s_sample, is_forward)

            score = score_segment(
                xy_w, psi_w, speed_w,
                seg_key, segment_registry, geometry_store,
                is_reverse=is_reverse,
                verbose=verbose, log=log,
            )

            if verbose and log is not None:
                log.debug(f"  [{seg_key}] entry={entry_idx} exit={exit_idx} "
                          f"reverse={is_reverse} score={score:.3f}")

            # Selection criterion based on chain position:
            #
            # Group A (is_fallback=False, entry_idx==0):
            #   Pure score — all candidates start at index 0.
            #
            # Group B (is_fallback=True):
            #   iteration 0 or after approach (next are turns):
            #     Pure score — entry_idx not meaningful (turns share
            #     the same intersection origin; iteration 0 finds best start)
            #   after turn (next are departure lanes):
            #     (entry_idx, score) — departure lane claiming the trajectory
            #     earliest is the correct one geographically.
            if is_fallback and prev_role == 'turn':
                # Departure lanes: prefer earliest entry
                candidate_key = (entry_idx, score)
                best_key      = (best_entry, best_score)
            else:
                # Turns, iteration 0, Group A: pure score
                candidate_key = (score, entry_idx)
                best_key      = (best_score, best_entry)

            if candidate_key < best_key:
                best_score   = score
                best_seg_key = seg_key
                best_entry   = entry_idx
                best_exit    = exit_idx
                best_reverse = is_reverse

        # ── Step 4: accept / reject ───────────────────────────────────────
        if best_seg_key is None or best_score > FORCED_MATCH_THRESHOLD:
            if verbose and log is not None:
                log.debug(f"[iter {iteration}] rejected — best score "
                          f"{best_score:.3f} > threshold")
            break

        if is_fallback:
            match_quality = 'fallback'
        elif best_score > POOR_MATCH_THRESHOLD:
            match_quality = 'poor'
        else:
            match_quality = 'good'

        # ── Step 5: handoff at s_change ───────────────────────────────────
        xy_win  = fragment_xy[best_entry:best_exit]
        entry_seg = segment_registry[best_seg_key]
        geom_key  = entry_seg['geometry_key']
        tck_h, unew_h, cum_h = geometry_store[geom_key]['spline']
        lut_h = geometry_store[geom_key].get('lut')

        s_win = np.array([
            project_point_full(xy_win[i], tck_h, unew_h, cum_h, lut=lut_h)[3]
            for i in range(len(xy_win))
        ])

        # Departure segments skip the s_change handoff — they start
        # past s_change by definition (cyclist just exited the turn).
        # Their window ends at the polygon exit naturally.
        # At iteration 0, also infer role early: if the segment only ever
        # appears as 'departure' in the registry (e.g. LangstrS_SB entered
        # mid-scene), treat as departure so handoff doesn't fire at s_change
        # index 0 and produce an empty seg_indices.
        _early_role   = _infer_role_from_registry(
            best_seg_key, segment_registry, movement_registry,
            is_reverse=best_reverse,
        ) if iteration == 0 else None
        _is_departure = (prev_role == 'turn') or (_early_role == 'departure')

        # ── _is_departure override at iteration 0 ────────────────────────
        # Two cases require adjusting _is_departure beyond the base value:
        #
        # Case A — reverse traversal of an approach segment (veh 12):
        #   Cyclist heads away from intersection on a normally-inbound road.
        #   Effectively a departure → suppress handoff, full window.
        #
        # Case B — mixed segment whose role flipped to 'departure' (veh 15):
        #   _infer_role_from_registry flipped approach→departure for a mixed
        #   segment traversed in reverse. But the cyclist may still genuinely
        #   cross s_change mid-trajectory. Trial run to check: if handoff
        #   fires at idx > 0, the crossing is real → let it fire.
        if iteration == 0:
            if best_reverse and _early_role == 'approach':
                # Case A: reverse on approach → true departure.
                _is_departure = True
            elif _early_role == 'departure':
                # Case B: role flip may have been too aggressive.
                _trial_handoff, _ = _find_handoff(
                    s_win, best_seg_key, segment_registry, geometry_store,
                    is_reverse   = best_reverse,
                    exit_idx     = len(s_win),
                    is_departure = False,
                )
                if _trial_handoff > 0:
                    # Genuine mid-trajectory crossing — let handoff fire.
                    _is_departure = False
                # _trial_handoff == 0: already past everything → keep
                # _is_departure=True (full window).

        handoff_local, s_change_key_fired = _find_handoff(
            s_win, best_seg_key, segment_registry, geometry_store,
            is_reverse    = best_reverse,
            exit_idx      = len(s_win),
            is_departure  = _is_departure,
        )

        # ── T-junction confirmation (secondary boundary only) ─────────────
        # When a secondary boundary fired (e.g. s_zollstr_east_stop on Zollstr),
        # the cyclist is on a through-road passing a T-junction. Only commit
        # to the handoff if there is evidence the cyclist actually entered the
        # minor road. If not confirmed, treat as straight-through: extend
        # the window to the full polygon exit and clear s_change_key_fired
        # so get_next_candidates uses primary intersection candidates instead.
        if s_change_key_fired is not None and s_change_key_fired != 's_change':
            remaining_xy = fragment_xy[best_entry + handoff_local :]
            confirmed = _confirm_minor_road_entry(
                remaining_xy, s_change_key_fired,
                best_seg_key, segment_registry,
                geometry_store, movement_registry,
            )
            if not confirmed:
                if verbose and log is not None:
                    log.debug(
                        f"[iter {iteration}] T-junction {s_change_key_fired} "
                        f"not confirmed — straight through on {best_seg_key}"
                    )
                # Straight through — revert to primary s_change handoff.
                # For departure segments (already past s_change), use the
                # full window — there is no primary boundary to revert to.
                if _is_departure:
                    handoff_local      = len(s_win)
                    s_change_key_fired = None
                else:
                    # Re-run _find_handoff restricted to primary s_change only
                    # by temporarily treating extra keys as non-existent.
                    # This gives either s_change crossing or exit_idx.
                    geom_key_ht  = segment_registry[best_seg_key]['geometry_key']
                    geo_ht       = geometry_store[geom_key_ht]
                    s_change_ht  = geo_ht.get('s_change')
                    moving_inc   = (segment_registry[best_seg_key]['is_forward']
                                    and not best_reverse) or \
                                   (not segment_registry[best_seg_key]['is_forward']
                                    and best_reverse)
                    handoff_local = len(s_win)   # default: full window
                    for i, s_i in enumerate(s_win):
                        if moving_inc and s_i >= s_change_ht:
                            if i > 0:
                                handoff_local = i
                            break
                        elif not moving_inc and s_i <= s_change_ht:
                            if i > 0:
                                handoff_local = i
                            break
                    s_change_key_fired = 's_change' if handoff_local < len(s_win) \
                                         else None
            else:
                if verbose and log is not None:
                    log.debug(
                        f"[iter {iteration}] T-junction {s_change_key_fired} "
                        f"confirmed — turn from {best_seg_key}"
                    )

        # Translate to global df indices
        # best_entry is relative to frag_global_start
        global_entry   = frag_global_start + best_entry
        global_handoff = frag_global_start + best_entry + handoff_local
        global_handoff = min(global_handoff, len(bike_df))

        seg_indices = list(range(global_entry, global_handoff))

        if len(seg_indices) < MIN_OVERLAP_PTS:
            if verbose and log is not None:
                log.debug(f"[iter {iteration}] break — seg_indices length "
                          f"{len(seg_indices)} < MIN_OVERLAP_PTS={MIN_OVERLAP_PTS} "
                          f"(handoff_local={handoff_local})")
            break

        # Role from chain position.
        # At iteration 0 pass registries so _infer_role_from_registry is used.
        role = _assign_role(
            iteration,
            segment_registry[best_seg_key]['type'],
            prev_role,
            seg_key          = best_seg_key        if iteration == 0 else None,
            segment_registry = segment_registry    if iteration == 0 else None,
            movement_registry= movement_registry   if iteration == 0 else None,
            is_reverse       = best_reverse,
        )

        chain.append({
            'seg_key':              best_seg_key,
            'role':                 role,
            'df_indices':           seg_indices,
            's_arr':                s_win[:handoff_local],
            'score':                best_score,
            'match_quality':        match_quality,
            'is_reverse_traversal': best_reverse,
            'is_fallback':          is_fallback,
            's_change_key_fired':   s_change_key_fired,
        })

        if verbose and log is not None:
            log.debug(f"[iter {iteration}] matched {best_seg_key} "
                      f"role={role} score={best_score:.3f} "
                      f"reverse={best_reverse} fallback={is_fallback} "
                      f"rows={seg_indices[0]}:{seg_indices[-1]}")

        # ── Step 6: next candidates ───────────────────────────────────────
        # For reverse traversal: look up via opposite-direction key + flipped role
        if best_reverse and entry_seg['type'] == 'lane':
            opp_dir = OPP_DIR.get(entry_seg['direction'])
            opp_key = f"{entry_seg['geometry_key']}_{opp_dir}" \
                      if opp_dir else best_seg_key
            flip_role  = 'approach' if role == 'departure' else 'departure'
            lookup_key  = opp_key if opp_key in segment_registry else best_seg_key
            lookup_role = flip_role
        else:
            lookup_key  = best_seg_key
            lookup_role = role

        next_cands = get_next_candidates(
            lookup_key, lookup_role, movement_registry, segment_registry,
            s_change_key_fired=s_change_key_fired,
        )
        candidates = list(next_cands) if next_cands else []
        if not candidates:
            break

        frag_global_start = global_handoff
        prev_role         = role
        iteration        += 1

    movement_key = derive_movement_key(chain, movement_registry)
    return chain, movement_key


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================
def to_lane_coordinates(bike_df, movement_registry,
                         segment_registry, geometry_store,
                         max_chain_length=3,
                         agent_mode='bike',
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
        s_dot, d_dot       — velocity components [m/s]
        s_ddot, d_ddot     — acceleration components [m/s²]
        s_decreasing       — bool
        in_bike_lane       — bool or NaN
        d_to_bike_boundary — float or NaN

    Parameters
    ----------
    bike_df           : DataFrame
    movement_registry : dict
    segment_registry  : dict
    geometry_store    : dict
    max_chain_length  : int
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
    new_cols = [
        'movement_key', 'segment_id', 'segment_type', 'segment_role',
        'match_quality', 'is_fallback', 'is_reverse',
        's_native', 'd_native', 's', 'd',
        's_dot', 'd_dot', 's_ddot', 'd_ddot',
        'in_bike_lane', 'd_to_bike_boundary',
    ]
    for col in new_cols:
        bike_df[col] = np.nan
    for col in ['movement_key', 'segment_id', 'segment_type',
                'segment_role', 'match_quality']:
        bike_df[col] = None
    bike_df['is_fallback'] = False
    bike_df['is_reverse']  = False

    # ── Matching ──────────────────────────────────────────────────────────
    chain, movement_key = assign_segments(
        bike_df, movement_registry,
        segment_registry, geometry_store,
        max_chain_length=max_chain_length,
        agent_mode=agent_mode,
        verbose=verbose, log=log,
    )

    if not chain:
        bike_df['movement_key']  = 'unmatched'
        bike_df['match_quality'] = 'unmatched'
        bike_df['is_fallback']   = True
        return bike_df

    # ── Transformation ────────────────────────────────────────────────────
    for seg_entry in chain:
        seg_key    = seg_entry['seg_key']
        role       = seg_entry['role']
        df_indices = seg_entry['df_indices']
        is_reverse = seg_entry['is_reverse_traversal']
        is_fallback= seg_entry['is_fallback']
        entry      = segment_registry[seg_key]

        result = transform_segment(
            bike_df, seg_key, df_indices, is_reverse,
            segment_registry, geometry_store,
        )

        idx = bike_df.index[df_indices]
        for col, vals in result.items():
            bike_df.loc[idx, col] = vals

        bike_df.loc[idx, 'movement_key']  = movement_key
        bike_df.loc[idx, 'segment_id']    = seg_key
        bike_df.loc[idx, 'segment_type']  = entry['type']
        bike_df.loc[idx, 'segment_role']  = role
        bike_df.loc[idx, 'match_quality'] = seg_entry['match_quality']
        bike_df.loc[idx, 'is_fallback']   = is_fallback
        bike_df.loc[idx, 'is_reverse']    = is_reverse

    return bike_df


# =============================================================================
# FORCED CHAIN TRANSFORM
# =============================================================================
def to_lane_coordinates_forced(
    bike_df,
    forced_chain,
    segment_registry,
    geometry_store,
    movement_registry,
    verbose=False,
    log=None,
):
    """
    Lane coordinate transform for a manually specified movement chain.

    Skips polygon walk and scoring entirely. The chain is taken as ground
    truth; the function finds handoff points geometrically and computes
    (s, d, s_dot, d_dot, s_ddot, d_ddot) for each segment.

    Parameters
    ----------
    bike_df          : DataFrame — one vehicle trajectory
    forced_chain     : list of str — ordered segment keys, e.g.
                       ['LangstrS_NB', 'turn_LangstrS_NB_2_LangstrN_NB',
                        'LangstrN_NB']
    segment_registry : dict
    geometry_store   : dict
    movement_registry: dict
    verbose          : bool
    log              : logger or None

    Returns
    -------
    bike_df : DataFrame with lane coordinate columns added.
              match_quality = 'forced' for all matched rows.
    """
    if log is None:
        log = _get_logger(debug=verbose)

    veh_id = bike_df['veh_id'].iloc[0] if 'veh_id' in bike_df.columns else '?'
    if verbose and log is not None:
        log.debug(f"=== to_lane_coordinates_forced | veh={veh_id} ===")
        log.debug(f"  chain: {forced_chain}")

    # Initialise output columns to NaN / defaults
    new_cols = [
        'movement_key', 'segment_id', 'segment_type', 'segment_role',
        'match_quality', 'is_fallback', 'is_reverse',
        's_native', 'd_native', 's', 'd',
        's_dot', 'd_dot', 's_ddot', 'd_ddot',
        'in_bike_lane', 'd_to_bike_boundary',
    ]
    for col in new_cols:
        bike_df[col] = np.nan
    for col in ['movement_key', 'segment_id', 'segment_type',
                'segment_role', 'match_quality']:
        bike_df[col] = None
    bike_df['is_fallback'] = False
    bike_df['is_reverse']  = False

    # Infer movement key from the chain via movement registry.
    # Build minimal chain entries that derive_movement_key needs.
    _chain_for_key = []
    _prev_role     = None
    for i, k in enumerate(forced_chain):
        if k not in segment_registry:
            continue
        _role = _assign_role(i, segment_registry[k]['type'], _prev_role)
        _chain_for_key.append({'seg_key': k, 'role': _role})
        _prev_role = _role
    movement_key = derive_movement_key(_chain_for_key, movement_registry)

    frag_global_start = 0
    prev_role         = None

    for iteration, seg_key in enumerate(forced_chain):
        if frag_global_start >= len(bike_df):
            break

        if seg_key not in segment_registry:
            if verbose and log is not None:
                log.debug(f"  [forced iter {iteration}] unknown seg_key "
                          f"'{seg_key}' — stopping")
            break

        entry    = segment_registry[seg_key]
        seg_type = entry['type']

        fragment    = bike_df.iloc[frag_global_start:].reset_index(drop=False)
        fragment_xy = fragment[['x_ekf', 'y_ekf']].to_numpy()

        if len(fragment_xy) < MIN_OVERLAP_PTS:
            break

        # ── Detect is_reverse from s progression ─────────────────────────
        geom_key             = entry['geometry_key']
        tck, unew, cum_dist  = geometry_store[geom_key]['spline']
        lut                  = geometry_store[geom_key].get('lut')
        is_forward           = entry['is_forward']

        n_sample = min(len(fragment_xy), 10)
        idx_s    = np.round(
            np.linspace(0, len(fragment_xy) - 1, n_sample)
        ).astype(int)
        s_sample = np.array([
            project_point_full(fragment_xy[i], tck, unew, cum_dist, lut=lut)[3]
            for i in idx_s
        ])
        is_reverse = _detect_reverse(s_sample, is_forward)

        # ── Project full remaining fragment for _find_handoff ─────────────
        s_win = np.array([
            project_point_full(fragment_xy[i], tck, unew, cum_dist, lut=lut)[3]
            for i in range(len(fragment_xy))
        ])

        # ── Determine role ────────────────────────────────────────────────
        role = _assign_role(
            iteration, seg_type, prev_role,
            seg_key           = seg_key          if iteration == 0 else None,
            segment_registry  = segment_registry if iteration == 0 else None,
            movement_registry = movement_registry if iteration == 0 else None,
            is_reverse        = is_reverse,
        )

        # ── _is_departure: same logic as assign_segments ──────────────────
        _early_role   = _infer_role_from_registry(
            seg_key, segment_registry, movement_registry,
            is_reverse=is_reverse,
        ) if iteration == 0 else None
        _is_departure = (prev_role == 'turn') or (_early_role == 'departure')

        if iteration == 0:
            if is_reverse and _early_role == 'approach':
                _is_departure = True
            elif _early_role == 'departure':
                _trial_handoff, _ = _find_handoff(
                    s_win, seg_key, segment_registry, geometry_store,
                    is_reverse   = is_reverse,
                    exit_idx     = len(s_win),
                    is_departure = False,
                )
                if _trial_handoff > 0:
                    _is_departure = False

        # ── Find handoff ──────────────────────────────────────────────────
        # Last segment: use full remaining fragment.
        # All other segments (including turns): use _find_handoff.
        #   Turns → last valid s within turn spline domain (s = L).
        #   Lane segments → s_change boundary or polygon exit fallback.
        # Additionally, bound handoff_local by the polygon exit — same as
        # assign_segments which uses best_exit from _polygon_walk.
        is_last = (iteration == len(forced_chain) - 1)

        # Polygon exit: find where cyclist leaves this segment's validity polygon.
        _, poly_exit = _polygon_walk(fragment_xy, seg_key, segment_registry)
        poly_exit = poly_exit if poly_exit is not None else len(s_win)

        if is_last:
            handoff_local      = poly_exit
            s_change_key_fired = None
        else:
            handoff_local, s_change_key_fired = _find_handoff(
                s_win[:poly_exit], seg_key, segment_registry, geometry_store,
                is_reverse   = is_reverse,
                exit_idx     = poly_exit,
                is_departure = _is_departure,
            )
            # Handoff fired at 0 (started past boundary) → use polygon exit
            if handoff_local == 0:
                handoff_local      = poly_exit
                s_change_key_fired = None

        # ── Row indices ───────────────────────────────────────────────────
        global_entry   = frag_global_start
        global_handoff = min(frag_global_start + handoff_local, len(bike_df))
        df_indices     = list(range(global_entry, global_handoff))

        if len(df_indices) < MIN_OVERLAP_PTS:
            if verbose and log is not None:
                log.debug(f"  [forced iter {iteration}] {seg_key} — "
                          f"too few rows ({len(df_indices)}), skipping")
            frag_global_start = global_handoff
            prev_role         = role
            continue

        if verbose and log is not None:
            log.debug(f"  [forced iter {iteration}] {seg_key} role={role} "
                      f"reverse={is_reverse} "
                      f"rows={global_entry}:{global_handoff}")

        # ── Transform ─────────────────────────────────────────────────────
        result = transform_segment(
            bike_df, seg_key, df_indices, is_reverse,
            segment_registry, geometry_store,
        )

        idx = bike_df.index[df_indices]
        for col, vals in result.items():
            bike_df.loc[idx, col] = vals

        bike_df.loc[idx, 'movement_key']  = movement_key
        bike_df.loc[idx, 'segment_id']    = seg_key
        bike_df.loc[idx, 'segment_type']  = seg_type
        bike_df.loc[idx, 'segment_role']  = role
        bike_df.loc[idx, 'match_quality'] = 'forced'
        bike_df.loc[idx, 'is_fallback']   = False
        bike_df.loc[idx, 'is_reverse']    = is_reverse

        frag_global_start = global_handoff
        prev_role         = role

    return bike_df


# =============================================================================
# RECOMPUTE DIRECTED S/D COORDINATES + CUMULATIVE S
# =============================================================================
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
    Recompute 's', 'd', and 'cumulative_s' for one bicycle trajectory
    from its reduced columns (must include 'segment_id', 'is_reverse',
    's_native', 'd_native', 'x_ekf', 'y_ekf').
 
    's' / 'd'
    ---------
    Reconstructed exactly as in transform_segment:
        eff_forward = is_forward XOR is_reverse
        d = d_native            if eff_forward else -d_native
        s = compute_directed_s(s_native, ..., is_reverse)   # handles
                                                              # s_change / L
    Rows with no segment match (segment_id is null / not in registry)
    are left as NaN, matching original behaviour for unmatched rows.
 
    'cumulative_s'
    --------------
    's' re-anchored to be continuous across segment boundaries, keeping
    its original shape (see _stitch_continuous_s):
      - Adjacent matched segments: offset so they connect with zero gap.
      - Runs separated by unmatched rows: bridged with the straight-line
        xy distance across the gap.
 
    Parameters
    ----------
    bike_df          : DataFrame — single bicycle trajectory, row order
                        assumed to be chronological.
    segment_registry : dict
    geometry_store    : dict
 
    Returns
    -------
    df : copy of bike_df with 's', 'd', 'cumulative_s' columns added/overwritten.
    """
    df = bike_df.copy().reset_index(drop=True)
    n  = len(df)
 
    if n == 0:
        df['s'] = df['d'] = df['cumulative_s'] = pd.Series(dtype=float)
        return df
 
    s_directed = np.full(n, np.nan)
    d_directed = np.full(n, np.nan)
 
    # ── Contiguous runs of the same segment_id (NaN/None = unmatched) ──────
    seg_id_filled = df['segment_id'].fillna('__UNMATCHED__')
    run_id = (seg_id_filled != seg_id_filled.shift()).cumsum().to_numpy()
 
    for r in np.unique(run_id):
        idx = np.where(run_id == r)[0]
        seg_key = df['segment_id'].iloc[idx[0]]
 
        if pd.isna(seg_key) or seg_key not in segment_registry:
            continue  # unmatched run — leave s/d as NaN
 
        s_native_vals = df['s_native'].to_numpy(dtype=float)[idx]
        d_native_vals = df['d_native'].to_numpy(dtype=float)[idx]
        is_rev        = bool(df['is_reverse'].iloc[idx[0]])
 
        is_forward = segment_registry[seg_key]['is_forward']
 
        s_dir = compute_directed_s(
            s_native_vals, seg_key, segment_registry, geometry_store, is_rev
        )
        eff_forward = is_forward ^ is_rev
        d_dir = d_native_vals if eff_forward else -d_native_vals
 
        s_directed[idx] = s_dir
        d_directed[idx] = d_dir
 
    df['s'] = s_directed
    df['d'] = d_directed
 
    # ── cumulative_s: shape-preserving stitch across segment boundaries ────
    xy = df[['x_ekf', 'y_ekf']].to_numpy(dtype=float)
    df['cumulative_s'] = _stitch_continuous_s(
        s_directed, df['segment_id'].to_numpy(), xy
    )
 
    return df
