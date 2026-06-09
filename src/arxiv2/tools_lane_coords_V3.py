"""
tools_lane_coords_V3.py
-----------------------
Phase B: Lane coordinate transform pipeline — Version 3.

Changes vs V2:
  - Layer 0: validity_polygon start-point filter in assign_segments
             (replaces loose bbox pre-filter in score_segment)
  - Layer 1: sign-aware hard lateral veto (d_left / d_right) in score_segment
  - Layer 2: arc_term added to scoring — penalises perpendicular crossings
  - Layer 3: minimum inlier coverage fraction (replaces absolute MIN_OVERLAP_PTS)

Authors: ETH Zürich IVT
"""

# #############################################################################
# IMPORTS
# #############################################################################
import logging
import numpy as np
import pandas as pd

from scipy.interpolate import splev
from shapely.geometry import Point

from tools_coordinate_transform import convert_xy2056_to_roadway_coordinates
from _logger import Logger


# =============================================================================
# LOGGER
# =============================================================================
def _get_logger(debug: bool) -> logging.Logger:
    """
    Return a logger configured to DEBUG level when debug=True,
    WARNING otherwise. Call once at the start of each public function.
    """
    logger = logging.getLogger(__name__)
    level = logging.DEBUG if debug else logging.WARNING
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            '[%(levelname)s] %(name)s — %(message)s'
        ))
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


# #############################################################################
# CONSTANTS
# #############################################################################
MIN_OVERLAP_PTS         = 5       # minimum inlier points (absolute floor)
MIN_OVERLAP_FRAC        = 0.20    # minimum inlier FRACTION of scoring window (V3)
SIGMA_DIST_M            = 3.0     # [m] lane segments
SIGMA_DIST_TURN_M       = 7.0     # [m] turn segments — cyclists deviate widely
SIGMA_HEAD_RAD          = np.pi / 4  # [rad] normalizer — 45°
W_DIST                  = 1.0     # weight: lateral proximity
W_HEAD                  = 0.5     # weight: heading alignment
W_PILE                  = 1.0     # weight: pileup penalty
W_ARC                   = 1.5     # weight: arc-length consistency (V3)
MIN_TRAVEL_PILEUP_M     = 5.0     # [m] minimum window travel to use pileup/arc terms
CLAMP_TOL               = 0.001   # t values within this of 0 or 1 are clamped
POOR_MATCH_THRESHOLD    = 3.0     # normalized score — tune empirically
FORCED_MATCH_THRESHOLD  = 6.0     # relaxed fallback ceiling
HEADING_RELIABLE_SPEED  = 0.5     # Threshold to ignore headings for near-zero speeds (in km/h)

# Layer 0: number of leading fragment points tested for polygon membership.
# At least 1 of LAYER0_N_PTS must be inside poly.buffer(tolerance).
# Using N>1 guards against a single GPS outlier at the fragment start.
LAYER0_N_PTS            = 5

# Entrance tolerance for validity_polygon filter [m].
# Polygon is expanded by this margin for both Layer 0 and window extraction.
POLYGON_ENTRY_TOLERANCE = 1.0     # [m]

# Threshold for s_decreasing flag [km/h]
S_DECREASING_THRESHOLD = -0.5

# Sparse sample size for matching (evenly spaced points from scoring window)
N_MATCH_SAMPLE = 50

BIKE_LANE_TOLERANCE = 0.2


# #############################################################################
# WARM-STARTED PROJECTION
# #############################################################################
def project_point_warm(point_local, tck, t_init=None, delta=0.1,
                       lut=None):
    """
    Project a point onto a spline with warm-started local search.

    Two-phase search:
      1. Coarse: vectorised nearest-neighbour on a precomputed LUT
                 (Opt 2) or single vectorised splev call (Opt 1).
                 Skipped entirely when t_init is provided (warm start).
      2. Refine: Newton iterations on d/dt dist_sq (3-4 steps max)

    Parameters
    ----------
    point_local : (2,) array in local EPSG:2056 coords
    tck         : spline representation
    t_init      : float or None — warm start; skips coarse search if given
    delta       : float — search half-width around t_init
    lut         : tuple (t_lut, xy_lut) or None
                  t_lut  : (M,) array of t values
                  xy_lut : (M, 2) array of spline points
                  Precomputed via build_spline_lut(). When provided,
                  coarse search is a free numpy nearest-neighbour lookup
                  instead of a splev call.

    Returns
    -------
    t_star        : float — spline parameter at closest point
    closest_point : (2,) array
    """
    point = np.asarray(point_local)

    if t_init is None:
        if lut is not None:
            # Opt 2: O(M) numpy nearest-neighbour on precomputed LUT
            t_lut, xy_lut = lut
            diffs  = xy_lut - point           # (M, 2)
            dists  = (diffs * diffs).sum(1)   # (M,)  no sqrt needed
            t_init = float(t_lut[np.argmin(dists)])
        else:
            # Opt 1: single vectorised splev (replaces 50 scalar calls)
            t_coarse   = np.linspace(0, 1, 200)
            xs, ys     = splev(t_coarse, tck)
            dists      = (xs - point[0])**2 + (ys - point[1])**2
            t_init     = float(t_coarse[np.argmin(dists)])

    # Refine with Newton iterations on d/dt dist_sq
    # f (t)  = (x-px)*x' + (y-py)*y'          [first derivative of dist_sq / 2]
    # f'(t)  = x'^2 + y'^2 + (x-px)*x'' + (y-py)*y''  [second derivative / 2]
    # step   = -f(t) / f'(t)   clamped to [-delta, +delta] and [0, 1]
    px, py = float(point[0]), float(point[1])
    t = float(t_init)
    for _ in range(4):
        xp,  yp  = splev(t, tck, der=0)
        xp1, yp1 = splev(t, tck, der=1)
        xp2, yp2 = splev(t, tck, der=2)
        ex   = float(xp) - px
        ey   = float(yp) - py
        f1   = ex * float(xp1) + ey * float(yp1)
        f2   = float(xp1)**2 + float(yp1)**2 + ex * float(xp2) + ey * float(yp2)
        if abs(f2) < 1e-12:
            break
        step = -f1 / f2
        step = max(-delta, min(delta, step))   # trust region
        t    = max(0.0, min(1.0, t + step))
        if abs(step) < 1e-7:
            break
    t_star = t
    xp, yp = splev(t_star, tck, der=0)
    return t_star, np.array([float(xp), float(yp)])


def project_point_full(point_local, tck, unew, cum_dist,
                       t_init=None, lut=None):
    """
    Full projection: warm-started closest point + roadway coordinates.

    Parameters
    ----------
    point_local         : (2,) array in local EPSG:2056
    tck, unew, cum_dist : spline representation
    t_init              : float or None — warm start
    lut                 : tuple (t_lut, xy_lut) or None
                          passed to project_point_warm for fast coarse search

    Returns
    -------
    t_star, tangent, normal, s, d
    """
    t_star, closest = project_point_warm(point_local, tck, t_init,
                                         lut=lut)

    s            = float(np.interp(t_star, unew, cum_dist))
    dx, dy       = splev(t_star, tck, der=1)
    tangent      = np.array([dx, dy])
    tangent     /= np.linalg.norm(tangent)
    normal       = np.array([-tangent[1], tangent[0]])
    d            = float(np.dot(point_local - closest, normal))

    return t_star, tangent, normal, s, d


# #############################################################################
# REGISTRY HELPERS
# #############################################################################
def get_next_candidates(seg_key, role, movement_registry,
                        segment_registry=None):
    """
    Given a matched segment and its role, return the set of possible
    next segment keys from movement_registry.

    For every registered next candidate, also adds the opposite-direction
    counterpart (same geometry key, flipped cardinal direction) when it
    exists in segment_registry. This allows matching wrong-side / reverse
    cyclists — e.g. a cyclist who correctly turned onto Roentgenstr_WB
    but is physically riding on the EB carriageway. The opposite-direction
    segment (Roentgenstr_EB) will be offered as a candidate and scored in
    reverse mode by score_segment (which auto-detects reverse traversal
    from s progression sign).

    Parameters
    ----------
    seg_key           : str — e.g. 'LangstrS_NB'
    role              : str — 'approach', 'turn', or 'departure'
    movement_registry : dict
    segment_registry  : dict or None — if provided, opposite-direction
                        counterpart is added for each candidate

    Returns
    -------
    next_candidates : set of str — possible next segment keys
                      empty if no next segment exists
    """
    OPP_DIR = {'EB': 'WB', 'WB': 'EB', 'NB': 'SB', 'SB': 'NB'}

    next_candidates = set()
    for mov_key, sequence in movement_registry.items():
        for i, (s_key, s_role) in enumerate(sequence):
            if s_key == seg_key and s_role == role:
                if i + 1 < len(sequence):
                    next_candidates.add(sequence[i + 1][0])

    # Add opposite-direction counterpart for each registered candidate.
    # Only for lane segments (turns have no opposite direction).
    if segment_registry is not None:
        for cand_key in list(next_candidates):
            entry = segment_registry.get(cand_key)
            if entry is None or entry['type'] != 'lane':
                continue
            geom_key  = entry['geometry_key']
            direction = entry['direction']
            opp_dir   = OPP_DIR.get(direction)
            if opp_dir is None:
                continue
            opp_key = f'{geom_key}_{opp_dir}'
            if opp_key in segment_registry and opp_key != cand_key:
                next_candidates.add(opp_key)

    return next_candidates


def derive_movement_key(chain, movement_registry):
    """
    Derive movement key from segment chain.

    Returns exact key if found in movement_registry,
    otherwise builds partial key:
      'approach_seg_2_unknown'
      'unknown_2_departure_seg'
      'unknown_2_unknown'

    Parameters
    ----------
    chain            : list of dicts with 'seg_key' and 'role'
    movement_registry: dict

    Returns
    -------
    movement_key : str
    """
    seg_keys = [e['seg_key'] for e in chain]

    # Try exact match — all chain segment keys appear in one movement
    for mov_key, sequence in movement_registry.items():
        mov_seg_keys = [s for s, r in sequence]
        if all(s in mov_seg_keys for s in seg_keys):
            return mov_key

    # Build partial key
    approach_seg  = next(
        (e['seg_key'] for e in chain if e['role'] == 'approach'),
        'unknown'
    )
    departure_seg = next(
        (e['seg_key'] for e in chain if e['role'] == 'departure'),
        'unknown'
    )
    return f'{approach_seg}_2_{departure_seg}'


def get_segment_s_boundary(seg_key, segment_registry, geometry_store):
    """
    Get the handoff arc-length in native spline coords.
    This is always s_stop — the boundary between inside/outside
    intersection in native coords.

    For forward segments:  handoff when s >= s_stop
    For reverse segments:  handoff when s <= s_stop

    Returns
    -------
    s_stop     : float — native arc-length of handoff boundary
    is_forward : bool
    """
    entry      = segment_registry[seg_key]
    geom_key   = entry['geometry_key']
    is_forward = entry['is_forward']
    s_stop     = geometry_store[geom_key]['s_stop']
    return s_stop, is_forward


# #############################################################################
# SEGMENT MATCHING
# #############################################################################
def score_segment(xy_local, psi, speed, seg_key,
                  segment_registry, geometry_store,
                  is_first_segment=True,
                  verbose=False, log=None):
    """
    Score how well a trajectory window matches a directed segment.
    Lower score = better match.

    V3 design:
      xy_local is already the scoring window — the contiguous leading
      portion of the fragment that lies inside the segment's validity
      polygon (extracted by _extract_scoring_window in match_segment).
      coverage normalization is removed: all candidates are scored on
      their own window, so scores are directly comparable.

    Four-term weighted sum:

      dist_term  : median |d| of inliers / SIGMA_DIST_M
      head_term  : RMSE heading error of moving inliers / SIGMA_HEAD_RAD
      pile_term  : s compressed toward entry boundary (departure/turn only)
      arc_term   : 1 - clip(s_spread / inlier_arc, 0, 1)
                   inlier_arc = arc length of inlier xy points (not full window).
                   → 0 when longitudinal progress ≈ inlier arc (correct match)
                   → 1 when trajectory crosses perpendicularly or wrong turn
                   Only active when inlier_arc > MIN_TRAVEL_PILEUP_M.

    Layer 1 hard veto: if median d of inliers falls outside
    [-d_right, +d_left] (travel-direction coords), returns np.inf.

    Parameters
    ----------
    xy_local  : (N, 2) array — scoring window positions in local EPSG:2056
    psi       : (N,) array — headings [rad]
    speed     : (N,) array — speeds [km/h]
    seg_key   : str
    segment_registry, geometry_store : dicts
    verbose   : bool
    log       : logger or None

    Returns
    -------
    score          : float — combined score (lower = better), np.inf if vetoed
    role           : str or None
    in_domain_mask : (N,) bool array — True for inlier points
    """
    entry      = segment_registry[seg_key]
    geom_key   = entry['geometry_key']
    tck, unew, cum_dist = geometry_store[geom_key]['spline']
    is_forward = entry['is_forward']
    seg_type   = entry['type']
    d_left     = entry.get('d_left',  entry.get('d_max', 30.0))
    d_right    = entry.get('d_right', entry.get('d_max', 30.0))

    N = len(xy_local)

    # ── Opt 2: precomputed LUT for fast coarse search ─────────────────────
    lut = geometry_store[geom_key].get('lut', None)

    # ── Project all window points ─────────────────────────────────────────
    s_arr    = np.zeros(N)
    d_arr    = np.zeros(N)   # travel-direction coords (+ = left, - = right)
    t_arr    = np.zeros(N)
    psi_lane = np.zeros(N)

    t_prev = None
    for i, pt in enumerate(xy_local):
        t_star, tangent, _, s_i, d_i = project_point_full(
            pt, tck, unew, cum_dist, t_init=t_prev, lut=lut
        )
        s_arr[i]  = s_i
        t_arr[i]  = t_star
        psi_raw   = float(np.arctan2(tangent[1], tangent[0]))
        if is_forward:
            d_arr[i]    = d_i
            psi_lane[i] = psi_raw
        else:
            d_arr[i]    = -d_i
            psi_lane[i] = psi_raw + np.pi
        t_prev = t_star

    # ── Clamping filter ───────────────────────────────────────────────────
    not_clamped = (t_arr > CLAMP_TOL) & (t_arr < 1.0 - CLAMP_TOL)

    # ── Reverse traversal detection ───────────────────────────────────────
    # A cyclist may traverse a segment in reverse (wrong-side / wrong-way).
    # Detection uses s-step sign majority AND negative net displacement.
    #
    # Gated by role and chain position to avoid false positives:
    #
    #   role == 'departure':
    #     Always allow reverse detection. Cyclist in departure domain going
    #     backward is approaching intersection from wrong side (veh_id=37/24).
    #
    #   role == 'approach' AND NOT is_first_segment:
    #     Allow reverse detection for post-turn approach segments. A cyclist
    #     who turned onto a segment and is riding in reverse on the approach
    #     domain (veh_id=50: WB on EB approach after turn_Zollstr_WB_2_Roentgenstr_WB).
    #
    #   role == 'approach' AND is_first_segment:
    #     Suppress reverse detection. First-iteration approach cyclists may
    #     dither near the stop line producing n_neg > n_pos by GPS noise,
    #     even though they ultimately proceed forward (veh_id=27).
    #
    # Turns: always False — no meaningful reverse traversal for turns.
    if seg_type == 'turn':
        is_reverse_traversal = False
    elif not_clamped.sum() >= 2:
        s_unclamped = s_arr[not_clamped]
        s_diff      = np.diff(s_unclamped)
        n_neg       = int((s_diff < 0).sum())
        n_pos       = int((s_diff > 0).sum())
        s_net       = float(s_unclamped[-1] - s_unclamped[0])
        is_reverse_traversal = False   # default
        # role is determined after this block — use a local check here
        # by inferring role from domain membership (approach vs departure)
        # We need role to gate, so compute it inline briefly:
        app = entry['approach_native']
        dep = entry['departure_native']
        app_min, app_max = app if app is not None else (None, None)
        dep_min, dep_max = dep if dep is not None else (None, None)
        not_cl = not_clamped
        n_app = int(((s_arr >= app_min) & (s_arr <= app_max) & not_cl).sum()) \
                if app_min is not None else 0
        n_dep = int(((s_arr >= dep_min) & (s_arr <= dep_max) & not_cl).sum()) \
                if dep_min is not None else 0
        inferred_role = 'approach' if n_app >= n_dep else 'departure'
        
        # s_net in travel direction: positive = forward, negative = reverse
        s_net_travel = s_net if is_forward else -s_net
        s_predominantly_decreasing = (n_neg > n_pos) and (s_net_travel < 0)
        if s_predominantly_decreasing:
            if inferred_role == 'departure':
                is_reverse_traversal = True
            elif inferred_role == 'approach' and not is_first_segment:
                is_reverse_traversal = True
    else:
        is_reverse_traversal = False

    if is_reverse_traversal:
        psi_lane = psi_lane + np.pi

    # ── Role and inlier domain ────────────────────────────────────────────
    if seg_type == 'turn':
        in_domain_mask = not_clamped
        role           = 'turn'
    else:
        app = entry['approach_native']
        dep = entry['departure_native']
        app_min, app_max = app if app is not None else (None, None)
        dep_min, dep_max = dep if dep is not None else (None, None)

        in_approach  = (
            not_clamped & (s_arr >= app_min) & (s_arr <= app_max)
            if app_min is not None else np.zeros(N, dtype=bool)
        )
        in_departure = (
            not_clamped & (s_arr >= dep_min) & (s_arr <= dep_max)
            if dep_min is not None else np.zeros(N, dtype=bool)
        )

        if in_approach.sum() >= in_departure.sum():
            in_domain_mask = in_approach
            role           = 'approach'
        else:
            in_domain_mask = in_departure
            role           = 'departure'

    n_in = int(in_domain_mask.sum())

    # ── Coverage gate ─────────────────────────────────────────────────────
    min_pts = max(MIN_OVERLAP_PTS, int(np.ceil(MIN_OVERLAP_FRAC * N)))
    if n_in < min_pts:
        return np.inf, role, in_domain_mask, is_reverse_traversal

    # ── Layer 1: sign-aware hard lateral veto ─────────────────────────────
    # d_arr: positive = LEFT, negative = RIGHT of travel direction.
    # Valid range: [-d_right, +d_left].
    median_d = float(np.median(d_arr[in_domain_mask]))
    if median_d > d_left + 1.0 or median_d < -d_right - 1.0:
        if verbose and log is not None:
            log.debug(f"  [{seg_key}] HARD VETO: median_d={median_d:.2f}m "
                      f"outside [-{d_right:.1f}, +{d_left:.1f}]")
        return np.inf, role, np.zeros(N, dtype=bool), is_reverse_traversal

    # ── Term 1: lateral proximity ─────────────────────────────────────────
    sigma_dist = SIGMA_DIST_TURN_M if seg_type == 'turn' else SIGMA_DIST_M
    dist_term  = float(np.median(np.abs(d_arr[in_domain_mask]))) / sigma_dist

    # ── Term 2: heading alignment ─────────────────────────────────────────
    ang_diff   = np.abs(
        (psi[in_domain_mask] - psi_lane[in_domain_mask] + np.pi)
        % (2 * np.pi) - np.pi
    )
    speed_mask = speed[in_domain_mask] > HEADING_RELIABLE_SPEED
    if speed_mask.any():
        head_term       = float(np.sqrt(np.mean(ang_diff[speed_mask] ** 2))) / SIGMA_HEAD_RAD
        head_term       = float(speed_mask.mean()) * head_term
    else:
        head_term = 0.0

    # ── Term 3: start-pileup penalty ─────────────────────────────────────
    s_inliers = s_arr[in_domain_mask]

    # Inlier arc length — arc of inlier xy points, not full window.
    # This is what arc_term compares s_spread against.
    xy_inliers  = xy_local[in_domain_mask]
    inlier_arc  = float(np.sum(np.linalg.norm(np.diff(xy_inliers, axis=0), axis=1))) \
                  if n_in > 1 else 0.0

    if seg_type == 'turn':
        domain_min = 0.0
        domain_len = float(geometry_store[geom_key]['total_length'])
        if domain_len > 0 and inlier_arc > MIN_TRAVEL_PILEUP_M:
            pile_term = float(np.clip(
                abs(s_inliers.min() - domain_min) / domain_len, 0.0, 1.0))
        else:
            pile_term = 0.0
    elif role == 'departure':
        # if is_forward:
        #     domain_min, domain_len = dep_min, dep_max - dep_min
        #     s_entry = s_inliers.min()   # forward: entry at low s
        # else:
        #     domain_min, domain_len = dep_max, dep_max - dep_min
        #     s_entry = s_inliers.max()   # reverse spline: entry at high s
        moving_increasing_s = (is_forward and not is_reverse_traversal) or \
                              (not is_forward and is_reverse_traversal)
        s_entry = s_inliers.min() if moving_increasing_s else s_inliers.max()
        if moving_increasing_s:
            domain_min = dep_min   # entry at low s
        else:
            domain_min = dep_max   # entry at high s
        domain_len = dep_max - dep_min
        if domain_len > 0 and inlier_arc > MIN_TRAVEL_PILEUP_M:
            pile_term = float(np.clip(
                abs(s_entry - domain_min) / domain_len, 0.0, 1.0))
        else:
            pile_term = 0.0
    else:   # approach
        if is_reverse_traversal:
            # Reverse approach: cyclist entered at high s (near s_change)
            # and moves toward low s. pile_term measures entry at s_change.
            domain_min_rev = app_max if is_forward else app_min
            domain_len_rev = app_max - app_min if app_min is not None else 0.0
            s_entry = s_inliers.max()
            if domain_len_rev > 0 and inlier_arc > MIN_TRAVEL_PILEUP_M:
                pile_term = float(np.clip(
                    abs(s_entry - domain_min_rev) / domain_len_rev, 0.0, 1.0))
            else:
                pile_term = 0.0
        else:
            pile_term = 0.0

    # ── Term 4: arc-length consistency ───────────────────────────────────
    # s_spread / inlier_arc ≈ 1 for correct match along segment.
    # → 0 for perpendicular crossing or wrong turn (s barely advances).
    # Uses inlier arc, not full window arc, so it measures only the
    # portion the segment can actually claim.
    if inlier_arc > MIN_TRAVEL_PILEUP_M:
        s_spread = float(s_inliers.max() - s_inliers.min())
        arc_term = 1.0 - float(np.clip(s_spread / inlier_arc, 0.0, 1.0))
    else:
        arc_term = 0.0

    # ── Combined score ────────────────────────────────────────────────────
    score = (W_DIST * dist_term +
             W_HEAD * head_term +
             W_PILE * pile_term +
             W_ARC  * arc_term)

    if verbose and log is not None:
        log.debug(f"  [{seg_key}] role={role}  n_in={n_in}  "
                  f"median_d={median_d:.2f}m  "
                  f"reverse={'YES' if is_reverse_traversal else 'no'}")
        log.debug(f"    dist_term  = {W_DIST * dist_term:.3f}  "
                  f"(median|d|={np.median(np.abs(d_arr[in_domain_mask])):.2f}m)")
        log.debug(f"    head_term  = {W_HEAD * head_term:.3f}")
        log.debug(f"    pile_term  = {W_PILE * pile_term:.3f}")
        log.debug(f"    arc_term   = {W_ARC  * arc_term:.3f}  "
                  f"(s_spread={s_inliers.max()-s_inliers.min():.2f}m  "
                  f"inlier_arc={inlier_arc:.2f}m)")
        log.debug(f"    score      = {score:.3f}")

    return score, role, in_domain_mask, is_reverse_traversal


def _extract_scoring_window(fragment_xy, seg_key, segment_registry,
                             tolerance=POLYGON_ENTRY_TOLERANCE):
    """
    Extract the contiguous leading portion of the fragment that lies
    inside the segment's validity polygon.

    Walks fragment_xy from the start; stops at the first point that
    falls outside poly.buffer(tolerance). Returns the index one past
    the last inside point (exclusive end), so window is fragment_xy[:end].

    Parameters
    ----------
    fragment_xy      : (M, 2) array — full fragment positions
    seg_key          : str
    segment_registry : dict
    tolerance        : float — polygon buffer margin [m]

    Returns
    -------
    window_end : int — exclusive end index of the scoring window.
                 Minimum 1 (always include at least the start point).
                 equals len(fragment_xy) if all points are inside.
    """
    poly = segment_registry[seg_key].get('validity_polygon')
    if poly is None or poly.is_empty:
        # No polygon available — use full fragment
        return len(fragment_xy)
    
    expanded = segment_registry[seg_key].get('_validity_polygon_expanded', 
                                             poly.buffer(tolerance))
    window_end = 0
    for i in range(len(fragment_xy)):
        pt = Point(float(fragment_xy[i, 0]), float(fragment_xy[i, 1]))
        if expanded.contains(pt):
            window_end = i + 1   # extend window to include this point
        else:
            break   # strict: stop at first exit

    return max(1, window_end)


def match_segment(fragment_df, candidates,
                  segment_registry, geometry_store,
                  n_sample=N_MATCH_SAMPLE,
                  is_first_segment=True,
                  log=None, verbose=False):
    """
    Match a trajectory fragment to the best segment from candidates.

    V3 architecture:
      For each candidate:
        1. Extract scoring window: contiguous leading fragment points
           inside the candidate's validity polygon (strict, stop at
           first exit). Cheap — point-in-polygon on full fragment.
        2. Sparse-sample within the window (N_MATCH_SAMPLE points).
        3. Project samples and score. No coverage normalization —
           all candidates scored on their own window, scores comparable.

    Parameters
    ----------
    fragment_df      : DataFrame — current trajectory fragment
    candidates       : list of str — segment keys to score
    segment_registry : dict
    geometry_store   : dict
    n_sample         : int — sparse sample size within window

    Returns
    -------
    best_seg_key  : str or None
    best_role     : str or None
    best_score    : float
    best_mask     : bool array or None
    window_end    : int — window end index of best candidate
    """
    if len(fragment_df) == 0 or len(candidates) == 0:
        return None, None, np.inf, None, 0, False

    fragment_xy  = fragment_df[['x_ekf', 'y_ekf']].to_numpy()
    psi_full     = fragment_df['angle_ekf'].to_numpy()
    vel_full     = fragment_df['speed_ekf'].to_numpy()

    best_seg_key = None
    best_role    = None
    best_score   = np.inf
    best_mask    = None
    best_window  = 0
    best_reverse = False

    for seg_key in candidates:
        # Step 1: window extraction (cheap — point-in-polygon on all points)
        w_end = _extract_scoring_window(fragment_xy, seg_key, segment_registry)

        # Step 2: sparse sample within window
        w_size     = w_end
        sample_idx = np.linspace(0, w_size - 1, min(n_sample, w_size), dtype=int)

        xy_sample  = fragment_xy[sample_idx]
        psi_sample = psi_full[sample_idx]
        vel_sample = vel_full[sample_idx]
        
        # Step 3: score
        score, role, mask, is_rev = score_segment(
            xy_sample, psi_sample, vel_sample, seg_key,
            segment_registry, geometry_store,
            is_first_segment=is_first_segment,
            verbose=verbose, log=log
        )

        if score < best_score:
            best_score   = score
            best_seg_key = seg_key
            best_role    = role
            best_mask    = mask
            best_window  = w_end
            best_reverse = is_rev

    return best_seg_key, best_role, best_score, best_mask, best_window, best_reverse


# #############################################################################
# HANDOFF DETECTION
# #############################################################################
def find_handoff_index(bike_df, seg_key, role,
                       segment_registry, geometry_store,
                       is_reverse_traversal=False,
                       max_lateral_dist=15.0):
    """
    Find the first trajectory row index where the vehicle crosses
    the segment boundary in native spline coords.

    For normal traversal:
      - approach: handoff when s crosses approach_native end (toward intersection)
      - departure: handoff when s exits departure domain
      - turn:      handoff when s reaches 95% of turn length

    For reverse traversal (is_reverse_traversal=True, lane segments only):
      The cyclist moves opposite to the segment's registered direction.
      Handoff fires when s decreases to the approach/departure boundary
      (s_change), i.e. when the cyclist crosses from departure into approach
      territory — which is the intersection boundary from the other side.
      Falls back to geometry_store s_stop if approach_native is None
      (nulled by restrict_segment_roles).

    Parameters
    ----------
    bike_df              : DataFrame — trajectory fragment
    seg_key              : str
    role                 : str — geometric role ('approach', 'departure', 'turn')
    segment_registry     : dict
    geometry_store       : dict
    is_reverse_traversal : bool
    max_lateral_dist     : float — fallback lateral exit threshold [m]

    Returns
    -------
    handoff_idx : int
    s_arr       : (N,) array of projected s values (native coords)
    """
    entry               = segment_registry[seg_key]
    geom_key            = entry['geometry_key']
    tck, unew, cum_dist = geometry_store[geom_key]['spline']
    is_forward          = entry['is_forward']
    L                   = geometry_store[geom_key]['total_length']

    app_domain = entry['approach_native']
    dep_domain = entry['departure_native']
    dep_min, dep_max = dep_domain if dep_domain is not None else (None, None)

    # Normal approach handoff boundary
    if app_domain is not None:
        s_handoff_approach = app_domain[1] if is_forward else app_domain[0]
    else:
        s_handoff_approach = None

    # Reverse traversal handoff boundary:
    # The cyclist enters from the departure end and the handoff fires when
    # s crosses s_change (the approach/departure split).
    # s_change = app_domain[1] for forward spline, app_domain[0] for reverse.
    # Falls back to geometry_store s_stop if approach_native was nulled by
    # restrict_segment_roles (segment only registered as departure).
    if is_reverse_traversal and entry['type'] == 'lane':
        if app_domain is not None:
            s_handoff_reverse = app_domain[1] if is_forward else app_domain[0]
        else:
            # Fallback: use s_stop from geometry_store
            s_handoff_reverse = 0.5 * (geometry_store[geom_key].get('s_stop', L * 0.5) + \
                                       geometry_store[geom_key].get('s_yield', L * 0.5))
    else:
        s_handoff_reverse = None

    xy     = bike_df[['x_ekf','y_ekf']].to_numpy()
    n      = len(xy)
    s_arr  = np.full(n, np.nan)
    t_prev = None

    for i in range(n):
        pt_local               = xy[i]
        t_star, _, _, s_i, d_i = project_point_full(
            pt_local, tck, unew, cum_dist, t_init=t_prev
        )
        s_arr[i] = s_i
        t_prev   = t_star

        # ── Reverse traversal lane handoff ────────────────────────────────
        # Cyclist moving opposite to registered direction: s decreases for
        # forward spline, increases for reverse spline.
        # Handoff when s crosses s_handoff_reverse from the departure side.
        if s_handoff_reverse is not None:
            if is_forward and s_i <= s_handoff_reverse + 0.001:
                return i, s_arr
            if not is_forward and s_i >= s_handoff_reverse - 0.001:
                return i, s_arr

        elif role == 'approach':
            if s_handoff_approach is None:
                continue
            if is_forward and s_i >= s_handoff_approach - 0.001:
                return i, s_arr
            if not is_forward and s_i <= s_handoff_approach + 0.001:
                return i, s_arr

        elif role == 'departure':
            if dep_min is None or dep_max is None:
                if abs(d_i) > max_lateral_dist:
                    return i, s_arr
            else:
                if is_forward and s_i >= dep_max - 0.001:
                    return i, s_arr
                if not is_forward and s_i <= dep_min + 0.001:
                    return i, s_arr
                if abs(d_i) > max_lateral_dist:
                    return i, s_arr

        elif role == 'turn':
            poly = segment_registry[seg_key].get('validity_polygon')
            if poly is not None and not poly.is_empty:
                # Use validity polygon exit as handoff
                # Require the bicycle to have entered the polygon first,
                # then fire when it exits.
                pt = Point(float(xy[i, 0]), float(xy[i, 1]))
                inside = poly.contains(pt)
                if i == 0:
                    entered_turn = inside
                else:
                    if not entered_turn and inside:
                        entered_turn = True
                    elif entered_turn and (not inside or s_i >= L * 0.95):
                        return i, s_arr
            else:
                # Fallback: s-based handoff
                if s_i >= L * 0.95:
                    return i, s_arr

    return n, s_arr

# #############################################################################
# LUT PRECOMPUTATION
# #############################################################################
LUT_RESOLUTION = 500   # number of points in spline LUT


def build_spline_lut(tck, n=LUT_RESOLUTION):
    """
    Build a lookup table for fast spline nearest-neighbour search.

    Parameters
    ----------
    tck : spline representation
    n   : int — number of uniformly sampled points

    Returns
    -------
    lut : tuple (t_lut, xy_lut)
          t_lut  : (n,) float array
          xy_lut : (n, 2) float array of spline points
    """
    t_lut     = np.linspace(0, 1, n)
    xs, ys    = splev(t_lut, tck)
    xy_lut    = np.column_stack([xs, ys])
    return t_lut, xy_lut


def build_registry_luts(geometry_store):
    """
    Precompute spline LUTs for all geometry entries (lanes + turns).

    Stores result in geometry_store[geom_key]['lut'] = (t_lut, xy_lut).
    Idempotent: skips entries that already have 'lut' populated.
    Call once after loading the registry, before processing vehicles.

    Parameters
    ----------
    geometry_store : dict — modified in-place

    Returns
    -------
    geometry_store : dict (same object, modified in-place)
    """
    skip = {'x_offset', 'y_offset'}
    for geom_key, geom in geometry_store.items():
        if geom_key in skip or 'spline' not in geom:
            continue
        if 'lut' not in geom:
            tck        = geom['spline'][0]
            geom['lut'] = build_spline_lut(tck)
    return geometry_store


# #############################################################################
# REGISTERED ENTRY DETECTION
# #############################################################################
def _build_segment_bboxes(segment_registry, geometry_store, padding=0.0):
    """
    Precompute axis-aligned bounding boxes for all lane segment splines.
 
    Call once per registry, store result, pass to _find_registered_entry.
 
    Parameters
    ----------
    segment_registry : dict
    geometry_store   : dict
    padding          : float — extra margin around bbox [m]
 
    Returns
    -------
    bboxes : dict  seg_key -> (x_min, x_max, y_min, y_max)
    """
    bboxes = {}
    for seg_key, entry in segment_registry.items():
        if entry['type'] not in ('lane', 'turn'):
            continue
        geom_key        = entry['geometry_key']
        tck             = geometry_store[geom_key]['spline'][0]
        unew            = np.linspace(0, 1, 200)
        xs, ys          = splev(unew, tck)
        bboxes[seg_key] = (
            float(xs.min()) - padding,
            float(xs.max()) + padding,
            float(ys.min()) - padding,
            float(ys.max()) + padding,
        )
    return bboxes
 
 
def _find_registered_entry(bike_df, segment_registry, geometry_store,
                            max_dist=20.0, bboxes=None):
    """
    Find the first trajectory index where the bicycle enters the domain
    of any registered lane segment.

    V3: uses validity_polygon from segment_registry for spatial filtering
    when available. Falls back to bbox+projection when polygon is absent.

    Points before the returned index are an unregistered prefix
    (e.g. Mattengasse) and remain labelled unmatched.

    Parameters
    ----------
    bike_df          : DataFrame — must have x_ekf, y_ekf
    segment_registry : dict
    geometry_store   : dict
    max_dist         : float — fallback max lateral distance [m]
    bboxes           : ignored in V3 (kept for signature compatibility)

    Returns
    -------
    first_idx : int — first index inside a registered domain.
                0 if trajectory starts inside a registered domain already.
    """
    lane_entries = [
        (k, e) for k, e in segment_registry.items()
        if e['type'] in ('lane', 'turn')
    ]

    xy = bike_df[['x_ekf', 'y_ekf']].to_numpy()
    N  = len(xy)

    # # Pre-expand validity polygons by max_dist once
    # expanded_polys = {}
    # for k, e in lane_entries:
    #     poly = e.get('validity_polygon')
    #     if poly is not None and not poly.is_empty:
    #         expanded_polys[k] = poly.buffer(max_dist)

    for i in range(N):
        pt = Point(float(xy[i, 0]), float(xy[i, 1]))

        for k, entry in lane_entries:
            # Spatial check
            # if k in expanded_polys:
            #     if not expanded_polys[k].contains(pt):
            #         continue
            if 'validity_polygon' in entry.keys():
                expanded_poly = entry.get(
                    '_validity_polygon_expanded', entry.get('validity_polygon').buffer(max_dist)
                )
                if not expanded_poly.contains(pt):
                        continue
            else:
                # Fallback: bbox check then projection
                geom_key = entry['geometry_key']
                tck      = geometry_store[geom_key]['spline'][0]
                unew_arr = np.linspace(0, 1, 200)
                xs, ys   = splev(unew_arr, tck)
                if not (xs.min() - max_dist <= xy[i, 0] <= xs.max() + max_dist and
                        ys.min() - max_dist <= xy[i, 1] <= ys.max() + max_dist):
                    continue

            # Full projection to confirm s-domain membership
            geom_key            = entry['geometry_key']
            tck, unew, cum_dist = geometry_store[geom_key]['spline']
            app = entry['approach_native']
            dep = entry['departure_native']

            _, _, _, s_i, d_i = project_point_full(
                xy[i], tck, unew, cum_dist, t_init=None
            )

            if abs(d_i) > max_dist:
                continue

            in_app = (app is not None) and (app[0] <= s_i <= app[1])
            in_dep = (dep is not None) and (dep[0] <= s_i <= dep[1])
            if in_app or in_dep:
                return i

    return 0   # no unregistered prefix — start from beginning



# #############################################################################
# V3: VALIDITY POLYGON CANDIDATE FILTER (Layer 0)
# #############################################################################
def _filter_candidates_by_validity_polygon(fragment_xy, candidates,
                                            segment_registry,
                                            n_pts=LAYER0_N_PTS,
                                            tolerance=POLYGON_ENTRY_TOLERANCE):
    """
    Layer 0 coarse spatial filter: retain only candidate segments whose
    validity_polygon contains at least one of the first n_pts fragment
    points (inside poly.buffer(tolerance)).

    Using n_pts > 1 guards against a single GPS outlier at the fragment
    start causing a false rejection. Requiring at least 1 of n_pts is
    permissive enough for the coarse pre-filter role: elimination, not
    confirmation.

    Parameters
    ----------
    fragment_xy      : (M, 2) array — full fragment positions
    candidates       : list of str
    segment_registry : dict
    n_pts            : int — number of leading points to test
    tolerance        : float — polygon expansion margin [m]

    Returns
    -------
    filtered : list of str — candidates that pass.
               Falls back to all candidates if none pass (safety net).
    """
    n_check = min(n_pts, len(fragment_xy))
    pts     = [Point(float(fragment_xy[i, 0]), float(fragment_xy[i, 1]))
               for i in range(n_check)]

    filtered = []
    for seg_key in candidates:
        poly = segment_registry[seg_key].get('validity_polygon')
        if poly is None or poly.is_empty:
            filtered.append(seg_key)
            continue
        expanded = segment_registry[seg_key].get('_validity_polygon_expanded', 
                                                 poly.buffer(tolerance))
        # Pass if at least one of the first n_pts points is inside
        if any(expanded.contains(pt) for pt in pts):
            filtered.append(seg_key)

    if not filtered:
        return list(candidates)   # safety net
    return filtered


# #############################################################################
# MAIN MATCHING PHASE
# #############################################################################
def assign_segments(bike_df, movement_registry,
                    segment_registry, geometry_store,
                    max_chain_length=3,
                    poor_match_threshold=POOR_MATCH_THRESHOLD,
                    verbose=False,
                    log=None):
    """
    Sequential segment chaining for one vehicle/bike trajectory.

    V3 matching logic:
      Pass 1: match fragment against lane segments filtered by Layer 0.
              If match is good → accept.
      Pass 2: if Pass 1 score is poor OR matched role is departure
              (bicycle entered the frame already inside the intersection)
              → retry against all segments (lanes + turns), Layer 0 filtered.
              No coverage normalization — window-based scoring makes
              scores directly comparable.

    Pass 2A (old: retry all segs on poor lane match) is removed.
    Pass 1 poor → Pass 2 directly. This is cleaner and avoids the
    coverage normalization artifact that caused wrong turns to win.

    Parameters
    ----------
    bike_df              : DataFrame — full trajectory for one vehicle
    movement_registry    : dict
    segment_registry     : dict
    geometry_store       : dict
    max_chain_length     : int
    poor_match_threshold : float

    Returns
    -------
    chain        : list of dicts with seg_key, role, df_indices, s_arr,
                   score, match_quality
    movement_key : str
    """
    skip_keys = {'x_offset', 'y_offset'}

    lane_segs = [k for k, e in segment_registry.items() if e['type'] == 'lane']
    all_segs  = [k for k in segment_registry if k not in skip_keys]

    chain = []
    build_registry_luts(geometry_store)

    entry_idx = _find_registered_entry(bike_df, segment_registry, geometry_store)
    if entry_idx > 0:
        log.info(f"[prefix trim] veh skipping {entry_idx} unregistered pts "
                 f"before first registered domain")
    remaining_indices  = list(range(entry_idx, len(bike_df)))
    candidates         = lane_segs
    iteration          = 0
    is_first_iteration = True

    while remaining_indices and iteration < max_chain_length:

        fragment    = bike_df.iloc[remaining_indices].reset_index(drop=False)
        fragment_xy = fragment[['x_ekf', 'y_ekf']].to_numpy()

        # ── Layer 0: filter candidates by validity polygon ────────────────
        candidates_filtered = _filter_candidates_by_validity_polygon(
            fragment_xy, candidates, segment_registry
        )

        if verbose:
            log.debug(
                f"--- Iter {iteration} | frag={len(fragment)} pts | "
                f"candidates={candidates} | layer0={candidates_filtered}"
            )

        # ── Pass 1: match against filtered candidates ─────────────────────
        seg_key, role, score, mask, window_end, is_reverse = match_segment(
            fragment, candidates_filtered,
            segment_registry, geometry_store,
            is_first_segment=is_first_iteration,
            log=log, verbose=verbose
        )

        # ── Pass 2: poor match OR departure-first ─────────────────────────
        need_pass2 = is_first_iteration and (
            seg_key is None
            or score > poor_match_threshold
            or role == 'departure'
        )
        if need_pass2:
            all_filtered = _filter_candidates_by_validity_polygon(
                fragment_xy, all_segs, segment_registry
            )
            if verbose:
                log.debug(f"--- Iter {iteration} PASS 2 | "
                          f"all_filtered={all_filtered}")

            seg_key2, role2, score2, mask2, window_end2, is_reverse2 = match_segment(
                fragment, all_filtered,
                segment_registry, geometry_store,
                is_first_segment=is_first_iteration,
                log=log, verbose=verbose
            )

            if seg_key is None or score2 < score:
                if verbose and seg_key is not None:
                    log.info(
                        f"[pass 2] override: "
                        f"{seg_key}({score:.3f}) → {seg_key2}({score2:.3f})"
                    )
                seg_key, role, score, mask, window_end, is_reverse = (
                    seg_key2, role2, score2, mask2, window_end2, is_reverse2
                )

        is_first_iteration = False

        # ── Accept / reject ───────────────────────────────────────────────
        if len(candidates_filtered) == 1 and seg_key is not None and score < np.inf:
            match_quality = 'good' if score <= poor_match_threshold else 'poor'
        elif seg_key is None or score > FORCED_MATCH_THRESHOLD:
            break
        elif score > poor_match_threshold:
            match_quality = 'poor'
        else:
            match_quality = 'good'

        # ── Handoff detection ─────────────────────────────────────────────
        handoff_local, s_arr = find_handoff_index(
            fragment, seg_key, role,
            segment_registry, geometry_store,
            is_reverse_traversal=is_reverse
        )

        if verbose:
            log.debug(f"--- Handoff | index={handoff_local} | "
                      f"seg={seg_key} role={role} score={score:.3f} "
                      f"reverse={'YES' if is_reverse else 'no'}")

        seg_indices    = [remaining_indices[i] for i in range(handoff_local)]
        beyond_handoff = [remaining_indices[i]
                          for i in range(handoff_local, len(remaining_indices))]

        if len(seg_indices) < MIN_OVERLAP_PTS:
            break

        chain.append({
            'seg_key':              seg_key,
            'role':                 role,
            'df_indices':           seg_indices,
            's_arr':                s_arr[:handoff_local],
            'score':                score,
            'match_quality':        match_quality,
            'is_reverse_traversal': is_reverse,
        })

        iteration         += 1
        remaining_indices  = beyond_handoff

        # ── Get next candidates ───────────────────────────────────────────
        # For reverse traversal on a lane segment: look up using the
        # opposite-direction segment key and flipped role so the movement
        # registry returns the correct turning candidates.
        #   reverse on departure → approaching intersection → lookup 'approach'
        #   reverse on approach  → leaving intersection    → lookup 'departure'
        OPP_DIR = {'EB': 'WB', 'WB': 'EB', 'NB': 'SB', 'SB': 'NB'}
        if is_reverse and segment_registry[seg_key]['type'] == 'lane':
            entry     = segment_registry[seg_key]
            geom_key  = entry['geometry_key']
            opp_key   = f"{geom_key}_{OPP_DIR.get(entry['direction'], entry['direction'])}"
            flip_role = 'approach' if role == 'departure' else 'departure'
            lookup_key  = opp_key if opp_key in segment_registry else seg_key
            lookup_role = flip_role
            if verbose:
                log.debug(f"[reverse] next candidates via "
                          f"{lookup_key}[{lookup_role}] "
                          f"(matched: {seg_key}[{role}] reverse=YES)")
        else:
            lookup_key  = seg_key
            lookup_role = role

        candidates = list(get_next_candidates(
            lookup_key, lookup_role, movement_registry, segment_registry
        ))
        if not candidates:
            break

    movement_key = derive_movement_key(chain, movement_registry)
    return chain, movement_key
def compute_directed_s(s_native, seg_key, segment_registry, geometry_store):
    """
    Convert native arc-length s to directed s (always increasing
    in travel direction, starting from 0 at segment entry).

    Forward segment: s_directed = s_native           (approach)
                     s_directed = s_native - s_change (departure)
    Reverse segment: s_directed = L - s_native        (approach)
                     s_directed = s_change - s_native  (departure)

    The role boundary s_change is read from the registry domains
    (approach_native / departure_native), not from geometry_store['s_stop'],
    to correctly reflect s_change = 0.5*(s_stop+s_yield) set in
    build_segment_registry.

    Parameters
    ----------
    s_native   : float or array — native spline arc-length
    seg_key    : str
    segment_registry, geometry_store : dicts

    Returns
    -------
    s_directed : float or array
    """
    entry      = segment_registry[seg_key]
    geom_key   = entry['geometry_key']
    is_forward = entry['is_forward']
    L          = geometry_store[geom_key]['total_length']
    s_stop_geo = geometry_store[geom_key].get('s_stop', None)

    if s_stop_geo is None:
        # Turn segment — s always starts at 0, no role split
        return np.asarray(s_native)

    # Derive s_change from the registry domains.
    # For forward:  approach = (0, s_change)   → s_change = approach_native[1]
    # For reverse:  approach = (s_change, L)   → s_change = approach_native[0]
    app = entry.get('approach_native')
    dep = entry.get('departure_native')

    if app is not None and dep is not None:
        s_change = app[1] if is_forward else app[0]
    else:
        # Fallback: domain was nulled by restrict_segment_roles.
        # Use geometry_store s_stop as a safe default.
        s_change = 0.5 * (s_stop_geo + geometry_store[geom_key].get('s_yield', s_stop_geo))

    s_native = np.asarray(s_native)

    if is_forward:
        # Approach: s_native in [0, s_change]   → directed = s_native
        # Departure: s_native in [s_change, L]  → directed = s_native - s_change
        if np.mean(s_native) <= s_change:
            return s_native
        else:
            return s_native - s_change
    else:
        # Approach: s_native in [s_change, L]  → directed = L - s_native
        # Departure: s_native in [0, s_change] → directed = s_change - s_native
        if np.mean(s_native) >= s_change:
            return L - s_native
        else:
            return s_change - s_native


def transform_segment(bike_df, seg_key, df_indices,
                       segment_registry, geometry_store):
    """
    Project trajectory points for one segment and compute all
    lane coordinate outputs.

    Uses warm-started projection along the trajectory for efficiency.

    Parameters
    ----------
    bike_df          : DataFrame — full trajectory (original indices)
    seg_key          : str
    df_indices       : list of int — row indices belonging to this segment
    segment_registry : dict
    geometry_store   : dict

    Returns
    -------
    result : dict of column_name → array of length len(df_indices)
    """
    entry           = segment_registry[seg_key]
    geom_key        = entry['geometry_key']
    tck, unew, cum_dist = geometry_store[geom_key]['spline']
    is_forward      = entry['is_forward']
    seg_type        = entry['type']
    bike_lane       = entry.get('bike_lane', None)

    xy      = bike_df.iloc[df_indices][['x_ekf','y_ekf']].to_numpy()
    speed   = bike_df.iloc[df_indices]['speed_ekf'].to_numpy()
    psi     = bike_df.iloc[df_indices]['angle_ekf'].to_numpy()
    accel   = bike_df.iloc[df_indices]['a_ekf'].to_numpy()
    n       = len(df_indices)

    # Vectorised projection: LUT coarse init + batched Newton + splev
    lut = geometry_store[geom_key].get('lut', None)
    if lut is not None:
        t_lut, xy_lut = lut
        diff       = xy_lut[np.newaxis, :, :] - xy[:, np.newaxis, :]  # (N,M,2)
        dists2     = (diff * diff).sum(axis=2)                         # (N,M)
        t_init_arr = t_lut[np.argmin(dists2, axis=1)]                  # (N,)
    else:
        t_coarse   = np.linspace(0, 1, 200)
        xs_c, ys_c = splev(t_coarse, tck)
        pts_c      = np.stack([xs_c, ys_c], axis=1)                    # (200,2)
        diff       = pts_c[np.newaxis] - xy[:, np.newaxis, :]          # (N,200,2)
        t_init_arr = t_coarse[np.argmin((diff*diff).sum(2), axis=1)]   # (N,)

    # Batched Newton refinement (4 iters, all N points in parallel)
    delta  = 0.1
    t_v    = t_init_arr.copy()
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
        step = np.clip(step, -delta, delta)
        t_v  = np.clip(t_v + step, 0.0, 1.0)
        if np.max(np.abs(step)) < 1e-7:
            break

    # Final splev: closest point, tangent, s, d
    xp_f,  yp_f  = splev(t_v, tck, der=0)
    xp1_f, yp1_f = splev(t_v, tck, der=1)
    s_native = np.interp(t_v, unew, cum_dist)

    tang_len  = np.sqrt(xp1_f**2 + yp1_f**2)
    tang_len  = np.where(tang_len > 1e-12, tang_len, 1.0)
    tx = xp1_f / tang_len;  ty = yp1_f / tang_len
    tang_arr  = np.stack([tx, ty], axis=1)    # (N, 2)
    norm_arr  = np.stack([-ty, tx], axis=1)   # (N, 2)

    ex_f = px - xp_f;  ey_f = py - yp_f
    d_nat = ex_f * norm_arr[:, 0] + ey_f * norm_arr[:, 1]

    if not is_forward:
        tang_arr = -tang_arr
        norm_arr = -norm_arr
        d_arr    = -d_nat
    else:
        d_arr    = d_nat

    s_native = np.asarray(s_native)
    d_arr    = np.asarray(d_arr)

    # Directed s (always increases in travel direction from 0)
    s_directed = compute_directed_s(
        s_native, seg_key, segment_registry, geometry_store
    )

    # Speed decomposition using velocity vector
    vx = speed * np.cos(psi)
    vy = speed * np.sin(psi)
    s_dot = tang_arr[:, 0] * vx + tang_arr[:, 1] * vy   # dot(v, tangent)
    d_dot = norm_arr[:, 0] * vx + norm_arr[:, 1] * vy   # dot(v, normal)

    # Acceleration decomposition
    ax    = accel * np.cos(psi)
    ay    = accel * np.sin(psi)
    s_ddot = tang_arr[:, 0] * ax + tang_arr[:, 1] * ay
    d_ddot = norm_arr[:, 0] * ax + norm_arr[:, 1] * ay

    # s_decreasing flag
    s_decreasing = s_dot < S_DECREASING_THRESHOLD

    # Bike lane membership
    in_bike_lane       = np.full(n, np.nan)
    d_to_bike_boundary = np.full(n, np.nan)

    if bike_lane is not None and 'd_boundary_spline' in bike_lane:
        d_bnd_spl        = bike_lane['d_boundary_spline']
        w_bike           = bike_lane['w_bike']
        side             = bike_lane['side']
        s_bl_min, s_bl_max = bike_lane['s_domain']

        for i in range(n):
            s_i = s_native[i]
            d_i = d_arr[i]
            if not (s_bl_min <= s_i <= s_bl_max):
                continue   # outside boundary coverage — leave NaN
                
            # d_arr is in travel-direction coords (negated for reverse segments)
            # d_boundary_spline is in native spline coords
            # → convert d_i to native before comparing
            d_i_native = d_i if is_forward else -d_i

            d_bnd  = float(d_bnd_spl(s_i))
            d_far  = d_bnd + side * w_bike
            d_lo   = min(d_bnd, d_far) - BIKE_LANE_TOLERANCE   # tolerance
            d_hi   = max(d_bnd, d_far) + BIKE_LANE_TOLERANCE

            in_bike_lane[i]       = bool(d_lo <= d_i_native <= d_hi)
            d_to_bike_boundary[i] = d_i_native - d_bnd

    # elif seg_type == 'lane' and entry.get('mode') == 'bike':
    #     # Entire lane is dedicated bike infrastructure
    #     in_bike_lane[:] = True
    
    return {
        's':                   s_directed,
        'd':                   d_arr,
        's_dot':               s_dot,
        'd_dot':               d_dot,
        's_ddot':              s_ddot,
        'd_ddot':              d_ddot,
        's_decreasing':        s_decreasing,
        'in_bike_lane':        in_bike_lane,
        'd_to_bike_boundary':  d_to_bike_boundary,
    }


# #############################################################################
# MAIN ENTRY POINT
# #############################################################################
def to_lane_coordinates(bike_df, movement_registry,
                         segment_registry, geometry_store,
                         max_chain_length=3,
                         verbose=False, log=None):
    """
    Full lane coordinate transform pipeline for one vehicle/bike.

    V3: seg_bboxes parameter removed — validity_polygon stored in
    segment_registry is used directly for Layer 0 spatial filtering.

    Runs matching phase then transformation phase. Adds all lane
    coordinate columns to bike_df. Unmatched rows get NaN.

    Parameters
    ----------
    bike_df           : DataFrame — trajectory for one vehicle
                        must have: x_ekf, y_ekf, speed_ekf, angle_ekf, a
    movement_registry : dict
    segment_registry  : dict
    geometry_store    : dict
    max_chain_length  : int

    Returns
    -------
    bike_df : DataFrame with added columns:
        movement_key, segment_id, segment_type, segment_role,
        s, d, s_dot, d_dot, s_ddot, d_ddot,
        in_bike_lane, d_to_bike_boundary, s_decreasing, match_quality
    """
    veh_id = bike_df['veh_id'].iloc[0]
    if log is None:
        log = _get_logger(debug=True)
        log.debug(f'=== to_lane_coordinates | veh={veh_id} ===')
    else:
        log.section(f'=== to_lane_coordinates | veh={veh_id} ===')

    # Initialise all output columns to NaN
    new_cols = [
        'movement_key', 'segment_id', 'segment_type', 'segment_role',
        's', 'd', 's_dot', 'd_dot', 's_ddot', 'd_ddot',
        'in_bike_lane', 'd_to_bike_boundary', 's_decreasing',
        'match_quality',
    ]
    for col in new_cols:
        bike_df[col] = np.nan
    bike_df['movement_key']  = None
    bike_df['segment_id']    = None
    bike_df['segment_type']  = None
    bike_df['segment_role']  = None
    bike_df['match_quality'] = None

    # ── Matching phase ────────────────────────────────────────────────────
    chain, movement_key = assign_segments(
        bike_df, movement_registry,
        segment_registry, geometry_store,
        max_chain_length=max_chain_length,
        verbose=verbose, log=log
    )

    if not chain:
        bike_df['movement_key']  = 'unmatched'
        bike_df['match_quality'] = 'unmatched'
        return bike_df

    # ── Transformation phase ──────────────────────────────────────────────
    for seg_entry in chain:
        seg_key    = seg_entry['seg_key']
        role       = seg_entry['role']
        df_indices = seg_entry['df_indices']
        entry      = segment_registry[seg_key]

        result = transform_segment(
            bike_df, seg_key, df_indices,
            segment_registry, geometry_store
        )

        idx = bike_df.index[df_indices]
        for col, vals in result.items():
            bike_df.loc[idx, col] = vals

        bike_df.loc[idx, 'movement_key']  = movement_key
        bike_df.loc[idx, 'segment_id']    = seg_key
        bike_df.loc[idx, 'segment_type']  = entry['type']
        bike_df.loc[idx, 'segment_role']  = role
        bike_df.loc[idx, 'match_quality'] = seg_entry['match_quality']

    return bike_df