"""
tools_lane_coords.py
--------------------
Phase B: Lane coordinate transform pipeline.

Implements:
  - Segment matching (sequential chaining)
  - Warm-started spline projection
  - Lane coordinate transformation
  - Bike lane membership
  - U-turn flagging

Authors: ETH Zürich IVT
"""

# #############################################################################
# IMPORTS
# #############################################################################
import numpy as np
import pandas as pd

from scipy.interpolate import splev
from scipy.optimize    import minimize_scalar

from tools_coordinate_transform import convert_xy2056_to_roadway_coordinates


# #############################################################################
# CONSTANTS
# #############################################################################
MIN_OVERLAP_PTS      = 5       # minimum inlier points to attempt scoring
SIGMA_DIST_M         = 3.0     # [m] lane segments
SIGMA_DIST_TURN_M    = 7.0     # [m] turn segments — cyclists deviate widely
SIGMA_HEAD_RAD       = np.pi / 4  # [rad] normalizer — 45°
W_DIST               = 1.0     # weight: lateral proximity
W_HEAD               = 1.0     # weight: heading alignment
W_PILE               = 1.0     # weight: pileup penalty
MIN_TRAVEL_PILEUP_M  = 5.0     # [m] minimum trajectory travel to use pileup term
CLAMP_TOL            = 0.001   # t values within this of 0 or 1 are clamped
POOR_MATCH_THRESHOLD = 3.0     # normalized score — tune empirically

# Threshold for s_decreasing flag [m/s]
S_DECREASING_THRESHOLD = -0.5

# Sparse sample size for matching (evenly spaced points from trajectory)
N_MATCH_SAMPLE = 50

BIKE_LANE_TOLERANCE = 0.2


# #############################################################################
# WARM-STARTED PROJECTION
# #############################################################################
def project_point_warm(point_local, tck, t_init=None, delta=0.1):
    """
    Project a point onto a spline with warm-started local search.

    Two-phase search:
      1. Coarse: evaluate at 50 uniform t values, find nearest
      2. Refine: minimize_scalar in [t_init - delta, t_init + delta]

    Parameters
    ----------
    point_local : (2,) array in local EPSG:2056 coords
    tck         : spline representation
    t_init      : float or None — warm start parameter
                  if None, coarse search covers full [0, 1]
    delta       : float — search half-width around t_init

    Returns
    -------
    t_star        : float — spline parameter at closest point
    closest_point : (2,) array
    """
    point = np.asarray(point_local)

    def dist_sq(t):
        xp, yp = splev(t, tck)
        return (xp - point[0])**2 + (yp - point[1])**2

    if t_init is None:
        # Full coarse search
        t_coarse   = np.linspace(0, 1, 50)
        dists      = [dist_sq(t) for t in t_coarse]
        t_init     = t_coarse[np.argmin(dists)]

    # Refine in neighbourhood
    bounds  = (max(0.0, t_init - delta),
               min(1.0, t_init + delta))
    res     = minimize_scalar(dist_sq, bounds=bounds, method='bounded')
    t_star  = res.x
    xp, yp  = splev(t_star, tck)
    return t_star, np.array([xp, yp])


def project_point_full(point_local, tck, unew, cum_dist, t_init=None):
    """
    Full projection: warm-started closest point + roadway coordinates.

    Parameters
    ----------
    point_local   : (2,) array in local EPSG:2056
    tck, unew, cum_dist : spline representation
    t_init        : float or None — warm start

    Returns
    -------
    t_star, tangent, normal, s, d
    """
    t_star, closest = project_point_warm(point_local, tck, t_init)

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
def get_next_candidates(seg_key, role, movement_registry):
    """
    Given a matched segment and its role, return the set of possible
    next segment keys from movement_registry.

    Parameters
    ----------
    seg_key          : str — e.g. 'LangstrS_NB'
    role             : str — 'approach', 'turn', or 'departure'
    movement_registry: dict

    Returns
    -------
    next_candidates : set of str — possible next segment keys
                      empty if no next segment exists
    """
    next_candidates = set()
    for mov_key, sequence in movement_registry.items():
        for i, (s_key, s_role) in enumerate(sequence):
            if s_key == seg_key and s_role == role:
                if i + 1 < len(sequence):
                    next_candidates.add(sequence[i + 1][0])
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
def score_segment(xy_local, psi, seg_key,
                  segment_registry, geometry_store, 
                  coverage=False, verbose=False):
    """
    Score how well a trajectory fragment matches a directed segment.
    Lower score = better match.

    Three-term weighted sum (all terms normalised to comparable scale):

      dist_term  : median |d| / SIGMA_DIST_M
                   Lateral offset from centerline. Penalises spatially
                   wrong segments and adjacent parallel streets.

      head_term  : RMSE(heading_error) / SIGMA_HEAD_RAD
                   Heading alignment with spline tangent. Penalises
                   wrong-direction segments and bidirectional confusion.

      pile_term  : 1 - s_spread / traj_cumulative_dist
                   Pileup penalty. Penalises collinear adjacent segments
                   (e.g. LangstrN_SB vs LangstrS_SB) where projected s
                   compresses to the domain boundary instead of spreading
                   proportionally to the trajectory travel distance.
                   Only active when traj_cumulative_dist > MIN_TRAVEL_PILEUP_M.

    Scoring uses only inlier points: unclamped projections whose s falls
    within the role-appropriate s-domain. Role is inferred from whichever
    domain (approach vs departure) captures more inlier points.

    Parameters
    ----------
    xy_local  : (N, 2) array — trajectory positions in local EPSG:2056
    psi       : (N,) array — trajectory headings [rad]
    seg_key   : str
    segment_registry, geometry_store : dicts

    Returns
    -------
    score          : float — combined match score (lower = better)
                     np.inf if fewer than MIN_OVERLAP_PTS inliers
    role           : str — 'approach', 'departure', or 'turn'
    in_domain_mask : (N,) bool array — True for inlier points
    """
    entry    = segment_registry[seg_key]
    geom_key = entry['geometry_key']
    tck, unew, cum_dist = geometry_store[geom_key]['spline']
    is_forward = entry['is_forward']
    seg_type   = entry['type']

    N = len(xy_local)
    s_arr    = np.zeros(N)
    d_arr    = np.zeros(N)
    t_arr    = np.zeros(N)
    psi_lane = np.zeros(N)

    # ── Project all points ────────────────────────────────────────────────
    for i, pt in enumerate(xy_local):
        t_star, tangent, _, s_i, d_i = project_point_full(
            pt, tck, unew, cum_dist, t_init=None
        )
        s_arr[i]    = s_i
        d_arr[i]    = d_i
        t_arr[i]    = t_star
        psi_raw     = float(np.arctan2(tangent[1], tangent[0]))
        psi_lane[i] = psi_raw if is_forward else psi_raw + np.pi
    
    
    # import matplotlib.pyplot as plt
    
    # plt.figure(f'{seg_key}_tsd')
    # plt.plot(t_arr, label='Spline t')
    # plt.plot(s_arr, label='s')
    # plt.plot(d_arr, label='d')
    # plt.legend()
    # plt.show()
    
    # dist_travlled = np.cumsum(np.linalg.norm(np.diff(xy_local, axis=0), axis=1))
    # s_cum = np.cumsum(np.abs(np.diff(s_arr)))
    # plt.figure(f'{seg_key}_cum')
    # plt.plot(s_cum, label='Cumulative s')
    # plt.plot(dist_travlled, label='Cumulative dist from xy')
    # plt.legend()
    # plt.show()

    # ── Clamping filter ───────────────────────────────────────────────────
    # Points whose t is at the spline boundary projected onto the endpoint,
    # not onto the true nearest point — discard before domain check.
    not_clamped = (t_arr > CLAMP_TOL) & (t_arr < 1.0 - CLAMP_TOL)

    # ── Role and inlier domain ────────────────────────────────────────────
    if seg_type == 'turn':
        in_domain_mask = not_clamped
        role           = 'turn'
    else:
        app_min, app_max = entry['approach_native']
        dep_min, dep_max = entry['departure_native']
        in_approach  = not_clamped & (s_arr >= app_min) & (s_arr <= app_max)
        in_departure = not_clamped & (s_arr >= dep_min) & (s_arr <= dep_max)
        if in_approach.sum() >= in_departure.sum():
            in_domain_mask = in_approach
            role           = 'approach'
        else:
            in_domain_mask = in_departure
            role           = 'departure'

    n_in = int(in_domain_mask.sum())
    if n_in < MIN_OVERLAP_PTS:
        return np.inf, role, in_domain_mask

    # ── Term 1: lateral proximity ─────────────────────────────────────────
    sigma_dist = SIGMA_DIST_TURN_M if seg_type == 'turn' else SIGMA_DIST_M
    dist_term = float(np.median(np.abs(d_arr[in_domain_mask]))) / sigma_dist

    # ── Term 2: heading alignment ─────────────────────────────────────────
    ang_diff  = np.abs(
        (psi[in_domain_mask] - psi_lane[in_domain_mask] + np.pi)
        % (2 * np.pi) - np.pi
    )
    head_term = float(np.sqrt(np.mean(ang_diff ** 2))) / SIGMA_HEAD_RAD

    # ── Term 3: pileup penalty ────────────────────────────────────────────
    # Trajectory cumulative distance (odometry), full fragment
    traj_arc = float(np.sum(np.linalg.norm(np.diff(xy_local, axis=0), axis=1)))

    if traj_arc > MIN_TRAVEL_PILEUP_M:
        s_inliers  = s_arr[in_domain_mask]
        s_spread   = float(s_inliers.max() - s_inliers.min())
        spread_ratio = float(np.clip(s_spread / traj_arc, 0.0, 1.0))
        pile_term  = 1.0 - spread_ratio
    else:
        pile_term  = 0.0   # too little travel to distinguish pileup from stop
    
    pile_term = 0.5 * pile_term + 0.5 * (n_in / N)

    # ── Combined score ────────────────────────────────────────────────────
    score = (W_DIST * dist_term +
             W_HEAD * head_term +
             W_PILE * pile_term)
    if coverage:
        score = score / (1e-06 + (n_in / len(xy_local)))
    
    if verbose:
        s_inliers  = s_arr[in_domain_mask]
        s_spread   = float(s_inliers.max() - s_inliers.min()) if n_in > 1 else 0.0
        print(f"  [{seg_key}] role={role}  n_in={n_in}")
        print(f"    dist_term  = {W_DIST * dist_term:.3f}  "
              f"(median|d|={np.median(np.abs(d_arr[in_domain_mask])):.2f}m)")
        print(f"    head_term  = {W_HEAD * head_term:.3f}  "
              f"(heading_rmse={np.degrees(float(np.sqrt(np.mean(ang_diff**2)))):.1f}°)")
        print(f"    pile_term  = {W_PILE * pile_term:.3f}  "
              f"(s_spread={s_spread:.2f}m  traj_arc={traj_arc:.2f}m  "
              f"ratio={s_spread/(traj_arc+1e-6):.3f})")
        print(f"    → score    = {score:.3f}")
        
    return score, role, in_domain_mask


def match_segment(bike_df, candidates,
                  segment_registry, geometry_store,
                  n_sample=N_MATCH_SAMPLE, coverage=False):
    """
    Match a trajectory fragment to the best segment from candidates.

    Parameters
    ----------
    bike_df          : DataFrame — trajectory fragment (subset of rows)
    candidates       : list or set of str — segment keys to score
    segment_registry : dict
    geometry_store   : dict
    n_sample         : int — sparse sample size for scoring

    Returns
    -------
    best_seg_key   : str or None
    best_role      : str or None
    best_score     : float
    best_mask      : (n_sample,) bool array or None
    sample_idx     : (n_sample,) int array — indices into bike_df
    """
    if len(bike_df) == 0 or len(candidates) == 0:
        return None, None, np.inf, None, None

    # Sparse sample
    sample_idx  = np.linspace(
        0, len(bike_df) - 1, min(n_sample, len(bike_df)), dtype=int
    )
    xy_local     = bike_df[['x_ekf', 'y_ekf']].to_numpy()
    psi_full    = bike_df['angle_ekf'].to_numpy()

    xy_sample   = xy_local[sample_idx]
    psi_sample  = psi_full[sample_idx]

    best_seg_key = None
    best_role    = None
    best_score   = np.inf
    best_mask    = None

    for seg_key in candidates:
        score, role, mask = score_segment(
            xy_sample, psi_sample, seg_key,
            segment_registry, geometry_store,
            coverage=coverage
        )
        if score < best_score:
            best_score   = score
            best_seg_key = seg_key
            best_role    = role
            best_mask    = mask

    return best_seg_key, best_role, best_score, best_mask, sample_idx


# #############################################################################
# HANDOFF DETECTION
# #############################################################################
def find_handoff_index(bike_df, seg_key, role,
                       segment_registry, geometry_store, 
                       max_lateral_dist=15.0):
    """
    Find the first trajectory row index where the vehicle crosses
    the segment boundary (s_stop in native coords).

    Projects all trajectory points onto the segment spline and finds
    the first index where s crosses s_stop in the travel direction.

    Parameters
    ----------
    bike_df          : DataFrame — full trajectory fragment for this segment
    seg_key          : str
    role             : str
    segment_registry : dict
    geometry_store   : dict
    x_offset, y_offset : float

    Returns
    -------
    handoff_idx : int — first row index beyond segment boundary
                  len(bike_df) if boundary never reached
    s_arr       : (N,) array of projected s values (native coords)
    """
    entry               = segment_registry[seg_key]
    geom_key            = entry['geometry_key']
    tck, unew, cum_dist = geometry_store[geom_key]['spline']
    is_forward          = entry['is_forward']
    L                   = geometry_store[geom_key]['total_length']
    dep_min, dep_max    = entry['departure_native']

    xy     = bike_df[['x_ekf','y_ekf']].to_numpy()
    n      = len(xy)
    s_arr  = np.full(n, np.nan)
    t_prev = None

    for i in range(n):
        pt_local         = xy[i]
        t_star, _, _, s_i, d_i = project_point_full(
            pt_local, tck, unew, cum_dist, t_init=t_prev
        )
        s_arr[i] = s_i
        t_prev   = t_star

        if role == 'approach':
            s_stop = geometry_store[geom_key]['s_stop']
            if is_forward and s_i >= s_stop - 0.001:
                return i, s_arr
            if not is_forward and s_i <= s_stop + 0.001:
                return i, s_arr

        elif role == 'departure':
            # Primary: s exits the departure domain in travel direction
            if is_forward and s_i >= dep_max - 0.001:
                return i, s_arr
            if not is_forward and s_i <= dep_min + 0.001:
                return i, s_arr
            # Fallback: lateral exit (e.g. bicycle mounts pavement)
            if abs(d_i) > max_lateral_dist:
                return i, s_arr

        elif role == 'turn':
            if s_i >= L * 0.95:
                return i, s_arr

    return n, s_arr

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
        if entry['type'] != 'lane':
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
 
    Two-stage filter per point per segment:
      1. Bounding-box check (vectorised, free) — skip if point outside
         bbox expanded by max_dist.
      2. Full spline projection (expensive) — only if bbox passed.
 
    Points before the returned index are an unregistered prefix
    (e.g. Mattengasse) and remain labelled unmatched.
 
    Parameters
    ----------
    bike_df          : DataFrame — must have x_ekf, y_ekf
    segment_registry : dict
    geometry_store   : dict
    max_dist         : float — maximum lateral distance to accept [m]
    bboxes           : dict or None — precomputed from _build_segment_bboxes.
                       If None, bboxes are computed on the fly (slower).
 
    Returns
    -------
    first_idx : int — first index inside a registered domain.
                0 if trajectory starts inside a registered domain already.
    """
    lane_entries = [
        (k, e) for k, e in segment_registry.items()
        if e['type'] == 'lane'
    ]
 
    if bboxes is None:
        bboxes = _build_segment_bboxes(segment_registry, geometry_store,
                                       padding=max_dist)
 
    xy  = bike_df[['x_ekf', 'y_ekf']].to_numpy()   # (N, 2)
    N   = len(xy)
 
    # Build expanded bbox arrays for vectorised filtering
    # shapes: (n_segs,)
    seg_keys = [k for k, _ in lane_entries]
    entries  = [e for _, e in lane_entries]
    x_mins   = np.array([bboxes[k][0] - max_dist for k in seg_keys])
    x_maxs   = np.array([bboxes[k][1] + max_dist for k in seg_keys])
    y_mins   = np.array([bboxes[k][2] - max_dist for k in seg_keys])
    y_maxs   = np.array([bboxes[k][3] + max_dist for k in seg_keys])
 
    for i in range(N):
        px, py = xy[i]
 
        # Vectorised bbox test: which segments could contain this point?
        in_bbox = (
            (px >= x_mins) & (px <= x_maxs) &
            (py >= y_mins) & (py <= y_maxs)
        )
        candidate_idxs = np.where(in_bbox)[0]
 
        # Full projection only for bbox-passing segments
        for ci in candidate_idxs:
            entry                = entries[ci]
            geom_key             = entry['geometry_key']
            tck, unew, cum_dist  = geometry_store[geom_key]['spline']
            app_min, app_max     = entry['approach_native']
            dep_min, dep_max     = entry['departure_native']
 
            _, _, _, s_i, d_i = project_point_full(
                xy[i], tck, unew, cum_dist, t_init=None
            )
 
            if abs(d_i) > max_dist:
                continue
 
            if (app_min <= s_i <= app_max) or (dep_min <= s_i <= dep_max):
                return i
 
    return 0   # no unregistered prefix — start from beginning



# #############################################################################
# MAIN MATCHING PHASE
# #############################################################################
def assign_segments(bike_df, movement_registry,
                    segment_registry, geometry_store,
                    max_chain_length=3,
                    poor_match_threshold=POOR_MATCH_THRESHOLD,
                    verbose=False):
    """
    Sequential segment chaining for one vehicle/bike trajectory.

    Step 1: Match against lane segments only (pass 1).
            If score is poor, retry with all segments (pass 2).
    Step 2: From matched segment, get next candidates from registry.
    Step 3: Match remaining trajectory against next candidates.
    Repeat until chain is complete or max_chain_length reached.

    Parameters
    ----------
    bike_df           : DataFrame — full trajectory for one vehicle
    movement_registry : dict
    segment_registry  : dict
    geometry_store    : dict
    max_chain_length  : int — maximum number of segments in chain
    poor_match_threshold : float — score above which match is poor

    Returns
    -------
    chain : list of dicts, each with:
        seg_key    : str
        role       : str
        df_indices : list of int — row indices in original bike_df
        s_arr      : (N,) array of projected s values
    movement_key : str
    """
    skip_keys = {'x_offset', 'y_offset'}

    # All lane segment keys (pass 1 candidates)
    lane_segs = [
        k for k, e in segment_registry.items()
        if e['type'] == 'lane'
    ]
    # All segment keys including turns (pass 2 candidates)
    all_segs  = [
        k for k in segment_registry
        if k not in skip_keys
    ]

    chain              = []
    
    # Precompute segment bboxes once — reused by _find_registered_entry
    seg_bboxes = _build_segment_bboxes(segment_registry, geometry_store)
    # Trim unregistered prefix (e.g. Mattengasse) before matching
    entry_idx = _find_registered_entry(
        bike_df, segment_registry, geometry_store, bboxes=seg_bboxes
    )
    if entry_idx > 0 and verbose:
        print(f"  [prefix trim] skipping {entry_idx} unregistered pts "
              f"before first registered domain")
    remaining_indices  = list(range(entry_idx, len(bike_df)))
    
    candidates         = lane_segs   # start with lane segments only
    iteration          = 0
    is_first_iteration = True

    while remaining_indices and iteration < max_chain_length:

        fragment = bike_df.iloc[remaining_indices].reset_index(drop=False)
        
        # ── DIAGNOSTIC ───────────────────────────────────────────────────────
        if verbose:
            print(f"\n--- Iteration {iteration} ---")
            print(f"  fragment size: {len(fragment)} pts")
            xy_sample = fragment[['x_ekf','y_ekf']].iloc[
                np.linspace(0, len(fragment)-1, min(5, len(fragment)), dtype=int)
            ].to_numpy()
            print(f"  sample pts (local):     {xy_sample}")
            print(f"  candidates: {candidates}")
    
            # Score each candidate individually
            sample_idx = np.linspace(
                0, len(fragment)-1, min(N_MATCH_SAMPLE, len(fragment)), dtype=int
            )
            xy_full    = fragment[['x_ekf','y_ekf']].to_numpy()
            psi_full   = fragment['angle_ekf'].to_numpy()
            xy_local   = xy_full[sample_idx]
            psi_sample = psi_full[sample_idx]
    
            for seg_key in candidates:
                score, role, mask = score_segment(
                    xy_local, psi_sample, seg_key,
                    segment_registry, geometry_store,
                    coverage=False
                )
                n_in = int(mask.sum()) if mask is not None else 0
                print(f"  {seg_key:30s}  score={score:8.3f}  "
                      f"role={role}  n_in_domain={n_in}")
                    
        # ── END DIAGNOSTIC ───────────────────────────────────────────────────
        

        # ── Match segment ─────────────────────────────────────────────────
        seg_key, role, score, mask, sample_idx = match_segment(
            fragment, candidates,
            segment_registry, geometry_store,
            coverage=False
        )

        # Pass 2a: if first iteration and poor match, retry with all segs
        if is_first_iteration and (
            seg_key is None or score > poor_match_threshold
        ):
            # ── DIAGNOSTIC ───────────────────────────────────────────────────────
            if verbose:
                print(f"\n--- Iteration {iteration} PASS 2A ---")
                print(f"  candidates: {all_segs}")
    
                for seg_key in all_segs:
                    score, role, mask = score_segment(
                        xy_local, psi_sample, seg_key,
                        segment_registry, geometry_store,
                        coverage=False
                    )
                    n_in = int(mask.sum()) if mask is not None else 0
                    print(f"  {seg_key:30s}  score={score:8.3f}  "
                          f"role={role}  n_in_domain={n_in}")
                        
            # ── END DIAGNOSTIC ───────────────────────────────────────────────────
            
            seg_key, role, score, mask, sample_idx = match_segment(
                fragment, all_segs,
                segment_registry, geometry_store, 
                coverage=False
            )
        
        # Pass 2b: first iteration matched departure → bicycle entered frame
        # already inside intersection. Check all segs (incl. turns) and
        # take the better score unconditionally.
        if is_first_iteration and role == 'departure':
            # ── DIAGNOSTIC ───────────────────────────────────────────────────────
            if verbose:
                print(f"\n--- Iteration {iteration} PASS 2B ---")
                print(f"  candidates: {all_segs}")
    
                for seg_key3 in all_segs:
                    score3, role3, mask3 = score_segment(
                        xy_local, psi_sample, seg_key3,
                        segment_registry, geometry_store,
                        coverage=True
                    )
                    n_in3 = int(mask3.sum()) if mask3 is not None else 0
                    print(f"  {seg_key3:30s}  score={score3:8.3f}  "
                          f"role={role3}  n_in_domain={n_in3}")
                        
            # ── END DIAGNOSTIC ───────────────────────────────────────────────────
            seg_key2, role2, score2, mask2, sample_idx2 = match_segment(
                fragment, all_segs,
                segment_registry, geometry_store,
                coverage=True
            )
            if score2 < score or int(mask2.sum()) > int(mask.sum()):
                print(f"  [pass 2b] departure-first overridden: "
                      f"{seg_key}({score:.3f}) → {seg_key2}({score2:.3f})")
                seg_key, role, score, mask, sample_idx = (
                    seg_key2, role2, score2, mask2, sample_idx2
                )

        is_first_iteration = False

        # # No valid match found
        # if seg_key is None or score > poor_match_threshold:
        #     break
        
        # Single candidate: accept regardless of score if enough points overlap
        if len(candidates) == 1 and seg_key is not None and score < np.inf:
            pass  # accept — registry already determined this is the only option
        elif seg_key is None or score > poor_match_threshold:
            break

        # ── Role-dependent handoff ────────────────────────────────────────────────
        handoff_local, s_arr = find_handoff_index(
            fragment, seg_key, role,
            segment_registry, geometry_store
        )
        
        # ── DIAGNOSTIC ───────────────────────────────────────────────────────
        if verbose:
            print("\n--- Handoff ---")
            print(f"  index: {handoff_local}")
        
        # dep_min, dep_max = segment_registry[seg_key]['departure_native']
        # import sys
        # import matplotlib.pyplot as plt
        
        # plt.figure(f'{seg_key}_handoff')
        # plt.plot(s_arr, label = 's_arr')
        # plt.plot(dep_min - s_arr, label = 'dep_min - s_arr')
        # plt.plot(s_arr - dep_max, label = 's_arr - dep_max')
        # plt.legend()
        # plt.show()
        # sys.exit(1)
                    
        # ── END DIAGNOSTIC ───────────────────────────────────────────────────


        # Map local indices back to original bike_df indices
        seg_indices  = [remaining_indices[i] for i in range(handoff_local)]
        beyond_handoff = [remaining_indices[i] for i in range(handoff_local, len(remaining_indices))]

        if len(seg_indices) < MIN_OVERLAP_PTS:
            break

        chain.append({
            'seg_key':    seg_key,
            'role':       role,
            'df_indices': seg_indices,
            's_arr':      s_arr[:handoff_local],
            'score':      score,
        })

        iteration        += 1
        remaining_indices = beyond_handoff

        # ── Get next candidates from registry ─────────────────────────────
        candidates = list(get_next_candidates(
            seg_key, role, movement_registry
        ))
        if not candidates:
            break

    # Derive movement key
    movement_key = derive_movement_key(chain, movement_registry)

    return chain, movement_key


# #############################################################################
# TRANSFORMATION PHASE
# #############################################################################
def compute_directed_s(s_native, seg_key, segment_registry, geometry_store):
    """
    Convert native arc-length s to directed s (always increasing
    in travel direction, starting from 0 at segment entry).

    Forward segment: s_directed = s_native - s_entry
    Reverse segment: s_directed = s_exit  - s_native

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
    role       = None  # determined below from s_native position

    # Determine role from s_native position relative to s_stop
    s_stop   = geometry_store[geom_key].get('s_stop', None)
    L        = geometry_store[geom_key]['total_length']

    if s_stop is None:
        # Turn segment — s always starts at 0
        return np.asarray(s_native)

    s_native = np.asarray(s_native)

    if is_forward:
        # Approach: s_native runs 0 → s_stop  → directed = s_native
        # Departure: s_native runs s_stop → L → directed = s_native - s_stop
        if np.mean(s_native) <= s_stop:
            return s_native                  # approach
        else:
            return s_native - s_stop         # departure
    else:
        # Approach: s_native runs L → s_stop  → directed = L - s_native
        # Departure: s_native runs s_stop → 0 → directed = s_stop - s_native
        if np.mean(s_native) >= s_stop:
            return L - s_native              # approach
        else:
            return s_stop - s_native         # departure


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
    accel   = bike_df.iloc[df_indices]['a'].to_numpy()
    n       = len(df_indices)

    # Storage
    s_native  = np.full(n, np.nan)
    d_arr     = np.full(n, np.nan)
    tang_arr  = np.zeros((n, 2))
    norm_arr  = np.zeros((n, 2))
    t_prev    = None

    # Project each point with warm start
    for i in range(n):
        pt_local           = xy[i]
        t_star, tang, norm, s_i, d_i = project_point_full(
            pt_local, tck, unew, cum_dist, t_init=t_prev
        )
        # Flip tangent/normal for reverse segments
        if not is_forward:
            tang = -tang
            norm = -norm
        s_native[i]  = s_i
        d_arr[i]     = d_i if is_forward else -d_i
        tang_arr[i]  = tang
        norm_arr[i]  = norm
        t_prev       = t_star

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
                         max_chain_length=3):
    """
    Full lane coordinate transform pipeline for one vehicle/bike.

    Runs matching phase then transformation phase. Adds all lane
    coordinate columns to bike_df. Unmatched rows get NaN.

    Parameters
    ----------
    bike_df           : DataFrame — trajectory for one vehicle
                        must have: x_ekf, y_ekf,
                                   speed_ekf, angle_ekf, a
    movement_registry : dict
    segment_registry  : dict
    geometry_store    : dict
    max_chain_length  : int

    Returns
    -------
    bike_df : DataFrame with added columns:
        movement_key, segment_id, segment_type, segment_role,
        s, d, s_dot, d_dot, s_ddot, d_ddot,
        in_bike_lane, d_to_bike_boundary, s_decreasing
    """
    # Initialise all output columns to NaN
    new_cols = [
        'movement_key', 'segment_id', 'segment_type', 'segment_role',
        's', 'd', 's_dot', 'd_dot', 's_ddot', 'd_ddot',
        'in_bike_lane', 'd_to_bike_boundary', 's_decreasing',
    ]
    for col in new_cols:
        bike_df[col] = np.nan
    bike_df['movement_key']  = None
    bike_df['segment_id']    = None
    bike_df['segment_type']  = None
    bike_df['segment_role']  = None

    # ── Matching phase ────────────────────────────────────────────────────────
    chain, movement_key = assign_segments(
        bike_df, movement_registry,
        segment_registry, geometry_store,
        max_chain_length=max_chain_length
    )

    if not chain:
        bike_df['movement_key'] = 'unmatched'
        return bike_df

    # ── Transformation phase ──────────────────────────────────────────────────
    for seg_entry in chain:
        seg_key    = seg_entry['seg_key']
        role       = seg_entry['role']
        df_indices = seg_entry['df_indices']
        entry      = segment_registry[seg_key]

        # Transform this segment
        result = transform_segment(
            bike_df, seg_key, df_indices,
            segment_registry, geometry_store
        )

        # Fill dataframe
        idx = bike_df.index[df_indices]   # convert positional → label index

        for col, vals in result.items():
            bike_df.loc[idx, col] = vals
        
        bike_df.loc[idx, 'movement_key']  = movement_key
        bike_df.loc[idx, 'segment_id']    = seg_key
        bike_df.loc[idx, 'segment_type']  = entry['type']
        bike_df.loc[idx, 'segment_role']  = role

    return bike_df