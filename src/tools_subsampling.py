"""
Trajectory Subsampling: 25 FPS → 10 FPS
-------------------------------------------
Authors:        Shaimaa El-Baklish
Organization:   ETH Zürich, Switzerland, IVT - Institute for Transportation Planning and Systems
Development:    2025
-------------------------------------------

Overview
--------
Subsamples EKF-filtered bicycle/vehicle trajectories from 25 FPS to a target
FPS (default 10 FPS) using timestamp-aware interpolation.

Column treatment
----------------
  x_ekf, y_ekf       : cubic spline  (CubicSpline, not-a-knot)
  speed_ekf           : linear interp + clamp >= 0
  angle_ekf           : unwrap → linear interp → rewrap
  a_ekf               : linear interp
  angular_vel_ekf     : linear interp
  datetime            : recomputed from interpolated time
  veh_id, veh_type    : passthrough (scalar per vehicle)
  off_grid            : True if this row is NOT an exact member of the
                         shared master grid (i.e. a spliced-in head/tail
                         from include_heads/include_tails, or the forced
                         fallback point for an empty-window trajectory).
                         False for every point taken directly from the
                         shared grid. Always all-False when
                         include_heads/include_tails are both left at
                         their default (False).

Design decisions
----------------
- Cubic spline for x/y to preserve path curvature through turns.
  Falls back to linear interpolation for short trajectories (< 4 frames)
  where CubicSpline cannot be fit reliably.
- Linear interp for all scalar signals — EKF+RTS smoothness makes higher-order
  unnecessary and risks overshoot near sharp maneuvers.
- Angle unwrapping before interp avoids wrap-around artefacts.
- Timestamps are used directly as the interp axis → handles drone time
  arbitration (irregular dt) correctly.
- SHARED MASTER GRID: the target time grid (grid_anchor + k*dt) is built
  ONCE per subsample_all() call, from the full multi-vehicle dataframe's
  [time.min(), time.max()] span. Each vehicle's output timestamps are a
  SLICE of this single master grid (via np.searchsorted), not a freshly
  built per-vehicle grid. This guarantees df['time'] is homogeneous
  across every vehicle in the output (nunique(time) is bounded by the
  master grid length, exactly), and — when subsample_all() is called
  separately on two dataframes (e.g. bikes and vehicles) sharing the same
  grid_anchor and target_fps — across dataframes too.
- No extrapolation: a vehicle's target timestamps are clipped to
  [t_actual[0], t_actual[-1]].
- include_heads / include_tails (both default False): opt-in flags to
  additionally splice in a vehicle's true first/last timestamp even if
  it falls off the shared grid. Useful when exact start/end coverage
  matters more than strict grid homogeneity; leave off to guarantee a
  perfectly homogeneous time grid across the whole output.
- Short/edge-case trajectories whose [t0, t1] window contains no master
  grid point (and include_heads/include_tails don't cover it) are never
  silently dropped: a single point at t0 is forced in, with a logged
  warning.
- datetime is recomputed from time using the same ref_datetime / ref_time
  convention used in the raw data loading step.
"""

# =============================================================================
# IMPORTS
# =============================================================================
import logging
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline


# =============================================================================
# LOGGER
# =============================================================================
def _get_logger(debug: bool) -> logging.Logger:
    logger = logging.getLogger(__name__)
    level  = logging.DEBUG if debug else logging.WARNING
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            '[%(levelname)s] %(name)s — %(message)s'
        ))
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


# =============================================================================
# INTERNAL HELPERS
# =============================================================================

# Tolerance for "is this timestamp already on the grid" comparisons.
_TOL = 1e-4   # seconds (0.1 ms)


def _build_master_grid(t_min: float,
                       t_max: float,
                       target_fps: float,
                       grid_anchor: float = 0.0) -> np.ndarray:
    """
    Build the single, shared uniform time grid at target_fps for an
    entire dataframe (all vehicles), covering [t_min, t_max].

    Grid points fall at grid_anchor + k*dt for integer k. This is built
    ONCE per subsample_all() call — every vehicle's output timestamps
    are a slice of this same array (see _slice_target_grid), which is
    what guarantees a homogeneous 'time' column across vehicles (and,
    when the same grid_anchor/target_fps are used, across dataframes).

    Parameters
    ----------
    t_min, t_max : full dataframe's time span [seconds]
    target_fps   : desired output frame rate [Hz]
    grid_anchor  : reference time [seconds] defining the grid's phase.
                   Default 0.0. Use the same value across dataframes you
                   want phase-aligned (see subsample_all docstring for
                   the datetime_anchor preprocessing pattern).

    Returns
    -------
    t_grid_master : 1-D sorted array of shared target timestamps [seconds]
    """
    dt = 1.0 / target_fps
    k0 = int(np.ceil((t_min - grid_anchor) / dt - 1e-9))
    k1 = int(np.floor((t_max - grid_anchor) / dt + 1e-9))

    if k1 < k0:
        return np.array([], dtype=float)

    return grid_anchor + np.arange(k0, k1 + 1) * dt


def _slice_target_grid(t_grid_master: np.ndarray,
                       t0: float,
                       t1: float,
                       include_heads: bool = False,
                       include_tails: bool = False,
                       veh_id=None,
                       log=None) -> np.ndarray:
    """
    Slice the shared master grid down to the portion covering a single
    vehicle's [t0, t1] span, optionally splicing in the true head/tail.

    Parameters
    ----------
    t_grid_master : shared master grid (see _build_master_grid), sorted
    t0, t1        : this vehicle's actual first/last timestamps [seconds]
    include_heads : if True, splice in t0 itself when the first sliced
                    grid point doesn't already coincide with it (within
                    _TOL). Off by default — keeps 'time' strictly on the
                    shared grid.
    include_tails : same, for t1 / the last point.
    veh_id        : optional, for logging only
    log           : optional logger

    Returns
    -------
    t_target : 1-D array of this vehicle's target timestamps [seconds].
               Guaranteed non-empty: if the [t0, t1] window contains no
               master grid point and include_heads/include_tails don't
               cover it, a single point at t0 is forced in (with a
               logged warning) so the vehicle is never silently dropped.
    off_grid : 1-D boolean array, same length as t_target. True for
               points that are NOT exact members of t_grid_master —
               i.e. spliced-in heads/tails, or the forced fallback point
               for an empty window. False for every point taken directly
               from the shared master grid.
    """
    lo = np.searchsorted(t_grid_master, t0, side='left')
    hi = np.searchsorted(t_grid_master, t1, side='right')
    t_target = t_grid_master[lo:hi]
    off_grid = np.zeros(len(t_target), dtype=bool)

    if include_heads and (len(t_target) == 0 or t_target[0] - t0 > _TOL):
        t_target = np.insert(t_target, 0, t0)
        off_grid = np.insert(off_grid, 0, True)

    if include_tails and (len(t_target) == 0 or t1 - t_target[-1] > _TOL):
        t_target = np.append(t_target, t1)
        off_grid = np.append(off_grid, True)

    if len(t_target) == 0:
        if log is not None:
            log.warning(
                f'  veh={veh_id}: trajectory [{t0:.3f}, {t1:.3f}] '
                f'(dur={t1 - t0:.3f}s) contains no master grid point and '
                f'include_heads/include_tails did not cover it — forcing '
                f'a single output point at t0={t0:.3f} to avoid dropping '
                f'this vehicle entirely.'
            )
        t_target = np.array([t0])
        off_grid = np.array([True])

    return t_target, off_grid


def _interp_angle(t_target: np.ndarray,
                  t_actual: np.ndarray,
                  angle: np.ndarray) -> np.ndarray:
    """
    Linear interpolation of a wrapped angle signal [rad].

    Steps:
      1. np.unwrap — removes 2π jumps so the signal is continuous.
      2. np.interp  — standard linear interpolation on the unwrapped signal.
      3. Rewrap     — maps result back to (−π, π].

    Parameters
    ----------
    t_target : target timestamps
    t_actual : source timestamps (25 FPS, possibly irregular)
    angle    : wrapped angle array [rad], same length as t_actual

    Returns
    -------
    angle_interp : interpolated angle, wrapped to (−π, π] [rad]
    """
    angle_uw       = np.unwrap(angle)
    angle_uw_interp = np.interp(t_target, t_actual, angle_uw)
    angle_interp   = (angle_uw_interp + np.pi) % (2.0 * np.pi) - np.pi
    return angle_interp


# Minimum number of input frames required to fit a cubic spline.
# CubicSpline (not-a-knot) needs at least 4 knots; below this threshold
# we fall back to linear interpolation to avoid a ValueError.
CUBIC_SPLINE_MIN_POINTS = 4


def _interp_xy_cubic(t_target: np.ndarray,
                     t_actual: np.ndarray,
                     x: np.ndarray,
                     y: np.ndarray,
                     log=None) -> tuple:
    """
    Cubic spline interpolation for x and y position, with linear fallback
    for short trajectories.

    Uses scipy.interpolate.CubicSpline with default not-a-knot boundary
    conditions. Works correctly with non-uniform input spacing (time
    arbitration).

    Falls back to np.interp (linear) when the number of input frames is
    below CUBIC_SPLINE_MIN_POINTS (< 4), since CubicSpline requires at
    least 4 knots for not-a-knot BCs.

    Parameters
    ----------
    t_target : target timestamps
    t_actual : source timestamps (25 FPS, possibly irregular)
    x, y     : position arrays [m], same length as t_actual
    log      : optional logger for fallback warning

    Returns
    -------
    x_interp, y_interp : interpolated positions [m]
    """
    if len(t_actual) < CUBIC_SPLINE_MIN_POINTS:
        if log is not None:
            log.warning(
                f'  _interp_xy_cubic: only {len(t_actual)} input frames '
                f'(< {CUBIC_SPLINE_MIN_POINTS}) — falling back to linear '
                f'interpolation for x_ekf / y_ekf.'
            )
        return np.interp(t_target, t_actual, x), np.interp(t_target, t_actual, y)

    cs_x = CubicSpline(t_actual, x)   # not-a-knot by default
    cs_y = CubicSpline(t_actual, y)
    return cs_x(t_target), cs_y(t_target)


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def subsample_trajectory(veh_df: pd.DataFrame,
                         ref_datetime: pd.Timestamp,
                         ref_time: float,
                         target_fps: float = 10.0,
                         debug: bool = False,
                         log=None,
                         include_heads: bool = False,
                         include_tails: bool = False,
                         t_grid_master: np.ndarray = None,
                         grid_anchor: float = 0.0) -> pd.DataFrame:
    """
    Subsample a single-vehicle EKF-filtered trajectory from 25 FPS to
    target_fps (default 10 FPS) using timestamp-aware interpolation.

    Parameters
    ----------
    veh_df        : single-vehicle DataFrame with columns:
                    veh_id, veh_type, time, datetime,
                    x_ekf, y_ekf, speed_ekf, angle_ekf,
                    a_ekf, angular_vel_ekf
    ref_datetime  : reference datetime when time == ref_time
                    (same convention as the raw data loader)
    ref_time      : reference time value [seconds] corresponding to ref_datetime
                    (used for datetime reconstruction only)
    target_fps    : desired output frame rate [Hz], default 10.0. Ignored
                    if t_grid_master is provided (the master grid's own
                    spacing is used).
    debug         : if True, enable DEBUG logging
    log           : optional external logger
    include_heads : if True, splice in this trajectory's true first
                    timestamp even if off the shared grid. Default False.
    include_tails : if True, splice in this trajectory's true last
                    timestamp even if off the shared grid. Default False.
    t_grid_master : precomputed shared master grid (see
                    _build_master_grid / subsample_all). If None, a grid
                    is built just for this trajectory's own span
                    (single-vehicle / standalone use).
    grid_anchor   : reference time [seconds] defining the grid's phase,
                    only used when t_grid_master is None. Default 0.0.

    Returns
    -------
    sub_df : subsampled DataFrame with the same columns at target_fps
    """
    if log is None:
        log = _get_logger(debug)

    veh_id   = veh_df['veh_id'].iloc[0]
    veh_type = veh_df['veh_type'].iloc[0]
    log.debug(f'=== subsample_trajectory | veh={veh_id} | target={target_fps} FPS ===')

    t_actual = veh_df['time'].to_numpy()
    t0, t1   = t_actual[0], t_actual[-1]

    # ------------------------------------------------------------------
    # Step 1: Slice this vehicle's target timestamps from the shared
    # master grid (building a standalone one if none was provided).
    # ------------------------------------------------------------------
    if t_grid_master is None:
        t_grid_master = _build_master_grid(t0, t1, target_fps, grid_anchor=grid_anchor)

    t_target, off_grid = _slice_target_grid(
        t_grid_master, t0, t1,
        include_heads=include_heads,
        include_tails=include_tails,
        veh_id=veh_id, log=log
    )
    log.debug(f'  input frames : {len(t_actual)} | output frames: {len(t_target)}')

    # ------------------------------------------------------------------
    # Step 2: Interpolate x_ekf, y_ekf — cubic spline
    # ------------------------------------------------------------------
    x_interp, y_interp = _interp_xy_cubic(
        t_target, t_actual,
        veh_df['x_ekf'].to_numpy(),
        veh_df['y_ekf'].to_numpy(),
        log=log
    )

    # ------------------------------------------------------------------
    # Step 3: Interpolate scalar signals — linear
    # ------------------------------------------------------------------
    speed_interp = np.interp(t_target, t_actual,
                             veh_df['speed_ekf'].to_numpy())
    speed_interp = np.maximum(0.0, speed_interp)   # clamp: speed >= 0

    a_interp     = np.interp(t_target, t_actual,
                             veh_df['a_ekf'].to_numpy())

    angvel_interp = np.interp(t_target, t_actual,
                              veh_df['angular_vel_ekf'].to_numpy())

    # ------------------------------------------------------------------
    # Step 4: Interpolate angle — unwrap → linear → rewrap
    # ------------------------------------------------------------------
    angle_interp = _interp_angle(
        t_target, t_actual,
        veh_df['angle_ekf'].to_numpy()
    )

    # ------------------------------------------------------------------
    # Step 5: Recompute datetime from interpolated time
    # ------------------------------------------------------------------
    datetime_interp = ref_datetime + pd.to_timedelta(
        t_target - ref_time, unit='s'
    )

    # ------------------------------------------------------------------
    # Step 6: Assemble output DataFrame
    # ------------------------------------------------------------------
    sub_df = pd.DataFrame({
        'veh_id'         : veh_id,
        'veh_type'       : veh_type,
        'time'           : t_target,
        'datetime'       : datetime_interp,
        'x_ekf'          : x_interp,
        'y_ekf'          : y_interp,
        'speed_ekf'      : speed_interp,
        'angle_ekf'      : angle_interp,
        'a_ekf'          : a_interp,
        'angular_vel_ekf': angvel_interp,
        'off_grid'       : off_grid,
    })

    # ------------------------------------------------------------------
    # Step 7: Sanity checks
    # ------------------------------------------------------------------
    assert not sub_df['x_ekf'].isna().any(),         f'veh={veh_id}: NaN in x_ekf'
    assert not sub_df['y_ekf'].isna().any(),         f'veh={veh_id}: NaN in y_ekf'
    assert not sub_df['speed_ekf'].isna().any(),     f'veh={veh_id}: NaN in speed_ekf'
    assert not sub_df['angle_ekf'].isna().any(),     f'veh={veh_id}: NaN in angle_ekf'
    assert (sub_df['speed_ekf'] >= 0).all(),         f'veh={veh_id}: negative speed_ekf'
    assert sub_df['angle_ekf'].between(-np.pi, np.pi).all(), \
                                                     f'veh={veh_id}: angle_ekf out of (−π, π]'

    log.debug(f'  subsampling complete | output shape: {sub_df.shape}')
    return sub_df


# =============================================================================
# FLEET-LEVEL ENTRY POINT
# =============================================================================

def subsample_all(df: pd.DataFrame,
                  target_fps: float = 10.0,
                  debug: bool = False,
                  log=None,
                  include_heads: bool = False,
                  include_tails: bool = False,
                  grid_anchor: float = 0.0) -> pd.DataFrame:
    """
    Subsample all vehicles in a multi-vehicle DataFrame.

    Builds ONE shared master time grid from the full dataframe's
    [time.min(), time.max()] span, then calls subsample_trajectory() per
    vehicle — each vehicle's output timestamps are a slice of that same
    master grid, so 'time' is homogeneous across the whole output
    (nunique(time) is bounded by the master grid length, exactly).

    Parameters
    ----------
    df             : full multi-vehicle DataFrame (all vehicles, 25 FPS)
    target_fps     : desired output frame rate [Hz], default 10.0
    debug          : if True, enable DEBUG logging
    log            : optional external logger
    include_heads  : if True, splice in each vehicle's true first
                     timestamp even if off the shared grid. Default
                     False — leave off for a perfectly homogeneous grid.
    include_tails  : if True, splice in each vehicle's true last
                     timestamp even if off the shared grid. Default
                     False, same rationale.
    grid_anchor    : reference time [seconds, in this dataframe's own
                     'time' coordinate] defining the PHASE of the target
                     grid (grid points fall at grid_anchor + k*dt).
                     Default 0.0.

                     To synchronize bike and vehicle dataframes, first
                     rebuild BOTH dataframes' 'time' columns from one
                     shared wall-clock reference, e.g.:

                         datetime_anchor = max(bike_df['datetime'].min(),
                                               veh_df['datetime'].min())
                         bike_df['time'] = bike_df['datetime'].apply(
                             lambda x: np.round((x - datetime_anchor).total_seconds(), 3))
                         veh_df['time'] = veh_df['datetime'].apply(
                             lambda x: np.round((x - datetime_anchor).total_seconds(), 3))

                     Once both 'time' columns are on the same absolute
                     clock, time=0 IS the shared reference point for
                     both dataframes, so the default grid_anchor=0.0
                     already phase-aligns them automatically — no need
                     to pass anything extra to either subsample_all()
                     call. (With datetime_anchor = max(...), the
                     earlier-starting dataframe will have negative
                     'time' values for its first frames — expected and
                     handled correctly.)

    Returns
    -------
    sub_df : concatenated subsampled DataFrame (all vehicles, target_fps)
    """
    if log is None:
        log = _get_logger(debug)

    # ref_datetime / ref_time are only used to reconstruct 'datetime'
    # from the interpolated 'time' grid — any single (datetime, time)
    # pair from the data works for this, since it's a fixed affine
    # mapping between the two columns. Using the first row keeps this
    # robust even when 'time' has been rebuilt against a shared
    # datetime_anchor and can legitimately be negative for some rows.
    ref_datetime = df['datetime'].iloc[0]
    ref_time     = df['time'].iloc[0]

    # ------------------------------------------------------------------
    # Build the ONE shared master grid for the whole dataframe.
    # ------------------------------------------------------------------
    t_min = df['time'].min()
    t_max = df['time'].max()
    t_grid_master = _build_master_grid(t_min, t_max, target_fps, grid_anchor=grid_anchor)

    log.debug(f'ref_datetime={ref_datetime}  ref_time={ref_time}  '
              f'grid_anchor={grid_anchor}  include_heads={include_heads}  '
              f'include_tails={include_tails}  '
              f'master grid: [{t_min:.3f}, {t_max:.3f}] -> {len(t_grid_master)} points')

    results    = []
    unique_ids = df['veh_id'].unique()

    for veh_id in unique_ids:
        veh_df = df[df['veh_id'] == veh_id].copy().reset_index(drop=True)
        sub    = subsample_trajectory(
            veh_df, ref_datetime, ref_time,
            target_fps=target_fps,
            debug=debug, log=log,
            include_heads=include_heads,
            include_tails=include_tails,
            t_grid_master=t_grid_master
        )
        results.append(sub)

    sub_df = pd.concat(results, ignore_index=True)
    log.debug(f'subsample_all complete | {len(unique_ids)} vehicles | '
              f'{len(df)} → {len(sub_df)} rows | '
              f'{sub_df["time"].nunique()} unique time values')
    return sub_df