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
- No extrapolation: target grid is clipped to [t_actual[0], t_actual[-1]].
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

def _build_target_grid(t_actual: np.ndarray, target_fps: float) -> np.ndarray:
    """
    Build a uniform time grid at target_fps, anchored to t_actual[0],
    clipped to [t_actual[0], t_actual[-1]].

    Anchoring to t_actual[0] (not rounding to nearest 0.1s boundary)
    avoids any extrapolation at the start of the trajectory.

    Parameters
    ----------
    t_actual   : actual timestamps from the 25 FPS data [seconds]
    target_fps : desired output frame rate [Hz]

    Returns
    -------
    t_target : 1-D array of target timestamps [seconds]
    """
    dt     = 1.0 / target_fps
    t0, t1 = t_actual[0], t_actual[-1]
    n      = int(np.floor((t1 - t0) / dt)) + 1
    t_target = t0 + np.arange(n) * dt

    # Safety clip — floating point can push the last point just past t1
    t_target = t_target[t_target <= t1 + 1e-9]
    t_target = np.clip(t_target, t0, t1)

    # Append the true endpoint if the last grid point doesn't reach it.
    # With a non-integer FPS ratio (25→10), floor() leaves a tail of up to
    # (dt - ε) seconds unrepresented. Appending t1 guarantees the subsampled
    # trajectory always ends at exactly the same point as the 25 FPS original.
    # This produces one slightly irregular interval at the end, which is
    # handled correctly by all downstream np.interp / CubicSpline calls.
    if t1 - t_target[-1] > 1e-4:   # > 0.1 ms gap → worth appending
        t_target = np.append(t_target, t1)

    return t_target


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
                         log=None) -> pd.DataFrame:
    """
    Subsample a single-vehicle EKF-filtered trajectory from 25 FPS to
    target_fps (default 10 FPS) using timestamp-aware interpolation.

    Parameters
    ----------
    veh_df       : single-vehicle DataFrame with columns:
                   veh_id, veh_type, time, datetime,
                   x_ekf, y_ekf, speed_ekf, angle_ekf,
                   a_ekf, angular_vel_ekf
    ref_datetime : reference datetime when time == ref_time
                   (same convention as the raw data loader)
    ref_time     : reference time value [seconds] corresponding to ref_datetime
    target_fps   : desired output frame rate [Hz], default 10.0
    debug        : if True, enable DEBUG logging
    log          : optional external logger

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

    # ------------------------------------------------------------------
    # Step 1: Build target time grid
    # ------------------------------------------------------------------
    t_target = _build_target_grid(t_actual, target_fps)
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
                  log=None) -> pd.DataFrame:
    """
    Subsample all vehicles in a multi-vehicle DataFrame.

    Computes ref_datetime and ref_time once from the full DataFrame
    (same convention as the raw data loader), then calls
    subsample_trajectory() per vehicle and concatenates results.

    Parameters
    ----------
    df         : full multi-vehicle DataFrame (all vehicles, 25 FPS)
    target_fps : desired output frame rate [Hz], default 10.0
    debug      : if True, enable DEBUG logging
    log        : optional external logger

    Returns
    -------
    sub_df : concatenated subsampled DataFrame (all vehicles, target_fps)
    """
    if log is None:
        log = _get_logger(debug)

    # Replicate the ref_datetime / ref_time convention from the data loader
    ref_datetime = df['datetime'].min()
    ref_time     = df.loc[
        (df['datetime'] == ref_datetime) & (df['time'] >= 0),
        'time'
    ].unique()[0]

    log.debug(f'ref_datetime={ref_datetime}  ref_time={ref_time}')

    results    = []
    unique_ids = df['veh_id'].unique()

    for veh_id in unique_ids:
        veh_df = df[df['veh_id'] == veh_id].copy().reset_index(drop=True)
        sub    = subsample_trajectory(
            veh_df, ref_datetime, ref_time,
            target_fps=target_fps,
            debug=debug, log=log
        )
        results.append(sub)

    sub_df = pd.concat(results, ignore_index=True)
    log.debug(f'subsample_all complete | {len(unique_ids)} vehicles | '
              f'{len(df)} → {len(sub_df)} rows')
    return sub_df