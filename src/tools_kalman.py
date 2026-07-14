"""
EKF + Clothoid Gap Inference for Bicycle Trajectory Filtering
-------------------------------------------
Authors:        Shaimaa El-Baklish
Organization:   ETH Zürich, Switzerland, IVT - Institute for Transportation Planning and Systems
Development:    2025
Submitted to:   JOURNAL
-------------------------------------------

Overview
--------
This module implements a four-phase trajectory filtering pipeline for bicycle
tracking data with missing observation gaps:

  Phase 0 — Geometric pre-processing
      Detects all gaps in the trajectory. For each long gap, builds a G2
      clothoid path from raw boundary measurements (position, heading,
      curvature). Optimizes a Bernstein speed profile v(t) along the fixed
      clothoid geometry. Gap timesteps are filled as pseudo-observations in
      the dataframe.

  Phase 1 — Forward EKF
      Runs a standard Extended Kalman Filter over the full trajectory
      (observed + clothoid pseudo-observations). Gap pseudo-observations use
      a slightly inflated measurement noise R_gap = 4 * R_t to reflect
      clothoid uncertainty. No special gap handling inside the EKF loop.

  Phase 2 — RTS backward smoother
      Standard Rauch-Tung-Striebel smoother over the full trajectory.
      No gap-aware gain clipping needed since gap states are pre-filled.

Key design decisions
--------------------
- Clothoid geometry (SolveG2) ensures G2 continuity (pos, heading, curvature)
  at gap boundaries — no wrong turns possible.
- v(t) is parameterized as a Bernstein polynomial (all ctrl pts >= 0) → v >= 0
  guaranteed by the convex hull property.
- ω(t) = κ(s(t)) · v(t) is recovered geometrically — not optimized separately.
- Boundary headings are always computed from xy finite differences, not from
  angle_estimation, which can be corrupted near gap boundaries.
- v_profile is normalized post-optimization so ∫v dt = S_total exactly,
  guaranteeing the rollout reaches the clothoid endpoint.
"""

# =============================================================================
# IMPORTS
# =============================================================================
import sys
import logging
import warnings
warnings.simplefilter('ignore', RuntimeWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from math import comb
from numba import njit
from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter1d
from pyclothoids import SolveG2

from _constants import SKIP_KALMAN_FILTERING_MAX_GAP

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


# =============================================================================
# NUMBA-COMPILED CORE DYNAMICS
# =============================================================================

@njit
def _f_dyn(x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
    """
    Unicycle kinematic model: one Euler step.

    State  x = [x_pos, y_pos, speed, heading]
    Input  u = [accel, omega]

    Speed is clamped to >= 0 (bicycles do not reverse).
    """
    x_new    = np.zeros(4, dtype=np.float64)
    x_new[0] = x[0] + dt * x[2] * np.cos(x[3])
    x_new[1] = x[1] + dt * x[2] * np.sin(x[3])
    x_new[2] = np.maximum(0.0, x[2] + dt * u[0])
    x_new[3] = x[3] + dt * u[1]
    return x_new


@njit
def _A_jacobian(x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
    """
    Jacobian of _f_dyn w.r.t. state x — used in EKF predict and RTS smoother.
    """
    return np.array([
        [1.0, 0.0, dt * np.cos(x[3]), -dt * x[2] * np.sin(x[3])],
        [0.0, 1.0, dt * np.sin(x[3]),  dt * x[2] * np.cos(x[3])],
        [0.0, 0.0, 1.0,                0.0                      ],
        [0.0, 0.0, 0.0,                1.0                      ],
    ], dtype=np.float64)


@njit
def _ekf_predict(x: np.ndarray, P: np.ndarray,
                 u: np.ndarray, Q: np.ndarray,
                 dt: float):
    """
    EKF prediction step.

    Returns
    -------
    x_pred : predicted state  (4,)
    P_pred : predicted covariance  (4, 4)
    """
    x_pred = _f_dyn(x, u, dt)
    A      = _A_jacobian(x, u, dt)
    P_pred = A @ P @ A.T + Q
    return x_pred, P_pred


@njit
def _ekf_correct(x_pred: np.ndarray, P_pred: np.ndarray,
                 y: np.ndarray, C: np.ndarray,
                 R: np.ndarray):
    """
    EKF correction (update) step.

    Parameters
    ----------
    y : measurement vector
    C : measurement matrix (full-state observation → identity)
    R : measurement noise covariance

    Returns
    -------
    x_corr : corrected state  (4,)
    P_corr : corrected covariance  (4, 4)
    """
    K      = P_pred @ C.T @ np.linalg.inv(C @ P_pred @ C.T + R)
    x_corr = x_pred + K @ (y - C @ x_pred)
    P_corr = (np.eye(4) - K) @ P_pred 
    return x_corr, P_corr


@njit
def _rts_smooth(xs_filt: np.ndarray, Ps_filt: np.ndarray,
                xs_pred: np.ndarray, Ps_pred: np.ndarray,
                us: np.ndarray, times: np.ndarray):
    """
    Rauch-Tung-Striebel backward smoother.

    Runs backward over the full trajectory (including gap pseudo-observations)
    without any special gap handling — gap states are already optimal since
    they were pre-filled from the clothoid in Phase 0.

    Parameters
    ----------
    xs_filt : EKF filtered states       (4, N)
    Ps_filt : EKF filtered covariances  (4, 4, N)
    xs_pred : EKF predicted states      (4, N)
    Ps_pred : EKF predicted covariances (4, 4, N)
    us      : inputs [a, omega]         (2, N)
    times   : timestamps                (N,)

    Returns
    -------
    xs_smooth : smoothed states       (4, N)
    Ps_smooth : smoothed covariances  (4, 4, N)
    """
    xs_smooth = np.copy(xs_filt)
    Ps_smooth = np.copy(Ps_filt)
    for i in range(len(times) - 2, -1, -1):
        dt  = times[i + 1] - times[i]
        A_t = _A_jacobian(xs_filt[:, i], us[:, i], dt)
        K_s = Ps_filt[:, :, i] @ A_t.T @ np.linalg.inv(Ps_pred[:, :, i + 1])
        xs_smooth[:, i] = (
            xs_filt[:, i] + K_s @ (xs_smooth[:, i + 1] - xs_pred[:, i + 1])
        )
        Ps_smooth[:, :, i] = (
            Ps_filt[:, :, i]
            + K_s @ (Ps_smooth[:, :, i + 1] - Ps_pred[:, :, i + 1]) @ K_s.T
        )
    return xs_smooth, Ps_smooth


# =============================================================================
# CLOTHOID GEOMETRY UTILITIES
# =============================================================================

def _wrap_angle(x: float) -> float:
    """Wrap angle to (-π, π]."""
    return (x + np.pi) % (2 * np.pi) - np.pi


def _build_clothoid_path(start_state: np.ndarray, target_state: np.ndarray,
                          k0: float = 0.0, k1: float = 0.0):
    """
    Build a G2-continuous clothoid path between two boundary states.

    Uses pyclothoids.SolveG2 which matches position, heading, and curvature
    at both endpoints simultaneously. The result is three clothoid segments.

    Parameters
    ----------
    start_state  : [x, y, v, theta]  at gap start
    target_state : [x, y, v, theta]  at gap end
    k0           : curvature [1/m] at start  (from xy geometry)
    k1           : curvature [1/m] at end    (from xy geometry)

    Returns
    -------
    pieces  : list of clothoid segments
    lengths : arc-length of each segment  [m]
    S_total : total arc-length  [m]
    """
    x0, y0, _, theta0 = start_state
    x1, y1, _, theta1 = target_state
    pieces  = list(SolveG2(x0, y0, theta0, k0, x1, y1, theta1, k1))
    lengths = [seg.length for seg in pieces]
    S_total = float(np.sum(lengths))
    return pieces, lengths, S_total


def _eval_clothoid(pieces, lengths, s_vals: np.ndarray) -> np.ndarray:
    """
    Evaluate clothoid path at given arc-length positions.

    Parameters
    ----------
    pieces  : clothoid segments from SolveG2
    lengths : arc-length of each segment
    s_vals  : arc-length query positions  (N,)

    Returns
    -------
    out : (N, 4) array — columns [x, y, theta, kappa]
    """
    seg_cum = np.concatenate(([0.0], np.cumsum(lengths)))
    out     = np.zeros((len(s_vals), 4))
    for i, s in enumerate(s_vals):
        s       = float(np.clip(s, 0.0, seg_cum[-1]))
        idx     = int(np.searchsorted(seg_cum, s, side='right') - 1)
        idx     = int(np.clip(idx, 0, len(lengths) - 1))
        local_s = s - seg_cum[idx]
        seg     = pieces[idx]
        out[i]  = [seg.X(local_s), seg.Y(local_s),
                   seg.Theta(local_s), seg.ThetaD(local_s)]
    return out


def _smooth_pts(pts: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """
    Gaussian smoothing of a 2D point sequence along each axis.

    Pins the last point exactly so the boundary position is not moved.
    Used to reduce noise before curvature estimation.

    Parameters
    ----------
    pts   : (N, 2) array of (x, y) points
    sigma : smoothing bandwidth in point-index units

    Returns
    -------
    smoothed : (N, 2) smoothed point array
    """
    if len(pts) < 3:
        return pts
    smoothed          = pts.astype(float).copy()
    smoothed[:, 0]    = gaussian_filter1d(pts[:, 0], sigma=sigma)
    smoothed[:, 1]    = gaussian_filter1d(pts[:, 1], sigma=sigma)
    smoothed[-1]      = pts[-1]   # pin boundary — must not move
    return smoothed


def _circumscribed_kappa(p1: np.ndarray,
                          p2: np.ndarray,
                          p3: np.ndarray) -> float:
    """
    Signed curvature of the circumscribed circle through three points.

    Sign convention (standard math, y-axis up):
      κ > 0 → CCW / left turn
      κ < 0 → CW  / right turn

    Returns 0.0 if the three points are collinear or too close.
    """
    a     = np.linalg.norm(p2 - p1)
    b     = np.linalg.norm(p3 - p2)
    c     = np.linalg.norm(p3 - p1)
    cross = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
    denom = a * b * c
    if abs(denom) < 1e-9 or abs(cross) < 1e-9:
        return 0.0
    R = denom / (2.0 * abs(cross))
    return float(np.sign(cross) / R)


def _estimate_boundary_curvatures(feat_veh_df: pd.DataFrame,
                                   times: np.ndarray,
                                   first_missing_idx: int,
                                   next_avail_idx: int,
                                   window: int = 7,
                                   debug: bool = False,
                                   log: logging.Logger = None) -> tuple:
    """
    Estimate signed curvature at both gap boundaries from observed xy geometry.

    Uses the circumscribed circle of the three points nearest each boundary,
    applied to a Gaussian-smoothed window of observations. This is more robust
    than omega/v because it uses multiple points and needs no speed estimate.

    Parameters
    ----------
    feat_veh_df      : trajectory dataframe
    times            : full time array
    first_missing_idx: index of first missing timestep
    next_avail_idx   : dataframe index of first observation after gap
    window           : number of observed points to collect on each side
    debug            : if True, log curvature triplet values

    Returns
    -------
    k0 : curvature at gap start [1/m]
    k1 : curvature at gap end   [1/m]
    """

    def _boundary_kappa(pts: np.ndarray, eval_at_end: bool = True) -> float:
        """Curvature at boundary using 3 nearest points on smoothed pts."""
        if len(pts) < 3:
            return 0.0
        if eval_at_end:
            return _circumscribed_kappa(pts[-3], pts[-2], pts[-1])
        else:
            return _circumscribed_kappa(pts[0], pts[1], pts[2])

    def _triplet_kappas(pts: np.ndarray) -> np.ndarray:
        """Curvature at every consecutive triplet — for debug logging."""
        return np.array([
            _circumscribed_kappa(pts[k], pts[k + 1], pts[k + 2])
            for k in range(len(pts) - 2)
        ])

    # --- Gap start boundary (k0): window of observed points BEFORE gap ---
    start_boundary_time = times[first_missing_idx - 1]
    obs_before_raw = (
        feat_veh_df[~feat_veh_df['missing'] &
                    (feat_veh_df['time'] <= start_boundary_time)]
        .tail(window)[['x', 'y']].to_numpy()
    )
    obs_before_smoothed      = _smooth_pts(obs_before_raw, sigma=5.0)
    k0                       = _boundary_kappa(obs_before_smoothed, eval_at_end=True)

    log.debug(f'k0 estimation | window={len(obs_before_raw)} pts before gap')
    log.debug(f"  raw triplets      : {[f'{v:.5f}' for v in _triplet_kappas(obs_before_raw)]}")
    log.debug(f"  smoothed triplets : {[f'{v:.5f}' for v in _triplet_kappas(obs_before_smoothed)]}")
    log.debug(f"  k0 = {k0:.5f} 1/m")

    # --- Gap end boundary (k1): window of observed points AFTER gap ---
    next_avail_time = feat_veh_df.loc[next_avail_idx, 'time']
    obs_after_raw   = (
        feat_veh_df[~feat_veh_df['missing'] &
                    (feat_veh_df['time'] >= next_avail_time)]
        .head(window)[['x', 'y']].to_numpy()
    )
    obs_after_smoothed      = _smooth_pts(obs_after_raw, sigma=1.0)
    obs_after_smoothed[0]   = obs_after_raw[0]   # pin boundary point
    k1                      = _boundary_kappa(obs_after_smoothed, eval_at_end=False)

    log.debug(f'k1 estimation | window={len(obs_after_raw)} pts after gap')
    log.debug(f'  k1 = {k1:.5f} 1/m')

    return float(k0), float(k1)


# =============================================================================
# STAGE 1: CLOTHOID PATH CONSTRUCTION
# =============================================================================

def _build_gap_geometry(start_state: np.ndarray,
                         target_state: np.ndarray,
                         last_avail_input: np.ndarray,
                         next_avail_input: np.ndarray,
                         feat_veh_df: pd.DataFrame,
                         times: np.ndarray,
                         first_missing_idx: int,
                         next_avail_idx: int,
                         k_max: float = 1.0,
                         window: int = 10,
                         debug: bool = False,
                         debug_title: str = '',
                         log: logging.Logger = None) -> tuple:
    """
    Stage 1: build a G2 clothoid path from raw boundary measurements.

    Boundary curvatures are estimated from the xy trajectory geometry
    (circumscribed circle method) rather than from omega/v, which is
    more robust to speed/angular-velocity estimation noise near gaps.

    A progressive fallback strategy relaxes curvature constraints until
    the clothoid construction succeeds:
      1. Full G2: k0, k1 from xy geometry
      2. Relax k1 → 0 only
      3. Relax k0 → 0 only
      4. Both k0 = k1 = 0

    Parameters
    ----------
    start_state       : [x, y, v, theta]  at gap start (empirical heading)
    target_state      : [x, y, v, theta]  at gap end   (empirical heading)
    last_avail_input  : [a, omega]  just before gap
    next_avail_input  : [a, omega]  just after gap
    feat_veh_df       : trajectory dataframe (for curvature window lookup)
    times             : full time array
    first_missing_idx : index of first missing timestep
    next_avail_idx    : dataframe index of first observation after gap
    k_max             : hard curvature ceiling [1/m]  (radius >= 1/k_max)
    window            : xy window size for curvature estimation
    debug             : if True, show clothoid geometry plot and log details
    debug_title       : string appended to plot/log titles for identification

    Returns
    -------
    pieces  : clothoid segments
    lengths : arc-lengths per segment [m]
    S_total : total arc-length [m]
    k0      : curvature used at start [1/m]
    k1      : curvature used at end   [1/m]
    """
    # log = _get_logger(debug)

    # --- Estimate boundary curvatures from xy geometry ---
    k0, k1 = _estimate_boundary_curvatures(
        feat_veh_df, times, first_missing_idx, next_avail_idx,
        window=window, debug=debug, log=log
    )
    k0 = float(np.clip(k0, -k_max, k_max))
    k1 = float(np.clip(k1, -k_max, k_max))

    log.debug(f'Stage 1 [{debug_title}] | k0={k0:.4f} 1/m  k1={k1:.4f} 1/m  (window={window})')

    # --- Progressive clothoid construction with fallback ---
    attempts = [
        (k0,   k1,   'full G2 (k0, k1)'),
        (k0,   0.0,  'relax k1=0'),
        (0.0,  k1,   'relax k0=0'),
        (0.0,  0.0,  'k0=k1=0'),
    ]
    pieces = lengths = S_total = None
    k0_used = k1_used = 0.0

    for ka, kb, label in attempts:
        try:
            pieces, lengths, S_total = _build_clothoid_path(
                start_state, target_state, k0=ka, k1=kb
            )
            k0_used, k1_used = ka, kb
            log.debug(f'  Clothoid success: {label}  S={S_total:.3f} m')
            break
        except Exception as e:
            log.warning(f'  Clothoid failed ({label}): {e} — trying next fallback')

    if pieces is None:
        raise RuntimeError(
            f'[_build_gap_geometry] Clothoid construction failed for all '
            f'fallback attempts. {debug_title}'
        )

    # --- Debug plot: clothoid geometry with curvature profile ---
    if debug:
        _plot_clothoid_geometry(
            pieces, lengths, S_total,
            start_state, target_state,
            k0_used, k1_used,
            feat_veh_df, times,
            first_missing_idx, next_avail_idx,
            window, debug_title
        )

    return pieces, lengths, S_total, k0_used, k1_used


# =============================================================================
# STAGE 2: DYNAMIC OPTIMIZATION
# =============================================================================

def _build_v_profile(params: np.ndarray, t_norm: np.ndarray,
                      v0: float, v1: float, K: int) -> tuple:
    """
    Build a non-negative velocity profile v(t) as a Bernstein polynomial.

    The profile is pinned at v(0) = v0 and v(1) = v1. Interior control
    points are bounded >= 0, so by the Bernstein convex-hull property
    v(t) >= 0 for all t in [0, 1] — no post-hoc clamping needed.

    Parameters
    ----------
    params : (K,) interior Bernstein control points  (all >= 0)
    t_norm : normalized time in [0, 1]  (N,)
    v0     : speed at gap start  [m/s]
    v1     : speed at gap end    [m/s]
    K      : number of interior control points

    Returns
    -------
    v_profile : (N,)  velocity profile  [m/s]
    a_profile : (N,)  dv/dt_norm  (scale by 1/T_gap for physical units)
    """
    n    = K + 1           # Bernstein polynomial degree
    N    = len(t_norm)
    v_cp = np.concatenate([[v0], params[:K], [v1]])   # full ctrl point vector

    Bv  = np.zeros((N, K + 2))
    dBv = np.zeros((N, K + 2))
    for j in range(K + 2):
        Bv[:, j]  = comb(n, j) * t_norm ** j * (1 - t_norm) ** (n - j)
    for j in range(K + 2):
        b_prev     = (comb(n-1, j-1) * t_norm**(j-1) * (1-t_norm)**(n-j)
                      if j > 0 else np.zeros(N))
        b_curr     = (comb(n-1, j)   * t_norm**j     * (1-t_norm)**(n-j-1)
                      if j < n else np.zeros(N))
        dBv[:, j] = n * (b_prev - b_curr)

    v_profile = Bv  @ v_cp
    a_profile = dBv @ v_cp
    return v_profile, a_profile


def _rollout_on_clothoid(start_state: np.ndarray,
                          times: np.ndarray,
                          v_profile: np.ndarray,
                          pieces, lengths, S_total: float) -> tuple:
    """
    Roll out trajectory along a fixed clothoid using a speed profile.

    Arc-length: s(t) = ∫₀ᵗ v(τ) dτ  (trapezoidal integration).
    Position and heading are read from the clothoid geometry — no heading
    drift, no position integration error.
    ω(t) = κ(s(t)) · v(t) is recovered geometrically.

    Returns
    -------
    states : (N, 4)  [x, y, v, theta]
    s_vals : (N,)    arc-length at each timestep
    omega  : (N,)    angular velocity [rad/s]
    """
    N      = len(times)
    s_vals = np.zeros(N)
    for i in range(N - 1):
        dt          = times[i + 1] - times[i]
        ds          = 0.5 * (v_profile[i] + v_profile[i + 1]) * dt
        s_vals[i+1] = min(s_vals[i] + ds, S_total)

    path   = _eval_clothoid(pieces, lengths, s_vals)
    states = np.zeros((N, 4))
    omega  = np.zeros(N)
    for i in range(N):
        states[i] = [path[i, 0], path[i, 1], v_profile[i], path[i, 2]]
        omega[i]  = path[i, 3] * v_profile[i]   # κ · v

    return states, s_vals, omega


def _rollout_backward_on_clothoid(target_state: np.ndarray,
                                   times: np.ndarray,
                                   v_profile: np.ndarray,
                                   pieces, lengths, S_total: float) -> tuple:
    """
    Backward rollout from target along the same clothoid.

    Integrates arc-length backward from S_total. Used to compute
    forward-backward consistency — if both rollouts agree on s(t),
    the speed profile is symmetric and naturalistic.

    Returns
    -------
    states : (N, 4)  [x, y, v, theta]
    s_vals : (N,)    arc-length at each timestep
    """
    N      = len(times)
    s_vals = np.zeros(N)
    s_vals[-1] = S_total
    for i in range(N - 2, -1, -1):
        dt        = times[i + 1] - times[i]
        ds        = 0.5 * (v_profile[i] + v_profile[i + 1]) * dt
        s_vals[i] = max(s_vals[i + 1] - ds, 0.0)

    path   = _eval_clothoid(pieces, lengths, s_vals)
    states = np.column_stack([
        path[:, 0], path[:, 1], v_profile, path[:, 2]
    ])
    return states, s_vals


def _gap_inference_objective(params: np.ndarray,
                              start_state: np.ndarray,
                              target_state: np.ndarray,
                              last_avail_input: np.ndarray,
                              next_avail_input: np.ndarray,
                              times: np.ndarray,
                              t_norm: np.ndarray,
                              K: int,
                              pieces, lengths, S_total: float,
                              a_min: float, a_max: float,
                              w_min: float, w_max: float,
                              w_pos: float = 10.0,
                              w_vel: float = 10.0,
                              w_ang: float = 0.0,
                              w_smooth_a: float = 0.1,
                              w_smooth_w: float = 0.1,
                              w_mag_a: float = 0.0,
                              w_mag_w: float = 0.0,
                              w_continuity: float = 1.0,
                              w_consistency: float = 0.0,
                              w_arc: float = 0.0) -> float:
    """
    Objective for Stage 2: optimize v(t) along a fixed clothoid path.

    Decision variables : K interior Bernstein control points for v(t).
    ω(t) = κ(s(t)) · v(t) — determined geometrically, not optimized.
    a(t) = dv/dt         — recovered analytically from Bernstein derivative.

    Objective terms
    ---------------
    w_pos          : endpoint position matching (dominant term)
    w_vel          : endpoint velocity matching
    w_ang          : endpoint heading matching
    w_smooth_a     : first-difference smoothness of a(t)
    w_smooth_w     : first-difference smoothness of ω(t)
    w_mag_a        : magnitude regularization of a(t)
    w_mag_w        : magnitude regularization of ω(t)  (min-turn principle)
    w_continuity   : rate-of-change of a and ω at gap boundaries
    w_consistency  : forward-backward arc-length and speed agreement
    w_arc          : integral of v(t) matching S_total  (prevents short-stop)
    """
    T_gap = float(times[-1] - times[0])
    v0    = float(start_state[2])
    v1    = float(target_state[2])

    v_profile, a_profile_norm = _build_v_profile(params, t_norm, v0, v1, K)
    a_profile = np.clip(a_profile_norm / T_gap, a_min, a_max)

    fwd_states, fwd_s, omega = _rollout_on_clothoid(
        start_state, times, v_profile, pieces, lengths, S_total
    )
    omega = np.clip(omega, w_min, w_max)

    bwd_states, bwd_s = _rollout_backward_on_clothoid(
        target_state, times, v_profile, pieces, lengths, S_total
    )

    final = fwd_states[-1]

    # 1. Endpoint matching
    pos_err = float(np.linalg.norm(final[:2] - target_state[:2]))
    vel_err = float(abs(final[2] - target_state[2]))
    ang_err = float(abs(_wrap_angle(final[3] - target_state[3])))

    # 2. Arc-length matching: ∫v dt should equal S_total
    dt_arr     = np.diff(times)
    v_integral = float(np.sum(0.5 * (v_profile[:-1] + v_profile[1:]) * dt_arr))
    arc_err    = (v_integral - S_total) ** 2

    # 3. Forward-backward consistency
    s_consistency   = float(np.mean((fwd_s - bwd_s) ** 2))
    vel_consistency = float(np.mean((fwd_states[:, 2] - bwd_states[:, 2]) ** 2))
    consistency_pen = s_consistency + vel_consistency

    # 4. Input smoothness
    smooth_a = float(np.mean(np.diff(a_profile) ** 2))
    smooth_w = float(np.mean(np.diff(omega)     ** 2))

    # 5. Input magnitude
    mag_a = float(np.mean(a_profile ** 2))
    mag_w = float(np.mean(omega     ** 2))

    # 6. Boundary continuity: rate-of-change at gap edges
    dt_start       = times[1]  - times[0]
    dt_end         = times[-1] - times[-2]
    a0_k, a1_k     = float(last_avail_input[0]),  float(next_avail_input[0])
    w0_k, w1_k     = float(last_avail_input[1]),  float(next_avail_input[1])
    continuity_pen = (
        ((a_profile[0]  - a0_k) / dt_start) ** 2 +
        ((a_profile[-1] - a1_k) / dt_end)   ** 2 +
        ((omega[0]      - w0_k) / dt_start) ** 2 +
        ((omega[-1]     - w1_k) / dt_end)   ** 2
    )

    return (w_pos         * pos_err         +
            w_vel         * vel_err         +
            w_ang         * ang_err         +
            w_arc         * arc_err         +
            w_consistency * consistency_pen +
            w_smooth_a    * smooth_a        +
            w_smooth_w    * smooth_w        +
            w_mag_a       * mag_a           +
            w_mag_w       * mag_w           +
            w_continuity  * continuity_pen)


def _infer_gap_dynamics(start_state: np.ndarray,
                         target_state: np.ndarray,
                         last_avail_input: np.ndarray,
                         next_avail_input: np.ndarray,
                         times: np.ndarray,
                         pieces, lengths, S_total: float,
                         K: int = 5,
                         a_min: float = -3.0, a_max: float = 3.0,
                         w_min: float = -2.0, w_max: float = 2.0,
                         debug: bool = False,
                         debug_title: str = '',
                         log: logging.Logger = None) -> np.ndarray:
    """
    Stage 2: optimize v(t) along a pre-built clothoid path.

    Clothoid geometry is fixed (from Stage 1) — this function only solves
    for the speed profile. ω(t) is recovered geometrically as κ(s) · v(t).

    After optimization, v_profile is normalized so ∫v dt = S_total exactly,
    guaranteeing the rollout reaches the clothoid endpoint regardless of
    the optimizer's convergence quality.

    Parameters
    ----------
    start_state      : [x, y, v, theta]  at times[0]   (v in m/s)
    target_state     : [x, y, v, theta]  at times[-1]  (v in m/s)
    last_avail_input : [a, omega]  just before gap
    next_avail_input : [a, omega]  just after gap
    times            : timesteps covering the gap, including endpoints
    pieces           : clothoid segments (from _build_gap_geometry)
    lengths          : arc-lengths per segment
    S_total          : total arc-length [m]
    K                : max interior Bernstein control points  (auto-capped)
    a_min, a_max     : acceleration bounds [m/s²]
    w_min, w_max     : angular velocity bounds [rad/s]
    debug            : if True, show time-profile plots and log details
    debug_title      : string appended to plot/log titles

    Returns
    -------
    missing_inputs : (2, len(times)-1)
                     Row 0 = a(t),  Row 1 = ω(t),  per interval
    """
    # log   = _get_logger(debug)
    T_gap = float(times[-1] - times[0])
    N     = len(times)
    t_norm = (times - times[0]) / T_gap
    K      = min(K, max(2, (N - 1) // 5))
    v0     = float(start_state[2])
    v1     = float(target_state[2])

    # Initialization: linear ramp from v0 to v1, clamped >= 0
    t_ctrl  = np.linspace(0, 1, K + 2)[1:-1]
    v_ctrl0 = np.maximum(0.0, v0 + t_ctrl * (v1 - v0))
    bounds  = [(0.0, max(v0, v1, 20.0))] * K

    res = minimize(
        _gap_inference_objective,
        v_ctrl0,
        method='L-BFGS-B',
        bounds=bounds,
        args=(start_state, target_state,
              last_avail_input, next_avail_input,
              times, t_norm, K,
              pieces, lengths, S_total,
              a_min, a_max, w_min, w_max),
        options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-7, 'disp': False}
    )
    log.debug(f'Stage 2 [{debug_title}] | optimizer success={res.success}  nit={res.nit}  fun={res.fun:.5f}')

    v_profile, _ = _build_v_profile(res.x, t_norm, v0, v1, K)

    # --- Normalize v_profile so ∫v dt = S_total exactly ---
    dt_arr     = np.diff(times)
    v_integral = float(np.sum(0.5 * (v_profile[:-1] + v_profile[1:]) * dt_arr))
    if v_integral > 1e-6:
        scale     = S_total / v_integral
        v_profile = v_profile * scale
    else:
        scale = 1.0

    log.debug(f'  v normalization | integral={v_integral:.4f} m  S_total={S_total:.4f} m  scale={scale:.6f}')
    if scale > 1.2:
        log.warning(f'  Large v scale factor ({scale:.3f}) — consider increasing w_arc')

    # Re-derive a_profile from scaled v_profile
    a_profile = np.clip(np.gradient(v_profile, times), a_min, a_max)

    fwd_states, fwd_s, omega = _rollout_on_clothoid(
        start_state, times, v_profile, pieces, lengths, S_total
    )
    omega      = np.clip(omega, w_min, w_max)
    bwd_states, bwd_s = _rollout_backward_on_clothoid(
        target_state, times, v_profile, pieces, lengths, S_total
    )

    log.debug(f'  Rollout | pos_err={np.linalg.norm(fwd_states[-1, :2] - target_state[:2]):.4f} m  vel_err={abs(fwd_states[-1, 2] - v1):.4f} m/s  '
              f'fwd-bwd Δs={float(np.mean(np.abs(fwd_s - bwd_s))):.4f} m')
    log.debug(f'  v range [{fwd_states[:, 2].min():.3f}, {fwd_states[:, 2].max():.3f}] m/s  a range [{a_profile.min():.3f}, {a_profile.max():.3f}] m/s²  '
              f'ω range [{omega.min():.3f}, {omega.max():.3f}] rad/s')

    if debug:
        _plot_gap_dynamics(
            pieces, lengths, S_total,
            start_state, target_state,
            last_avail_input, next_avail_input,
            times, v_profile, a_profile, omega,
            fwd_states, fwd_s, bwd_states, bwd_s,
            v0, v1, K, T_gap, res,
            a_min, a_max, w_min, w_max,
            debug_title
        )

    missing_inputs       = np.zeros((2, N - 1))
    missing_inputs[0, :] = a_profile[:-1]
    missing_inputs[1, :] = omega[:-1]
    return missing_inputs


# =============================================================================
# GAP REGISTRY HELPERS
# =============================================================================

def _gap_mask_from_registry(gap_registry: dict, idx: int) -> bool:
    """Return True if timestep idx falls inside any registered gap."""
    for gap_start, (_, gap_len, _, _, _) in gap_registry.items():
        if gap_start <= idx < gap_start + gap_len - 1:
            return True
    return False


def _fill_gap_pseudo_observations(feat_veh_df: pd.DataFrame,
                                   times: np.ndarray,
                                   gap_registry: dict) -> pd.DataFrame:
    """
    Fill gap timesteps in feat_veh_df with clothoid-derived pseudo-observations.

    For each registered gap:
      - Reconstructs v_profile by forward integration of a_profile.
      - Computes arc-length s(t) from v_profile.
      - Evaluates clothoid (x, y, theta) at each arc-length.
      - Writes results into feat_veh_df and marks rows as not missing.

    After this step, the EKF loop sees no missing entries — it treats
    gap timesteps as observations with slightly elevated noise.

    Parameters
    ----------
    feat_veh_df  : trajectory dataframe (modified in place)
    times        : full time array
    gap_registry : {first_missing_idx: (missing_inputs, gap_len, pieces,
                                        lengths, S_total)}

    Returns
    -------
    feat_veh_df : updated dataframe with gap pseudo-observations filled
    """
    col = {c: feat_veh_df.columns.get_loc(c)
           for c in ['x', 'y', 'speed', 'angle_estimation',
                     'a', 'angle_vel_estimation', 'missing']}

    for gap_start, (missing_inputs, gap_len,
                     pieces, lengths, S_total) in gap_registry.items():
        gap_end   = gap_start + gap_len - 1
        gap_times = times[gap_start:gap_end + 1]

        # Reconstruct v_profile from a_profile
        v0        = max(0.0, feat_veh_df.iloc[gap_start - 1]['speed'] / 3.6)
        v_profile = np.zeros(gap_len)
        v_profile[0] = v0
        for j in range(gap_len - 1):
            dt_j           = gap_times[j + 1] - gap_times[j]
            v_profile[j+1] = max(0.0, v_profile[j] + missing_inputs[0, j] * dt_j)

        # Arc-lengths from v_profile (trapezoidal)
        s_vals    = np.zeros(gap_len)
        for j in range(gap_len - 1):
            dt_j         = gap_times[j + 1] - gap_times[j]
            ds           = 0.5 * (v_profile[j] + v_profile[j + 1]) * dt_j
            s_vals[j+1]  = min(s_vals[j] + ds, S_total)

        path = _eval_clothoid(pieces, lengths, s_vals)   # (gap_len, 4)

        for j in range(gap_len):
            idx = gap_start + j
            feat_veh_df.iloc[idx, col['x']]                 = path[j, 0]
            feat_veh_df.iloc[idx, col['y']]                 = path[j, 1]
            feat_veh_df.iloc[idx, col['speed']]             = v_profile[j] * 3.6
            feat_veh_df.iloc[idx, col['angle_estimation']]  = path[j, 2]
            if j < gap_len - 1:
                feat_veh_df.iloc[idx, col['a']]                     = missing_inputs[0, j]
                feat_veh_df.iloc[idx, col['angle_vel_estimation']]  = missing_inputs[1, j]
            feat_veh_df.iloc[idx, col['missing']] = False

    return feat_veh_df


# =============================================================================
# PHASE 0: GEOMETRIC PRE-PROCESSING
# =============================================================================

def _precompute_gap_geometries(feat_veh_df: pd.DataFrame,
                                times: np.ndarray,
                                fps: float,
                                a_min: float, a_max: float,
                                w_min: float, w_max: float,
                                debug: bool = False,
                                log: logging.Logger = None) -> dict:
    """
    Phase 0: scan the full trajectory, detect all gaps, and compute
    clothoid-based inputs for each long gap before the EKF runs.

    For each detected gap longer than SKIP_KALMAN_FILTERING_MAX_GAP / fps:
      1. Identifies raw boundary measurements (last observed before gap,
         first observed after gap).
      2. Overrides boundary headings with empirical arctan2(dy, dx) estimates
         — more reliable than angle_estimation near gaps.
      3. Calls _build_gap_geometry (Stage 1) to construct the G2 clothoid.
      4. Calls _infer_gap_dynamics (Stage 2) to optimize v(t) along it.
      5. Stores results in gap_registry keyed by first_missing_idx.

    Short gaps (below threshold) are left to the EKF predict-only path.

    Parameters
    ----------
    feat_veh_df : trajectory dataframe
    times       : full time array
    fps         : frames per second (used for gap-length threshold)
    a_min/max   : acceleration bounds [m/s²]
    w_min/max   : angular velocity bounds [rad/s]
    debug       : if True, show Stage 1 and Stage 2 debug plots

    Returns
    -------
    gap_registry : dict
        {first_missing_idx: (missing_inputs, gap_len, pieces, lengths, S_total)}
        missing_inputs : (2, gap_len-1) — a(t) and ω(t) per interval
        gap_len        : number of timesteps in gap (including endpoints)
        pieces, lengths, S_total : clothoid geometry for later reuse
    """
    # log          = _get_logger(debug)
    gap_registry = {}
    veh_id       = feat_veh_df['veh_id'].iloc[0]
    i            = 0

    while i < len(times) - 1:
        if not feat_veh_df.loc[feat_veh_df['time'] == times[i], 'missing'].item():
            i += 1
            continue

        # --- Identify gap extent ---
        first_missing_idx = i
        curr_idx          = feat_veh_df[feat_veh_df['time'] == times[i]].index[0]
        next_avail_idx    = feat_veh_df.loc[curr_idx:, 'missing'].idxmin()
        next_avail_time   = feat_veh_df.loc[next_avail_idx, 'time']
        gap_length        = next_avail_time - times[i]

        log.debug(f'[Phase 0] veh={veh_id} | gap idx={first_missing_idx}  t={times[i]:.3f}s  length={gap_length:.3f}s')

        # Skip short gaps — handled by EKF predict-only
        # Still overwrite the poisoned 'a' sentinel (-1) with a linear
        # interpolation between the last known-good 'a' before the gap
        # and the first known-good 'a' after it — 'a' is a smooth input,
        # not a raw measurement, so interpolation (not a constant) is the
        # right prior across a short, unobserved stretch.
        if gap_length <= SKIP_KALMAN_FILTERING_MAX_GAP / fps:
            prev_row = (feat_veh_df[~feat_veh_df['missing'] &
                                     (feat_veh_df['time'] <= times[i - 1])]
                        .iloc[-1])
            next_row = feat_veh_df.loc[next_avail_idx]

            a_start = float(prev_row['a'])
            a_end   = float(next_row['a'])

            gap_row_mask = ((feat_veh_df['time'] >= times[first_missing_idx]) &
                             (feat_veh_df['time'] <  next_avail_time))
            gap_row_times = feat_veh_df.loc[gap_row_mask, 'time'].to_numpy()

            # Linear interpolation in time between the two real boundary a's
            frac  = (gap_row_times - prev_row['time']) / (next_row['time'] - prev_row['time'])
            a_use = a_start + frac * (a_end - a_start)

            feat_veh_df.loc[gap_row_mask, 'a'] = a_use

            log.debug(
                f'[Phase 0] veh={veh_id} | short gap idx={first_missing_idx} '
                f'a_start={a_start:.3f} a_end={a_end:.3f} m/s²  '
                f'(interpolated over {len(gap_row_times)} rows, dur={gap_length:.2f}s)'
            )
            
            # i += 1
            i = int(np.searchsorted(times, next_avail_time))
            continue

        # --- Raw boundary rows (always from observed rows) ---
        prev_obs_time = times[i - 1]
        start_row     = (feat_veh_df[~feat_veh_df['missing'] &
                                      (feat_veh_df['time'] <= prev_obs_time)]
                         .iloc[-1])
        target_row    = feat_veh_df.loc[next_avail_idx]

        start_state  = np.array([start_row['x'],   start_row['y'],
                                  max(0.0, start_row['speed'] / 3.6),
                                  start_row['angle_estimation']])
        target_state = np.array([target_row['x'],  target_row['y'],
                                  max(0.0, target_row['speed'] / 3.6),
                                  target_row['angle_estimation']])
        last_avail_input = np.array([start_row['a'],
                                      start_row['angle_vel_estimation']])
        next_avail_input = np.array([target_row['a'],
                                      target_row['angle_vel_estimation']])

        # --- Override headings with empirical arctan2(dy, dx) ---
        # angle_estimation is unreliable near gaps even for observed rows.
        obs_before = (feat_veh_df[~feat_veh_df['missing'] &
                                   (feat_veh_df['time'] <= prev_obs_time)]
                      .tail(5)[['x', 'y']].to_numpy())
        obs_after  = (feat_veh_df[~feat_veh_df['missing'] &
                                   (feat_veh_df['time'] >= next_avail_time)]
                      .head(5)[['x', 'y']].to_numpy())

        if len(obs_before) >= 2:
            dp               = obs_before[-1] - obs_before[-2]
            start_state[3]   = float(np.arctan2(dp[1], dp[0]))
        if len(obs_after) >= 2:
            dp               = obs_after[1] - obs_after[0]
            target_state[3]  = float(np.arctan2(dp[1], dp[0]))

        log.debug(
            f"  start : ({start_state[0]:.4f}, {start_state[1]:.4f}) "
            f"v={start_state[2]:.3f} m/s theta={np.degrees(start_state[3]):.2f}°"
        )
        
        log.debug(
            f"  target: ({target_state[0]:.4f}, {target_state[1]:.4f}) "
            f"v={target_state[2]:.3f} m/s theta={np.degrees(target_state[3]):.2f}°"
        )

        # --- Gap time arrays ---
        gap_times  = times[(times >= times[first_missing_idx]) &
                            (times <= next_avail_time)]
        gap_len    = len(gap_times)
        full_times = np.concatenate([[prev_obs_time], gap_times])
        dbg_title  = f'veh={veh_id}  t={times[first_missing_idx]:.2f}s'

        # --- Stage 1: clothoid geometry ---
        pieces, lengths, S_total, k0, k1 = _build_gap_geometry(
            start_state, target_state,
            last_avail_input, next_avail_input,
            feat_veh_df=feat_veh_df, times=times,
            first_missing_idx=first_missing_idx,
            next_avail_idx=next_avail_idx,
            window=5, debug=debug, debug_title=dbg_title, log=log
        )

        # --- Stage 2: dynamic optimization ---
        missing_inputs = _infer_gap_dynamics(
            start_state, target_state,
            last_avail_input, next_avail_input,
            full_times, pieces, lengths, S_total,
            a_min=a_min, a_max=a_max,
            w_min=w_min, w_max=w_max,
            debug=debug, debug_title=dbg_title, log=log
        )
        missing_inputs = missing_inputs[:, 1:]   # drop pre-gap interval

        gap_registry[first_missing_idx] = (missing_inputs, gap_len,
                                            pieces, lengths, S_total)
        i = int(np.searchsorted(times, next_avail_time))

    log.debug(f'[Phase 0] veh={veh_id} | {len(gap_registry)} gap(s) registered')
    return gap_registry


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def calculate_kalman_filtered_trajectory(veh_df: pd.DataFrame,
                                          Q_t: np.ndarray,
                                          R_t: np.ndarray,
                                          fps: float = 25.0,
                                          debug: bool = False,
                                          log: logging.Logger = None) -> pd.DataFrame:
    """
    Four-phase trajectory filtering pipeline for a single vehicle.

    Phases
    ------
    0. Geometric pre-processing  (_precompute_gap_geometries)
       Detect gaps, build clothoid paths, optimize speed profiles,
       fill gap timesteps as pseudo-observations.

    1. Forward EKF  (uniform loop, no gap-specific branching)
       Predict + correct at every timestep. Gap pseudo-observations
       use R_gap = 4 * R_t to reflect clothoid uncertainty.

    2. RTS backward smoother  (standard, no gap-aware modification)
       Retroactively tightens all estimates using future information.

    Parameters
    ----------
    veh_df : single-vehicle trajectory DataFrame with columns:
             time, x, y, speed [km/h], angle [rad], angular_vel [rad/s],
             a [m/s²], missing [bool], veh_id
    Q_t    : process noise covariance  (4, 4)
    R_t    : measurement noise covariance  (4, 4)
    fps    : camera frame rate  [Hz]
    debug  : if True, show all diagnostic plots and enable DEBUG logging

    Returns
    -------
    filt_veh_df : filtered trajectory DataFrame with columns:
                  x, y, speed [km/h], angle [rad], time,
                  cov_mat, cov_norm, a [m/s²], angular_vel [rad/s]
    """
    veh_id = veh_df['veh_id'].iloc[0]
    if log is None:
        log = _get_logger(debug)
        log.debug(f'=== calculate_kalman_filtered_trajectory | veh={veh_id} ===')
    else:
        log.section(f'=== calculate_kalman_filtered_trajectory | veh={veh_id} ===')

    # Rename columns to internal names
    feat_veh_df = veh_df.copy()
    feat_veh_df = feat_veh_df.rename(columns={
        'angle':       'angle_estimation',
        'angular_vel': 'angle_vel_estimation'
    })

    C_t   = np.diag([1, 1, 1, 1]).astype(np.float64)
    times = feat_veh_df['time'].to_numpy()
    R_gap = 4.0 * R_t   # elevated noise for clothoid pseudo-observations

    # ------------------------------------------------------------------
    # Phase 0: Geometric pre-processing
    # ------------------------------------------------------------------
    gap_registry = _precompute_gap_geometries(
        feat_veh_df, times, fps,
        a_min=-3.0, a_max=3.0,
        w_min=-1.5, w_max=1.5,
        debug=debug, log=log
    )
    feat_veh_df = _fill_gap_pseudo_observations(
        feat_veh_df, times, gap_registry
    )

    # ------------------------------------------------------------------
    # Phase 1: Forward EKF
    # ------------------------------------------------------------------
    N = len(feat_veh_df)

    states_kalman              = np.zeros((4, N), dtype=np.float64)
    states_kalman[:, 0]        = feat_veh_df.loc[
        feat_veh_df['time'] == times[0],
        ['x', 'y', 'speed', 'angle_estimation']
    ].to_numpy().flatten()
    states_kalman[-2, 0]      /= 3.6   # km/h → m/s

    states_pred                = np.copy(states_kalman)
    states_cov_kalman          = np.zeros((4, 4, N), dtype=np.float64)
    states_cov_kalman[:, :, 0] = np.eye(4, dtype=np.float64)
    states_cov_pred            = np.copy(states_cov_kalman)
    inputs_all                 = np.zeros((2, N), dtype=np.float64)
    
    for i in range(N - 1):
        dt = times[i + 1] - times[i]

        u_t              = feat_veh_df.loc[
            feat_veh_df['time'] == times[i],
            ['a', 'angle_vel_estimation']
        ].to_numpy().flatten()
        inputs_all[:, i] = u_t

        states_pred[:, i+1], states_cov_pred[:, :, i+1] = _ekf_predict(
            states_kalman[:, i], states_cov_kalman[:, :, i],
            inputs_all[:, i], Q_t, dt
        )

        y_t        = feat_veh_df.loc[
            feat_veh_df['time'] == times[i],
            ['x', 'y', 'speed', 'angle_estimation']
        ].to_numpy().flatten()
        y_t[-2]   /= 3.6
        y_t[2]     = max(0.0, y_t[2])

        # R_use = R_gap if _gap_mask_from_registry(gap_registry, i) else R_t
        
        # states_kalman[:, i+1], states_cov_kalman[:, :, i+1] = _ekf_correct(
        #     states_pred[:, i+1], states_cov_pred[:, :, i+1],
        #     y_t, C_t, R_use
        # )
        
        is_missing = feat_veh_df.loc[feat_veh_df['time'] == times[i], 'missing'].item()
        if is_missing:
            # Short gap, no clothoid pseudo-observation: no trustworthy
            # measurement this step — predict-only, using a_use/omega=0
            # computed above as the control input.
            states_kalman[:, i+1]        = states_pred[:, i+1]
            states_cov_kalman[:, :, i+1] = states_cov_pred[:, :, i+1]
        else:
            R_use = R_gap if _gap_mask_from_registry(gap_registry, i) else R_t
            states_kalman[:, i+1], states_cov_kalman[:, :, i+1] = _ekf_correct(
                states_pred[:, i+1], states_cov_pred[:, :, i+1], y_t, C_t, R_use
            )
        
        
        # Clamp speed >= 0 after unconstrained linear correction
        states_kalman[2, i+1] = max(0.0, states_kalman[2, i+1])
        states_pred[2, i+1]   = max(0.0, states_pred[2, i+1])

    # Last timestep input
    inputs_all[:, -1] = feat_veh_df.loc[
        feat_veh_df['time'] == times[-1],
        ['a', 'angle_vel_estimation']
    ].to_numpy().flatten()

    # ------------------------------------------------------------------
    # Phase 2: RTS backward smoother
    # ------------------------------------------------------------------
    states_rts, states_cov_rts = _rts_smooth(
        states_kalman, states_cov_kalman,
        states_pred,   states_cov_pred,
        inputs_all, times
    )
    # Safety clamp: RTS is a linear smoother and can produce v < 0
    states_rts[2, :] = np.maximum(0.0, states_rts[2, :])

    # ------------------------------------------------------------------
    # Debug plots
    # ------------------------------------------------------------------
    if debug:
        _plot_full_trajectory(
            veh_id, feat_veh_df, times,
            gap_registry, states_kalman, states_rts
        )
        _plot_time_profiles(
            veh_id, times, feat_veh_df,
            gap_registry, states_kalman, states_rts, inputs_all
        )

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    filt_veh_df = pd.DataFrame(
        states_rts.T, columns=['x', 'y', 'speed', 'angle']
    )
    filt_veh_df['speed']        = states_rts[2, :] * 3.6
    filt_veh_df['time']         = times
    filt_veh_df['cov_mat']      = [states_cov_rts[:, :, i] for i in range(N)]
    filt_veh_df['cov_norm']     = np.linalg.norm(
        states_cov_rts, ord='fro', axis=(0, 1)
    )
    filt_veh_df['a']            = inputs_all[0, :]
    filt_veh_df['angular_vel']  = inputs_all[1, :]
    
    log.debug(f'=== Filtering complete | veh={veh_id} ===')
    return filt_veh_df


# =============================================================================
# DEBUG PLOT HELPERS
# =============================================================================

def _plot_clothoid_geometry(pieces, lengths, S_total,
                             start_state, target_state,
                             k0, k1,
                             feat_veh_df, times,
                             first_missing_idx, next_avail_idx,
                             window, debug_title):
    """Plot Stage 1 clothoid geometry with boundary curvature profile."""
    s_fine    = np.linspace(0, S_total, 500)
    path_fine = _eval_clothoid(pieces, lengths, s_fine)
    arrow_len = max(S_total * 0.07, 0.5)

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.suptitle(
        f'Stage 1: G2 Clothoid  {debug_title}\n'
        f'start: v={start_state[2]:.2f} m/s  θ={np.degrees(start_state[3]):.1f}°  '
        f'k0={k0:.4f} 1/m   '
        f'target: v={target_state[2]:.2f} m/s  θ={np.degrees(target_state[3]):.1f}°  '
        f'k1={k1:.4f} 1/m   S={S_total:.2f} m',
        fontsize=8
    )

    ax.plot(path_fine[:, 0], path_fine[:, 1], 'b-', linewidth=2,
            label='Clothoid path')
    ax.plot([start_state[0], target_state[0]],
            [start_state[1], target_state[1]],
            'k--', linewidth=1, alpha=0.35, label='Straight line')

    # xy window points used for curvature estimation
    start_boundary_time = times[first_missing_idx - 1]
    next_avail_time     = feat_veh_df.loc[next_avail_idx, 'time']
    obs_before = (feat_veh_df[~feat_veh_df['missing'] &
                               (feat_veh_df['time'] <= start_boundary_time)]
                  .tail(window)[['x', 'y']].to_numpy())
    obs_after  = (feat_veh_df[~feat_veh_df['missing'] &
                               (feat_veh_df['time'] >= next_avail_time)]
                  .head(window)[['x', 'y']].to_numpy())
    ax.scatter(obs_before[:, 0], obs_before[:, 1], color='green',
               s=30, zorder=4, marker='s',
               label=f'xy window start (n={len(obs_before)})')
    ax.scatter(obs_after[:, 0],  obs_after[:, 1],  color='red',
               s=30, zorder=4, marker='s',
               label=f'xy window end (n={len(obs_after)})')

    # Segment junctions
    seg_cum = np.concatenate(([0.0], np.cumsum(lengths)))
    for k_seg, s_seg in enumerate(seg_cum):
        pt = _eval_clothoid(pieces, lengths, [s_seg])
        ax.scatter(pt[0, 0], pt[0, 1], color='gray', s=40, zorder=3)
        ax.annotate(f'  s{k_seg}={s_seg:.1f}m',
                    (pt[0, 0], pt[0, 1]), fontsize=7, color='gray')

    # Heading arrows
    for pos, theta, color, lbl in [
        (start_state[:2],  start_state[3],  'green', f'Start k={k0:.4f}'),
        (target_state[:2], target_state[3], 'red',   f'Target k={k1:.4f}'),
    ]:
        ax.annotate('',
            xy=(pos[0] + arrow_len * np.cos(theta),
                pos[1] + arrow_len * np.sin(theta)),
            xytext=(pos[0], pos[1]),
            arrowprops=dict(arrowstyle='->', color=color, lw=2))
        ax.scatter(*pos, color=color, s=90, zorder=5, label=lbl)

    # Curvature profile on twin axis
    ax_twin = ax.twinx()
    ax_twin.plot(s_fine, path_fine[:, 3], color='purple',
                 linewidth=1, linestyle=':', alpha=0.7, label='κ(s)')
    ax_twin.axhline(k0, color='green', linewidth=0.8,
                    linestyle='--', alpha=0.6, label=f'k0={k0:.4f}')
    ax_twin.axhline(k1, color='red',   linewidth=0.8,
                    linestyle='--', alpha=0.6, label=f'k1={k1:.4f}')
    ax_twin.axhline(0,  color='purple', linewidth=0.5, linestyle=':')
    ax_twin.set_ylabel('Curvature κ [1/m]', color='purple', fontsize=8)
    ax_twin.tick_params(axis='y', labelcolor='purple', labelsize=7)
    ax_twin.legend(fontsize=6, loc='upper right')

    ax.set_xlabel('x [m]', fontsize=8)
    ax.set_ylabel('y [m]', fontsize=8)
    ax.legend(fontsize=7)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show(block=False)


def _plot_gap_dynamics(pieces, lengths, S_total,
                        start_state, target_state,
                        last_avail_input, next_avail_input,
                        times, v_profile, a_profile, omega,
                        fwd_states, fwd_s, bwd_states, bwd_s,
                        v0, v1, K, T_gap, res,
                        a_min, a_max, w_min, w_max,
                        debug_title):
    """Plot Stage 2 dynamic optimization: xy trajectory + time profiles."""
    s_fine    = np.linspace(0, S_total, 500)
    path_fine = _eval_clothoid(pieces, lengths, s_fine)
    arrow_len = max(S_total * 0.07, 0.5)

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(
        f'Stage 2: Dynamic Optimization  {debug_title}\n'
        f'start v={v0:.2f} m/s   target v={v1:.2f} m/s   '
        f'T_gap={T_gap:.2f}s   K={K}   S_total={S_total:.2f}m   '
        f'optimizer success={res.success}  nit={res.nit}',
        fontsize=8
    )
    gs    = gridspec.GridSpec(4, 2, figure=fig, hspace=0.5, wspace=0.38)
    ax_xy = fig.add_subplot(gs[:, 0])
    ax_v  = fig.add_subplot(gs[0, 1])
    ax_a  = fig.add_subplot(gs[1, 1])
    ax_w  = fig.add_subplot(gs[2, 1])
    ax_s  = fig.add_subplot(gs[3, 1])

    # xy panel
    ax_xy.plot(path_fine[:, 0], path_fine[:, 1],
               'b--', linewidth=1, alpha=0.3, label='Clothoid ref')
    ax_xy.plot(fwd_states[:, 0], fwd_states[:, 1],
               'b-', linewidth=2, label='Forward rollout')
    ax_xy.plot(bwd_states[:, 0], bwd_states[:, 1],
               'r-.', linewidth=1.5, alpha=0.7, label='Backward rollout')
    sc = ax_xy.scatter(fwd_states[:, 0], fwd_states[:, 1],
                       c=fwd_states[:, 2], cmap='RdYlGn', s=18, zorder=3,
                       vmin=0, vmax=max(v0, v1, fwd_states[:, 2].max() + 0.1))
    plt.colorbar(sc, ax=ax_xy, label='Speed [m/s]', shrink=0.7)
    for pos, theta, color, lbl in [
        (start_state[:2],  start_state[3],  'green', f'Start v={v0:.2f}'),
        (target_state[:2], target_state[3], 'red',   f'Target v={v1:.2f}'),
    ]:
        ax_xy.annotate('',
            xy=(pos[0] + arrow_len * np.cos(theta),
                pos[1] + arrow_len * np.sin(theta)),
            xytext=(pos[0], pos[1]),
            arrowprops=dict(arrowstyle='->', color=color, lw=2))
        ax_xy.scatter(*pos, color=color, s=90, zorder=5, label=lbl)
    ax_xy.scatter(fwd_states[-1, 0], fwd_states[-1, 1],
                  color='orange', s=100, marker='*', zorder=6,
                  label=f'Arrived v={fwd_states[-1, 2]:.2f}')
    ax_xy.set_title('Dynamic Trajectory (xy)', fontsize=9)
    ax_xy.set_xlabel('x [m]', fontsize=8)
    ax_xy.set_ylabel('y [m]', fontsize=8)
    ax_xy.legend(fontsize=7)
    ax_xy.set_aspect('equal')
    ax_xy.grid(True, alpha=0.3)

    # Speed
    ax_v.plot(times, fwd_states[:, 2], 'b-', linewidth=2, label='v fwd')
    ax_v.plot(times, bwd_states[:, 2], 'r--', linewidth=1.5,
              alpha=0.7, label='v bwd')
    ax_v.axhline(v0, color='green', linestyle=':', linewidth=1,
                 label=f'v_start={v0:.2f}')
    ax_v.axhline(v1, color='red', linestyle=':', linewidth=1,
                 label=f'v_target={v1:.2f}')
    ax_v.axhline(0, color='k', linewidth=0.5)
    ax_v.set_ylabel('Speed [m/s]', fontsize=8)
    ax_v.set_title('Time Profiles', fontsize=9)
    ax_v.legend(fontsize=7, ncol=2)
    ax_v.grid(True, alpha=0.3)
    ax_v.set_xticklabels([])

    # Acceleration
    ax_a.plot(times[:-1], a_profile[:-1], 'r-', linewidth=2, label='a(t)')
    ax_a.axhline(last_avail_input[0], color='green', linestyle=':', linewidth=1,
                 label=f'a_last={last_avail_input[0]:.2f}')
    ax_a.axhline(next_avail_input[0], color='red', linestyle=':', linewidth=1,
                 label=f'a_next={next_avail_input[0]:.2f}')
    ax_a.axhline(a_min, color='k', linestyle='--', linewidth=0.8, alpha=0.4)
    ax_a.axhline(a_max, color='k', linestyle='--', linewidth=0.8, alpha=0.4)
    ax_a.axhline(0, color='k', linewidth=0.5)
    ax_a.set_ylabel('Accel [m/s²]', fontsize=8)
    ax_a.legend(fontsize=7, ncol=2)
    ax_a.grid(True, alpha=0.3)
    ax_a.set_xticklabels([])

    # Angular velocity
    ax_w.plot(times, omega, 'g-', linewidth=2, label='ω(t) = κ·v')
    ax_w.axhline(last_avail_input[1], color='green', linestyle=':', linewidth=1,
                 label=f'ω_last={last_avail_input[1]:.3f}')
    ax_w.axhline(next_avail_input[1], color='red', linestyle=':', linewidth=1,
                 label=f'ω_next={next_avail_input[1]:.3f}')
    ax_w.axhline(w_min, color='k', linestyle='--', linewidth=0.8, alpha=0.4)
    ax_w.axhline(w_max, color='k', linestyle='--', linewidth=0.8, alpha=0.4)
    ax_w.axhline(0, color='k', linewidth=0.5)
    ax_w.set_ylabel('Ang vel [rad/s]', fontsize=8)
    ax_w.legend(fontsize=7, ncol=2)
    ax_w.grid(True, alpha=0.3)
    ax_w.set_xticklabels([])

    # Arc-length
    ax_s.plot(times, fwd_s, 'b-', linewidth=2, label='s(t) fwd')
    ax_s.plot(times, bwd_s, 'r--', linewidth=1.5, alpha=0.7, label='s(t) bwd')
    ax_s.axhline(S_total, color='k', linestyle='--', linewidth=1,
                 label=f'S_total={S_total:.2f}m')
    ax_s.axhline(0, color='k', linewidth=0.5)
    ax_s.set_ylabel('Arc-length [m]', fontsize=8)
    ax_s.set_xlabel('Time [s]', fontsize=8)
    ax_s.legend(fontsize=7, ncol=2)
    ax_s.grid(True, alpha=0.3)
    ax_s.tick_params(labelsize=7)

    plt.tight_layout()
    plt.show(block=False)


def _plot_full_trajectory(veh_id, feat_veh_df, times,
                           gap_registry, states_kalman, states_rts):
    """
    Plot full xy trajectory showing all processing stages:
    observed (green), clothoid (orange), EKF forward (blue), RTS backward (red).
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.suptitle(
        f'Trajectory Debug — Vehicle {veh_id}\n'
        f'green=observed  gray=missing  '
        f'orange=clothoid  blue=EKF fwd  red=RTS bwd',
        fontsize=9
    )

    obs_mask  = ~feat_veh_df['missing'].to_numpy()
    miss_mask =  feat_veh_df['missing'].to_numpy()
    first_gap = list(gap_registry.keys())[0] if gap_registry else None

    ax.scatter(feat_veh_df.loc[obs_mask,  'x'],
               feat_veh_df.loc[obs_mask,  'y'],
               color='green', s=12, zorder=5, label='Observed (raw)')
    ax.scatter(feat_veh_df.loc[miss_mask, 'x'],
               feat_veh_df.loc[miss_mask, 'y'],
               color='lightgray', s=8, zorder=2, alpha=0.6, label='Missing (raw)')

    for gap_start, (_, gap_len, pieces, lengths, S_total) in gap_registry.items():
        gap_end   = gap_start + gap_len - 1
        s_fine    = np.linspace(0, S_total, max(50, gap_len * 5))
        path_fine = _eval_clothoid(pieces, lengths, s_fine)
        ax.plot(path_fine[:, 0], path_fine[:, 1], color='orange',
                linewidth=2.5, zorder=3,
                label='Clothoid (Phase 0)' if gap_start == first_gap
                else '_nolegend_')
        ax.scatter(feat_veh_df.iloc[gap_start - 1]['x'],
                   feat_veh_df.iloc[gap_start - 1]['y'],
                   color='orange', s=80, marker='^', zorder=6,
                   edgecolors='black', linewidths=0.5)
        ax.scatter(feat_veh_df.iloc[min(gap_end, len(feat_veh_df)-1)]['x'],
                   feat_veh_df.iloc[min(gap_end, len(feat_veh_df)-1)]['y'],
                   color='orange', s=80, marker='v', zorder=6,
                   edgecolors='black', linewidths=0.5)

    ax.plot(states_kalman[0, :], states_kalman[1, :],
            color='blue', linewidth=1.5, zorder=4, alpha=0.7,
            label='EKF forward')
    ax.plot(states_rts[0, :],    states_rts[1, :],
            color='red',  linewidth=1.5, zorder=4, alpha=0.7,
            label='RTS backward')

    for gap_start, (_, gap_len, _, _, _) in gap_registry.items():
        gap_end = gap_start + gap_len - 1
        lbl_ekf = 'EKF fwd (gap)' if gap_start == first_gap else '_nolegend_'
        lbl_rts = 'RTS bwd (gap)' if gap_start == first_gap else '_nolegend_'
        ax.plot(states_kalman[0, gap_start:gap_end+1],
                states_kalman[1, gap_start:gap_end+1],
                color='blue', linewidth=3.0, linestyle='--',
                zorder=4, alpha=0.9, label=lbl_ekf)
        ax.plot(states_rts[0, gap_start:gap_end+1],
                states_rts[1, gap_start:gap_end+1],
                color='red', linewidth=3.0, linestyle='--',
                zorder=4, alpha=0.9, label=lbl_rts)

        # Shaded gap region + annotation
        gap_x = states_kalman[0, gap_start:gap_end+1]
        gap_y = states_kalman[1, gap_start:gap_end+1]
        if len(gap_x) > 1:
            ax.fill(np.concatenate([gap_x, gap_x[::-1]]),
                    np.concatenate([gap_y - 0.3, (gap_y + 0.3)[::-1]]),
                    alpha=0.08, color='orange', zorder=1)
        ax.annotate(
            f'gap\n{times[gap_end] - times[gap_start]:.1f}s',
            xy=(states_kalman[0, gap_start], states_kalman[1, gap_start]),
            fontsize=7, color='darkorange',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                      alpha=0.7, edgecolor='orange')
        )

    ax.scatter(states_kalman[0, 0],  states_kalman[1, 0],
               color='black', s=120, marker='*', zorder=7,
               label='Trajectory start')
    ax.scatter(states_kalman[0, -1], states_kalman[1, -1],
               color='black', s=120, marker='X', zorder=7,
               label='Trajectory end')

    ax.set_xlabel('x [m]', fontsize=9)
    ax.set_ylabel('y [m]', fontsize=9)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc='best')
    plt.tight_layout()
    plt.show(block=False)


def _plot_time_profiles(veh_id, times, feat_veh_df,
                         gap_registry, states_kalman, states_rts, inputs_all):
    """Plot speed, acceleration, and angular velocity over time."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(f'Speed / Accel / Omega — Vehicle {veh_id}', fontsize=9)
    first_gap = list(gap_registry.keys())[0] if gap_registry else None

    # Speed
    axes[0].plot(times, feat_veh_df['speed'].to_numpy() / 3.6,
                 'g.', markersize=4, alpha=0.5, label='Observed (raw)')
    axes[0].plot(times, states_kalman[2, :], 'b-',
                 linewidth=1.5, alpha=0.7, label='EKF forward')
    axes[0].plot(times, states_rts[2, :], 'r-',
                 linewidth=1.5, alpha=0.7, label='RTS backward')
    axes[0].axhline(0, color='k', linewidth=0.5)

    # Acceleration and omega
    axes[1].plot(times, inputs_all[0, :], 'r-', linewidth=1.5, label='a(t)')
    axes[1].axhline(0, color='k', linewidth=0.5)
    axes[2].plot(times, inputs_all[1, :], 'g-', linewidth=1.5, label='ω(t)')
    axes[2].axhline(0, color='k', linewidth=0.5)

    # Gap shading
    for gap_start, (_, gap_len, _, _, _) in gap_registry.items():
        gap_end = gap_start + gap_len - 1
        lbl     = 'Gap region' if gap_start == first_gap else '_nolegend_'
        for ax in axes:
            ax.axvspan(times[gap_start], times[gap_end],
                       alpha=0.1, color='orange', label=lbl)
            lbl = '_nolegend_'

    axes[0].set_ylabel('Speed [m/s]', fontsize=8)
    axes[0].legend(fontsize=7, ncol=3)
    axes[0].grid(True, alpha=0.3)
    axes[1].set_ylabel('Accel [m/s²]', fontsize=8)
    axes[1].legend(fontsize=7)
    axes[1].grid(True, alpha=0.3)
    axes[2].set_ylabel('Ang vel [rad/s]', fontsize=8)
    axes[2].set_xlabel('Time [s]', fontsize=8)
    axes[2].legend(fontsize=7)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show(block=False)