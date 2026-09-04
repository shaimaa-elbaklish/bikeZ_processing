"""
TITLE OF PAPAER
-------------------------------------------
Authors:        Shaimaa El-Baklish
Organization:   ETH Zürich, Switzerland, IVT - Institute for Transportation Planning and Systems
Development:    2025-2026
Submitted to:   JOURNAL
-------------------------------------------

Heading and angular-velocity estimation from raw (x, y) trajectories,
gap-aware: computations are restricted to contiguous observed segments
so smoothing and differencing never bleed across occlusion gaps, and
gap frames are filled by propagation rather than interpolation.
Used upstream of the Kalman filter (tools_kalman.py) to seed heading
and turn-rate estimates from raw detections.
"""

# #############################################################################
# IMPORTS
# #############################################################################
import sys
import warnings
warnings.simplefilter('ignore', RuntimeWarning) # Ignore all RuntimeWarnings

import numpy as np
import pandas as pd

from scipy.signal import savgol_filter
from scipy.ndimage import uniform_filter1d

# #############################################################################
# METHODS
# #############################################################################
def estimate_heading(df, speed_threshold=0.5, window_s=0.5, min_periods=3, 
                     fps=25, smooth_method='savgol'):
    """
    Estimate heading angle (radians, 0=East, CCW positive) per vehicle.
    
    - Uses central differences on (x, y) for heading
    - Masks low-speed frames (unreliable displacement direction)
    - Smooths with a rolling circular mean
    - Fills masked frames by forward/backward propagation
    
    Args:
        df:                DataFrame with columns [veh_id, x, y, speed, time, missing]
        speed_threshold:   below this (km/h) heading is considered unreliable
        window_s:          smoothing window in seconds
        min_periods:       minimum valid samples in smoothing window
        fps:               frames per second
        smooth_method:    'rolling' for rolling MA, 'savgol' for Savitzky-Golay, None to skip
    
    Returns:
        Copy of `df` with an added 'angle' column (radians), concatenated
        across vehicles and re-sorted to the original index order.
    """
    window = int(window_s * fps)
    if window % 2 == 0:
        window += 1

    results = []

    for veh_id, grp in df.groupby('veh_id'):
        grp      = grp.sort_values('time').copy()
        n        = len(grp)
        missing  = grp['missing'].to_numpy()
        observed = ~missing
        speed    = grp['speed'].to_numpy()
        x        = grp['x'].to_numpy()
        y        = grp['y'].to_numpy()

        heading  = np.full(n, np.nan)

        # ------------------------------------------------------------------ #
        # Step 1: Raw heading ONLY at observed frames using central           #
        # differences over observed neighbors — never crossing gap boundaries #
        # ------------------------------------------------------------------ #
        obs_idx = np.where(observed)[0]

        for k, i in enumerate(obs_idx):
            # Find previous and next OBSERVED neighbor
            i_prev = obs_idx[k-1] if k > 0           else None
            i_next = obs_idx[k+1] if k < len(obs_idx)-1 else None

            if i_prev is None and i_next is None:
                continue
            elif i_prev is None:
                # Forward difference
                dx = x[i_next] - x[i]
                dy = y[i_next] - y[i]
            elif i_next is None:
                # Backward difference
                dx = x[i] - x[i_prev]
                dy = y[i] - y[i_prev]
            else:
                # Central difference — but only if no gap between neighbors
                # i.e. i_prev and i_next are consecutive observed frames
                # If gap between them, use local one-sided difference
                gap_before = (i - i_prev) > 1   # missing frames between
                gap_after  = (i_next - i) > 1

                if not gap_before and not gap_after:
                    # True central difference — most accurate
                    dx = x[i_next] - x[i_prev]
                    dy = y[i_next] - y[i_prev]
                elif not gap_before:
                    # Gap after: use backward difference
                    dx = x[i] - x[i_prev]
                    dy = y[i] - y[i_prev]
                elif not gap_after:
                    # Gap before: use forward difference
                    dx = x[i_next] - x[i]
                    dy = y[i_next] - y[i]
                else:
                    # Isolated observed frame between two gaps — skip
                    continue

            ds = np.hypot(dx, dy)
            if ds < 5e-02 or speed[i] < speed_threshold:
                continue  # unreliable direction

            heading[i] = np.arctan2(dy, dx)   # radians

        # ------------------------------------------------------------------ #
        # Step 2: Unwrap on observed frames only                              #
        # ------------------------------------------------------------------ #
        obs_valid = np.where(~np.isnan(heading))[0]
        if len(obs_valid) > 1:
            heading[obs_valid] = np.unwrap(heading[obs_valid])

        # ------------------------------------------------------------------ #
        # Step 3: Smooth ONLY observed frames — no interpolation across gaps  #
        # Split trajectory at gap boundaries and smooth each segment          #
        # independently                                                       #
        # ------------------------------------------------------------------ #
        if smooth_method in ('savgol', 'rolling') and len(obs_valid) >= 3:
            # Find contiguous observed segments
            segments = []
            seg_start = obs_valid[0]
            for k in range(1, len(obs_valid)):
                if obs_valid[k] - obs_valid[k-1] > 1:
                    # Gap detected — close current segment
                    segments.append((seg_start, obs_valid[k-1]))
                    seg_start = obs_valid[k]
            segments.append((seg_start, obs_valid[-1]))

            for seg_s, seg_e in segments:
                seg_idx = [j for j in obs_valid
                           if seg_s <= j <= seg_e]
                if len(seg_idx) < 3:
                    continue
                h_seg = heading[seg_idx]
                w     = min(window, len(seg_idx))
                w     = w if w % 2 == 1 else w - 1
                if w < 3:
                    continue
                if smooth_method == 'savgol':
                    h_seg = savgol_filter(h_seg, window_length=w,
                                          polyorder=min(2, w-1))
                elif smooth_method == 'rolling':
                    h_seg = (pd.Series(h_seg)
                             .rolling(window=w, center=True,
                                      min_periods=min_periods)
                             .mean()
                             .to_numpy())
                heading[seg_idx] = h_seg

        # ------------------------------------------------------------------ #
        # Step 4: Re-wrap to (-pi, pi] BEFORE propagation                    #
        # ------------------------------------------------------------------ #
        valid_h = ~np.isnan(heading)
        heading[valid_h] = (heading[valid_h] + np.pi) % (2*np.pi) - np.pi

        # ------------------------------------------------------------------ #
        # Step 5: Fill missing by propagation — ffill then bfill             #
        # This correctly gives gap frames the heading of their nearest        #
        # observed neighbor, not an interpolated value across the gap         #
        # ------------------------------------------------------------------ #
        heading_series = pd.Series(heading, index=grp.index)
        heading_series = heading_series.ffill().bfill()

        grp['angle'] = heading_series
        results.append(grp)

    return pd.concat(results).sort_index()


def estimate_angular_velocity(df, heading_col='angle', time_col='time', 
                               smooth_window_s=0.2, fps=25.0, 
                               max_angvel_rad=3.0, smooth_method='rolling'):
    """
    Estimate angular velocity (rad/s) from heading angle series.
    Processes each vehicle independently to avoid cross-vehicle unwrap artifacts.
    
    - Uses central differences for derivative
    - Handles angle wraparound before differencing
    - Smooths result to reduce noise
    
    Args:
        df:               DataFrame per vehicle, sorted by time
        heading_col:      column with heading in radians
        time_col:         column with time in seconds
        smooth_window_s:  smoothing window in seconds (set 0 to disable)
        fps:              frames per second
        max_angvel_rad:   physically plausible cap for bicycle angular velocity (rad/s).
                          Values beyond this are clipped and flagged.
                          Typical bicycle sharp turn: ~1-2 rad/s. Default 3.0 is conservative.
        smooth_method:    'rolling' for rolling MA, 'savgol' for Savitzky-Golay, None to skip

    Returns:
        Copy of `df` with added 'angular_vel' (rad/s) and 'angvel_clipped'
        (bool, True where the cap was applied) columns.
    """
    results = []

    for veh_id, grp in df.groupby('veh_id'):
        grp     = grp.sort_values(time_col).copy()
        n       = len(grp)
        missing = grp['missing'].to_numpy()
        observed = ~missing
        heading  = grp[heading_col].to_numpy(dtype=float)
        times    = grp[time_col].to_numpy()

        omega    = np.zeros(n)

        # Find contiguous observed segments
        obs_idx  = np.where(observed)[0]
        if len(obs_idx) < 2:
            grp['angular_vel']  = 0.0
            grp['angvel_clipped'] = False
            results.append(grp)
            continue

        segments = []
        seg_start = obs_idx[0]
        for k in range(1, len(obs_idx)):
            if obs_idx[k] - obs_idx[k-1] > 1:
                segments.append((seg_start, obs_idx[k-1]))
                seg_start = obs_idx[k]
        segments.append((seg_start, obs_idx[-1]))

        clipped = np.zeros(n, dtype=bool)

        for seg_s, seg_e in segments:
            seg_idx = [j for j in obs_idx if seg_s <= j <= seg_e]
            if len(seg_idx) < 2:
                continue

            h_seg = heading[seg_idx]
            t_seg = times[seg_idx]

            # Unwrap within segment only
            h_seg_unwrapped = np.unwrap(h_seg)

            # Gradient within segment
            d_seg = np.gradient(h_seg_unwrapped, t_seg)

            # Clip implausible values
            clip_mask         = np.abs(d_seg) > max_angvel_rad
            d_seg             = np.clip(d_seg, -max_angvel_rad, max_angvel_rad)
            clipped[seg_idx] |= clip_mask

            # Smooth within segment only — no bleed across gap
            w = max(3, int(smooth_window_s * fps))
            w = w if w % 2 == 1 else w - 1
            w = min(w, len(seg_idx))
            w = w if w % 2 == 1 else w - 1
            if w >= 3:
                if smooth_method == 'rolling':
                    d_seg = (pd.Series(d_seg)
                             .rolling(window=w, center=True, min_periods=2)
                             .mean()
                             .fillna(method='bfill')
                             .fillna(method='ffill')
                             .to_numpy())
                elif smooth_method == 'savgol':
                    d_seg = savgol_filter(d_seg, window_length=w,
                                          polyorder=min(2, w-1))

            omega[seg_idx] = d_seg

        # Missing frames stay at omega=0 — set explicitly after all segments
        omega[missing] = 0.0

        grp['angular_vel']    = omega
        grp['angvel_clipped'] = clipped
        results.append(grp)

    return pd.concat(results).sort_index()


def _pca_heading(veh_df: pd.DataFrame) -> float:
    """
    Estimate a single dominant heading from a stationary trajectory
    using PCA on observed xy positions.
    Disambiguated by net displacement direction.
    """
    obs  = veh_df[~veh_df['missing']][['x', 'y']].to_numpy(dtype=float)
    if len(obs) < 2:
        return 0.0   # degenerate fallback
    
    centered = obs - obs.mean(axis=0)
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    axis     = Vt[0]
    
    # Disambiguate with net displacement
    net = obs[-1] - obs[0]
    if np.dot(axis, net) < 0:
        axis = -axis
    
    return float(np.arctan2(axis[1], axis[0]))
