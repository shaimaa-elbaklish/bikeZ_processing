"""
TITLE OF PAPAER
-------------------------------------------
Authors:        Shaimaa El-Baklish
Organization:   ETH Zürich, Switzerland, IVT - Institute for Transportation Planning and Systems
Development:    2025
Submitted to:   JOURNAL
-------------------------------------------
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

from _constants import SPEED_ESTIMATION_HORIZON
from _constants import VEHICLE_DIMENSION_MOVING_AVERAGE_WINDOW_LENGTH
from _constants import ANGLE_VELOCITY_THRESHOLD
from _constants import PROCESSING_MAX_VELOCITY
from _constants import POST_FILTERING_KERNEL_A
from _constants import PROCESSING_THR_VELOCITY

# #############################################################################
# METHODS
# #############################################################################
def calculate_features(veh_df: pd.DataFrame, fps: float = 25.0):
    speed_delta = SPEED_ESTIMATION_HORIZON if len(veh_df) > SPEED_ESTIMATION_HORIZON else 1
    feat_veh_df = veh_df.copy()
    feat_veh_df['frame_nr'] = np.round(feat_veh_df['time'] * fps + 1e-05, decimals=0)
    feat_veh_df['frame_nr'] = feat_veh_df['frame_nr'].astype(int)
    # Calculate time features
    feat_veh_df['delta_time'] = feat_veh_df['time'].diff()
    feat_veh_df['delta_timeX'] = feat_veh_df['time'].diff(speed_delta)
    feat_veh_df = _deltaForward(feat_veh_df, 'time', 'delta_time_forward', speed_delta)
    feat_veh_df = _deltaBackward(feat_veh_df, 'time', 'delta_time_backward', speed_delta)
    # Calculate Moving Average Features from Annotation
    feat_veh_df = _rollingMA_limHorizon(feat_veh_df, "x", "x_ma")
    feat_veh_df = _rollingMA_limHorizon(feat_veh_df, "y", "y_ma")
    # Calculate Speed Features
    feat_veh_df = _deltaForward(feat_veh_df,  "x_ma", "delta_x_forward", speed_delta)
    feat_veh_df = _deltaForward(feat_veh_df,  "y_ma", "delta_y_forward", speed_delta)
    feat_veh_df = _deltaBackward(feat_veh_df, "x_ma", "delta_x_backward", speed_delta)
    feat_veh_df = _deltaBackward(feat_veh_df, "y_ma", "delta_y_backward", speed_delta)
    feat_veh_df = _rollingMA_limHorizon(feat_veh_df, "delta_x_forward", "delta_x_forward")
    feat_veh_df = _rollingMA_limHorizon(feat_veh_df, "delta_y_forward", "delta_y_forward")
    feat_veh_df = _rollingMA_limHorizon(feat_veh_df, "delta_x_backward", "delta_x_backward")
    feat_veh_df = _rollingMA_limHorizon(feat_veh_df, "delta_y_backward", "delta_y_backward")
    feat_veh_df["v_x_forward"] = feat_veh_df["delta_x_forward"] / feat_veh_df["delta_time_forward"]
    feat_veh_df["v_y_forward"] = feat_veh_df["delta_y_forward"] / feat_veh_df["delta_time_forward"]
    feat_veh_df["v_forward"] = np.sqrt(feat_veh_df["v_x_forward"]**2 + feat_veh_df["v_y_forward"]**2)
    feat_veh_df["v_x_backward"] = feat_veh_df["delta_x_backward"] / feat_veh_df["delta_time_backward"]
    feat_veh_df["v_y_backward"] = feat_veh_df["delta_y_backward"] / feat_veh_df["delta_time_backward"]
    feat_veh_df["v_backward"] = np.sqrt(feat_veh_df["v_x_backward"]**2 + feat_veh_df["v_y_backward"]**2)
    feat_veh_df = _rollingMA_limHorizon(feat_veh_df, "v_backward", "v_backward_ma")
    feat_veh_df = _rollingMA_limHorizon(feat_veh_df, "v_forward", "v_forward_ma")
    feat_veh_df["v_estimation_backward"] = -feat_veh_df["v_backward_ma"]
    feat_veh_df["v_estimation_forward"] = feat_veh_df["v_forward_ma"] 
    tmp_arr = np.stack([feat_veh_df['v_estimation_forward'].to_numpy(), -feat_veh_df['v_estimation_backward'].to_numpy()], axis=-1)
    feat_veh_df['v_estimation']  = np.nanmean(tmp_arr, axis=-1)
    # Calculate Acceleration Features
    feat_veh_df = _deltaForward(feat_veh_df,  "v_estimation", "delta_v_forward", speed_delta)
    feat_veh_df = _deltaBackward(feat_veh_df,  "v_estimation", "delta_v_backward", speed_delta)
    feat_veh_df = _rollingMA_limHorizon(feat_veh_df, "delta_v_forward", "delta_v_forward")
    feat_veh_df = _rollingMA_limHorizon(feat_veh_df, "delta_v_backward", "delta_v_backward")
    feat_veh_df["a_forward"] = feat_veh_df["delta_v_forward"] / feat_veh_df["delta_time_forward"]
    feat_veh_df["a_backward"] = feat_veh_df["delta_v_backward"] / feat_veh_df["delta_time_backward"]
    feat_veh_df = _rollingMA_limHorizon(feat_veh_df, "a_backward", "a_backward_ma")
    feat_veh_df = _rollingMA_limHorizon(feat_veh_df, "a_forward", "a_forward_ma")
    feat_veh_df["a_estimation_backward"] = -feat_veh_df["a_backward_ma"]
    feat_veh_df["a_estimation_forward"] = feat_veh_df["a_forward_ma"] 
    tmp_arr = np.stack([feat_veh_df['a_estimation_forward'].to_numpy(), -feat_veh_df['a_estimation_backward'].to_numpy()], axis=-1)
    feat_veh_df['a_estimation']  = np.nanmean(tmp_arr, axis=-1)
    
    # Calculate Angle Features
        # Estimated Angle Based on Trajectory - Forward
    feat_veh_df["angle_estim1_forward"] = np.atan2(feat_veh_df["v_y_forward"], feat_veh_df["v_x_forward"]) 
    frame_start = feat_veh_df.loc[feat_veh_df['angle_estim1_forward'].isna(), 'frame_nr'].max() + 1
    angles = feat_veh_df.loc[feat_veh_df["frame_nr"] >= frame_start, "angle_estim1_forward"].to_numpy()
    angles = np.unwrap(angles, period = 2*np.pi)
    feat_veh_df.loc[feat_veh_df["frame_nr"] >= frame_start, "angle_estim1_forward"] = angles
    feat_veh_df["angle_estimation_forward"] = feat_veh_df["angle_estim1_forward"]
    feat_veh_df = _rollingMA_limHorizon(feat_veh_df, "angle_estimation_forward", "angle_estimation_forward")
    
    # feat_veh_df["angle_estim1_forward"] = boundAngleListPositive(feat_veh_df["angle_estim1_forward"], "rad")
    # feat_veh_df["angle_estimation_forward"] = feat_veh_df["angle_estim1_forward"]
    # feat_veh_df = _rollingMA_limHorizon(feat_veh_df, "angle_estimation_forward", "angle_estimation_forward")
    
    feat_veh_df = _deltaForward(feat_veh_df,  "angle_estimation_forward", "angle_vel_estimation_forward", speed_delta)
    feat_veh_df["angle_vel_estimation_forward"] = [angle if abs(angle)<ANGLE_VELOCITY_THRESHOLD else ANGLE_VELOCITY_THRESHOLD for angle in feat_veh_df["angle_vel_estimation_forward"]]
    feat_veh_df["angle_vel_estimation_forward"] = feat_veh_df["angle_vel_estimation_forward"]/feat_veh_df["delta_time_forward"]  
    feat_veh_df = _rollingMA_limHorizon(feat_veh_df, "angle_vel_estimation_forward", "angle_vel_estimation_forward")
        # Estimated Angle Based on Trajectory - Backward
    feat_veh_df["angle_estim1_backward"] = np.atan2(feat_veh_df["v_y_backward"], feat_veh_df["v_x_backward"]) 
    frame_end = feat_veh_df.loc[feat_veh_df['angle_estim1_backward'].isna(), 'frame_nr'].min() - 1
    angles = feat_veh_df.loc[feat_veh_df["frame_nr"] <= frame_end, "angle_estim1_backward"].to_numpy()
    angles = np.unwrap(angles, period = 2*np.pi)
    feat_veh_df.loc[feat_veh_df["frame_nr"] <= frame_end, "angle_estim1_backward"] = angles
    feat_veh_df["angle_estimation_backward"] = feat_veh_df["angle_estim1_backward"]
    feat_veh_df = _rollingMA_limHorizon(feat_veh_df, "angle_estimation_backward", "angle_estimation_backward")
    
    # feat_veh_df["angle_estim1_backward"] = boundAngleListPositive(feat_veh_df["angle_estim1_backward"], "rad")
    # feat_veh_df["angle_estimation_backward"] = feat_veh_df["angle_estim1_backward"]
    # feat_veh_df = _rollingMA_limHorizon(feat_veh_df, "angle_estimation_backward", "angle_estimation_backward")
    
    feat_veh_df = _deltaForward(feat_veh_df,  "angle_estimation_backward", "angle_vel_estimation_backward", speed_delta)
    feat_veh_df["angle_vel_estimation_backward"] = [angle if abs(angle)<ANGLE_VELOCITY_THRESHOLD else ANGLE_VELOCITY_THRESHOLD for angle in feat_veh_df["angle_vel_estimation_backward"]]
    feat_veh_df["angle_vel_estimation_backward"] = feat_veh_df["angle_vel_estimation_backward"]/feat_veh_df["delta_time_backward"]  
    feat_veh_df = _rollingMA_limHorizon(feat_veh_df, "angle_vel_estimation_backward", "angle_vel_estimation_backward")
    
    feat_veh_df['angle_estimation'] = np.nanmean(feat_veh_df[['angle_estimation_backward', 'angle_estimation_forward']].to_numpy(), axis=-1)
    tmp_arr = np.stack([feat_veh_df['angle_vel_estimation_forward'].to_numpy(), -feat_veh_df['angle_vel_estimation_backward'].to_numpy()], axis=-1)
    feat_veh_df['angle_vel_estimation']  = np.nanmean(tmp_arr, axis=-1)
    return feat_veh_df


def _rollingMA_limHorizon(df_orig, av_field, ma_field):
    lst_ma_val = []
    for frame_nr in df_orig["frame_nr"].tolist():
        df_sub = df_orig[df_orig["frame_nr"]<=frame_nr+int(VEHICLE_DIMENSION_MOVING_AVERAGE_WINDOW_LENGTH)/2]
        df_sub = df_sub[df_sub["frame_nr"]>=frame_nr-int(VEHICLE_DIMENSION_MOVING_AVERAGE_WINDOW_LENGTH)/2]
        lst_ma_val.append(np.nanmean(df_sub[av_field]))
    df_orig[ma_field] = lst_ma_val
    return df_orig


def _deltaForward(df, col, new_col, speed_delta):
    new_vals = []
    vals = df[col].tolist()
    frams = df["frame_nr"].tolist()
    for idx in range(0, len(vals)):
        if idx>=speed_delta:
            this_frame = frams[idx]
            other_frame = frams[idx-speed_delta]
            if not abs(this_frame-other_frame)>speed_delta:                
                new_vals.append(vals[idx]-vals[idx-speed_delta])
            else:
                foundOj = -1
                for oj in range(speed_delta, 0, -1):
                    other_frame = frams[idx-oj]
                    if not abs(this_frame-other_frame)>speed_delta:                
                        foundOj = oj
                        break
                if foundOj==-1:
                    new_vals.append(np.nan)
                else:
                    new_vals.append(vals[idx]-vals[idx-foundOj])
        else:
            new_vals.append(np.nan)
    df[new_col] = new_vals
    return df        


def _deltaBackward(df, col, new_col, speed_delta):
    new_vals = []
    vals = df[col].tolist()
    frams = df["frame_nr"].tolist()
    for idx in range(0, len(vals)):
        if idx<=len(vals)-speed_delta-1:
            this_frame = frams[idx]
            other_frame = frams[idx+speed_delta]
            if not abs(this_frame-other_frame)>speed_delta:       
                new_vals.append(vals[idx]-vals[idx+speed_delta])
            else:
                foundOj = -1
                for oj in range(speed_delta, 0, -1):
                    other_frame = frams[idx+oj]
                    if not abs(this_frame-other_frame)>speed_delta:                
                        foundOj = oj
                        break
                if foundOj==-1:
                    new_vals.append(np.nan)
                else:
                    new_vals.append(vals[idx]-vals[idx+foundOj])
        else:
            new_vals.append(np.nan)
    df[new_col] = new_vals
    return df


def boundAnglePositive(angle, angle_format="rad"):
    """
    This method filters any angle into the positive range (0 - 360°) resp. (0 - 2*PI).

    Parameters
    ----------
    angle : float
        The angle to be bounded.
    angle_format : str
        The format of the angle. Options are "deg" (degree) or "rad" (radians). Default is "deg".
        
    Returns
    -------
    angle_bounded : float
        The bounded angle.
    """
    angle_bounded = angle
    if angle_format=="rad":
        while angle_bounded <0:
            angle_bounded += 2*np.pi
        while angle_bounded > 2*np.pi:
            angle_bounded -= 2*np.pi
    elif angle_format=="deg":
        while angle_bounded <0:
            angle_bounded += 360
        while angle_bounded > 360:
            angle_bounded -= 360
    else:
        raise Exception("Invalid angle_format '"+str(angle_format)+"'. Supported Formats are 'rad' and 'deg'!")
    return angle_bounded


def boundAngleListPositive(lst_angles, angle_format="deg"):
    """
    This method filters any list of angles into the positive range (0 - 360°) resp. (0 - 2*PI).

    Parameters
    ----------
    lst_angle : List[float]
        The angles to be bounded.
    angle_format : str
        The format of the angle. Options are "deg" (degree) or "rad" (radians). Default is "deg".
        
    Returns
    -------
    angle_bounded : List[float]
        The bounded angles.
    """
    lst_angles_bounded = []
    for angle in lst_angles:
        lst_angles_bounded.append(boundAnglePositive(angle, angle_format))
    return lst_angles_bounded


def _estimateBestAngle(lst_angle_cos, lst_angle_sin):
    lst_angle_cos = lst_angle_cos.tolist()
    lst_angle_sin = lst_angle_sin.tolist()
    lst_new = []
    for idx in range(0, len(lst_angle_cos)):
        if lst_angle_sin[idx] < np.pi:
            lst_new.append(np.pi + (np.pi - lst_angle_cos[idx]) )
        else:
            lst_new.append(lst_angle_cos[idx])

    return lst_new


def _filter_velocity(veh_df):
    # VELOCITY (State 4)
    # Velocity MAX Capping
    veh_df["velocity_cartesian"] = veh_df["velocity_cartesian"].clip(upper=PROCESSING_MAX_VELOCITY)
    # Velocity Outlier Correction
    veh_df["velocity_cartesian"] = _filter_median_deviation(veh_df["velocity_cartesian"], kernel_size=int(len(veh_df)*POST_FILTERING_KERNEL_A), threshold=PROCESSING_THR_VELOCITY)
    # Velocity Tail Correction
    to_idx = int(len(veh_df)*POST_FILTERING_KERNEL_A)
    const_val = veh_df["velocity_cartesian"].iloc[to_idx]
    veh_df.iloc[0:to_idx+1, veh_df.columns.get_loc("velocity_cartesian")] = const_val
    to_idx = len(veh_df)-int(len(veh_df)*POST_FILTERING_KERNEL_A)
    if to_idx >= len(veh_df):
        return veh_df
    const_val = veh_df["velocity_cartesian"].iloc[to_idx]
    veh_df.iloc[to_idx+1:, veh_df.columns.get_loc("velocity_cartesian")] = const_val
    return veh_df


def _filter_median_deviation(series, kernel_size, threshold):
    if kernel_size <= 0:
        kernel_size = 1
    rolling_median = series.rolling(window=kernel_size, center=True, min_periods=1).median()
    deviation = np.abs(series - rolling_median)
    mask = deviation > threshold
    # Calculate weights
    weight_series = 1 / (np.maximum(deviation - threshold, 0) + 1)
    weight_median = 1 - weight_series
    # Apply weighted average where deviation > threshold
    filtered = pd.Series(index=series.index)
    filtered[mask] = (series[mask] * weight_series[mask] + 
                      rolling_median[mask] * weight_median[mask])
    filtered[~mask] = series[~mask]
    return filtered


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
            if ds < 1e-6 or speed[i] < speed_threshold:
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
