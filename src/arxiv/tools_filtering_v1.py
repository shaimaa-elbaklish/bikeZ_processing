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

from numba import njit

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
    # =========================
    # Extract numpy arrays once
    # =========================
    x = veh_df["x"].to_numpy()
    y = veh_df["y"].to_numpy()
    t = veh_df["time"].to_numpy()
    frames = np.round(t * fps + 1e-05, decimals=0).astype(np.int64)
    # =========================
    # Time deltas
    # =========================
    delta_time = np.full_like(t, np.nan)
    delta_time[1:] = t[1:] - t[:-1]
    delta_time_fwd = _deltaForward(t, frames, speed_delta)
    delta_time_bwd = _deltaBackward(t, frames, speed_delta)
    # =========================
    # Moving average positions
    # =========================
    x_ma = _rollingMA_limHorizon(frames, x, VEHICLE_DIMENSION_MOVING_AVERAGE_WINDOW_LENGTH)
    y_ma = _rollingMA_limHorizon(frames, y, VEHICLE_DIMENSION_MOVING_AVERAGE_WINDOW_LENGTH)
    # =========================
    # Velocity
    # =========================
    dx_fwd = _deltaForward(x_ma, frames, speed_delta)
    dy_fwd = _deltaForward(y_ma, frames, speed_delta)
    dx_bwd = _deltaBackward(x_ma, frames, speed_delta)
    dy_bwd = _deltaBackward(y_ma, frames, speed_delta)
    v_x_fwd = dx_fwd / delta_time_fwd
    v_y_fwd = dy_fwd / delta_time_fwd
    v_fwd = np.sqrt(v_x_fwd**2 + v_y_fwd**2)
    v_x_bwd = dx_bwd / delta_time_bwd
    v_y_bwd = dy_bwd / delta_time_bwd
    v_bwd = np.sqrt(v_x_bwd**2 + v_y_bwd**2)
    v_fwd_ma = _rollingMA_limHorizon(frames, v_fwd, VEHICLE_DIMENSION_MOVING_AVERAGE_WINDOW_LENGTH)
    v_bwd_ma = _rollingMA_limHorizon(frames, v_bwd, VEHICLE_DIMENSION_MOVING_AVERAGE_WINDOW_LENGTH)
    v_est = np.nanmean(
        np.vstack((v_fwd_ma, v_bwd_ma)),
        axis=0
    )
    # =========================
    # Acceleration
    # =========================
    dv_fwd = _deltaForward(v_est, frames, speed_delta)
    dv_bwd = _deltaBackward(v_est, frames, speed_delta)
    a_fwd = dv_fwd / delta_time_fwd
    a_bwd = dv_bwd / delta_time_bwd
    a_fwd_ma = _rollingMA_limHorizon(frames, a_fwd, VEHICLE_DIMENSION_MOVING_AVERAGE_WINDOW_LENGTH)
    a_bwd_ma = _rollingMA_limHorizon(frames, a_bwd, VEHICLE_DIMENSION_MOVING_AVERAGE_WINDOW_LENGTH)
    a_est = np.nanmean(
        np.vstack((a_fwd_ma, a_bwd_ma)),
        axis=0
    )
    # =========================
    # Angle (from velocity)
    # =========================
    angle_fwd = np.arctan2(v_y_fwd, v_x_fwd)
    angle_fwd = np.unwrap(angle_fwd) # unwrap (manual for numba safety)
    angle_fwd_ma = _rollingMA_limHorizon(frames, angle_fwd, VEHICLE_DIMENSION_MOVING_AVERAGE_WINDOW_LENGTH)
    angle_bwd = np.arctan2(v_y_bwd, v_x_bwd)
    angle_bwd = np.unwrap(angle_bwd) # unwrap (manual for numba safety)
    angle_bwd_ma = _rollingMA_limHorizon(frames, angle_bwd, VEHICLE_DIMENSION_MOVING_AVERAGE_WINDOW_LENGTH)
    angle_est = np.nanmean(
        np.vstack((angle_fwd_ma, angle_bwd_ma)),
        axis=0
    )
    # =========================
    # Angular velocity
    # =========================
    dtheta_fwd = _deltaForward(angle_fwd_ma, frames, speed_delta)
    dtheta = np.clip(dtheta, -ANGLE_VELOCITY_THRESHOLD, ANGLE_VELOCITY_THRESHOLD) # clamp
    ang_vel = dtheta / delta_time_fwd
    ang_vel_ma = _rollingMA_limHorizon(frames, ang_vel, VEHICLE_DIMENSION_MOVING_AVERAGE_WINDOW_LENGTH)

    
    
    
    
    
    return feat_veh_df


# def _rollingMA_limHorizon(df_orig, av_field, ma_field):
#     lst_ma_val = []
#     for frame_nr in df_orig["frame_nr"].tolist():
#         df_sub = df_orig[df_orig["frame_nr"]<=frame_nr+int(VEHICLE_DIMENSION_MOVING_AVERAGE_WINDOW_LENGTH)/2]
#         df_sub = df_sub[df_sub["frame_nr"]>=frame_nr-int(VEHICLE_DIMENSION_MOVING_AVERAGE_WINDOW_LENGTH)/2]
#         lst_ma_val.append(np.nanmean(df_sub[av_field]))
#     df_orig[ma_field] = lst_ma_val
#     return df_orig


# def _deltaForward(df, col, new_col, speed_delta):
#     new_vals = []
#     vals = df[col].tolist()
#     frams = df["frame_nr"].tolist()
#     for idx in range(0, len(vals)):
#         if idx>=speed_delta:
#             this_frame = frams[idx]
#             other_frame = frams[idx-speed_delta]
#             if not abs(this_frame-other_frame)>speed_delta:                
#                 new_vals.append(vals[idx]-vals[idx-speed_delta])
#             else:
#                 foundOj = -1
#                 for oj in range(speed_delta, 0, -1):
#                     other_frame = frams[idx-oj]
#                     if not abs(this_frame-other_frame)>speed_delta:                
#                         foundOj = oj
#                         break
#                 if foundOj==-1:
#                     new_vals.append(np.nan)
#                 else:
#                     new_vals.append(vals[idx]-vals[idx-foundOj])
#         else:
#             new_vals.append(np.nan)
#     df[new_col] = new_vals
#     return df        


# def _deltaBackward(df, col, new_col, speed_delta):
#     new_vals = []
#     vals = df[col].tolist()
#     frams = df["frame_nr"].tolist()
#     for idx in range(0, len(vals)):
#         if idx<=len(vals)-speed_delta-1:
#             this_frame = frams[idx]
#             other_frame = frams[idx+speed_delta]
#             if not abs(this_frame-other_frame)>speed_delta:       
#                 new_vals.append(vals[idx]-vals[idx+speed_delta])
#             else:
#                 foundOj = -1
#                 for oj in range(speed_delta, 0, -1):
#                     other_frame = frams[idx+oj]
#                     if not abs(this_frame-other_frame)>speed_delta:                
#                         foundOj = oj
#                         break
#                 if foundOj==-1:
#                     new_vals.append(np.nan)
#                 else:
#                     new_vals.append(vals[idx]-vals[idx+foundOj])
#         else:
#             new_vals.append(np.nan)
#     df[new_col] = new_vals
#     return df


@njit
def _rollingMA_limHorizon(frames, values, window):
    n = len(values)
    out = np.empty(n)
    
    half = window // 2
    left = 0
    right = 0
    
    running_sum = 0.0
    count = 0
    
    for i in range(n):
        lower = frames[i] - half
        upper = frames[i] + half
        
        # move right pointer forward
        while right < n and frames[right] <= upper:
            if not np.isnan(values[right]):
                running_sum += values[right]
                count += 1
            right += 1
        
        # move left pointer forward
        while left < n and frames[left] < lower:
            if not np.isnan(values[left]):
                running_sum -= values[left]
                count -= 1
            left += 1
        
        if count > 0:
            out[i] = running_sum / count
        else:
            out[i] = np.nan
            
    return out


@njit
def _deltaForward(vals, frames, speed_delta):
    n = len(vals)
    out = np.full(n, np.nan)
    
    j = 0
    
    for i in range(n):
        target = frames[i] - speed_delta
        
        # move j forward until frame[j] >= target
        while j < i and frames[j] < target:
            j += 1
        
        if j < i and abs(frames[i] - frames[j]) <= speed_delta:
            out[i] = vals[i] - vals[j]
            
    return out


@njit
def _deltaBackward(vals, frames, speed_delta):
    n = len(vals)
    out = np.full(n, np.nan)
    
    j = 0
    
    for i in range(n):
        # ensure j is always ahead of i
        if j <= i:
            j = i + 1
        
        target = frames[i] + speed_delta
        
        # move j forward until frames[j] >= target
        while j < n and frames[j] < target:
            j += 1
        
        if j < n and abs(frames[j] - frames[i]) <= speed_delta:
            out[i] = vals[i] - vals[j]
            
    return out


@njit
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
        angle = np.mod(angle, 2*np.pi)
        while angle_bounded > 2*np.pi:
            angle_bounded -= 2*np.pi
    elif angle_format=="deg":
        angle = np.mod(angle, 360)
        while angle_bounded > 360:
            angle_bounded -= 360
    else:
        raise Exception("Invalid angle_format '"+str(angle_format)+"'. Supported Formats are 'rad' and 'deg'!")
    return angle_bounded


@njit
def boundAngleListPositive(arr_angles, angle_format="rad"):
    """
    This method filters any list of angles into the positive range (0 - 360°) resp. (0 - 2*PI).

    Parameters
    ----------
    arr_angles : np.ndarray[float]
        The angles to be bounded.
    angle_format : str
        The format of the angle. Options are "deg" (degree) or "rad" (radians). Default is "deg".
        
    Returns
    -------
    angle_bounded : List[float]
        The bounded angles.
    """
    for i in range(arr_angles.shape[0]):
        arr_angles[i] = boundAnglePositive(arr_angles[i], angle_format)
    return arr_angles


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