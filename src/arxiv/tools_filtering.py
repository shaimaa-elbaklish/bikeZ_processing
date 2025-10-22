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
import pytz
import warnings
warnings.simplefilter('ignore', RuntimeWarning) # Ignore all RuntimeWarnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
    feat_veh_df = veh_df.copy()
    feat_veh_df['frame_nr'] = feat_veh_df['time'] * fps
    feat_veh_df['frame_nr'] = feat_veh_df['frame_nr'].astype(int)
    # Calculate time features
    feat_veh_df['delta_time'] = feat_veh_df['time'].diff()
    feat_veh_df['delta_timeX'] = feat_veh_df['time'].diff(SPEED_ESTIMATION_HORIZON)
    feat_veh_df = _deltaForward(feat_veh_df, 'time', 'delta_time_forward', SPEED_ESTIMATION_HORIZON)
    feat_veh_df = _deltaBackward(feat_veh_df, 'time', 'delta_time_backward', SPEED_ESTIMATION_HORIZON)
    # Calculate Moving Average Features from Annotation
    feat_veh_df = _rollingMA_limHorizon(feat_veh_df, "x", "x_ma")
    feat_veh_df = _rollingMA_limHorizon(feat_veh_df, "y", "y_ma")
    # Calculate Speed Features
    feat_veh_df = _deltaForward(feat_veh_df,  "x_ma", "delta_x_forward", SPEED_ESTIMATION_HORIZON)
    feat_veh_df = _deltaForward(feat_veh_df,  "y_ma", "delta_y_forward", SPEED_ESTIMATION_HORIZON)
    feat_veh_df = _deltaBackward(feat_veh_df, "x_ma", "delta_x_backward", SPEED_ESTIMATION_HORIZON)
    feat_veh_df = _deltaBackward(feat_veh_df, "y_ma", "delta_y_backward", SPEED_ESTIMATION_HORIZON)
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
    # Calculate Angle Features
        # Estimated Angle Based on Trajectory - Forward
    # feat_veh_df["angle_estim1_forward"] = np.arctan(feat_veh_df["v_y_forward"]/feat_veh_df["v_x_forward"]) 
    feat_veh_df["angle_estim1_forward"] = np.atan2(feat_veh_df["v_y_forward"], feat_veh_df["v_x_forward"]) 
    feat_veh_df["angle_estim1_forward"] = boundAngleListPositive(feat_veh_df["angle_estim1_forward"], "rad")
    # feat_veh_df["angle_estim2_forward"] = np.arcsin(feat_veh_df["v_y_forward"]/feat_veh_df["v_forward"]) 
    # feat_veh_df["angle_estim2_forward"] = boundAngleListPositive(feat_veh_df["angle_estim2_forward"], "rad")
    # feat_veh_df["angle_estim3_forward"] = np.arccos(feat_veh_df["v_x_forward"]/feat_veh_df["v_forward"]) 
    # feat_veh_df["angle_estim3_forward"] = boundAngleListPositive(feat_veh_df["angle_estim3_forward"], "rad")
    # feat_veh_df["angle_estim_final_forward"] = _estimateBestAngle(feat_veh_df["angle_estim3_forward"], feat_veh_df["angle_estim2_forward"])
    # feat_veh_df["angle_estimation_forward"] = feat_veh_df["angle_estim_final_forward"]
    feat_veh_df["angle_estimation_forward"] = feat_veh_df["angle_estim1_forward"]
    feat_veh_df = _rollingMA_limHorizon(feat_veh_df, "angle_estimation_forward", "angle_estimation_forward")
    feat_veh_df = _deltaForward(feat_veh_df,  "angle_estimation_forward", "angle_vel_estimation_forward", SPEED_ESTIMATION_HORIZON)
    feat_veh_df["angle_vel_estimation_forward"] = [angle if abs(angle)<ANGLE_VELOCITY_THRESHOLD else ANGLE_VELOCITY_THRESHOLD for angle in feat_veh_df["angle_vel_estimation_forward"]]
    feat_veh_df["angle_vel_estimation_forward"] = feat_veh_df["angle_vel_estimation_forward"]/feat_veh_df["delta_time_forward"]  
    feat_veh_df = _rollingMA_limHorizon(feat_veh_df, "angle_vel_estimation_forward", "angle_vel_estimation_forward")
        # Estimated Angle Based on Trajectory - Backward
    # feat_veh_df["angle_estim1_backward"] = np.arctan(feat_veh_df["v_y_backward"]/feat_veh_df["v_x_backward"]) 
    feat_veh_df["angle_estim1_backward"] = np.atan2(feat_veh_df["v_y_backward"], feat_veh_df["v_x_backward"]) 
    feat_veh_df["angle_estim1_backward"] = boundAngleListPositive(feat_veh_df["angle_estim1_backward"], "rad")
    # feat_veh_df["angle_estim2_backward"] = np.arcsin(feat_veh_df["v_y_backward"]/feat_veh_df["v_backward"]) 
    # feat_veh_df["angle_estim2_backward"] = boundAngleListPositive(feat_veh_df["angle_estim2_backward"], "rad")
    # feat_veh_df["angle_estim3_backward"] = np.arccos(feat_veh_df["v_x_backward"]/feat_veh_df["v_backward"]) 
    # feat_veh_df["angle_estim3_backward"] = boundAngleListPositive(feat_veh_df["angle_estim3_backward"], "rad")
    # feat_veh_df["angle_estim_final_backward"] = _estimateBestAngle(feat_veh_df["angle_estim3_backward"], feat_veh_df["angle_estim2_backward"])
    # feat_veh_df["angle_estimation_backward"] = feat_veh_df["angle_estim_final_backward"]
    feat_veh_df["angle_estimation_backward"] = feat_veh_df["angle_estim1_backward"]
    feat_veh_df = _rollingMA_limHorizon(feat_veh_df, "angle_estimation_backward", "angle_estimation_backward")
    feat_veh_df = _deltaForward(feat_veh_df,  "angle_estimation_backward", "angle_vel_estimation_backward", SPEED_ESTIMATION_HORIZON)
    feat_veh_df["angle_vel_estimation_backward"] = [angle if abs(angle)<ANGLE_VELOCITY_THRESHOLD else ANGLE_VELOCITY_THRESHOLD for angle in feat_veh_df["angle_vel_estimation_backward"]]
    feat_veh_df["angle_vel_estimation_backward"] = feat_veh_df["angle_vel_estimation_backward"]/feat_veh_df["delta_time_backward"]  
    feat_veh_df = _rollingMA_limHorizon(feat_veh_df, "angle_vel_estimation_backward", "angle_vel_estimation_backward")
    return feat_veh_df

def _rollingMA_limHorizon(df_orig, av_field, ma_field):
    lst_ma_val = []
    for frame_nr in df_orig["frame_nr"].tolist():
        df_sub = df_orig[df_orig["frame_nr"]<=frame_nr+int(VEHICLE_DIMENSION_MOVING_AVERAGE_WINDOW_LENGTH)/2]
        df_sub = df_sub[df_sub["frame_nr"]>=frame_nr-int(VEHICLE_DIMENSION_MOVING_AVERAGE_WINDOW_LENGTH)/2]
        lst_ma_val.append(np.nanmean(df_sub[av_field]))
    df_orig[ma_field] = lst_ma_val
    return df_orig


def _deltaForward(df, col, new_col, SPEED_ESTIMATION_HORIZON):
    new_vals = []
    vals = df[col].tolist()
    frams = df["frame_nr"].tolist()
    for idx in range(0, len(vals)):
        if idx>=SPEED_ESTIMATION_HORIZON:
            this_frame = frams[idx]
            other_frame = frams[idx-SPEED_ESTIMATION_HORIZON]
            if not abs(this_frame-other_frame)>SPEED_ESTIMATION_HORIZON:                
                new_vals.append(vals[idx]-vals[idx-SPEED_ESTIMATION_HORIZON])
            else:
                foundOj = -1
                for oj in range(SPEED_ESTIMATION_HORIZON, 0, -1):
                    other_frame = frams[idx-oj]
                    if not abs(this_frame-other_frame)>SPEED_ESTIMATION_HORIZON:                
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


def _deltaBackward(df, col, new_col, SPEED_ESTIMATION_HORIZON):
    new_vals = []
    vals = df[col].tolist()
    frams = df["frame_nr"].tolist()
    for idx in range(0, len(vals)):
        if idx<=len(vals)-SPEED_ESTIMATION_HORIZON-1:
            this_frame = frams[idx]
            other_frame = frams[idx+SPEED_ESTIMATION_HORIZON]
            if not abs(this_frame-other_frame)>SPEED_ESTIMATION_HORIZON:       
                new_vals.append(vals[idx]-vals[idx+SPEED_ESTIMATION_HORIZON])
            else:
                foundOj = -1
                for oj in range(SPEED_ESTIMATION_HORIZON, 0, -1):
                    other_frame = frams[idx+oj]
                    if not abs(this_frame-other_frame)>SPEED_ESTIMATION_HORIZON:                
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