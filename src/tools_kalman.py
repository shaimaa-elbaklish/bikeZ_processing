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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import minimize

from _constants import SPEED_ESTIMATION_HORIZON
from _constants import SKIP_KALMAN_FILTERING_MAX_GAP
from _constants import KALMAN_INITIAL_ESTIMATION_WINDOW_LENGTH
from _constants import KALMAN_TRANSIENT_PERIOD
from _constants import POST_FILTERING_KERNEL_B
from tools_filtering import calculate_features, boundAnglePositive, _filter_velocity

# #############################################################################
# CONSTANTS
# #############################################################################
C = np.asarray([[1,0,0,0,0], [0,1,0,0,0], [0,0,1,0,0], [0,0,0,1,0], [0,0,0,0,1]]) # system output matrix
I = np.eye(5)
sel_columns = ["y_1_m", "y_2_m", "y_3_m", "y_4_m", "y_5_m"]

# #############################################################################
# METHODS: Extended Kalman Filter
# #############################################################################
def calculate_kalman_filtered_trajectory(veh_df: pd.DataFrame, Qk: np.ndarray, Rk:np.ndarray, 
                                         first_frame: int, last_frame: int, fps: float = 25.0):
    feat_veh_df = calculate_features(veh_df, fps)
    kf_traj_forward = _kalman_filter(feat_veh_df, Qk, Rk, first_frame, last_frame, fps, rev=False)
    kf_traj_backward = _kalman_filter(feat_veh_df, Qk, Rk, first_frame, last_frame, fps, rev=True)
    kf_traj_rts = kf_traj_forward.copy()
    kf_traj_rts = kf_traj_rts.merge(kf_traj_backward, on="frame_nr", how="left")
    kf_traj_rts["time"] = kf_traj_rts["time_x"]
    kf_traj_rts["measurement_available"] = kf_traj_rts["measurement_available_x"]  
    kf_traj_rts = _determine_trajectory_fusion_weights(kf_traj_rts)
    kf_traj_rts["weight_sum"] = kf_traj_rts[["weight_forward", "weight_backward"]].sum(axis=1)
    kf_traj_rts["weight_forward_final"] = (kf_traj_rts["weight_sum"]-kf_traj_rts["weight_forward"])/kf_traj_rts["weight_sum"]
    kf_traj_rts["weight_backward_final"] = (kf_traj_rts["weight_sum"]-kf_traj_rts["weight_backward"])/kf_traj_rts["weight_sum"]
    for col in ["x", "y", "state1", "state2", "state3", "state4", "state5"]:
            kf_traj_rts[col] = (kf_traj_rts[col+"_x"]*kf_traj_rts["weight_forward_final"]+kf_traj_rts[col+"_y"]*kf_traj_rts["weight_backward_final"])
            kf_traj_rts[col] = kf_traj_rts[col].rolling(window=POST_FILTERING_KERNEL_B, center=True, min_periods=1).mean()
    kf_traj_rts = kf_traj_rts[["frame_nr", "time", "measurement_available", "x", "y", "state1", "state2", "state3", "state4", "state5"]]
    
    # estimate and filter speed
    kf_traj_rts['velocity_x'] = kf_traj_rts['x'].diff().shift(-1).fillna(0) * fps
    kf_traj_rts['velocity_y'] = kf_traj_rts['y'].diff().shift(-1).fillna(0) * fps
    kf_traj_rts['velocity_cartesian'] = np.sqrt(kf_traj_rts['velocity_x']**2 + kf_traj_rts['velocity_y']**2)
    kf_traj_rts = _filter_velocity(kf_traj_rts)
    kf_traj_rts['speed'] = kf_traj_rts['velocity_cartesian'] * 3.6
    kf_traj_rts = kf_traj_rts.drop(columns=['velocity_x', 'velocity_y', 'velocity_cartesian'])
    return kf_traj_rts
    

def _kalman_filter(veh_df: pd.DataFrame, Qk: np.ndarray, Rk:np.ndarray,
                   first_frame: int, last_frame: int, fps: float = 25.0,
                   rev: bool = False):
    # Determine Initial Estimate of state x0 and covariance P0
    x_0, P_0 = _get_initial_estimate(veh_df, rev)
    kalman_data = _generate_kalman_trajectory_data(veh_df, rev)
    kalman_data = kalman_data.dropna()
    last_x_p = x_0.copy()
    last_x_c = x_0.copy()
    last_P_p = P_0.copy()
    last_P_c = P_0.copy()
    last_u = None
    state_kalman = []
    last_frame_had_measurement=False
    a = first_frame
    b = last_frame
    c = 1
    if rev:
        a = last_frame
        b = first_frame
        c = -1
    skip_counter = 0
    # For each Frame conduct Kalman Filter Step
    for frame_nr in range(a, b, c):   
        # Skip Kalman Iterations if very large gap was filled by inference approach
        if skip_counter > 0:
            skip_counter -= 1
            continue
        time = frame_nr*(1/fps)
        # Determine measurement availability
        measurement_available = len(kalman_data[kalman_data["frame_nr"]==frame_nr])==1
        if not rev:
            next_available_frame = _determine_next_available_frame(kalman_data, frame_nr+1, b, c)
        else:
            next_available_frame = _determine_next_available_frame(kalman_data, frame_nr-1, b, c)
        # Assess whether observation gap to large, whether too apply inference approach
        if measurement_available:
            if next_available_frame!=-1 and abs(next_available_frame - frame_nr) >= SKIP_KALMAN_FILTERING_MAX_GAP:
                # skip following frames
                if not rev:
                    skip_counter = next_available_frame - frame_nr - 1
                else:
                    skip_counter = frame_nr - next_available_frame - 1
                # Determine Speed and Angular Velocity based on optimization, so that arc hits next observation best
                target = np.asarray(kalman_data[kalman_data["frame_nr"]==next_available_frame][[*sel_columns]].iloc[0])                 
                x_initial_guess = [last_x_p[3], last_x_p[4]]
                res = minimize(_to_optimize_function, x_initial_guess, method="nelder-mead",
                               args=(last_x_p, target, frame_nr, next_available_frame, c, True, fps),
                               # (x, state_start, state_target, frame_start, frame_end, frame_steps, crit)
                               options={'xatol': 1e-8, 'disp': False})
                last_x_p[3] = res.x[0]
                last_x_p[4] = res.x[1]
                x_initial_guess = [last_x_p[3], last_x_p[4]]
                res = minimize(_to_optimize_function, x_initial_guess, method="nelder-mead",
                               args=(last_x_p, target, frame_nr, next_available_frame, c, False, fps),
                               # (x, state_start, state_target, frame_start, frame_end, frame_steps, crit)
                               options={'xatol': 1e-8, 'disp': False})
                last_x_p[3] = res.x[0]
                last_x_p[4] = res.x[1]                
                # Fill Gaps of Series with Inference Approach
                for frame_nr2 in range(frame_nr, next_available_frame, c):
                    time = frame_nr2*(1/fps)
                    # prediction step
                    u_measured = []
                    last_u = u_measured
                    delta_time = 1/fps
                    A = _get_linearized_matrix_A(last_x_p, delta_time)
                    next_x_p = _f_func( last_x_p, u_measured, fps)
                    next_P_p = ( A @ last_P_p @ A.T ) + Qk
                    state_kalman.append([frame_nr2, time, 2, next_x_p[0], next_x_p[1],   next_x_p[0],next_x_p[1],next_x_p[2],next_x_p[3],next_x_p[4]   ])
                    last_x_p = next_x_p
                    last_P_p = next_P_p
                continue 
 
        # Prediction Step
        if measurement_available:
            u_measured = [] 
            last_u = u_measured
        else:
            u_measured = last_u
        delta_time = 1/fps
        if last_frame_had_measurement: 
            A = _get_linearized_matrix_A(last_x_c, delta_time)
            next_x_p = _f_func( last_x_c, u_measured, fps)
            next_P_p = ( A @ last_P_c @ A.T ) + Qk
        else:
            A = _get_linearized_matrix_A(last_x_p, delta_time)
            next_x_p = _f_func( last_x_p, u_measured, fps)
            next_P_p = ( A @ last_P_p @ A.T ) + Qk
        # Correction Step
        if measurement_available:
            y_measured = np.asarray(kalman_data[kalman_data["frame_nr"]==frame_nr][[*sel_columns]].iloc[0]) 
            K_c = last_P_p @ C.T @ np.linalg.inv( C @ last_P_p @ C.T + Rk )
            next_x_c = last_x_p + K_c @ ( y_measured - _h_func( last_x_p, u_measured ) )
            next_P_c = (I - K_c @ C) @ last_P_c
        # Update Data & Variables
        if measurement_available:
            state_kalman.append([frame_nr, time, measurement_available, next_x_c[0], next_x_c[1],   next_x_c[0],next_x_c[1],next_x_c[2],next_x_c[3],next_x_c[4]   ])
        else:
            state_kalman.append([frame_nr, time, measurement_available, next_x_p[0], next_x_p[1],   next_x_p[0],next_x_p[1],next_x_p[2],next_x_p[3],next_x_p[4]   ])
        last_x_p = next_x_p
        last_frame_had_measurement = False
        if measurement_available:
            last_x_c = next_x_c
            last_P_c = next_P_c
            last_frame_had_measurement = True
        last_P_p = next_P_p
    state_kalman = np.asarray(state_kalman)
    kalman_filtered_trajectory = pd.DataFrame(state_kalman, columns=["frame_nr", "time", "measurement_available", "x", "y", "state1", "state2", "state3", "state4", "state5"])
    return kalman_filtered_trajectory
    

def _get_initial_estimate(veh_df: pd.DataFrame, rev: bool = False):
    if not rev:
        cols  = ["x", "y", "angle_estimation_forward",  "v_estimation_forward",  "angle_vel_estimation_forward"]
    else:
        cols  = ["x", "y", "angle_estimation_backward", "v_estimation_backward", "angle_vel_estimation_backward"]
    colsE = ["x_ma", "y_ma", "angle_estimation_forward", "v_estimation_forward", "angle_vel_estimation_forward", "x", "y", "angle_estimation_backward", "v_estimation_backward", "angle_vel_estimation_backward"]
    # initial estimate for state
    if not rev:
        x_0 = [np.nanmean(veh_df[ cols[0] ].tolist()[SPEED_ESTIMATION_HORIZON:SPEED_ESTIMATION_HORIZON+KALMAN_INITIAL_ESTIMATION_WINDOW_LENGTH]), 
               np.nanmean(veh_df[ cols[1] ].tolist()[SPEED_ESTIMATION_HORIZON:SPEED_ESTIMATION_HORIZON+KALMAN_INITIAL_ESTIMATION_WINDOW_LENGTH]),
               np.nanmean(veh_df[ cols[2] ].tolist()[SPEED_ESTIMATION_HORIZON:SPEED_ESTIMATION_HORIZON+KALMAN_INITIAL_ESTIMATION_WINDOW_LENGTH]), 
               np.nanmean(veh_df[ cols[3] ].tolist()[SPEED_ESTIMATION_HORIZON:SPEED_ESTIMATION_HORIZON+KALMAN_INITIAL_ESTIMATION_WINDOW_LENGTH]),
               np.nanmean(veh_df[ cols[4] ].tolist()[SPEED_ESTIMATION_HORIZON:SPEED_ESTIMATION_HORIZON+KALMAN_INITIAL_ESTIMATION_WINDOW_LENGTH]), 
               ]
    else:
        x_0 = [np.nanmean(veh_df[ cols[0] ].tolist()[-KALMAN_INITIAL_ESTIMATION_WINDOW_LENGTH-SPEED_ESTIMATION_HORIZON:-SPEED_ESTIMATION_HORIZON]), 
               np.nanmean(veh_df[ cols[1] ].tolist()[-KALMAN_INITIAL_ESTIMATION_WINDOW_LENGTH-SPEED_ESTIMATION_HORIZON:-SPEED_ESTIMATION_HORIZON]),
               np.nanmean(veh_df[ cols[2] ].tolist()[-KALMAN_INITIAL_ESTIMATION_WINDOW_LENGTH-SPEED_ESTIMATION_HORIZON:-SPEED_ESTIMATION_HORIZON]), 
               np.nanmean(veh_df[ cols[3] ].tolist()[-KALMAN_INITIAL_ESTIMATION_WINDOW_LENGTH-SPEED_ESTIMATION_HORIZON:-SPEED_ESTIMATION_HORIZON]),
               np.nanmean(veh_df[ cols[4] ].tolist()[-KALMAN_INITIAL_ESTIMATION_WINDOW_LENGTH-SPEED_ESTIMATION_HORIZON:-SPEED_ESTIMATION_HORIZON]), 
               ]
    # initial estimate for state error
    state_error = veh_df[[*colsE]]
    state_error["x_err"]  = veh_df[ colsE[0] ] - veh_df[ colsE[5] ]
    state_error["y_err"]  = veh_df[ colsE[1] ] - veh_df[ colsE[6] ]
    state_error["a_err"]  = veh_df[ colsE[2] ] - veh_df[ colsE[7] ]
    state_error["v_err"]  = veh_df[ colsE[3] ] - veh_df[ colsE[8] ]
    state_error["av_err"] = veh_df[ colsE[4] ] - veh_df[ colsE[9] ]
    state_error = np.asarray(state_error[["x_err", "y_err", "a_err", "v_err", "av_err"]])
    state_error_valid = np.sum(np.isnan(state_error), axis=1)
    P_0 = np.cov(state_error[state_error_valid==0].transpose())
    return x_0, P_0


def _generate_kalman_trajectory_data(veh_df: pd.DataFrame, rev: bool =False):
    # New Description For Kalman Filtering
    data = veh_df.copy()
    data["y_1_m"] = data["x"]
    data["y_2_m"] = data["y"]
    if not rev:
        data["y_3_m"] = data["angle_estimation_forward"]
        data["y_4_m"] = data["v_estimation_forward"]
        data["y_5_m"] = data["angle_vel_estimation_forward"]
    else:
        data["y_3_m"] = data["angle_estimation_backward"]
        data["y_4_m"] = data["v_estimation_backward"]
        data["y_5_m"] = data["angle_vel_estimation_backward"]
    data = data[["frame_nr", "time", "y_1_m", "y_2_m", "y_3_m", "y_4_m", "y_5_m"]]
    return data


def _determine_next_available_frame(df, a, b, c):
    next_frame = -1
    for frame_nr in range(a, b, c):
        if len(df[df["frame_nr"]==frame_nr])==1:
            next_frame = frame_nr
            break
    return next_frame


# Linearized Matrices 
# system state matrix:    x+1 = f(x,u)    ->      A = df/dx
# output state matrix:    y+1 = h(x,u)    ->      C = dh/dx
# state = [x, y, angle,  v, angle_vel]
# units = [m, m, rad,  m/s, rad/s]
def _f_func(x, u, video_frames_per_second):
    delta_time = 1/video_frames_per_second
    x_new = [x[0] + np.cos(x[2])*x[3]*delta_time,
             x[1] - np.sin(x[2])*x[3]*delta_time,
             boundAnglePositive(x[2] + x[4]*delta_time, "rad"),
             x[3],
             x[4]]
    return np.asarray(x_new)

def _h_func(x, u):
    return np.asarray([x[0], x[1], x[2], x[3], x[4]])

def _get_linearized_matrix_A(last_x_c, delta_time):
    A = np.asarray([
        [1,0,-np.sin(last_x_c[2])*last_x_c[3]*delta_time,+np.cos(last_x_c[2])*delta_time, 0], 
        [0,1,-np.cos(last_x_c[2])*last_x_c[3]*delta_time,-np.sin(last_x_c[2])*delta_time, 0],
        [0,0,1,0,delta_time],
        [0,0,0,1,0],
        [0,0,0,0,1]
        ])
    return A

def _predictEvaluateGuess(state_start, state_target, frame_start, frame_end, frame_steps, vel, angle_vel, crit, video_frames_per_second):
    last_x_p = state_start.copy()
    last_x_p[3] = vel
    last_x_p[4] = angle_vel
    distance_travelled = []
    for frame_nr in range(frame_start, frame_end, frame_steps):
        next_x_p = _f_func( last_x_p, [], video_frames_per_second )
        distance_travelled.append(np.linalg.norm(np.asarray([last_x_p[0], last_x_p[1]]) - np.asarray([state_target[0], state_target[1]])))    
        last_x_p = next_x_p
    distance_travelled.append(np.linalg.norm(np.asarray([last_x_p[0], last_x_p[1]]) - np.asarray([state_target[0], state_target[1]])))    
    weights = np.arange(len(distance_travelled))*np.arange(len(distance_travelled))
    weights = weights/np.sum(weights)
    pos_actual = np.asarray([last_x_p[0], last_x_p[1]])
    pos_target = np.asarray([state_target[0],state_target[1]])
    angle_diff = abs(last_x_p[2]-state_target[2])
    if crit:
        return np.linalg.norm(pos_actual-pos_target)
    else:
        return 10*np.linalg.norm(pos_actual-pos_target) + angle_diff

def _to_optimize_function(x, state_start, state_target, frame_start, frame_end, frame_steps, crit, video_frames_per_second):
    return _predictEvaluateGuess(state_start, state_target, frame_start, frame_end, frame_steps, x[0], x[1], crit, video_frames_per_second)


def _determine_trajectory_fusion_weights(df):
    weights_backward = []
    weights_forward = []
    dat = df["measurement_available"].tolist()
    for i in range(0, len(dat)):
        if i<KALMAN_TRANSIENT_PERIOD:
            weights_forward.append(10000)
        else:
            weights_forward.append(1)
        if i>len(dat)-KALMAN_TRANSIENT_PERIOD:
            weights_backward.append(10000)
        else:
            weights_backward.append(1)
    df["weight_forward"] = weights_forward 
    df["weight_backward"] = weights_backward
    return df
