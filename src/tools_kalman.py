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
import gc
import sys
import warnings
warnings.simplefilter('ignore', RuntimeWarning) # Ignore all RuntimeWarnings

import numpy as np
import pandas as pd

from scipy.optimize import minimize

from _constants import SKIP_KALMAN_FILTERING_MAX_GAP
from tools_filtering import calculate_features, boundAnglePositive

# #############################################################################
# METHODS
# #############################################################################
def _f_dyn(x, u, dt):
    # x_arr = [x, y, v, theta]
    # u = [accel, omega]
    return np.array([
        x[0] + dt*x[2]*np.cos(x[3]),
        x[1] + dt*x[2]*np.sin(x[3]),
        max(0, x[2] + dt*u[0]),
        boundAnglePositive(x[3] + dt*u[1], "rad")
    ])


def _A_jacobian(x, u, dt):
    return np.array([
        [1, 0, dt*np.cos(x[-1]), -dt*x[2]*np.sin(x[-1])],
        [0, 1, dt*np.sin(x[-1]), dt*x[2]*np.cos(x[-1])],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])


def _B_jacobian(x, u, dt):
    return np.array([[0], [0], [dt], [dt]])


def _determine_next_available_frame(df, a, b, c):
    next_frame = -1
    for frame_nr in range(a, b, c):
        if len(df[df["frame_nr"]==frame_nr])==1:
            next_frame = frame_nr
            break
    return next_frame


def _gap_inference_objective(x, start_state, target_state, start_input, target_input, start_frame, stop_frame, fps):
    # pos_error, vel_error, ang_error = 0, 0, 0
    inp_mag, inp_changes = 0, 0
    curr_state = start_state
    prev_input = start_input
    for f in range(start_frame, stop_frame, 1):
        i = f - start_frame
        next_state = _f_dyn(curr_state, u=x[2*i:2*i+2], dt=1/fps)
        # next_input = x[2*(i+1):2*(i+1)+2] if f < stop_frame-1 else target_input
        # next_state = _f_dyn(curr_state, u=x, dt=1/fps)
        
        # interm_target = start_state + (f-start_frame+1)*(target_state - start_state)/(stop_frame - start_frame)
        # pos_error += np.linalg.norm(next_state[:2] - interm_target[:2])
        # vel_error += abs(next_state[-2] - interm_target[-2])
        # ang_error += abs(next_state[-1] - interm_target[-1])        
        inp_mag += abs(x[2*i]) + abs(x[2*i+1])
        inp_changes += abs(x[2*i] - prev_input[0]) + abs(x[2*i+1] - prev_input[1])
        if f == stop_frame - 1:
            inp_changes += abs(x[2*i] - target_input[0]) + abs(x[2*i+1] - target_input[1])
        
        curr_state = next_state
        prev_input = x[2*i:2*i+2]
        
    final_pos_error = np.linalg.norm(next_state[:2] - target_state[:2])
    final_vel_error = abs(next_state[-2] - target_state[-2])
    final_ang_error = abs(next_state[-1] - target_state[-1])
    jerk_error = (fps*abs(x[0] - start_input[0]) - 10) + (fps*abs(x[-2] - target_input[0]) - 10)
    return 10*final_pos_error + 2*final_vel_error + final_ang_error + inp_changes + 0.1*inp_mag + 10*jerk_error
            

def _gap_inference(start_state, target_state, last_available_input, next_available_input, frame_nr, next_available_frame, fps):
    possible_accel = (target_state[-2] - start_state[-2])/(next_available_frame - frame_nr) * fps
    possible_ang_vel = (target_state[-1] - start_state[-1])/(next_available_frame - frame_nr) * fps
    
    n_vars = 2*(next_available_frame - frame_nr)
    
    x_initial_guess = np.asarray([[possible_accel, possible_ang_vel] for _ in range(frame_nr, next_available_frame, 1)]).flatten()
    # x_initial_guess = np.asarray([possible_accel, possible_ang_vel])
    bnds = ((-3, 3), (-0.5, 0.5))
    res = minimize(_gap_inference_objective, x_initial_guess, method="nelder-mead",
                   args=(start_state, target_state, last_available_input, next_available_input, frame_nr, next_available_frame, fps),
                   # (x, start_state, target_state, start_input, target_input, start_frame, stop_frame, fps)
                   options={'xatol': 1e-8, 'disp': False, 'maxiter': 800*n_vars, 'maxfev': 800*n_vars},
                   bounds=tuple([b for _ in range(frame_nr, next_available_frame, 1) for b in bnds]),
    )
    # print(res)
    
    res_dict = {}
    for f in range(frame_nr, next_available_frame, 1):
        # res_dict[f] = {
        #     'accel': res.x[0],
        #     'ang_vel': res.x[1]
        # }
        res_dict[f] = {
            'accel': res.x[2*(f-frame_nr)],
            'ang_vel': res.x[2*(f-frame_nr)+1]
        }
    
    return res_dict


def calculate_kalman_filtered_trajectory(veh_df: pd.DataFrame, Q_t: np.ndarray, R_t: np.ndarray, 
                                         first_frame: int, last_frame: int, fps: float = 25.0):
    feat_veh_df = calculate_features(veh_df, fps)      
    C_t = np.diag([1, 1, 1, 1])
    I = np.eye(4)
    
    # Forward Pass
    states_kalman = np.zeros(shape=(4, last_frame-first_frame+1))
    states_kalman[:, 0] = feat_veh_df.loc[feat_veh_df['frame_nr']==first_frame, ['x', 'y', 'speed', 'angle_estimation']].to_numpy().flatten()
    states_kalman[-2, 0] /= 3.6
    states_pred = np.copy(states_kalman)
    states_cov_kalman = np.zeros(shape=(4, 4, last_frame-first_frame+1))
    states_cov_kalman[:, :, 0] = I
    states_cov_pred = np.copy(states_cov_kalman)
    accel, ang_vel, skip_counter = 0, 0, 0
    missing_inputs = {}
    for frame_nr in range(first_frame, last_frame, 1):
        # Skip Kalman Iterations if very large gap was filled by inference approach
        if skip_counter > 0:
            skip_counter -= 1
            continue
        
        i = frame_nr-first_frame
        
        # Determine measurement availability
        measurement_available = len(feat_veh_df[feat_veh_df["frame_nr"]==frame_nr])==1
        next_available_frame = _determine_next_available_frame(feat_veh_df, frame_nr+1, last_frame, 1)
        if not measurement_available:
            if (next_available_frame - frame_nr) <= SKIP_KALMAN_FILTERING_MAX_GAP:
                # Impute based on prediction step only if gap is not too large
                missing_inputs.update({frame_nr: {'accel': accel, 'ang_vel': ang_vel}})
                A_t = _A_jacobian(states_kalman[:, i], u=[accel, ang_vel], dt=1/fps)
                states_pred[:, i+1] = _f_dyn(states_kalman[:, i], u=[accel, ang_vel], dt=1/fps)
                states_cov_pred[:, :, i+1] = A_t @ states_cov_kalman[:, :, i] @ A_t.T + Q_t
                states_kalman[:, i+1] = states_pred[:, i+1]
                states_cov_kalman[:, :, i+1] = states_cov_pred[:, :, i+1]
                continue
            # Get accel and omega that reach the target (i.e. next available measurement) best
            target = feat_veh_df.loc[feat_veh_df['frame_nr']==next_available_frame, ['x', 'y', 'speed', 'angle_estimation']].to_numpy().flatten()
            target[-2] /= 3.6 # km/h to m/s
            target_accel = feat_veh_df.loc[feat_veh_df['frame_nr']==next_available_frame, 'a'].item()
            target_ang_vel = feat_veh_df.loc[feat_veh_df['frame_nr']==next_available_frame, 'angle_vel_estimation'].item()
            res = _gap_inference(states_kalman[:, i], target, np.asarray([accel, ang_vel]), np.asarray([target_accel, target_ang_vel]), frame_nr, next_available_frame, fps)
            missing_inputs.update(res)
            skip_counter = next_available_frame - frame_nr - 1
            for f in range(frame_nr, next_available_frame, 1):
                j = f - first_frame
                accel, ang_vel = missing_inputs[f]['accel'], missing_inputs[f]['ang_vel']
                A_t = _A_jacobian(states_kalman[:, j], u=[accel, ang_vel], dt=1/fps)
                states_pred[:, j+1] = _f_dyn(states_kalman[:, j], u=[accel, ang_vel], dt=1/fps)
                states_cov_pred[:, :, j+1] = A_t @ states_cov_kalman[:, :, j] @ A_t.T + Q_t
                states_kalman[:, j+1] = states_pred[:, j+1]
                states_cov_kalman[:, :, j+1] = states_cov_pred[:, :, j+1]
            continue
        
        accel = feat_veh_df.loc[feat_veh_df['frame_nr']==frame_nr, 'a'].item()
        ang_vel = feat_veh_df.loc[feat_veh_df['frame_nr']==frame_nr, 'angle_vel_estimation'].item()
        A_t = _A_jacobian(states_kalman[:, i], u=[accel, ang_vel], dt=1/fps)
        
        states_pred[:, i+1] = _f_dyn(states_kalman[:, i], u=[accel, ang_vel], dt=1/fps)
        states_cov_pred[:, :, i+1] = A_t @ states_cov_kalman[:, :, i] @ A_t.T + Q_t
        
        K_t = states_cov_pred[:, :, i+1] @ C_t.T @ np.linalg.inv(C_t @ states_cov_pred[:, :, i+1] @ C_t.T + R_t)
        y_t = feat_veh_df.loc[feat_veh_df['frame_nr']==frame_nr, ['x', 'y', 'speed', 'angle_estimation']].to_numpy().flatten()
        y_t[-2] /= 3.6 # km/h to m/s
        states_kalman[:, i+1] = states_pred[:, i+1] + K_t @ (y_t - C_t @ states_pred[:, i+1])
        states_cov_kalman[:, :, i+1] = (I - K_t @ C_t) @ states_cov_pred[:, :, i+1]
    
    
    # Backward Pass
    states_rts = np.copy(states_kalman)
    states_cov_rts = np.copy(states_cov_kalman)
    for frame_nr in range(last_frame-1, first_frame-1, -1):
        i = frame_nr-first_frame
        
        # Determine measurement availability
        measurement_available = len(feat_veh_df[feat_veh_df["frame_nr"]==frame_nr])==1
        if not measurement_available:
            accel = missing_inputs[frame_nr]['accel']
            ang_vel = missing_inputs[frame_nr]['ang_vel']
        else:
            accel = feat_veh_df.loc[feat_veh_df['frame_nr']==frame_nr, 'a'].item()
            ang_vel = feat_veh_df.loc[feat_veh_df['frame_nr']==frame_nr, 'angle_vel_estimation'].item()
        A_t = _A_jacobian(states_kalman[:, i], u=[accel, ang_vel], dt=1/fps)
        
        K_s = states_cov_kalman[:, :, i] @ A_t.T @ np.linalg.inv(states_cov_pred[:, :, i+1])
        states_rts[:, i] = states_kalman[:, i] + K_s @ (states_rts[:, i+1] - states_pred[:, i+1])
        states_cov_rts[:, :, i] = states_cov_kalman[:, :, i] + K_s @ (states_cov_rts[:, :, i+1] - states_cov_pred[:, :, i+1]) @ K_s.T
      
    filt_veh_df = pd.DataFrame(states_rts.T, columns=['x', 'y', 'speed', 'angle'])
    filt_veh_df['speed'] *= 3.6 # m/s to km/h 
    filt_veh_df['frame_nr'] = np.arange(first_frame, last_frame+1, 1)
    filt_veh_df['time'] = filt_veh_df['frame_nr'] / fps
    filt_veh_df['cov_mat'] = [states_cov_rts[:, :, i] for i in range(states_cov_rts.shape[-1])]
    filt_veh_df['cov_norm'] = np.linalg.norm(states_cov_rts, ord='fro', axis=(0,1))
    filt_veh_df['a'] = -1
    for frame_nr in range(first_frame, last_frame+1, 1):
        measurement_available = len(feat_veh_df[feat_veh_df["frame_nr"]==frame_nr])==1
        if measurement_available:
            filt_veh_df.loc[filt_veh_df['frame_nr']==frame_nr, 'a'] = feat_veh_df.loc[feat_veh_df['frame_nr']==frame_nr, 'a'].item()
        else:
            filt_veh_df.loc[filt_veh_df['frame_nr']==frame_nr, 'a'] = missing_inputs[frame_nr]['accel']
    
    return filt_veh_df