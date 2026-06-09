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

from numba import njit
from scipy.optimize import minimize

from _constants import SKIP_KALMAN_FILTERING_MAX_GAP
from tools_filtering import calculate_features
from tools_gap_inference import reconstruct_gap
from tools_gap_inference import estimate_curvature

# #############################################################################
# METHODS
# #############################################################################
@njit
def _f_dyn(x, u, dt):
    # x_arr = [x, y, v, theta]
    # u = [accel, omega]
    x_new = np.zeros(shape=(4,), dtype=np.float64)
    x_new[0] = x[0] + dt * x[2] * np.cos(x[3])
    x_new[1] = x[1] + dt * x[2] * np.sin(x[3])
    x_new[2] = np.maximum(0, x[2] + dt * u[0])
    x_new[3] = x[3] + dt * u[1]
    return x_new


@njit
def _A_jacobian(x, u, dt):
    return np.array([
        [1.0, 0.0, dt*np.cos(x[-1]), -dt*x[2]*np.sin(x[-1])],
        [0.0, 1.0, dt*np.sin(x[-1]),  dt*x[2]*np.cos(x[-1])],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ], dtype=np.float64)


@njit
def _B_jacobian(x, u, dt):
    return np.array([[0.0], [0.0], [dt], [dt]], dtype=np.float64)


@njit
def _ekf_predict(x: np.ndarray, P: np.ndarray, u: np.ndarray, 
                 Q: np.ndarray, dt: float):
    # Predict state, cov
    x_pred = _f_dyn(x, u, dt)
    A = _A_jacobian(x, u, dt)
    P_pred = A @ P @ A.T + Q
    return x_pred, P_pred


@njit
def _ekf_correct(x_pred: np.ndarray, P_pred: np.ndarray, y: np.ndarray, 
                 Ct: np.ndarray, R: np.ndarray):
    # Kalman gain
    K = P_pred @ Ct.T @ np.linalg.inv(Ct @ P_pred @ Ct.T + R)
    # Update state, cov
    x_corr = x_pred + K @ (y - Ct @ x_pred)
    I = np.eye(4)
    P_corr = (I - K) @ P_pred
    return x_corr, P_corr


@njit
def _rts_smooth(xs_filt: np.ndarray, Ps_filt: np.ndarray, 
                xs_pred: np.ndarray, Ps_pred: np.ndarray, 
                us: np.ndarray, times: np.ndarray):
    xs_smooth = np.copy(xs_filt)
    Ps_smooth = np.copy(Ps_filt)
    for i in range(len(times)-2, -1, -1):
        dt = times[i+1] - times[i]
        A_t = _A_jacobian(xs_filt[:, i], u=us[:, i], dt=dt)
        K_s = Ps_filt[:, :, i] @ A_t.T @ np.linalg.inv(Ps_pred[:, :, i+1])
        xs_smooth[:, i] = xs_filt[:, i] + K_s @ (xs_smooth[:, i+1] - xs_pred[:, i+1])
        Ps_smooth[:, :, i] = Ps_filt[:, :, i] + K_s @ (Ps_smooth[:, :, i+1] - Ps_pred[:, :, i+1]) @ K_s.T
    return xs_smooth, Ps_smooth


def calculate_kalman_filtered_trajectory(veh_df: pd.DataFrame, Q_t: np.ndarray, 
                                         R_t: np.ndarray, fps: float = 25.0):
    tmp_df = calculate_features(veh_df[~veh_df['missing']], fps)   
    feat_veh_df = veh_df.merge(tmp_df[['time', 'angle_estimation', 'angle_vel_estimation']], on=['time'], how='left')
    feat_veh_df = feat_veh_df.reset_index()
    C_t = np.diag([1, 1, 1, 1]).astype(np.float64)
    
    # Get time
    times = feat_veh_df['time'].to_numpy()
    
    # Forward Pass: EKF
    states_kalman = np.zeros(shape=(4, len(feat_veh_df)), dtype=np.float64) # not removing the missing parts
    states_kalman[:, 0] = feat_veh_df.loc[feat_veh_df['time']==times[0], ['x', 'y', 'speed', 'angle_estimation']].to_numpy().flatten()
    states_kalman[-2, 0] /= 3.6
    states_pred = np.copy(states_kalman)
    states_cov_kalman = np.zeros(shape=(4, 4, len(feat_veh_df)), dtype=np.float64)
    states_cov_kalman[:, :, 0] = np.eye(4, dtype=np.float64)
    states_cov_pred = np.copy(states_cov_kalman)
    inputs_all = np.zeros(shape=(2, len(feat_veh_df)), dtype=np.float64)
    skip_counter = 0
    
    for i in range(len(times)-1):
        # Skip Kalman Iterations if very large gap was filled by inference approach
        if skip_counter > 0:
            skip_counter -= 1
            continue
        
        dt = times[i+1] - times[i]
        
        # Determine measurement availability
        measurement_missing = feat_veh_df.loc[feat_veh_df['time'] == times[i], 'missing'].item()
        if measurement_missing:
            curr_idx = feat_veh_df[feat_veh_df['time'] == times[i]].index[0]
            next_avail_measurement_idx = feat_veh_df.loc[curr_idx:, 'missing'].idxmin()
            next_avail_time = feat_veh_df.loc[next_avail_measurement_idx, 'time']
            gap_length = next_avail_time - times[i]
            print(f"\nBicycle ID = {veh_df['veh_id'].iloc[0]}")
            print(f"Current time = {times[i]:.3f}, Next Available Time = {next_avail_time:.3f}, Gap = {gap_length:.3f} seconds.")
            print(f"Last time = {times[-1]:.3f}, Next Available Time = {next_avail_time:.3f} seconds.")
            if gap_length <= SKIP_KALMAN_FILTERING_MAX_GAP/fps:
                # Impute based on prediction step only if gap is not too large
                inputs_all[:, i] = inputs_all[:, i-1]
                states_pred[:, i+1], states_cov_pred[:, :, i+1] = _ekf_predict(
                    states_kalman[:, i], states_cov_kalman[:, :, i], inputs_all[:, i], Q_t, dt
                )
                states_kalman[:, i+1] = states_pred[:, i+1]
                states_cov_kalman[:, :, i+1] = states_cov_pred[:, :, i+1]
                continue
            
            # Get accel and omega that reach the target (i.e. next available measurement) best
            missing_times = times[times <= next_avail_time]
            missing_times = missing_times[i-1:]
            target = feat_veh_df.loc[next_avail_measurement_idx, ['x', 'y', 'speed', 'angle_estimation']].to_numpy().flatten()
            target[-2] /= 3.6 # km/h to m/s
            target_accel = feat_veh_df.loc[next_avail_measurement_idx, 'a'].item()
            target_ang_vel = feat_veh_df.loc[next_avail_measurement_idx, 'angle_vel_estimation'].item()
            
            k0 = estimate_curvature(feat_veh_df, curr_idx-1, window=5)
            k1 = estimate_curvature(feat_veh_df, next_avail_measurement_idx, window=5)
            res = reconstruct_gap(
                states_kalman[:, i-1], target, inputs_all[:, i-1], np.asarray([target_accel, target_ang_vel]), 
                missing_times, k0=k0, k1=k1, degree=8, lambda_jerk=1.0, 
                lambda_acc=100.0, lambda_beta=1.0, verbose=True
            )
            
            inputs_all[0, (times >= times[i]) & (times < next_avail_time)] = res['a'][1:-1]
            inputs_all[1, (times >= times[i]) & (times < next_avail_time)] = res['omega'][1:-1]
            skip_counter = len(res['a'][1:-1]) - 1
            for j in range(i, i+len(res['a'][1:-1])):
                dt = times[j+1] - times[j]
                states_pred[:, j+1], states_cov_pred[:, :, j+1] = _ekf_predict(
                    states_kalman[:, j], states_cov_kalman[:, :, j], inputs_all[:, j], 10*Q_t, dt
                )
                y_t = np.asarray([
                    res['x'][1+j-i],
                    res['y'][1+j-i],
                    res['v'][1+j-i],
                    res['theta'][1+j-i]
                ])
                states_kalman[:, j+1], states_cov_kalman[:, :, j+1] = _ekf_correct(
                    states_pred[:, j+1], states_cov_pred[:, :, j+1], y_t, C_t, 10*R_t
                )
            continue
        
        u_t = feat_veh_df.loc[feat_veh_df['time']==times[i], ['a', 'angle_vel_estimation']].to_numpy().flatten()
        inputs_all[:, i] = u_t
        states_pred[:, i+1], states_cov_pred[:, :, i+1] = _ekf_predict(
            states_kalman[:, i], states_cov_kalman[:, :, i], inputs_all[:, i], Q_t, dt
        )
        y_t = feat_veh_df.loc[feat_veh_df['time']==times[i], ['x', 'y', 'speed', 'angle_estimation']].to_numpy().flatten()
        y_t[-2] /= 3.6
        states_kalman[:, i+1], states_cov_kalman[:, :, i+1] = _ekf_correct(
            states_pred[:, i+1], states_cov_pred[:, :, i+1], y_t, C_t, R_t
        )
    # Last input @ len(times)-1
    u_t = feat_veh_df.loc[feat_veh_df['time']==times[-1], ['a', 'angle_vel_estimation']].to_numpy().flatten()
    inputs_all[:, -1] = u_t
    
    # Backward Pass: RTS
    states_rts, states_cov_rts = _rts_smooth(
        states_kalman, states_cov_kalman, states_pred, states_cov_pred, inputs_all, times
    )
    
    
    filt_veh_df = pd.DataFrame(states_rts.T, columns=['x', 'y', 'speed', 'angle'])
    filt_veh_df['speed'] *= 3.6 # m/s to km/h 
    filt_veh_df['time'] = times
    filt_veh_df['cov_mat'] = [states_cov_rts[:, :, i] for i in range(states_cov_rts.shape[-1])]
    filt_veh_df['cov_norm'] = np.linalg.norm(states_cov_rts, ord='fro', axis=(0,1))
    filt_veh_df['a'] = inputs_all[0, :]
        
    return filt_veh_df
