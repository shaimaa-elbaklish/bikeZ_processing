"""
TITLE OF PAPAER
-------------------------------------------
Authors:        Shaimaa El-Baklish
Organization:   ETH Zürich, Switzerland, IVT - Institute for Transportation Planning and Systems
Development:    2025-2026
Submitted to:   JOURNAL
-------------------------------------------

Command-line entry point: runs the EKF + clothoid gap-inference trajectory
filter (tools_kalman.calculate_kalman_filtered_trajectory) on one
(date, mode, intersection, code, timeslot) recording, and writes the
filtered trajectory to a .parquet file alongside MAE/RMSE summary stats.

Usage: python main_kf.py <date> <mode> <intersection> <code> <timeslot> <debug>
"""

# #############################################################################
# IMPORTS
# #############################################################################
import gc
import sys
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm

from _logger import Logger
from _constants import BikeZ_Config 
from tools_filtering import estimate_heading
from tools_filtering import estimate_angular_velocity
from tools_filtering import _pca_heading
from tools_kalman import calculate_kalman_filtered_trajectory


# #############################################################################
# CONSTANTS
# #############################################################################
# Configuration
BikeZ_Config = BikeZ_Config()

parser = argparse.ArgumentParser(description="EKF+Gap Inference for BikeZ trajectories")
parser.add_argument("date",          type=str, help="Date string, e.g. 2025-06-16")
parser.add_argument("mode",          type=str, help="Mode: bike or vehicle")
parser.add_argument("intersection",  type=str, help="Intersection ID, e.g. D3")
parser.add_argument("code",          type=str, help="Code letter, e.g. E")
parser.add_argument("timeslot",      type=str, help="Timeslot, e.g. AM1")
parser.add_argument("debug",         type=str, help="Enable debug plots: True or False")
args = parser.parse_args()

date         = args.date
mode         = args.mode
intersection = args.intersection
code         = args.code
timeslot     = args.timeslot
debug_mode   = args.debug.lower() == "true"

campaign  = f"Zurich_2025{date[5:7]}"
data_root = BikeZ_Config.data_root[campaign][mode]

XY_2056_Bounds = BikeZ_Config.XY_2056_Bounds[date][(intersection, code)]
X_2056_offset = XY_2056_Bounds[0][0]
Y_2056_offset = XY_2056_Bounds[1][0]

log = Logger(date, intersection, code, timeslot, f"KF_{mode}")

# #############################################################################
# MAIN
# #############################################################################
if mode == "bike":
    filename = f"trajectories_bikes_{date}_{intersection}_{timeslot}_{code}-1.csv"
else:
    filename = f"trajectories_vehicles_{date}_{intersection}_{timeslot}_{code}-1.csv"
df = pd.read_csv(data_root + f"{date}/{intersection}/{filename}")
# COLUMNS: ['veh_id', 'veh_type', 'speed(km/h)', 'a(m/s2)', 'time(s)', 'X_2056(m)', 'Y_2056(m)', 'longitude', 'latitude', 'datetime']
# add a column as a missing flag
df['missing'] = (df['speed(km/h)'] == -1)
# print(df.loc[df['missing'], 'veh_id'].unique())

df = df.rename(columns={
    'speed(km/h)': 'speed',
    'a(m/s2)': 'a',
    'time(s)': 'time',
    'X_2056(m)': 'x_act',
    'Y_2056(m)': 'y_act',
    'longitude': 'lon',
    'latitude': 'lat'
})
df['x'] = df['x_act'] - X_2056_offset
df['y'] = df['y_act'] - Y_2056_offset
df['datetime'] = pd.to_datetime(df['datetime'], format='ISO8601')

# Fix time = -1 issues
# Find ref. datetime (i.e. datetime when time == 0)
ref_datetime = df['datetime'].min()
ref_time = df.loc[(df['datetime'] == ref_datetime) & (df['time'] >= 0), 'time'].unique()[0]
df['time'] = df['datetime'].apply(lambda x: np.round((x - ref_datetime).total_seconds() + ref_time, decimals=3))
df = df.sort_values(by=['veh_id', 'time'], ascending=True)


# Estimate heading angle (radians)
df = estimate_heading(df, speed_threshold=1.0, window_s=0.8, fps=BikeZ_Config.fps, smooth_method='savgol')
# Estimate angular velocity (rad/s)
df = estimate_angular_velocity(df, smooth_window_s=0.4, fps=BikeZ_Config.fps, smooth_method='rolling')
# Handle stationary vehicles
for veh_id, veh_df in df.groupby('veh_id'):
    if veh_df['angle'].isna().all():
        h = _pca_heading(veh_df)
        df.loc[df['veh_id'] == veh_id, 'angle']       = h
        df.loc[df['veh_id'] == veh_id, 'angular_vel'] = 0.0
        log.warning(
            f'veh={veh_id}: stationary, no heading data — '
            f'PCA heading={np.degrees(h):.1f}° assigned as constant.'
        )

# #############################################################################
# MAIN: Perform EKF for all bicycles
# #############################################################################
Qk = np.diag([1.0, 1.0, 1.0, 10.0]).astype(np.float64)   # covariance matrix of error of state
Rk = np.diag([1.0, 1.0, 2.0, 10.0]).astype(np.float64)   # covariance matrix of error of output

filt_df = None
# unique_ids = [35, 86, 22, 72, 152, 161] # test
unique_ids = df['veh_id'].unique()
for veh_id in tqdm(unique_ids, desc="Processing EKF on Bicycles"):
    veh_df = df[df['veh_id'] == veh_id].copy()
    veh_df = veh_df.sort_values(by='time', ascending=True)
    filt_bike_df = calculate_kalman_filtered_trajectory(
        veh_df, Qk, Rk, fps=BikeZ_Config.fps, debug=debug_mode, log=log
    )
    filt_bike_df = filt_bike_df[['time', 'x', 'y', 'speed', 'angle', 
                                 'a', 'angular_vel']]
    filt_bike_df = filt_bike_df.rename(
        columns={'x': 'x_ekf', 'y': 'y_ekf', 'speed': 'speed_ekf', 
                 'angle': 'angle_ekf', 'a': 'a_ekf',
                 'angular_vel': 'angular_vel_ekf'}
    )
    veh_df = veh_df.merge(filt_bike_df, on=['time'], how='left')    
    if filt_df is None:
        filt_df = veh_df.copy()
    else:
        filt_df = pd.concat((filt_df, veh_df), ignore_index=True)
    gc.collect()

if mode == "bike":
    filename = f"trajectories_bikes_{date}_{intersection}_{timeslot}_{code}-1-ekf"
else:
    filename = f"trajectories_vehicles_{date}_{intersection}_{timeslot}_{code}-1-ekf"
# filt_df.to_csv(data_root + f"{date}/{intersection}/{filename}.csv", index=False)
filt_df.to_parquet(data_root + f"{date}/{intersection}/{filename}.parquet", compression='zstd', index=False)


# Get some statistics
mae_x = np.nanmean(abs(filt_df.loc[(
    ~filt_df['missing']), 'x'] - filt_df.loc[(~filt_df['missing']), 'x_ekf']))
mae_y = np.nanmean(abs(filt_df.loc[(
    ~filt_df['missing']), 'y'] - filt_df.loc[(~filt_df['missing']), 'y_ekf']))
mae_v = np.nanmean(abs(filt_df.loc[(
    ~filt_df['missing']), 'speed'] - filt_df.loc[(~filt_df['missing']), 'speed_ekf']))

rmse_x = np.sqrt(np.nanmean(np.square(filt_df.loc[(
    ~filt_df['missing']), 'x'] - filt_df.loc[(~filt_df['missing']), 'x_ekf'])))
rmse_y = np.sqrt(np.nanmean(np.square(filt_df.loc[(
    ~filt_df['missing']), 'y'] - filt_df.loc[(~filt_df['missing']), 'y_ekf'])))
rmse_v = np.sqrt(np.nanmean(np.square(filt_df.loc[(
    ~filt_df['missing']), 'speed'] - filt_df.loc[(~filt_df['missing']), 'speed_ekf'])))

log.section('EKF + Gap Inference Statistics')
log.info(f'MAE: x = {mae_x:.6f} m, y = {mae_y:.6f} m, v = {mae_v:.6f} km/h')
log.info(f'RMSE: x = {rmse_x:.6f} m, y = {rmse_y:.6f} m, v = {rmse_v:.6f} km/h')
