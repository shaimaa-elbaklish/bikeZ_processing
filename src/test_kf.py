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
import pytz
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm

from _constants import BikeZ_Config
from tools_kalman import calculate_kalman_filtered_trajectory

# #############################################################################
# CONSTANTS
# #############################################################################
date = BikeZ_Config.avail_dates[0]
intersection = BikeZ_Config.avail_intersections[2]
time_slot = 'AM1'
code= 'E'

# #############################################################################
# MAIN
# #############################################################################
filename = f"trajectories_bikes_{date}_{intersection}_{time_slot}_{code}-1.csv"
df = pd.read_csv(BikeZ_Config.data_root + f"{date}/{intersection}/{filename}")
# COLUMNS: ['veh_id', 'veh_type', 'speed(km/h)', 'a(m/s2)', 'time(s)', 'X_2056(m)', 'Y_2056(m)', 'longitude', 'latitude', 'datetime']
# add a column as a missing flag
df['missing'] = (df['speed(km/h)'] == -1)
# print(df.loc[df['missing'], 'veh_id'].unique())
# IDs with missing values: 22,  72, 152, 161

df = df.rename(columns={
    'speed(km/h)': 'speed', 
    'a(m/s2)': 'a', 
    'time(s)': 'time', 
    'X_2056(m)': 'x_act', 
    'Y_2056(m)': 'y_act',
    'longitude': 'lon', 
    'latitude': 'lat'
})
df['x'] = df['x_act'] - BikeZ_Config.X_2056_Bounds[0]
df['y'] = df['y_act'] - BikeZ_Config.Y_2056_Bounds[0]
df['datetime'] = pd.to_datetime(df['datetime'], format='ISO8601')

# Fix time = -1 issues
# Find ref. datetime (i.e. datetime when time == 0)
ref_datetime = df.loc[df['time'] == 0, 'datetime'].unique()[0]
fix_df = df[df['time'] == -1]
# print(fix_df['veh_id'].unique()) # [35, 86, 101, 110, 146]
for idx, _ in fix_df.iterrows():
    df.loc[idx, 'time'] = np.round((df.loc[idx, 'datetime'] - ref_datetime).total_seconds(), decimals=2)
del fix_df
gc.collect()
df = df.sort_values(by=['veh_id', 'time'], ascending=True)

# #############################################################################
# MAIN: Test EKF for a single bike with missing data
# #############################################################################
# from tools_filtering import calculate_features
# from _constants import SPEED_ESTIMATION_HORIZON


sel_bike_id = 12 # 141, 22 (with missing)
bike_df = df[(~df['missing']) & (df['veh_id'] == sel_bike_id)]
bike_df = bike_df.sort_values(by='time', ascending=True)

Qk = np.diag([1.0, 1.0, 10.0, 10.0])  # covariance matrix of error of state
Rk = np.diag([5.0, 5.0, 1.0, 1.0])   # covariance matrix of error of output
first_frame = int(np.round(bike_df['time'].iloc[0]*BikeZ_Config.fps, decimals=0))
last_frame = int(np.round(bike_df['time'].iloc[-1]*BikeZ_Config.fps, decimals=0))
filtered_bike_df = calculate_kalman_filtered_trajectory(
    bike_df, Qk, Rk, first_frame, last_frame, fps=BikeZ_Config.fps
)

# filtered_bike_df = filtered_bike_df.merge(df.loc[df['veh_id'] == sel_bike_id, ['time', 'missing']], on=['time'], how='left').dropna()
# start_frame_missing = filtered_bike_df.loc[filtered_bike_df['missing'], 'frame_nr'].min()
# end_frame_missing = filtered_bike_df.loc[filtered_bike_df['missing'], 'frame_nr'].max()
# missing_filt_bike_df = filtered_bike_df[(filtered_bike_df['frame_nr'] >= start_frame_missing - SPEED_ESTIMATION_HORIZON) &
#                                         (filtered_bike_df['frame_nr'] <= end_frame_missing + SPEED_ESTIMATION_HORIZON)].copy()
# missing_filt_bike_df = calculate_features(missing_filt_bike_df, BikeZ_Config.fps)
# missing_filt_bike_df['vx'] = missing_filt_bike_df['x'].diff().shift(-1).fillna(0) * BikeZ_Config.fps
# missing_filt_bike_df['vy'] = missing_filt_bike_df['y'].diff().shift(-1).fillna(0) * BikeZ_Config.fps
# missing_filt_bike_df['v_est'] = np.sqrt(missing_filt_bike_df['vx']**2 + missing_filt_bike_df['vy']**2)
# missing_filt_bike_df['a_est'] = missing_filt_bike_df['v_est'].diff().shift(-1).fillna(0) * BikeZ_Config.fps

# draw individual bicycle
fig, axs = plt.subplots(1, 3, figsize=(12, 4))
axs[0].scatter(bike_df['x'], bike_df['y'], s=5, label='Original')
axs[0].scatter(filtered_bike_df['x'], filtered_bike_df['y'], s=1, label='EKF')
# axs[0].plot(filtered_bike_df['x'], filtered_bike_df['y'], label='EKF', color='tab:orange')
axs[0].set_xlabel('X_2056 - X_ref [m]')
axs[0].set_ylabel('Y_2056 - Y_ref [m]')
axs[0].legend()

axs[1].hist(bike_df['speed'], bins=100, density=True, label='Original', alpha=0.5)
axs[1].hist(filtered_bike_df['speed'], bins=100, density=True, label='EKF', alpha=0.5)
axs[1].set_xlabel('Speed [km/h]')
axs[1].set_ylabel('PDF')
axs[1].legend()

axs[2].plot(bike_df['time'], bike_df['speed'], label='Original')
axs[2].plot(filtered_bike_df['time'], filtered_bike_df['speed'], label='EKF')
axs[2].set_xlabel('Time [s]')
axs[2].set_ylabel('Speed [km/h]')
axs[2].legend()

fig.tight_layout()

prev_filename = f"trajectories_bikes_{date}_{intersection}_{time_slot}_{code}-1-ekf-prev.csv"
prev_df = pd.read_csv(BikeZ_Config.data_root + f"{date}/{intersection}/{prev_filename}")
prev_filtered_bike_df = prev_df[(prev_df['veh_id'] == sel_bike_id)]
prev_filtered_bike_df['x_ekf'] = prev_filtered_bike_df['x_ekf'].interpolate(method='linear').ffill().bfill()
prev_filtered_bike_df['y_ekf'] = prev_filtered_bike_df['y_ekf'].interpolate(method='linear').ffill().bfill()
prev_filtered_bike_df['vx'] = prev_filtered_bike_df['x_ekf'].diff().shift(-1).fillna(0) * 25
prev_filtered_bike_df['vy'] = prev_filtered_bike_df['y_ekf'].diff().shift(-1).fillna(0) * 25
prev_filtered_bike_df['speed_ekf'] = np.sqrt(prev_filtered_bike_df['vx']**2 + prev_filtered_bike_df['vy']**2) * 3.6
fig, axs = plt.subplots(1, 3, figsize=(12, 4))
axs[0].scatter(bike_df['x'], bike_df['y'], s=5, label='Original')
axs[0].scatter(filtered_bike_df['x'], filtered_bike_df['y'], s=1, label='Current EKF')
axs[0].plot(prev_filtered_bike_df['x_ekf'], prev_filtered_bike_df['y_ekf'], label='Previous EKF')
axs[0].set_xlabel('X_2056 - X_ref [m]')
axs[0].set_ylabel('Y_2056 - Y_ref [m]')
axs[0].legend()

axs[1].hist(bike_df['speed'], bins=100, density=True, label='Original')
axs[1].hist(filtered_bike_df['speed'], bins=100, density=True, label=' Current EKF', alpha=0.75)
axs[1].hist(prev_filtered_bike_df['speed_ekf'], bins=100, density=True, label='Previous EKF', alpha=0.5)
axs[1].set_xlabel('Speed [km/h]')
axs[1].set_ylabel('PDF')
axs[1].legend()

axs[2].plot(bike_df['time'], bike_df['speed'], label='Original')
axs[2].plot(filtered_bike_df['time'], filtered_bike_df['speed'], label='Current EKF')
axs[2].plot(prev_filtered_bike_df['time'], prev_filtered_bike_df['speed_ekf'], label='Previous EKF')
axs[2].set_xlabel('Time [s]')
axs[2].set_ylabel('Speed [km/h]')
axs[2].legend()

fig.tight_layout()

bike_df['jerk'] = bike_df['a'].diff().shift(-1).fillna(0) * 25
filtered_bike_df['jerk'] = filtered_bike_df['a'].diff().shift(-1).fillna(0) * 25
prev_filtered_bike_df['a_ekf'] = prev_filtered_bike_df['speed_ekf'].diff().shift(-1).fillna(0) / 3.6 * 25
prev_filtered_bike_df['jerk'] = prev_filtered_bike_df['a_ekf'].diff().shift(-1).fillna(0) * 25

fig, axs = plt.subplots(1, 2, figsize=(8, 4))
axs[0].plot(bike_df['time'], bike_df['a'], label='Original')
axs[0].plot(filtered_bike_df['time'], filtered_bike_df['a'], label='Current EKF', alpha=0.75)
axs[0].plot(prev_filtered_bike_df['time'], prev_filtered_bike_df['a_ekf'], label='Previous EKF', alpha=0.5)
axs[0].set_xlabel('Time [s]')
axs[0].set_ylabel('Acceleration [m/s$^2$]')
axs[0].legend()
axs[0].set_ylim([-3, 3])

axs[1].plot(bike_df['time'], bike_df['jerk'], label='Original')
axs[1].plot(filtered_bike_df['time'], filtered_bike_df['jerk'], label='Current EKF', alpha=0.75)
axs[1].plot(prev_filtered_bike_df['time'], prev_filtered_bike_df['jerk'], label='Previous EKF', alpha=0.5)
axs[1].set_xlabel('Time [s]')
axs[1].set_ylabel('Jerk [m/s$^3$]')
axs[1].legend()
axs[1].set_ylim([-5, 5])

fig.tight_layout()
