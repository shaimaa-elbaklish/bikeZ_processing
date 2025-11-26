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
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from _constants import BikeZ_Config
from tools_kalman import calculate_kalman_filtered_trajectory

# #############################################################################
# CONSTANTS
# #############################################################################
# Configuration
BikeZ_Config = BikeZ_Config()

# Specify Trajectory File
date = BikeZ_Config.avail_dates[0]
campaign = f"Zurich_2025{date[5:7]}" # June or September
mode = BikeZ_Config.avail_modes[0] # Bike
data_root = BikeZ_Config.data_root[campaign][mode]

intersection, code = BikeZ_Config.avail_intersections[date][3]
# all_timeslots = BikeZ_Config.avail_timeslots[date][(intersection, code)]
timeslot = BikeZ_Config.avail_timeslots[date][(intersection, code)][0] # 'AM2'

XY_2056_Bounds = BikeZ_Config.XY_2056_Bounds[date][(intersection, code)]
X_2056_offset = XY_2056_Bounds[0][0]
Y_2056_offset = XY_2056_Bounds[1][0]

# #############################################################################
# MAIN
# #############################################################################
filename = f"trajectories_bikes_{date}_{intersection}_{timeslot}_{code}-1.csv"
df = pd.read_csv(data_root + f"{date}/{intersection}/{filename}")
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
df['x'] = df['x_act'] - X_2056_offset
df['y'] = df['y_act'] - Y_2056_offset
df['datetime'] = pd.to_datetime(df['datetime'], format='ISO8601')

# Fix time = -1 issues
# Find ref. datetime (i.e. datetime when time == 0)
ref_datetime = df.loc[df['time'] == 0, 'datetime'].unique()
if len(ref_datetime) >= 1:
    ref_datetime = ref_datetime[0]
    fix_df = df[df['time'] == -1]
    for idx, _ in fix_df.iterrows():
        df.loc[idx, 'time'] = np.round((df.loc[idx, 'datetime'] - ref_datetime).total_seconds(), decimals=2)
    del fix_df
    gc.collect()
else:
    # to handle 'AM6'
    ref_datetime = df['datetime'].min()
    ref_time = df.loc[(df['datetime'] == ref_datetime) & (df['time'] >= 0), 'time'].unique()[0]
    df['time'] = df['datetime'].apply(lambda x: np.round((x - ref_datetime).total_seconds() + ref_time, decimals=3))
df = df.sort_values(by=['veh_id', 'time'], ascending=True)

# #############################################################################
# MAIN: Test EKF for a single bike with missing data
# #############################################################################
# from tools_filtering import calculate_features
# from _constants import SPEED_ESTIMATION_HORIZON


sel_bike_id = 22 # 141, 22 (with missing) # 299
bike_df = df[(~df['missing']) & (df['veh_id'] == sel_bike_id)]
bike_df = bike_df.sort_values(by='time', ascending=True)

Qk = np.diag([1.0, 1.0, 10.0, 10.0])  # covariance matrix of error of state
Rk = np.diag([5.0, 5.0, 1.0, 1.0])   # covariance matrix of error of output
first_frame = int(np.round(bike_df['time'].iloc[0]*BikeZ_Config.fps, decimals=0))
last_frame = int(np.round(bike_df['time'].iloc[-1]*BikeZ_Config.fps, decimals=0))
filtered_bike_df = calculate_kalman_filtered_trajectory(
    bike_df, Qk, Rk, first_frame, last_frame, fps=BikeZ_Config.fps
)

bike_df['frame_nr'] = np.round(bike_df['time'] * BikeZ_Config.fps + 1e-05, decimals=0)
bike_df['frame_nr'] = bike_df['frame_nr'].astype(int)

# draw individual bicycle
fig, axs = plt.subplots(1, 3, figsize=(12, 4))
axs[0].scatter(bike_df['x'], bike_df['y'], s=5, label='Original')
axs[0].scatter(filtered_bike_df['x'], filtered_bike_df['y'], s=1, label='EKF')
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

# prev_filename = f"trajectories_bikes_{date}_{intersection}_{timeslot}_{code}-1-ekf-prev.csv"
# prev_df = pd.read_csv(BikeZ_Config.data_root + f"{date}/{intersection}/{prev_filename}")
# prev_filtered_bike_df = prev_df[(prev_df['veh_id'] == sel_bike_id)]
# prev_filtered_bike_df['x_ekf'] = prev_filtered_bike_df['x_ekf'].interpolate(method='linear').ffill().bfill()
# prev_filtered_bike_df['y_ekf'] = prev_filtered_bike_df['y_ekf'].interpolate(method='linear').ffill().bfill()
# prev_filtered_bike_df['vx'] = prev_filtered_bike_df['x_ekf'].diff().shift(-1).fillna(0) * 25
# prev_filtered_bike_df['vy'] = prev_filtered_bike_df['y_ekf'].diff().shift(-1).fillna(0) * 25
# prev_filtered_bike_df['speed_ekf'] = np.sqrt(prev_filtered_bike_df['vx']**2 + prev_filtered_bike_df['vy']**2) * 3.6
# fig, axs = plt.subplots(1, 3, figsize=(12, 4))
# axs[0].scatter(bike_df['x'], bike_df['y'], s=5, label='Original')
# axs[0].scatter(filtered_bike_df['x'], filtered_bike_df['y'], s=1, label='Current EKF')
# axs[0].plot(prev_filtered_bike_df['x_ekf'], prev_filtered_bike_df['y_ekf'], label='Previous EKF')
# axs[0].set_xlabel('X_2056 - X_ref [m]')
# axs[0].set_ylabel('Y_2056 - Y_ref [m]')
# axs[0].legend()

# axs[1].hist(bike_df['speed'], bins=100, density=True, label='Original')
# axs[1].hist(filtered_bike_df['speed'], bins=100, density=True, label=' Current EKF', alpha=0.75)
# axs[1].hist(prev_filtered_bike_df['speed_ekf'], bins=100, density=True, label='Previous EKF', alpha=0.5)
# axs[1].set_xlabel('Speed [km/h]')
# axs[1].set_ylabel('PDF')
# axs[1].legend()

# axs[2].plot(bike_df['time'], bike_df['speed'], label='Original')
# axs[2].plot(filtered_bike_df['time'], filtered_bike_df['speed'], label='Current EKF')
# axs[2].plot(prev_filtered_bike_df['time'], prev_filtered_bike_df['speed_ekf'], label='Previous EKF')
# axs[2].set_xlabel('Time [s]')
# axs[2].set_ylabel('Speed [km/h]')
# axs[2].legend()

# fig.tight_layout()

bike_df['jerk'] = bike_df['a'].diff().shift(-1).fillna(0) * 25
filtered_bike_df['jerk'] = filtered_bike_df['a'].diff().shift(-1).fillna(0) * 25
# prev_filtered_bike_df['a_ekf'] = prev_filtered_bike_df['speed_ekf'].diff().shift(-1).fillna(0) / 3.6 * 25
# prev_filtered_bike_df['jerk'] = prev_filtered_bike_df['a_ekf'].diff().shift(-1).fillna(0) * 25

fig, axs = plt.subplots(1, 2, figsize=(8, 4))
axs[0].plot(bike_df['time'], bike_df['a'], label='Original')
axs[0].plot(filtered_bike_df['time'], filtered_bike_df['a'], label='Current EKF', alpha=0.75)
# axs[0].plot(prev_filtered_bike_df['time'], prev_filtered_bike_df['a_ekf'], label='Previous EKF', alpha=0.5)
axs[0].set_xlabel('Time [s]')
axs[0].set_ylabel('Acceleration [m/s$^2$]')
axs[0].legend()
axs[0].set_ylim([-3, 3])

axs[1].plot(bike_df['time'], bike_df['jerk'], label='Original')
axs[1].plot(filtered_bike_df['time'], filtered_bike_df['jerk'], label='Current EKF', alpha=0.75)
# axs[1].plot(prev_filtered_bike_df['time'], prev_filtered_bike_df['jerk'], label='Previous EKF', alpha=0.5)
axs[1].set_xlabel('Time [s]')
axs[1].set_ylabel('Jerk [m/s$^3$]')
axs[1].legend()
axs[1].set_ylim([-5, 5])

fig.tight_layout()
