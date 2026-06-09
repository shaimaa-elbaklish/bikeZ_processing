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
timeslot = BikeZ_Config.avail_timeslots[date][(intersection, code)][-3] # 'AM2'

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


from tools_filtering import estimate_heading, estimate_angular_velocity
# Estimate heading angle (radians)
df = estimate_heading(df, speed_threshold=1.0, window_s=0.8, fps=BikeZ_Config.fps, smooth_method='savgol')
# Estimate angular velocity (rad/s)
df = estimate_angular_velocity(df, smooth_window_s=0.4, fps=BikeZ_Config.fps, smooth_method='rolling')

# # Log clipped entries per vehicle
# clipped = df[df['angvel_clipped']]
# max_angvel_rad = 3.0
# if not clipped.empty:
#     # log.warning(f"{len(clipped)} frames clipped to ±{max_angvel_rad} rad/s")
#     print(f"{len(clipped)} frames clipped to ±{max_angvel_rad} rad/s")
#     for veh_id, count in clipped.groupby('veh_id').size().items():
#         # log.debug(f"  veh_id={veh_id}: {count} frames clipped")
#         print(f"  veh_id={veh_id}: {count} frames clipped")

from tools_filtering import _pca_heading
for veh_id, veh_df in df.groupby('veh_id'):
    if veh_df['angle'].isna().all():
        h = _pca_heading(veh_df)
        df.loc[df['veh_id'] == veh_id, 'angle']       = h
        df.loc[df['veh_id'] == veh_id, 'angular_vel'] = 0.0
        print(
            f'veh={veh_id}: stationary, no heading data — '
            f'PCA heading={np.degrees(h):.1f}° assigned as constant.'
        )

# #############################################################################
# MAIN: Test EKF for a single bike with missing data
# #############################################################################

# from _logger import Logger 
# log = Logger(date, intersection, code, timeslot, "KF_TEST")

# counts = df.groupby('veh_id').size()
# longest_id = counts.idxmax() # 189, 31, (misisng) 99, 116
# top2 = counts.nlargest(2)
# turning bikes: 68, (missing) 14

sel_bike_id = 415
bike_df = df[(df['veh_id'] == sel_bike_id)]
bike_df = bike_df.sort_values(by='time', ascending=True)

Qk = np.diag([1.0, 1.0, 1.0, 10.0]).astype(np.float64)   # covariance matrix of error of state
Rk = np.diag([1.0, 1.0, 5.0, 10.0]).astype(np.float64)   # covariance matrix of error of output

import time
start = time.perf_counter()
filtered_bike_df = calculate_kalman_filtered_trajectory(
    bike_df, Qk, Rk, fps=BikeZ_Config.fps, debug=True #, log=log
)
end = time.perf_counter()
print(f"Elapsed time: {end - start:.6f} seconds")

# draw individual bicycle
bike_df = bike_df[(~bike_df['missing'])]
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

axs[2].plot(bike_df['time'], bike_df['speed']/3.6, label='Original')
axs[2].plot(filtered_bike_df['time'], filtered_bike_df['speed']/3.6, label='EKF')
axs[2].set_xlabel('Time [s]')
axs[2].set_ylabel('Speed [m/s]')
axs[2].set_ylim([0, 6])
axs[2].legend()

fig.tight_layout()

bike_df['jerk'] = bike_df['a'].diff().shift(-1).fillna(0) * 25
filtered_bike_df['jerk'] = filtered_bike_df['a'].diff().shift(-1).fillna(0) * 25

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


fig, axs = plt.subplots(1, 2, figsize=(8, 4))
axs[0].plot(bike_df['time'], bike_df['angle']*180/np.pi, label='Original')
axs[0].plot(filtered_bike_df['time'], filtered_bike_df['angle']*180/np.pi, label='Current EKF', alpha=0.75)
axs[0].set_xlabel('Time [s]')
axs[0].set_ylabel('Heading Angle [deg]')
axs[0].legend()
axs[0].set_ylim([-180, 180])

axs[1].plot(bike_df['time'], bike_df['angular_vel'], label='Original')
axs[1].plot(filtered_bike_df['time'], filtered_bike_df['angular_vel'], label='Current EKF', alpha=0.75)
axs[1].set_xlabel('Time [s]')
axs[1].set_ylabel('Angular Velocity [rad/s]')
axs[1].legend()
axs[1].set_ylim([-4, 4])

fig.tight_layout()
