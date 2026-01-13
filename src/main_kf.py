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
from tools_kalman import calculate_kalman_filtered_trajectory
from _constants import BikeZ_Config
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import gc
import sys
import warnings
warnings.filterwarnings("ignore")


# #############################################################################
# CONSTANTS
# #############################################################################
# Configuration
BikeZ_Config = BikeZ_Config()

# Specify Trajectory File
date = BikeZ_Config.avail_dates[0]
campaign = f"Zurich_2025{date[5:7]}"  # June or September
mode = BikeZ_Config.avail_modes[0]  # Bike
data_root = BikeZ_Config.data_root[campaign][mode]

intersection, code = BikeZ_Config.avail_intersections[date][4]
timeslot = BikeZ_Config.avail_timeslots[date][(intersection, code)][0] # 'AM1'

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

# #############################################################################
# MAIN: Perform EKF for all bicycles
# #############################################################################
Qk = np.diag([1.0, 1.0, 10.0, 10.0]).astype(np.float64)  # covariance matrix of error of state
Rk = np.diag([5.0, 5.0, 1.0, 1.0]).astype(np.float64)    # covariance matrix of error of output

filt_df = None
# unique_ids = [35, 86, 22, 72, 152, 161] # test
unique_ids = df['veh_id'].unique()
for veh_id in tqdm(unique_ids, desc="Processing EKF on Bicycles"):
    veh_df = df[df['veh_id'] == veh_id].copy()
    veh_df = veh_df.sort_values(by='time', ascending=True)
    filt_bike_df = calculate_kalman_filtered_trajectory(
        veh_df, Qk, Rk, fps=BikeZ_Config.fps
    )
    filt_bike_df = filt_bike_df[['time', 'x', 'y', 'speed', 'angle', 'a']]
    filt_bike_df = filt_bike_df.rename(
        columns={'x': 'x_ekf', 'y': 'y_ekf', 'speed': 'speed_ekf', 
                 'angle': 'angle_ekf', 'a': 'a_ekf'}
    )
    veh_df = veh_df.merge(filt_bike_df, on=['time'], how='left')    
    if filt_df is None:
        filt_df = veh_df.copy()
    else:
        filt_df = pd.concat((filt_df, veh_df), ignore_index=True)
    gc.collect()

filename = f"trajectories_bikes_{date}_{
    intersection}_{timeslot}_{code}-1-ekf.csv"
filt_df.to_csv(data_root + f"{date}/{intersection}/{filename}", index=False)

# filename = f"trajectories_bikes_{date}_{intersection}_{time_slot}_{code}-1-ekf.csv"
# filt_df = pd.read_csv(data_root + f"{date}/{intersection}/{filename}")
# unique_ids = df['veh_id'].unique()

# oveview of trajectories
fig, axs = plt.subplots(1, 2, figsize=(8, 4))
for veh_id in unique_ids:
    veh_df = df[(df['veh_id'] == veh_id)].copy()  # (~df['missing']) &
    veh_df.loc[veh_df['missing'], 'x'] = pd.NA
    veh_df.loc[veh_df['missing'], 'y'] = pd.NA
    # axs[0].scatter(veh_df['x'], veh_df['y'], s=1, color='b')
    axs[0].plot(veh_df['x'], veh_df['y'], color='b')

    veh_df = filt_df[filt_df['veh_id'] == veh_id]
    if veh_df[['x_ekf', 'y_ekf', 'speed_ekf', 'angle_ekf']].isna().any().any():
        print(veh_id)
        sys.exit(1)
    # axs[1].scatter(veh_df['x_ekf'], veh_df['y_ekf'], s=1, color='b')
    axs[1].plot(veh_df['x_ekf'], veh_df['y_ekf'], color='b')

axs[0].set_xlabel('X_2056 - X_ref [m]')
axs[0].set_ylabel('Y_2056 - Y_ref [m]')
axs[0].set_xlim([10, 150])
axs[0].set_ylim([10, 125])
axs[0].set_title('Original')

axs[1].set_xlabel('X_2056 - X_ref [m]')
axs[1].set_ylabel('Y_2056 - Y_ref [m]')
axs[1].set_xlim([10, 150])
axs[1].set_ylim([10, 125])
axs[1].set_title('EKF')

fig.tight_layout()

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
