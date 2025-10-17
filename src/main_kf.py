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
df = df.sort_values(by=['veh_id', 'time(s)'], ascending=True)
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

# overview of trajectories
fig, axs = plt.subplots(1, 2, figsize=(8, 4))
grouped = df[~df['missing']].groupby(by='veh_id')
for veh_id, veh_df in grouped:
    axs[0].plot(veh_df['x_act'], veh_df['y_act'], 'b')
    axs[1].plot(veh_df['x'], veh_df['y'], 'b')

axs[0].set_xlabel('X_2056 [m]')
axs[0].set_ylabel('Y_2056 [m]')
axs[0].set_xlim(BikeZ_Config.X_2056_Bounds)
axs[0].set_ylim(BikeZ_Config.Y_2056_Bounds)

axs[1].set_xlabel('X_2056 - X_ref [m]')
axs[1].set_ylabel('Y_2056 - Y_ref [m]')

fig.tight_layout()

# #############################################################################
# MAIN: Test EKF for a single bike with missing data
# #############################################################################
sel_bike_id = 22
bike_df = df[(~df['missing']) & (df['veh_id'] == sel_bike_id)]

Qk = np.diag([1.0, 1.0, 1.0, 1.0, 1.0])  # covariance matrix of error of state
Rk = np.diag([1.0, 1.0, 1.0, 1.0, 1.0])  # covariance matrix of error of output
first_frame = int(bike_df['time'].iloc[0]*BikeZ_Config.fps)
last_frame = int(bike_df['time'].iloc[-1]*BikeZ_Config.fps)
filtered_bike_df = calculate_kalman_filtered_trajectory(
    bike_df, Qk, Rk, first_frame, last_frame, fps=BikeZ_Config.fps
)

# draw individual bicycle
fig, axs = plt.subplots(1, 2, figsize=(8, 4))
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

fig.tight_layout()

# sys.exit(1)
plt.close('all')

# #############################################################################
# MAIN: Perform EKF for all bicycles
# #############################################################################
filt_df = None
# unique_ids = [1, 2, 5, 8, 20, 80, 22, 72, 152, 161] # test
unique_ids = df['veh_id'].unique()
for veh_id in tqdm(unique_ids, desc="Processing EKF on Bicycles"):
    if veh_id in [35, 86, 101, 110, 146]:
        continue
    veh_df = df[df['veh_id'] == veh_id].copy()
    first_frame = int(veh_df['time'].iloc[0]*BikeZ_Config.fps)
    last_frame = int(veh_df['time'].iloc[-1]*BikeZ_Config.fps)
    filt_bike_df = calculate_kalman_filtered_trajectory(
        veh_df[(~veh_df['missing'])], Qk, Rk, first_frame, last_frame, fps=BikeZ_Config.fps
    )
    filt_bike_df = filt_bike_df[['time', 'x', 'y', 'speed']]
    filt_bike_df = filt_bike_df.rename(columns={'x': 'x_ekf', 'y': 'y_ekf', 'speed': 'speed_ekf'})
    veh_df = veh_df.merge(filt_bike_df, on=['time'], how='left')
    if filt_df is None:
        filt_df = veh_df.copy()
    else:
        filt_df = pd.concat((filt_df, veh_df), ignore_index=True)
    gc.collect()
        
filename = f"trajectories_bikes_{date}_{intersection}_{time_slot}_{code}-1-ekf.csv"
filt_df.to_csv(BikeZ_Config.data_root + f"{date}/{intersection}/{filename}", index=False)

# oveview of trajectories
fig, axs = plt.subplots(1, 2, figsize=(8, 4))
for veh_id in unique_ids:
    veh_df = df[(~df['missing']) & (df['veh_id'] == veh_id)]
    # axs[0].scatter(veh_df['x'], veh_df['y'], s=1, color='b')
    axs[0].plot(veh_df['x'], veh_df['y'], color='b')
    
    veh_df = filt_df[filt_df['veh_id'] == veh_id]
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
mae_x = np.mean(abs(filt_df.loc[(~filt_df['missing']), 'x'] - filt_df.loc[(~filt_df['missing']), 'x_ekf']))
mae_y = np.mean(abs(filt_df.loc[(~filt_df['missing']), 'y'] - filt_df.loc[(~filt_df['missing']), 'y_ekf']))
mae_v = np.mean(abs(filt_df.loc[(~filt_df['missing']), 'speed'] - filt_df.loc[(~filt_df['missing']), 'speed_ekf']))
    
rmse_x = np.sqrt(np.mean(np.square(filt_df.loc[(~filt_df['missing']), 'x'] - filt_df.loc[(~filt_df['missing']), 'x_ekf'])))
rmse_y = np.sqrt(np.mean(np.square(filt_df.loc[(~filt_df['missing']), 'y'] - filt_df.loc[(~filt_df['missing']), 'y_ekf'])))
rmse_v = np.sqrt(np.mean(np.square(filt_df.loc[(~filt_df['missing']), 'speed'] - filt_df.loc[(~filt_df['missing']), 'speed_ekf'])))

