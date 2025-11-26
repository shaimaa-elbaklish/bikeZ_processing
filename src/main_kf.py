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
time_slot = 'PM6'
code = 'E'

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
ref_datetime = df.loc[df['time'] == 0, 'datetime'].unique()
if len(ref_datetime) >= 1:
    ref_datetime = ref_datetime[0]
    fix_df = df[df['time'] == -1]
    # print(fix_df['veh_id'].unique()) # [35, 86, 101, 110, 146]
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
# MAIN: Perform EKF for all bicycles
# #############################################################################
Qk = np.diag([1.0, 1.0, 10.0, 10.0])  # covariance matrix of error of state
Rk = np.diag([5.0, 5.0, 1.0, 1.0])   # covariance matrix of error of output

filt_df = None
# unique_ids = [35, 86, 22, 72, 152, 161] # test
unique_ids = df['veh_id'].unique()
for veh_id in tqdm(unique_ids, desc="Processing EKF on Bicycles"):
    veh_df = df[df['veh_id'] == veh_id].copy()
    veh_df = veh_df.sort_values(by='time', ascending=True)
    first_frame = int(np.round(veh_df['time'].iloc[0]*BikeZ_Config.fps + 1e-05, decimals=0))
    last_frame = int(np.round(veh_df['time'].iloc[-1]*BikeZ_Config.fps + 1e-05, decimals=0))
    filt_bike_df = calculate_kalman_filtered_trajectory(
        veh_df[(~veh_df['missing'])], Qk, Rk, first_frame, last_frame, fps=BikeZ_Config.fps
    )
    filt_bike_df = filt_bike_df[['frame_nr', 'x', 'y', 'speed', 'angle']]
    filt_bike_df = filt_bike_df.rename(columns={'x': 'x_ekf', 'y': 'y_ekf', 'speed': 'speed_ekf', 'angle': 'angle_ekf'})
    veh_df['frame_nr'] = np.round(veh_df['time']*BikeZ_Config.fps, decimals=0)
    veh_df['frame_nr'] = veh_df['frame_nr'].astype(int)
    veh_df = veh_df.merge(filt_bike_df, on=['frame_nr'], how='left')    
    if filt_df is None:
        filt_df = veh_df.copy()
    else:
        filt_df = pd.concat((filt_df, veh_df), ignore_index=True)
    gc.collect()
        
filename = f"trajectories_bikes_{date}_{intersection}_{time_slot}_{code}-1-ekf.csv"
filt_df.to_csv(BikeZ_Config.data_root + f"{date}/{intersection}/{filename}", index=False)

# filename = f"trajectories_bikes_{date}_{intersection}_{time_slot}_{code}-1-ekf.csv"
# filt_df = pd.read_csv(BikeZ_Config.data_root + f"{date}/{intersection}/{filename}")
# unique_ids = df['veh_id'].unique()

# oveview of trajectories
# unique_ids = [22, 72, 152, 161]
fig, axs = plt.subplots(1, 2, figsize=(8, 4))
for veh_id in unique_ids:
    veh_df = df[(df['veh_id'] == veh_id)].copy() # (~df['missing']) & 
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
mae_x = np.nanmean(abs(filt_df.loc[(~filt_df['missing']), 'x'] - filt_df.loc[(~filt_df['missing']), 'x_ekf']))
mae_y = np.nanmean(abs(filt_df.loc[(~filt_df['missing']), 'y'] - filt_df.loc[(~filt_df['missing']), 'y_ekf']))
mae_v = np.nanmean(abs(filt_df.loc[(~filt_df['missing']), 'speed'] - filt_df.loc[(~filt_df['missing']), 'speed_ekf']))
    
rmse_x = np.sqrt(np.nanmean(np.square(filt_df.loc[(~filt_df['missing']), 'x'] - filt_df.loc[(~filt_df['missing']), 'x_ekf'])))
rmse_y = np.sqrt(np.nanmean(np.square(filt_df.loc[(~filt_df['missing']), 'y'] - filt_df.loc[(~filt_df['missing']), 'y_ekf'])))
rmse_v = np.sqrt(np.nanmean(np.square(filt_df.loc[(~filt_df['missing']), 'speed'] - filt_df.loc[(~filt_df['missing']), 'speed_ekf'])))

