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
import os
import sys
import pickle
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from scipy.interpolate import splprep, splev

from _constants import BikeZ_Config
from _logger import Logger
from tools_filtering import estimate_heading
from tools_coordinate_transform import match_bicycle_to_centerline
from tools_coordinate_transform import convert_xy2056_to_roadway_coordinates

# #############################################################################
# CONSTANTS
# #############################################################################
# Configuration
BikeZ_Config = BikeZ_Config()

parser = argparse.ArgumentParser(description="Coordinate transform for BikeZ trajectories")
parser.add_argument("date",          type=str, help="Date string, e.g. 2025-06-16")
parser.add_argument("mode",          type=str, help="Mode: bike or vehicle")
parser.add_argument("intersection",  type=str, help="Intersection ID, e.g. D3")
parser.add_argument("code",          type=str, help="Code letter, e.g. E")
parser.add_argument("timeslot",      type=str, help="Timeslot, e.g. AM1")
parser.add_argument("debug_plotting",type=str, help="Enable debug plots: True or False")
args = parser.parse_args()

date         = args.date
mode         = args.mode
intersection = args.intersection
code         = args.code
timeslot     = args.timeslot
DEBUG_PLOT   = args.debug_plotting.lower() == "true"

campaign  = f"Zurich_2025{date[5:7]}"
data_root = BikeZ_Config.data_root[campaign][mode]

XY_2056_Bounds = BikeZ_Config.XY_2056_Bounds[date][(intersection, code)]
X_2056_offset = XY_2056_Bounds[0][0]
Y_2056_offset = XY_2056_Bounds[1][0]

OPP_DIRECTIONS = {"N": "S", "S": "N", "W": "E", "E": "W"}

bike_lane_tol = 0.4

log = Logger(date, intersection, code, timeslot, "CT")

# #############################################################################
# MAIN: Load data
# #############################################################################
##  Load trajectories
if mode == "bike":
    filename = f"trajectories_bikes_{date}_{intersection}_{timeslot}_{code}-1.csv"
else:
    filename = f"trajectories_vehicles_{date}_{intersection}_{timeslot}_{code}-1.csv"
    
df = pd.read_csv(data_root + f"{date}/{intersection}/{filename}")
# COLUMNS: ['veh_id', 'veh_type', 'speed(km/h)', 'a(m/s2)', 'time(s)', 'X_2056(m)', 'Y_2056(m)', 'longitude', 'latitude', 'datetime']
# add a column as a missing flag
df['missing'] = (df['speed(km/h)'] == -1)

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
# Estimate heading angle (degrees)
df = estimate_heading(df, speed_threshold=1.0, window_s=0.5)
df['angle'] = df['heading'] * np.pi / 180 # convert to rad

center_lat, center_lon = df.loc[~df['missing'], "lat"].mean(), df.loc[~df['missing'], "lon"].mean()

## Load centerlines
with open(f"../data/centerlines_splines_{date}_{intersection}.pkl", "rb") as f:
    centerlines_spl_dict_orig = pickle.load(f)
# filter them based on mode
centerlines_spl_dict = {}
for key, spline in centerlines_spl_dict_orig.items():
    if mode == "bike" and key[-1] == 'V':
        continue
    if mode == "vehicle" and key[-1] == 'B':
        continue
    centerlines_spl_dict[key[:-2]] = spline

# get centerlines start and end points
centerlines_start_end_pts_dict = {}
for key, spline in centerlines_spl_dict.items():
    x_start, y_start = splev(0.0, spline[0])
    x_end, y_end = splev(1.0, spline[0])
    centerlines_start_end_pts_dict[key] = [(x_start, y_start), (x_end, y_end)]

# sys.exit(1)
# #############################################################################
# MAIN: Perform Coordinate Transform for ALL Bicycles
# #############################################################################

mod_df = None
unique_ids = df['veh_id'].unique()
for bike_id in tqdm(unique_ids, desc="Processing Coordinate Transform on Bicycles"):
    bike_df = df[(df["veh_id"] == bike_id) & (~df['missing'])].copy()
    bike_df[['x_act_ekf', 'y_act_ekf']] = bike_df[['x_act', 'y_act']]
    bike_df[['speed_ekf', 'angle_ekf']] = bike_df[['speed', 'angle']]
    # Select appropriate centerline
    try:
        centerline_id = match_bicycle_to_centerline(bike_df, centerlines_spl_dict)
        bike_df['Centerline_ID'] = centerline_id
        centerline_spl = centerlines_spl_dict[centerline_id]
    except Exception as e:
        log.error(f"Failed during coordinate transform for ID {bike_id}: {e}")
        continue
    tck, unew, cum_dist = centerline_spl

    # Perform Transformation
    roadway_out = bike_df.apply(lambda row: convert_xy2056_to_roadway_coordinates([row['x_act_ekf'], row['y_act_ekf']], tck, unew, cum_dist), axis=1)
    bike_df["Position_Longitudinal"] = roadway_out.apply(lambda x: x[3])
    bike_df["Position_Lateral"] = roadway_out.apply(lambda x: x[4])
    bike_df["Spline_Param"] = roadway_out.apply(lambda x: x[0])
    bike_df["Spline_Tangent"] = roadway_out.apply(lambda x: x[1])
    bike_df["Spline_Normal"] = roadway_out.apply(lambda x: x[2])
    # t_star, tangent, normal, s, d

    bike_df['velocity_x'] = bike_df['speed_ekf'] * np.cos(bike_df['angle_ekf'])
    bike_df['velocity_y'] = bike_df['speed_ekf'] * np.sin(bike_df['angle_ekf'])
    bike_df["velocity_global"] = bike_df[['velocity_x', 'velocity_y']].to_numpy().tolist()
    bike_df['acceleration_x'] = bike_df['a'] * np.cos(bike_df['angle_ekf'])
    bike_df['acceleration_y'] = bike_df['a'] * np.sin(bike_df['angle_ekf'])
    bike_df["acceleration_global"] = bike_df[['acceleration_x', 'acceleration_y']].to_numpy().tolist()
    bike_df["Speed_Longitudinal"] = bike_df.apply(lambda row: np.dot(row["velocity_global"], row["Spline_Tangent"]), axis=1)
    bike_df["Speed_Lateral"] = bike_df.apply(lambda row: np.dot(row["velocity_global"], row["Spline_Normal"]), axis=1)
    bike_df["Accel_Longitudinal"] = bike_df.apply(lambda row: np.dot(row["acceleration_global"], row["Spline_Tangent"]), axis=1)
    bike_df["Accel_Lateral"] = bike_df.apply(lambda row: np.dot(row["acceleration_global"], row["Spline_Normal"]), axis=1)
    bike_df = bike_df.drop(columns=['velocity_x', 'velocity_y', 'velocity_global',
                                    'acceleration_x', 'acceleration_y', 'acceleration_global'])

    # Determining if bike is inside bike lane
    bike_df['In_Bike_Lane'] = pd.NA
    bike_df['Bike_Lane_ID'] = pd.NA
    bike_df = bike_df.drop(columns=["x_act_ekf", "y_act_ekf", "speed_ekf", "angle_ekf"])
    if mod_df is None:
        mod_df = bike_df.copy()
    else:
        mod_df = pd.concat((mod_df, bike_df), ignore_index=True)

if mode == "bike":
    filename = f"trajectories_bikes_{date}_{intersection}_{timeslot}_{code}-1-lane.csv"
else:
    filename = f"trajectories_vehicles_{date}_{intersection}_{timeslot}_{code}-1-lane.csv"
mod_df.to_csv(data_root + f"{date}/{intersection}/{filename}", index=False)

# #############################################################################
# MAIN: Plot Coordinate Transform for ALL Bicycles
# #############################################################################
save_path = os.path.join("../debugging/", f"{date}-{intersection}")
os.makedirs(save_path, exist_ok=True)

if DEBUG_PLOT:
    plt.ioff()
    
    for bike_id in tqdm(unique_ids, desc="Plotting Lane Coordinates on Bicycles"):
        bike_df = mod_df[(mod_df["veh_id"] == bike_id)].copy()
        if bike_df['Centerline_ID'].isna().all():
            continue
        centerline_id = bike_df['Centerline_ID'].iloc[0]
        centerline_spl = centerlines_spl_dict[centerline_id]
        tck, unew, cum_dist = centerline_spl 
        
        tol = 5
        fig, axs = plt.subplots(2, 2, figsize=(8, 8))
    
        mask = bike_df['In_Bike_Lane'].isna()
        axs[0, 0].scatter(bike_df.loc[mask, 'x_act'], bike_df.loc[mask, 'y_act'], label='Trajectory', alpha=0.5, color='tab:blue', s=1)
        mask = (bike_df['In_Bike_Lane'] == True)
        axs[0, 0].scatter(bike_df.loc[mask, 'x_act'], bike_df.loc[mask, 'y_act'], alpha=1, color='tab:olive', s=1)
        mask = (bike_df['In_Bike_Lane'] == False)
        axs[0, 0].scatter(bike_df.loc[mask, 'x_act'], bike_df.loc[mask, 'y_act'], alpha=0.1, color='tab:cyan', s=1)
    
        axs[0, 0].scatter(bike_df['x_act'].iloc[0], bike_df['y_act'].iloc[0], label='Start', color='black')
        axs[0, 0].scatter(bike_df['x_act'].iloc[-1], bike_df['y_act'].iloc[-1], label='End', color='red')
        x_spline, y_spline = splev(np.linspace(0, 1, 50), tck)
        axs[0, 0].plot(x_spline, y_spline, label='Centerline', linestyle='-.', color='gray')
    
        axs[0, 0].set(xlabel='X_2056 [m]', ylabel='Y_2056 [m]', 
                      xlim=[bike_df['x_act'].min()-tol, bike_df['x_act'].max()+tol],
                      ylim=[bike_df['y_act'].min()-tol, bike_df['y_act'].max()+tol])
        axs[0, 0].legend()
    
        axs[0, 1].plot(bike_df["Position_Longitudinal"], bike_df["Position_Lateral"], label='Trajectory')
        axs[0, 1].scatter(bike_df['Position_Longitudinal'].iloc[0], bike_df['Position_Lateral'].iloc[0], label='Start', color='black')
        axs[0, 1].scatter(bike_df['Position_Longitudinal'].iloc[-1], bike_df['Position_Lateral'].iloc[-1], label='End', color='red')
        axs[0, 1].set(xlabel='Longitudinal Position, $s$ [m]', ylabel='Lateral Offset, $d$ [m]')
        axs[0, 1].legend()
    
        axs[1, 0].plot(bike_df['time'], bike_df["Speed_Longitudinal"], label='Longitudinal')
        axs[1, 0].plot(bike_df['time'], bike_df["Speed_Lateral"], label='Lateral')
        axs[1, 0].plot(bike_df['time'], bike_df["speed"], label='Total', alpha=0.5, linestyle='--')
        axs[1, 0].set(xlabel='Time [s]', ylabel='Speed [km/h]')
        axs[1, 0].legend()
    
        axs[1, 1].plot(bike_df['time'], bike_df["Accel_Longitudinal"], label='Longitudinal')
        axs[1, 1].plot(bike_df['time'], bike_df["Accel_Lateral"], label='Lateral')
        axs[1, 1].plot(bike_df['time'], bike_df["a"], label='Total', alpha=0.5, linestyle='--')
        axs[1, 1].set(xlabel='Time [s]', ylabel='Acceleration [m/s$^2$]')
        axs[1, 1].legend()
        
        fig.suptitle(f'Bike ID = {bike_id}, Centerline ID = {centerline_id}')
        fig.tight_layout()
        if mode == "bike":
            fig.savefig(os.path.join(save_path, f'{timeslot}_{code}_lane_coordinates_bicycle_{bike_id}.png'), 
                        bbox_inches='tight')
        else:
            fig.savefig(os.path.join(save_path, f'{timeslot}_{code}_lane_coordinates_car_{bike_id}.png'), 
                        bbox_inches='tight')
        
        plt.close(fig)


