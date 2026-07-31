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
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pyproj import Transformer

from _logger import Logger
from _constants import BikeZ_Config 
from tools_subsampling import subsample_all

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

loc_num = BikeZ_Config.location_map[(date[5:7], intersection, code)]

log = Logger(date, intersection, code, timeslot, f"Subsample_{mode}")

# #############################################################################
# MAIN: Load data
# #############################################################################
# trajectories after EKF
if mode == "bike":
    filename = f"trajectories_bikes_{date}_{intersection}_{timeslot}_{code}-1-ekf"
    filename_other = f"trajectories_vehicles_{date}_{intersection}_{timeslot}_{code}-1-ekf"
    data_root_other = BikeZ_Config.data_root[campaign]['vehicle']
else:
    filename = f"trajectories_vehicles_{date}_{intersection}_{timeslot}_{code}-1-ekf"
    filename_other = f"trajectories_bikes_{date}_{intersection}_{timeslot}_{code}-1-ekf"
    data_root_other = BikeZ_Config.data_root[campaign]['bike']
# df = pd.read_csv(data_root + f"{date}/{intersection}/{filename}.csv")
# df['datetime'] = pd.to_datetime(df['datetime'], format='ISO8601')
df = pd.read_parquet(data_root + f"{date}/{intersection}/{filename}.parquet")
ref_datetime = df['datetime'].min()
ref_time     = df.loc[
    (df['datetime'] == ref_datetime) & (df['time'] >= 0),
    'time'
].unique()[0]


# df_other = pd.read_csv(data_root_other + f"{date}/{intersection}/{filename_other}.csv")
# df_other['datetime'] = pd.to_datetime(df_other['datetime'], format='ISO8601')
df_other = pd.read_parquet(data_root_other + f"{date}/{intersection}/{filename_other}.parquet")
ref_datetime_other   = df_other['datetime'].min()
ref_time_other       = df_other.loc[
    (df_other['datetime'] == ref_datetime_other) & (df_other['time'] >= 0),
    'time'
].unique()[0]
del df_other
gc.collect()


datetime_anchor = max(ref_datetime, ref_datetime_other)   # pick one common phase reference
df['time']      = df['datetime'].apply(lambda x: np.round((x - datetime_anchor).total_seconds(), decimals=3))
df = df.sort_values(by=['veh_id', 'time'], ascending=True)


df_10fps = subsample_all(df, target_fps=10.0, log=log, 
                         include_heads=True, include_tails=True)
df_10fps['x_act_ekf'] = df_10fps['x_ekf'] + X_2056_offset
df_10fps['y_act_ekf'] = df_10fps['y_ekf'] + Y_2056_offset
transformer = Transformer.from_crs("EPSG:2056", "EPSG:4326", always_xy=True)
df_10fps["lon_ekf"], df_10fps["lat_ekf"] = transformer.transform(
    df_10fps["x_act_ekf"].values, df_10fps["y_act_ekf"].values
)

# Flag rows where occlusion gap was present
df_10fps['in_gap'] = False
if df.missing.any():
    from tools_utils import extract_all_gaps
    all_gaps_df = extract_all_gaps(df, include_datetime=True)
    
    df_10fps = df_10fps.sort_values(['veh_id', 'datetime']).reset_index(drop=True)
    all_gaps_df = all_gaps_df.sort_values(['veh_id', 'start_datetime']).reset_index(drop=True)

    for veh_id, gap_sub in all_gaps_df.groupby('veh_id'):
        mask_veh = (df_10fps['veh_id'] == veh_id)
        for row in gap_sub.itertuples(index=False):
            mask_datetime = (
                (df_10fps['datetime'] >= row.start_datetime) &
                (df_10fps['datetime'] <= row.end_datetime)
            )
            df_10fps.loc[mask_veh & mask_datetime, 'in_gap'] = True

df_10fps = df_10fps[[
    'veh_id', 'veh_type', 'datetime', 'time',
    'x_act_ekf', 'y_act_ekf', 'x_ekf', 'y_ekf',
    'lon_ekf', 'lat_ekf', 
    'speed_ekf', 'a_ekf',
    'angle_ekf', 'angular_vel_ekf', 
    'in_gap', 'off_grid'
]]

output_path = f"C:/Users/ShaimaaElBaklish/OneDrive - ETH Zurich/BikeZ-Subsampled/location_{loc_num}/{loc_num}_{mode}s_{date}_{timeslot}.csv"
df_10fps.to_csv(output_path, index=False)


# bike_id = np.random.choice(df['veh_id'].unique())
# bike_df = df[df['veh_id'] == bike_id].copy()
# bike_df_10fps = df_10fps[df_10fps['veh_id'] == bike_id].copy()

# plt.figure()
# plt.plot(bike_df['x_ekf'], bike_df['y_ekf'], label='25 fps')
# plt.plot(bike_df_10fps['x_ekf'], bike_df_10fps['y_ekf'], 'o--', markersize=2, label='10 fps')
# plt.scatter(bike_df['x_ekf'].iloc[-1], bike_df['y_ekf'].iloc[-1], label='end', color='red')
# plt.legend()
# plt.title(f'XY Path - veh_id = {bike_id}')

# plt.figure()
# plt.plot(bike_df['time'], bike_df['angle_ekf'], label='25 fps')
# plt.plot(bike_df_10fps['time'], bike_df_10fps['angle_ekf'], 'o--', markersize=2, label='10 fps')
# plt.legend()
# plt.title(f'Angle timeseries - veh_id = {bike_id}')

# plt.figure()
# plt.plot(bike_df['time'], bike_df['speed_ekf'], label='25 fps')
# plt.plot(bike_df_10fps['time'], bike_df_10fps['speed_ekf'], 'o--', markersize=2, label='10 fps')
# plt.legend()
# plt.title(f'Speed timeseries - veh_id = {bike_id}')

# plt.figure()
# plt.plot(bike_df['time'], bike_df['a_ekf'], label='25 fps')
# plt.plot(bike_df_10fps['time'], bike_df_10fps['a_ekf'], 'o--', markersize=2, label='10 fps')
# plt.legend()
# plt.title(f'Acceleration timeseries - veh_id = {bike_id}')

# plt.show()
