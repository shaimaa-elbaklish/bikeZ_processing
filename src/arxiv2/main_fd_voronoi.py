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
import pathlib
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from pedpy import load_trajectory
from pedpy import plot_measurement_setup

from _constants import BikeZ_Config
from tools_voronoi import prepare_data_pedpy
from tools_voronoi import define_measurement_setup
from tools_voronoi import compute_voronoi_states, compute_voronoi_states_lines

# #############################################################################
# CONSTANTS
# #############################################################################
date = BikeZ_Config.avail_dates[0]
intersection = BikeZ_Config.avail_intersections[2]
all_time_slots = ['AM1', 'AM2', 'AM3', 'AM6'] # [f'AM{i}' for i in range(1, 7)]
code= 'E'

X_2056_offset = BikeZ_Config.X_2056_Bounds[0]
Y_2056_offset = BikeZ_Config.Y_2056_Bounds[0]

kml_path = "../maps/from_swisstopo/gessnerbrucke.kml"

# #############################################################################
# METHODS
# #############################################################################
def aggregate_FD(ts_df, max_density=180.0, bin_width=0.3, min_observations=15):
    ts_df['Density_Bin'] = pd.cut(x=ts_df['Density'], bins=np.arange(0, max_density, bin_width))
    agg_df = ts_df.groupby(["Density_Bin"], observed=False).agg({
        "Density": "mean", 
        "Flow": "mean",
        "Speed": "mean",
        "Density_Bin": "count"
    })
    agg_df = agg_df.rename(
        columns={"Density_Bin": "Num_Observations"}
    )
    agg_df = agg_df.dropna()
    print(agg_df["Num_Observations"].min(), agg_df["Num_Observations"].max())
    print(agg_df["Num_Observations"].mean(), agg_df["Num_Observations"].median())
    agg_df = agg_df[agg_df["Num_Observations"] >= min_observations]
    return agg_df


# #############################################################################
# MAIN: Load data
# #############################################################################
walkable_area, measurement_areas, measurement_lines = define_measurement_setup(
    kml_path, X_2056_offset, Y_2056_offset
)
# plt.figure()
# plot_measurement_setup(
#     walkable_area=walkable_area, traj=traj, traj_alpha=0.5, traj_width=1,
#     # measurement_areas=measurement_areas, ma_line_width=2, ma_alpha=0.25,
#     measurement_lines=measurement_lines, ma_line_width=2, ma_alpha=0.25,
# ).set_aspect("equal")
# plt.show()

# trajectories after EKF
ts_df_all = None
ts_df_per_line = [None]*len(measurement_lines)
for time_slot in all_time_slots:
    trajfilepath = f"../data/pedpy_traj/{date}_{intersection}_{time_slot}_{code}.txt"
    if not os.path.exists(trajfilepath):
        filename = f"trajectories_bikes_{date}_{intersection}_{time_slot}_{code}-1-ekf.csv"
        df = pd.read_csv(BikeZ_Config.data_root + f"{date}/{intersection}/{filename}")
        if time_slot == 'AM2':
            df = df[df['veh_id'] != 276] # temp fix for now
        df = prepare_data_pedpy(df, BikeZ_Config.fps, walkable_area)
        df.to_csv(trajfilepath, index=False, sep='\t', header=False)
        del df
        
    traj = load_trajectory(
        trajectory_file=pathlib.Path(trajfilepath),
        default_frame_rate=BikeZ_Config.fps,
        default_unit='m'
    )
    
    (
     voronoi_states_all, voronoi_states_lines, _
    ) = compute_voronoi_states_lines(traj, walkable_area, measurement_lines, return_full=True)
    if ts_df_all is None:
        ts_df_all = voronoi_states_all.copy()
    else:
        ts_df_all = pd.concat((ts_df_all, voronoi_states_all), ignore_index=True)
    
    for i in range(len(measurement_lines)):
        if ts_df_per_line[i] is None:
            ts_df_per_line[i] = voronoi_states_lines[i].copy()
        else:
            ts_df_per_line[i] = pd.concat((ts_df_per_line[i], voronoi_states_lines[i]), ignore_index=True)
        
print('TSE DONE!')
ts_df_all.to_csv(f"../data/Voronoi_TS_{date}_{intersection}_AM_{code}.csv", index=False)
sys.exit(1)
    
ts_df_all['Speed_Temp'] = ts_df_all['Speed']
ts_df_all['Speed'] = ts_df_all['Speed_CL']
agg_df = aggregate_FD(ts_df_all, max_density=180.0, bin_width=0.3, min_observations=50)

plt.figure()
plt.scatter(agg_df['Density'], agg_df['Flow'])

plt.figure()
plt.scatter(agg_df['Density'], agg_df['Speed'])




sys.exit(1)

# voronoi_states_all, voronoi_states_areas, individual_joined = compute_voronoi_states(
#     traj, walkable_area, measurement_areas, return_full=True
# )

# agg_df = aggregate_FD(voronoi_states_all, max_density=180.0, bin_width=0.3, min_observations=15)

# plt.figure()
# plt.scatter(agg_df['Density'], agg_df['Flow'])

# plt.figure()
# plt.scatter(agg_df['Density'], agg_df['Speed'])

