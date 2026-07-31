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
import sys
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

from _constants import BikeZ_Config

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

intersection, code = BikeZ_Config.avail_intersections[date][0]
# all_timeslots = BikeZ_Config.avail_timeslots[date][(intersection, code)]
timeslot = BikeZ_Config.avail_timeslots[date][(intersection, code)][0] # 'AM1' or 'PM1

XY_2056_Bounds = BikeZ_Config.XY_2056_Bounds[date][(intersection, code)]
X_2056_offset = XY_2056_Bounds[0][0]
Y_2056_offset = XY_2056_Bounds[1][0]

# #############################################################################
# MAIN: Load Data
# #############################################################################
if mode == "bike":
    filename = f"trajectories_bikes_{date}_{intersection}_{timeslot}_{code}-1-ekf-lane"
else:
    filename = f"trajectories_vehicles_{date}_{intersection}_{timeslot}_{code}-1-ekf-lane"
# df = pd.read_csv(data_root + f"{date}/{intersection}/{filename}.csv")
df = pd.read_parquet(data_root + f"{date}/{intersection}/{filename}.parquet")

# Load geometry, segment, and movement registries
registry_path = f"../data/registry_{date}_{intersection}_{code}.pkl"
registry = pickle.load(open(registry_path, 'rb'))
geometry_store    = registry['geometry_store']
segment_registry  = registry['segment_registry']
movement_registry = registry['movement_registry']
max_chain_length  = registry['metadata'].get('max_chain_length', 3)

# #############################################################################
# MAIN: Plot Longitudinal Position on a Segment
# #############################################################################
fig, axs = plt.subplots(2, 2, figsize=(8, 8))

subplots_def = [
    ('LangstrN_SB', axs[0, 0]),
    ('LangstrS_NB', axs[1, 1]),
    ('Zollstr_WB', axs[0, 1]),
    ('Roentgenstr_EB', axs[1, 0])
]
for seg_key, ax in subplots_def:
    seg_df = df[df['segment_id'] == seg_key].copy()
    geom_key = segment_registry[seg_key]['geometry_key']
    s_stop = geometry_store[geom_key]['s_stop']
    s_yield = geometry_store[geom_key]['s_yield']
    s_change = geometry_store[geom_key]['s_change']
    ax.axhline(y=s_stop, label='Stop Line', color='black', linestyle='solid')
    ax.axhline(y=s_yield, label='Yield Line', color='black', linestyle='dotted')
    ax.axhline(y=s_change, label='s_change', color='black', linestyle='dashed')
    ax.legend()
    norm = Normalize(vmin=0, vmax=seg_df['speed_ekf'].max())
    for veh_id, veh_df in seg_df.groupby('veh_id'):
        veh_df = veh_df.sort_values('time')
        x = veh_df['time'].to_numpy()
        y = veh_df['s_native'].to_numpy()
        c = veh_df['speed_ekf'].to_numpy()
        # Create line segments
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(
            segments, cmap='jet_r', norm=norm
        )
        # Color each segment by the average speed of its endpoints
        lc.set_array((c[:-1] + c[1:]) / 2)
        lc.set_linewidth(1.5)
        ax.add_collection(lc)
    ax.autoscale()
    cbar = plt.colorbar(lc, ax=ax)
    cbar.set_label('speed_ekf (km/h)')
    ax.set_title(seg_key)
    ax.set(xlabel='Time (s)', ylabel='s_native (m)')

fig.tight_layout()

# #############################################################################
# MAIN: Plot Matching versus Validity on a Segment
# #############################################################################
from shapely import Point
from shapely.plotting import plot_polygon
from scipy.interpolate import splev

fig, axs = plt.subplots(2, 5, figsize=(20, 8))

subplots_def = [
    ('LangstrN_SB', axs[0, 0]),
    ('LangstrN_NB', axs[0, 1]),
    ('Zollstr_WB', axs[0, 2]),
    ('Zollstr_EB', axs[0, 3]),
    ('Matteng_NB', axs[0, 4]),
    ('LangstrS_SB', axs[1, 0]),
    ('LangstrS_NB', axs[1, 1]),
    ('Roentgenstr_WB', axs[1, 2]),
    ('Roentgenstr_EB', axs[1, 3]),
    ('Matteng_SB', axs[1, 4]),
]
for seg_key, ax in subplots_def:
    seg_df = df[df['segment_id'] == seg_key].copy()
    seg_bike_ids = seg_df['veh_id'].unique()

    # Plot Centerline
    geom_key = segment_registry[seg_key]['geometry_key']
    tck, unew, cum_dist = geometry_store[geom_key]['spline']
    L = geometry_store[geom_key]['total_length']
    s_v  = np.linspace(0, L, 200)
    t_v  = np.interp(s_v, cum_dist, unew)
    xc, yc   = splev(t_v, tck, der=0)
    ax.plot(xc, yc, color='gray', alpha=0.75, zorder=5, label=geom_key)
    poly = segment_registry[seg_key].get('validity_polygon')
    plot_polygon(poly, ax=ax, facecolor='lightgray', edgecolor='gray', alpha=0.75, 
                 zorder=1, add_points=False, label='Validity Polygon')

    # Plot Matched Vehicles
    first_plot = True
    for bike_id in seg_bike_ids:
        bike_df = df[df['veh_id'] == bike_id]
        if first_plot:
            ax.plot(bike_df['x_ekf'], bike_df['y_ekf'], color='tab:blue', alpha=0.5, label='Matched')
            first_plot = False
        else:
            ax.plot(bike_df['x_ekf'], bike_df['y_ekf'], color='tab:blue', alpha=0.5)

    # Plot Unmatched Vehicles but Coincident inside Validity Polygon
    all_bike_ids = df['veh_id'].unique()
    unmatched_dict = {}
    first_plot = True
    for bike_id in all_bike_ids:
        if bike_id in seg_bike_ids:
            continue
        bike_df = df[df['veh_id'] == bike_id]
        traj_xy = bike_df[['x_ekf', 'y_ekf']].to_numpy()
        entry_idx, exit_idx = None, None
        for i in range(len(traj_xy)):
            pt     = Point(float(traj_xy[i, 0]), float(traj_xy[i, 1]))
            inside = poly.contains(pt)
            if entry_idx is None:
                if inside:
                    entry_idx = i
            else:
                if not inside:
                    exit_idx = i
                    break
        if entry_idx is not None and exit_idx is not None and exit_idx - entry_idx >= 5:
            if first_plot:
                ax.plot(bike_df['x_ekf'], bike_df['y_ekf'], color='tab:red', alpha=0.5, label='Unmatched')
                first_plot = False
            else:
                ax.plot(bike_df['x_ekf'], bike_df['y_ekf'], color='tab:red', alpha=0.5)
            unmatched_dict[bike_id] = (entry_idx, exit_idx)
        
    ax.legend()
    ax.set_xlim([XY_2056_Bounds[0][0]-X_2056_offset, XY_2056_Bounds[0][1]-X_2056_offset])
    ax.set_ylim([XY_2056_Bounds[1][0]-Y_2056_offset, XY_2056_Bounds[1][1]-Y_2056_offset])
    ax.set_title(seg_key)

fig.tight_layout()
plt.show()
