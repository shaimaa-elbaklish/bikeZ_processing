"""
example_lane_coords_usage.py
----------------------------
Practical examples of how to use the lane coordinate output dataframe
produced by to_lane_coordinates().

Assumes you have already run main_coordinate_transform.py and have
a saved CSV at the path below, or that mod_df is in memory.

Columns of interest:
    veh_id, time, movement_key, segment_id, segment_type, segment_role,
    s, d, s_dot, d_dot, s_ddot, d_ddot,
    in_bike_lane, d_to_bike_boundary, s_decreasing,
    speed_ekf, angle_ekf, x_act_ekf, y_act_ekf
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

from _constants import BikeZ_Config

# #############################################################################
# 0. LOAD DATA
# #############################################################################
BikeZ_Config = BikeZ_Config()

# Specify Trajectory File
date = BikeZ_Config.avail_dates[0]
campaign = f"Zurich_2025{date[5:7]}" # June or September
mode = BikeZ_Config.avail_modes[0] # 0: Bike, 1: Vehicle
data_root = BikeZ_Config.data_root[campaign][mode]

intersection, code = BikeZ_Config.avail_intersections[date][0]
timeslot = BikeZ_Config.avail_timeslots[date][(intersection, code)][0]

XY_2056_Bounds = BikeZ_Config.XY_2056_Bounds[date][(intersection, code)]
X_2056_offset = XY_2056_Bounds[0][0]
Y_2056_offset = XY_2056_Bounds[1][0]

# Load the lane coordinate CSV (output of main_coordinate_transform.py)
filename = f"trajectories_bikes_{date}_{intersection}_{timeslot}_{code}-1-ekf-lane"
# df       = pd.read_csv(data_root + f"{date}/{intersection}/{filename}.csv")
df       = pd.read_parquet(data_root + f"{date}/{intersection}/{filename}.parquet")

# Load the registry (needed for plotting only)
registry_path  = f"../data/registry_{date}_{intersection}_{code}.pkl"
registry       = pickle.load(open(registry_path, 'rb'))
geometry_store    = registry['geometry_store']
segment_registry  = registry['segment_registry']
movement_registry = registry['movement_registry']
XY_2056_Bounds = BikeZ_Config.XY_2056_Bounds[date][(intersection, code)]


# #############################################################################
# MAIN
# #############################################################################

# ── Basic filtering ───────────────────────────────────────────────────────────
# Keep only well-matched rows
df_good = df[df['match_quality'].isin(['good', 'poor'])]

# Keep only a specific movement
df_mov = df_good[df_good['movement_key'] == 'LangstrN_SB_2_LangstrS_SB']

# Exclude reverse traversals
df_mov = df_mov[~df_mov['is_reverse']]

# ── Per-vehicle chain reconstruction ─────────────────────────────────────────
# Each vehicle may span multiple segments; group by veh_id and segment_role
fig, axs = plt.subplots(1, 3, figsize=(12, 4))
for veh_id, grp in df_mov.groupby('veh_id'):
    approach = grp[grp['segment_role'] == 'approach']
    turn     = grp[grp['segment_role'] == 'turn']
    departure= grp[grp['segment_role'] == 'departure']

    # s is continuous across the chain — plot full trajectory in lane coords
    axs[0].plot(approach['s_native'], approach['d_native'], alpha=0.3, color='steelblue')
    axs[1].plot(turn['s_native'], turn['d_native'], alpha=0.3, color='steelblue')
    axs[2].plot(departure['s_native'], departure['d_native'], alpha=0.3, color='steelblue')

axs[0].set(xlabel='s [m]', ylabel='d [m]', title='Approach')
axs[1].set(xlabel='s [m]', ylabel='d [m]', title='Turn')
axs[2].set(xlabel='s [m]', ylabel='d [m]', title='Departure')
fig.suptitle('Movement=LangstrN_SB → LangstrS_SB')
fig.tight_layout()

# ── Bike lane usage ───────────────────────────────────────────────────────────
# in_bike_lane: 1.0 = in bike lane, 0.0 = outside, NaN = no bike lane defined
df_bike = df_good[df_good['in_bike_lane'].notna()]
bike_lane_usage = (
    df_bike.groupby('movement_key')['in_bike_lane']
    .mean()
    .rename('fraction_in_bike_lane')
)
print(bike_lane_usage)

# ── Coordinate inversion: recover (x, y) from (s_native, d_native) ───────────
# The (s_native, d_native, segment_id) triple is invertible.
# Use the spline directly: evaluate position at s_native, then offset by d_native
# along the normal.
from scipy.interpolate import splev

def invert_lane_coordinates(row, segment_registry, geometry_store):
    seg_key = row['segment_id']
    if pd.isna(seg_key) or seg_key not in segment_registry:
        return pd.Series({'x_reconstructed': np.nan, 'y_reconstructed': np.nan})

    geom_key            = segment_registry[seg_key]['geometry_key']
    tck, unew, cum_dist = geometry_store[geom_key]['spline']

    s_nat = row['s_native']
    d_nat = row['d_native']

    # Interpolate parameter t from arc-length s_native
    t = float(np.interp(s_nat, cum_dist, unew))

    # Spline position and tangent at t
    x,  y  = splev(t, tck, der=0)
    dx, dy = splev(t, tck, der=1)

    # Unit normal (left of spline = positive d_native)
    tang = np.sqrt(dx**2 + dy**2)
    if tang < 1e-12:
        tang = 1.0
    nx = -dy / tang
    ny =  dx / tang

    return pd.Series({
        'x_reconstructed': float(x) + d_nat * nx,
        'y_reconstructed': float(y) + d_nat * ny,
    })

reconstructed = df_good.apply(
    invert_lane_coordinates, axis=1,
    segment_registry=segment_registry,
    geometry_store=geometry_store,
)
df_good[['x_reconstructed', 'y_reconstructed']] = reconstructed

# Reconstruction error (should be < 0.1m for well-matched rows)
df_good['reconstruction_error'] = np.sqrt(
    (df_good['x_reconstructed'] - df_good['x_ekf'])**2 +
    (df_good['y_reconstructed'] - df_good['y_ekf'])**2
)
print(df_good['reconstruction_error'].describe())

# ── Longitudinal Position on Segment ─────────────────────────────────────
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