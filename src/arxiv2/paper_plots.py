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
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

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
timeslot = BikeZ_Config.avail_timeslots[date][(intersection, code)][3] # 'AM2'

XY_2056_Bounds = BikeZ_Config.XY_2056_Bounds[date][(intersection, code)]
X_2056_offset = XY_2056_Bounds[0][0]
Y_2056_offset = XY_2056_Bounds[1][0]

# #############################################################################
# MAIN: Kalman
# #############################################################################
if mode == "bike":
    filename = f"trajectories_bikes_{date}_{intersection}_{timeslot}_{code}-1-ekf.csv"
else:
    filename = f"trajectories_vehicles_{date}_{intersection}_{timeslot}_{code}-1-ekf.csv"
df = pd.read_csv(data_root + f"{date}/{intersection}/{filename}")
df = df.dropna()

gap_records = []

for veh_id, veh_df in df.groupby('veh_id'):
    missing = veh_df['missing'].values
    times = veh_df['time'].values
    
    in_gap = False
    gap_start = None
    
    for i, m in enumerate(missing):
        if m and not in_gap:
            in_gap = True
            gap_start = i
        elif not m and in_gap:
            gap_len = i - gap_start  # number of missing frames
            gap_duration = times[i] - times[gap_start]  # seconds
            gap_records.append({
                'veh_id': veh_id,
                'gap_start_idx': gap_start,
                'gap_len_frames': gap_len,
                'gap_duration_s': gap_duration
            })
            in_gap = False
    
    # Handle gap that runs to end of trajectory
    if in_gap:
        gap_len = len(missing) - gap_start
        gap_duration = times[-1] - times[gap_start]
        gap_records.append({
            'veh_id': veh_id,
            'gap_start_idx': gap_start,
            'gap_len_frames': gap_len,
            'gap_duration_s': gap_duration
        })

gaps_df = pd.DataFrame(gap_records)
top5_ids = (gaps_df.groupby('veh_id')['gap_duration_s']
            .max()
            .nlargest(5)
            .index.tolist())

bike_id = top5_ids[0]

veh = df[df['veh_id'] == bike_id].copy().reset_index(drop=True)
gaps = gaps_df[gaps_df['veh_id'] == bike_id]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle(f'Gap Inference + Filtering — Vehicle {bike_id} — Gap duration = {gaps["gap_duration_s"].item():.2f}', fontsize=12)

# =========================================================================
# TOP: XY Trajectory
# =========================================================================
ax = axes[0]

# Raw observations
obs = veh[~veh['missing']]
ax.scatter(obs['x'], obs['y'], s=8, color='gray', alpha=0.5, zorder=2,
           label='Raw observations')

# Full EKF+RTS trajectory
ax.plot(veh['x_ekf'], veh['y_ekf'], color='steelblue', linewidth=1.5,
        alpha=0.8, zorder=3, label='EKF+RTS')

# Highlight gap segments in orange
for _, gap in gaps.iterrows():
    s = int(gap['gap_start_idx'])
    e = int(s + gap['gap_len_frames'])
    gap_rows = veh.iloc[s:e+1]
    ax.plot(gap_rows['x_ekf'], gap_rows['y_ekf'], color='orange', linewidth=2.5,
            zorder=4, label='Inferred gap' if _ == gaps.index[0] else '_nolegend_')
    ax.scatter(gap_rows['x_ekf'].iloc[0],  gap_rows['y_ekf'].iloc[0],
               color='orange', s=80, marker='^', zorder=5, edgecolors='k', linewidths=0.5)
    ax.scatter(gap_rows['x_ekf'].iloc[-1], gap_rows['y_ekf'].iloc[-1],
               color='orange', s=80, marker='v', zorder=5, edgecolors='k', linewidths=0.5)

from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
# ── Inset zoom: first gap + context window ────────────────────────────────────
if len(gaps) > 0:
    first_gap  = gaps.iloc[0]
    s = int(first_gap['gap_start_idx'])
    e = int(s + first_gap['gap_len_frames'])
    context = 50  # frames before/after gap to include in zoom

    s_ctx = max(0, s - context)
    e_ctx = min(len(veh) - 1, e + context)
    zoom_rows  = veh.iloc[s_ctx:e_ctx+1]
    gap_rows   = veh.iloc[s:e+1]
    obs_zoom   = zoom_rows[~zoom_rows['missing']]

    # Inset axes — position: upper right; adjust loc as needed
    axins = inset_axes(ax, width='40%', height='35%', loc='lower left')

    axins.plot(zoom_rows['x_ekf'], zoom_rows['y_ekf'],
               color='steelblue', linewidth=1.5, alpha=0.8)
    axins.scatter(obs_zoom['x'], obs_zoom['y'],
                  s=12, color='gray', alpha=0.6, zorder=2)
    axins.plot(gap_rows['x_ekf'], gap_rows['y_ekf'],
               color='orange', linewidth=2.5, zorder=4)
    axins.scatter(gap_rows['x_ekf'].iloc[0],  gap_rows['y_ekf'].iloc[0],
                  color='orange', s=80, marker='^', zorder=5, edgecolors='k', linewidths=0.5)
    axins.scatter(gap_rows['x_ekf'].iloc[-1], gap_rows['y_ekf'].iloc[-1],
                  color='orange', s=80, marker='v', zorder=5, edgecolors='k', linewidths=0.5)

    # Zoom limits with a small margin
    margin = 1.0  # metres
    x_min, x_max = zoom_rows['x_ekf'].min() - margin, zoom_rows['x_ekf'].max() + margin
    y_min, y_max = zoom_rows['y_ekf'].min() - margin, zoom_rows['y_ekf'].max() + margin
    axins.set_xlim(x_min, x_max)
    axins.set_ylim(y_min, y_max)
    axins.set_aspect('equal')
    # axins.tick_params(labelsize=7)
    axins.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    axins.grid(True, alpha=0.3)

    # Draw connecting lines from inset to main axes
    mark_inset(ax, axins, loc1=2, loc2=4, fc='none', ec='black', linewidth=0.8)


ax.set_xlabel('x [m]', fontsize=10)
ax.set_ylabel('y [m]', fontsize=10)
# ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=9)

# =========================================================================
# BOTTOM: Time series — speed, acceleration, angular velocity
# =========================================================================
ax2 = axes[1]

t = veh['time'].to_numpy()
ax2.plot(t, veh['speed'] / 3.6,      color='gray',     linewidth=1,   alpha=0.6, label='Speed raw [m/s]')
ax2.plot(t, veh['speed_ekf'] / 3.6,  color='steelblue', linewidth=1.5, label='Speed EKF [m/s]')

# Twin axes for acceleration and angular velocity
ax2b = ax2.twinx()
ax2b.plot(t, veh['a_ekf'],           color='tomato',   linewidth=1.2, linestyle='--', label='a EKF [m/s²]')
ax2b.plot(t, veh['angular_vel_ekf'], color='seagreen', linewidth=1.2, linestyle=':',  label='ω EKF [rad/s]')
ax2b.set_ylabel('a [m/s²] / ω [rad/s]', fontsize=9)
ax2b.axhline(0, color='k', linewidth=0.5)

# Shade gap regions
for _, gap in gaps.iterrows():
    s = int(gap['gap_start_idx'])
    e = int(s + gap['gap_len_frames'])
    t_start = veh['time'].iloc[min(s, len(veh)-1)]
    t_end   = veh['time'].iloc[min(e, len(veh)-1)]
    ax2.axvspan(t_start, t_end, alpha=0.15, color='orange',
                label='Gap region' if _ == gaps.index[0] else '_nolegend_')

ax2.set_xlabel('Time [s]', fontsize=10)
ax2.set_ylabel('Speed [m/s]', fontsize=10)
ax2.grid(True, alpha=0.3)

# Combined legend
lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax2b.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc='lower right')

plt.tight_layout()
# plt.savefig(f'filtering_{bike_id}.png', dpi=150)
plt.show()

# #############################################################################
# MAIN: Queuing
# #############################################################################
if mode == "bike":
    filename = f"trajectories_bikes_{date}_{intersection}_{timeslot}_{code}-1-ekf-lane.csv"
else:
    filename = f"trajectories_vehicles_{date}_{intersection}_{timeslot}_{code}-1-ekf-lane.csv"
df = pd.read_csv(data_root + f"{date}/{intersection}/{filename}")

from matplotlib.collections import LineCollection
from shapely.geometry import Point

plt.figure()
seg_key = 'Zollstr_WB'

# Load geometry, segment, and movement registries
import pickle
registry_path = f"../data/registry_{date}_{intersection}_{code}.pkl"
registry = pickle.load(open(registry_path, 'rb'))
segment_registry  = registry['segment_registry']
polygon = segment_registry[seg_key]['validity_polygon']

for bike_id, bike_df in df.groupby('veh_id'):
    bike_df = bike_df[['x_ekf', 'y_ekf', 'segment_id']].copy()
    if bike_df.empty:
        continue
    # Condition 1: already matched
    is_matched = seg_key in bike_df['segment_id'].values
    # Condition 2: any point inside validity polygon
    intersects_polygon = any(
        polygon.contains(Point(x, y)) 
        for x, y in zip(bike_df['x_ekf'], bike_df['y_ekf'])
    )
    if is_matched:
        plt.plot(bike_df['x_ekf'], bike_df['y_ekf'], color='blue', alpha=0.7)
    elif intersects_polygon:
        plt.plot(bike_df['x_ekf'], bike_df['y_ekf'], color='red', alpha=0.7)

plt.xlabel('x_ekf')
plt.ylabel('y_ekf')
plt.title(f'Trajectories for segment {seg_key}')
plt.axis('equal')
plt.show()
sys.exit(1)



fig, axs = plt.subplots(1, 2, figsize=(10, 4))
cmap = cm.RdYlGn  # red=slow, green=fast

seg_key = 'LangstrN_SB'
df_seg = df[df['segment_id'] == seg_key].copy()
unique_ids = df_seg['veh_id'].unique()
# Global speed range for consistent colormap
vmin = df_seg['speed_ekf'].min()
vmax = df_seg['speed_ekf'].max()
norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

segments_list = []
colors_list = []
for veh_id, veh in df_seg.groupby('veh_id'):
    veh = veh.sort_values('time')
    t = veh['time'].values
    s = veh['s'].values
    spd = veh['speed_ekf'].values

    # Build (N-1, 2, 2) segments array
    points = np.stack([t, s], axis=1)          # (N, 2)
    segs = np.stack([points[:-1], points[1:]], axis=1)  # (N-1, 2, 2)
    segments_list.append(segs)
    colors_list.append(spd[:-1])               # color by start point speed

all_segments = np.concatenate(segments_list, axis=0)
all_colors   = np.concatenate(colors_list,   axis=0)

lc = LineCollection(all_segments, cmap=cmap, norm=norm,
                    linewidth=1.2, capstyle='round')
lc.set_array(all_colors)
axs[0].add_collection(lc)
axs[0].autoscale_view()
cbar = fig.colorbar(lc, ax=axs[0])
cbar.set_label('Speed [km/h]')
axs[0].set_xlabel('Time [s]')
axs[0].set_ylabel('s [m]')
axs[0].set_title(seg_key)
axs[0].grid(True, alpha=0.3)


seg_key = 'Zollstr_WB'
df_seg = df[df['segment_id'] == seg_key].copy()
unique_ids = df_seg['veh_id'].unique()
# Global speed range for consistent colormap
vmin = df_seg['speed_ekf'].min()
vmax = df_seg['speed_ekf'].max()
norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

segments_list = []
colors_list = []
for veh_id, veh in df_seg.groupby('veh_id'):
    veh = veh.sort_values('time')
    t = veh['time'].values
    s = veh['s'].values
    spd = veh['speed_ekf'].values

    # Build (N-1, 2, 2) segments array
    points = np.stack([t, s], axis=1)          # (N, 2)
    segs = np.stack([points[:-1], points[1:]], axis=1)  # (N-1, 2, 2)
    segments_list.append(segs)
    colors_list.append(spd[:-1])               # color by start point speed

all_segments = np.concatenate(segments_list, axis=0)
all_colors   = np.concatenate(colors_list,   axis=0)

lc = LineCollection(all_segments, cmap=cmap, norm=norm,
                    linewidth=1.2, capstyle='round')
lc.set_array(all_colors)
axs[1].add_collection(lc)
axs[1].autoscale_view()
cbar = fig.colorbar(lc, ax=axs[1])
cbar.set_label('Speed [km/h]')
axs[1].set_xlabel('Time [s]')
axs[1].set_ylabel('s [m]')
axs[1].set_title(seg_key)
axs[1].grid(True, alpha=0.3)

fig.tight_layout()

