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
import matplotlib.dates as mdates

from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
from matplotlib.ticker import MaxNLocator

from _constants import BikeZ_Config
from tools_utils import extract_all_gaps

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
timeslot = BikeZ_Config.avail_timeslots[date][(intersection, code)][1] # 'AM2'

XY_2056_Bounds = BikeZ_Config.XY_2056_Bounds[date][(intersection, code)]
X_2056_offset = XY_2056_Bounds[0][0]
Y_2056_offset = XY_2056_Bounds[1][0]

loc_num = BikeZ_Config.location_map[(date[5:7], intersection, code)]
subsampled_data_root = "C:/Users/ShaimaaElBaklish/OneDrive - ETH Zurich/BikeZ-Subsampled/"

# ── global font sizes ────────────────────────────────────────────────────────
FS_TITLE   = 13   # subplot titles
FS_LABEL   = 12   # axis labels
FS_ANNOT   = 12   # annotations
FS_TICK    = 10   # axis tick labels
FS_LEGEND  = 10   # legend entries

plt.rcParams.update({
    # 'font.family': 'sans-serif',
    # 'font.sans-serif': ['Arial', 'Helvetica'],
    'axes.labelsize': FS_LABEL,
    'axes.titlesize': FS_TITLE,
    'legend.fontsize': FS_LEGEND,
    'xtick.labelsize': FS_TICK,
    'ytick.labelsize': FS_TICK,
})

COLOR_BIKE  = '#0072B2'   # blue
COLOR_VEH   = '#009E73'   # green
COLOR_GRID  = '#333333'   # near-black — master grid

COLOR_RAW        = '#0072B2'   # blue — 25 fps observed samples
COLOR_GAP        = '#D55E00'   # orange — gap-bridged (originally missing) segments
COLOR_RESAMPLED  = '#009E73'   # green — 10 fps resampled points
COLOR_GRID       = '#333333'   # near-black — master grid (Panel A only)


# #############################################################################
# MAIN
# #############################################################################
# Select bike ID
bike_id = 43 # 383 (very long)

# 25 fps
filename = f"trajectories_{mode}s_{date}_{intersection}_{timeslot}_{code}-1-ekf"
# df = pd.read_csv(data_root + f"{date}/{intersection}/{filename}.csv")
df_25fps = pd.read_parquet(data_root + f"{date}/{intersection}/{filename}.parquet")
df_25fps = df_25fps[df_25fps['veh_id'] == bike_id]
gaps_df = extract_all_gaps(df_25fps, include_datetime=True)

# 10 fps
filename = f"location_{loc_num}/{loc_num}_{mode}s_{date}_{timeslot}.csv"
df_10fps = pd.read_csv(subsampled_data_root + filename)
df_10fps['datetime'] = pd.to_datetime(df_10fps['datetime'], format='ISO8601')
df_10fps = df_10fps[df_10fps['veh_id'] == bike_id]

# ensure datetime is tz-native
def _to_naive_datetime(s):
    if getattr(s.dt, 'tz', None) is not None:
        s = s.dt.tz_localize(None)
    return s

df_25fps['datetime'] = _to_naive_datetime(df_25fps['datetime'])
df_10fps['datetime'] = _to_naive_datetime(df_10fps['datetime'])
gaps_df['start_datetime'] = _to_naive_datetime(gaps_df['start_datetime'])
gaps_df['end_datetime']   = _to_naive_datetime(gaps_df['end_datetime'])

dt25 = mdates.date2num(df_25fps['datetime'])
dt10 = mdates.date2num(df_10fps['datetime'])

# plotting
fig, axes = plt.subplots(1, 3, figsize=(10, 4))

# -----------------------------------------------------------------------
# PANEL A — Homogeneous time-grid schematic (synthetic, illustrative)
axA = axes[0]

rng = np.random.default_rng(7)
dt_nom = 1.0 / 25.0
n_pts = 20

t_bike_raw = 0.0 + np.arange(n_pts) * dt_nom + rng.normal(0, 0.004, n_pts)
t_bike_raw[0] = 0.0
t_veh_raw  = 0.18 + np.arange(n_pts) * dt_nom + rng.normal(0, 0.004, n_pts)

t_span = (0.0, max(t_bike_raw[-1], t_veh_raw[-1]) + 0.05)
dt_master = 0.1
t_master = np.arange(np.ceil(t_span[0] / dt_master) * dt_master,
                      t_span[1], dt_master)

y_grid, y_bike, y_veh = 2, 1, 0   # grid on top
y_bottom = -0.8   # matches set_ylim lower bound below — define once, reuse
tick_half_h = 0.15                # half-height of the vline markers

# ---- restrict each row's resampled ticks to its own observed [t0, t1] span ----
# (mirrors the no-extrapolation rule in _slice_target_grid)
bike_t0, bike_t1 = t_bike_raw.min(), t_bike_raw.max()
veh_t0,  veh_t1  = t_veh_raw.min(),  t_veh_raw.max()

t_master_bike = t_master[(t_master >= bike_t0) & (t_master <= bike_t1)]
t_master_veh  = t_master[(t_master >= veh_t0)  & (t_master <= veh_t1)]

# ---- dashed guide lines: now reach the x-axis, not just y_veh ----
for t in t_master:
    axA.plot([t, t], [y_bottom, y_grid], ls=':', lw=0.6, color='0.75', zorder=0)

# ---- master grid timeline: solid horizontal line + tick markers ----
axA.hlines(y_grid, t_span[0], t_span[1], color=COLOR_GRID, lw=1.2, zorder=1)
axA.vlines(t_master, y_grid - tick_half_h, y_grid + tick_half_h,
           color=COLOR_GRID, lw=2.5, zorder=3)

# ---- bicycle row: native 25 fps samples + 10 fps ticks only within its own span ----
axA.scatter(t_bike_raw, np.full(n_pts, y_bike),
            s=45, color=COLOR_BIKE, alpha=0.4, edgecolors='none', zorder=2)
axA.vlines(t_master_bike, y_bike - tick_half_h, y_bike + tick_half_h,
           color=COLOR_BIKE, lw=2.5, zorder=3)

# ---- vehicle row: native 25 fps samples + 10 fps ticks only within its own span ----
axA.scatter(t_veh_raw, np.full(n_pts, y_veh),
            s=45, color=COLOR_VEH, alpha=0.4, edgecolors='none', zorder=2)
axA.vlines(t_master_veh, y_veh - tick_half_h, y_veh + tick_half_h,
           color=COLOR_VEH, lw=2.5, zorder=3)

axA.set_yticks([y_grid, y_bike, y_veh])
axA.set_yticklabels(['Master grid\n(10 Hz, shared)', 'Bicycles\n(raw, 25 fps)',
                      'Vehicles\n(raw, 25 fps)'], fontsize=FS_TICK)
axA.set_xlabel('Time [s]', fontsize=FS_LABEL)
axA.set_xlim([-0.04, 0.64])
axA.set_ylim(-0.8, 3.5)
axA.set_title('(a)', pad=5, fontsize=FS_TITLE, fontweight='bold')

# ---- legend distinguishing native samples vs. resampled grid ticks ----
legend_elems_A = [
    Line2D([0], [0], marker='o', color='none', markerfacecolor='0.6',
           alpha=0.5, markersize=6, label='Native 25 fps sample'),
    Line2D([0], [0], color='0.15', lw=2.5, label='Resampled (10 fps)'),
]
axA.legend(handles=legend_elems_A, loc='upper center',
           # bbox_to_anchor=(0.5, -0.28), ncol=1, frameon=True,
           fontsize=FS_LEGEND, handletextpad=0.6, columnspacing=1.2)

# -----------------------------------------------------------------------
# PANEL B — x–y path overlay: 25 fps vs. 10 fps reconstruction
axB = axes[1]

x25 = df_25fps['x_ekf'].to_numpy()
y25 = df_25fps['y_ekf'].to_numpy()
missing25 = df_25fps['missing'].to_numpy()

points = np.column_stack([x25, y25]).reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)

seg_missing = missing25[:-1] | missing25[1:]
seg_colors = np.where(seg_missing, COLOR_GAP, COLOR_RAW)

lc = LineCollection(segments, colors=seg_colors, linewidths=1.2, zorder=5)
axB.add_collection(lc)
axB.autoscale_view()
axB.set_xlim([18, None])

# ---- 10 fps reconstructed points, split by in_gap status ----
mask_gap = df_10fps['in_gap'].to_numpy(dtype=bool)

axB.plot(df_10fps.loc[~mask_gap, 'x_ekf'], df_10fps.loc[~mask_gap, 'y_ekf'],
         'o', color=COLOR_RESAMPLED, ms=4.5, mec='white', mew=0.4,
         zorder=2, label='10 fps', alpha=0.8)

axB.plot(df_10fps.loc[mask_gap, 'x_ekf'], df_10fps.loc[mask_gap, 'y_ekf'],
         'o', color=COLOR_GAP, ms=4.5, mec='white', mew=0.4,
         zorder=2, label='10 fps (in gap)', alpha=0.6)

axB.set_xlabel('$x$ [m]', fontsize=FS_LABEL)
axB.set_ylabel('$y$ [m]', fontsize=FS_LABEL)
axB.set_title('(b)', pad=5, fontsize=FS_TITLE, fontweight='bold')

legend_elems_B = [
    Line2D([0], [0], color=COLOR_RAW, lw=1.2, label='25 fps (observed)'),
    Line2D([0], [0], color=COLOR_GAP, lw=1.2, label='25 fps (gap)'),
    Line2D([0], [0], marker='o', color='none', markerfacecolor=COLOR_RESAMPLED,
           markeredgecolor='white', markersize=6, label='10 fps (resampled)'),
    Line2D([0], [0], marker='o', color='none', markerfacecolor=COLOR_GAP,
           markeredgecolor='white', markersize=6, label='10 fps (in gap)'),
]
# axB.legend(handles=legend_elems_B, loc='best', frameon=True, fontsize=FS_LEGEND)
axB.legend(handles=legend_elems_B, loc='lower left', 
           bbox_to_anchor=(0.01, 0.24),   # (x, y) in axes-fraction coords, 0-1
           frameon=True, fontsize=FS_LEGEND)

# -----------------------------------------------------------------------
# PANEL C — heading angle + speed vs. datetime (dual y-axis)
COLOR_HEADING = '#D62728'   # red — distinct from RAW (blue), GAP (orange), RESAMPLED

axC = axes[2]
axC2 = axC.twinx()

angle_25_deg = np.degrees(df_25fps['angle_ekf'].to_numpy())
angle_10_deg = np.degrees(df_10fps['angle_ekf'].to_numpy())

# ---- speed on primary (left) axis, solid blue ----
axC.plot(dt25, df_25fps['speed_ekf'] / 3.6, '-', color=COLOR_RAW, lw=1.2, zorder=5)
axC.plot(dt10, df_10fps['speed_ekf'] / 3.6, 's', color=COLOR_RESAMPLED,
         ms=4, mec='white', mew=0.4, alpha=0.7)
axC.set_ylabel('Speed [m/s]', fontsize=FS_LABEL)
axC.set_ylim([0, 6])

# ---- heading angle on secondary (right) axis, red ----
axC2.plot(dt25, angle_25_deg, '-', color=COLOR_HEADING, lw=1.2, zorder=5)
axC2.plot(dt10, angle_10_deg, 'o', color=COLOR_RESAMPLED,
          ms=4, mec='white', mew=0.4, alpha=0.7)
axC2.set_ylabel('Heading angle [deg]', fontsize=FS_LABEL)
axC2.set_ylim([0, 180])
axC2.yaxis.set_major_locator(MaxNLocator(nbins=6))   # at most 6 ticks

# axC.set_xlabel('Datetime', fontsize=FS_LABEL)
axC.set_title('(c)', pad=5, fontsize=FS_TITLE, fontweight='bold') # loc = 'left'

# ---- exactly 3 ticks: start, middle, end of the trajectory's time span ----
t_min = min(np.nanmin(dt25), np.nanmin(dt10))
t_max = max(np.nanmax(dt25), np.nanmax(dt10))
t_mid = t_min + (t_max - t_min) / 2
axC.set_xlim(t_min, t_max)

axC.set_xticks([t_min, t_mid, t_max])
axC.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
axC.tick_params(axis='x', rotation=30)

# -----------------------------------------------------------------------
# Shade gap-bridged intervals, skipping any invalid/NaT boundaries
GAP_ALPHA = 0.25

for i, row in enumerate(gaps_df.itertuples(index=False)):
    if pd.isna(row.start_datetime) or pd.isna(row.end_datetime):
        continue   # skip malformed gap entries rather than let them warp the axis
    t0 = mdates.date2num(row.start_datetime)
    t1 = mdates.date2num(row.end_datetime)
    axC.axvspan(t0, t1, color='darkorange', alpha=0.20, zorder=0, lw=0,
                label='Gap-bridged interval' if i == 0 else None)

legend_elems = [
    Line2D([0], [0], color=COLOR_RAW, lw=1.2, ls='-', label='Speed, 25 fps'),
    Line2D([0], [0], color=COLOR_RESAMPLED, marker='s', ls='', mec='white',
           label='Speed, 10 fps'),
    Line2D([0], [0], color=COLOR_HEADING, lw=1.2, ls='-', label='Angle, 25 fps'),
    Line2D([0], [0], color=COLOR_RESAMPLED, marker='o', ls='', mec='white',
           label='Angle, 10 fps'),
]
handles_gap, labels_gap = axC.get_legend_handles_labels()
axC.legend(handles=legend_elems + handles_gap, loc='lower center', 
           frameon=True, fontsize=FS_LEGEND)

# tick label sizes for twin axis (not covered by rcParams for axC2 alone)
axC2.tick_params(labelsize=FS_TICK)
axA.tick_params(labelsize=FS_TICK)
axB.tick_params(labelsize=FS_TICK)
axC.tick_params(labelsize=FS_TICK)

# =============================================================================
# LAYOUT & SAVE
# =============================================================================
fig.tight_layout()
fig.savefig(f"../figures/Downsampling_{date}_{intersection}_{code}_{timeslot}_{mode}{bike_id}.pdf", dpi=300, bbox_inches='tight')
fig.savefig(f"../figures/Downsampling_{date}_{intersection}_{code}_{timeslot}_{mode}{bike_id}.png", dpi=300, bbox_inches='tight')
plt.show()

