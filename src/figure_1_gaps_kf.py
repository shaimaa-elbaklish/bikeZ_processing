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
import matplotlib.patches as mpatches

from _constants import BikeZ_Config

# #############################################################################
# CONSTANTS
# #############################################################################
# Configuration
BikeZ_Config = BikeZ_Config()

# Specify Trajectory File
# date = BikeZ_Config.avail_dates[2]
# campaign = f"Zurich_2025{date[5:7]}" # June or September
# mode = BikeZ_Config.avail_modes[0] # Bike
# data_root = BikeZ_Config.data_root[campaign][mode]

# intersection, code = BikeZ_Config.avail_intersections[date][2]
# all_timeslots = BikeZ_Config.avail_timeslots[date][(intersection, code)]
# timeslot = BikeZ_Config.avail_timeslots[date][(intersection, code)][0] # 'AM1' or 'PM1

# XY_2056_Bounds = BikeZ_Config.XY_2056_Bounds[date][(intersection, code)]


# #############################################################################
# FIGURE 1: Distribution of gaps
# #############################################################################
# --- Step 1: Get gaps statsistics ---
all_gaps_df = None
gap_nums_df = []
for mode in BikeZ_Config.avail_modes:
    for date in BikeZ_Config.avail_dates:
        campaign = f"Zurich_2025{date[5:7]}" 
        data_root = BikeZ_Config.data_root[campaign][mode]
        for loc in BikeZ_Config.avail_intersections[date]:
            intersection, code = loc
            all_timeslots = BikeZ_Config.avail_timeslots[date][(intersection, code)]
            for timeslot in all_timeslots:
                filename = f"trajectories_{mode}s_{date}_{intersection}_{timeslot}_{code}-1.csv"
                df = pd.read_csv(data_root + f"{date}/{intersection}/{filename}")
                df['datetime'] = pd.to_datetime(df['datetime'], format='ISO8601')
                df['missing'] = (df['speed(km/h)'] == -1)
                df = df.rename(columns={'time(s)': 'time'})
                ref_datetime = df['datetime'].min()
                ref_time = df.loc[(df['datetime'] == ref_datetime) & (df['time'] >= 0), 'time'].unique()[0]
                df['time'] = df['datetime'].apply(lambda x: np.round((x - ref_datetime).total_seconds() + ref_time, decimals=3))
                df = df.sort_values(by=['veh_id', 'time'], ascending=True)
                
                # Group consecutive missing rows into gaps per vehicle
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
                gaps_df['mode'] = mode
                gaps_df['date'] = date
                gaps_df['intersection'] = intersection
                gaps_df['code'] = code
                gaps_df['timeslot'] = timeslot
                if all_gaps_df is None:
                    all_gaps_df = gaps_df.copy()
                else:
                    all_gaps_df = pd.concat((all_gaps_df, gaps_df), ignore_index=True)
                gap_nums_df.append({
                    'mode': mode, 'date': date, 'intersection': intersection,
                    'code': code, 'timeslot': timeslot,
                    'num_gap_vehicles': 0 if gaps_df.empty else gaps_df['veh_id'].nunique(),
                    'total_vehicles': df['veh_id'].nunique()
                })
                
                del df
                gc.collect()

gap_nums_df = pd.DataFrame(gap_nums_df)
print(all_gaps_df['gap_duration_s'].describe())

sys.exit(1)

all_gaps_df['location_num'] = all_gaps_df.apply(
    lambda r: BikeZ_Config.location_map.get((r['date'][5:7], r['intersection'], r['code'])), axis=1
)
gap_nums_df['location_num'] = gap_nums_df.apply(
    lambda r: BikeZ_Config.location_map.get((r['date'][5:7], r['intersection'], r['code'])), axis=1
)

# # --- Shared label ---
# all_gaps_df['loc_date'] = (
#     all_gaps_df['date'].astype(str).str[5:] + '\n' +
#     all_gaps_df['intersection'].astype(str)
# )
# gap_nums_df['loc_date'] = (
#     gap_nums_df['date'].astype(str).str[5:] + '\n' +
#     gap_nums_df['intersection'].astype(str)
# )

locations = sorted(all_gaps_df['location_num'].unique())
x = np.arange(len(locations))
colors = {'bike': '#2196F3', 'vehicle': '#FF7043'}
width = 0.35


# --- Step 2: Plot Gap Distributions ---
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# ── global font sizes ─────────────────────────────────────────────────────────
FS_TITLE   = 13   # subplot titles
FS_LABEL   = 12   # axis labels
FS_ANNOT   = 12   # annotations
FS_TICK    = 10   # axis tick labels
FS_LEGEND  = 10   # legend entries


# LEFT: Boxplot — bike gap duration only
ax = axes[0]
bike_data = [
    all_gaps_df[(all_gaps_df['location_num'] == loc) &
                (all_gaps_df['mode'] == 'bike')]['gap_duration_s'].dropna().values
    for loc in locations
]
bp = ax.boxplot(
    bike_data,
    positions=x,
    widths=0.5,
    patch_artist=True,
    boxprops=dict(facecolor=colors['bike'], alpha=0.7),
    medianprops=dict(color='black', linewidth=2),
    whiskerprops=dict(color=colors['bike']),
    capprops=dict(color=colors['bike']),
    flierprops=dict(marker='o', color=colors['bike'], alpha=0.4, markersize=4),
    manage_ticks=False
)
ax.set_xticks(x)
ax.set_xticklabels(locations)
ax.set_xlabel('Location', fontsize=FS_LABEL)
ax.set_ylabel('Gap Duration [s]', fontsize=FS_LABEL)
ax.set_title('Gap Duration Distribution\n(Bicycles only)', fontsize=FS_TITLE)
ax.tick_params(axis='both', labelsize=FS_TICK)
ax.grid(axis='y', alpha=0.3)

# RIGHT: Bar chart — frequency of vehicles with gaps (mean + std), both modes
ax = axes[1]
gap_nums_df['gap_freq'] = gap_nums_df['num_gap_vehicles'] / gap_nums_df['total_vehicles']

for i, mode in enumerate(['bike', 'vehicle']):
    agg = (
        gap_nums_df[gap_nums_df['mode'] == mode]
        .groupby('location_num')['gap_freq']
        .agg(['mean', 'std'])
        .reindex(locations)
    )
    offset = (i - 0.5) * width
    ax.bar(
        x + offset,
        agg['mean'],
        width=width * 0.9,
        color=colors[mode],
        alpha=0.7,
        label=mode
    )

    # # Clip only the lower error so mean - lower_err >= 0
    # lower_err = np.minimum(agg['std'], agg['mean'])
    # upper_err = agg['std']
    # yerr = np.vstack([lower_err, upper_err])
    ax.errorbar(
        x + offset,
        agg['mean'],
        yerr=agg['std'], #yerr,
        fmt='none',
        color='black',
        capsize=4,
        linewidth=1.2
    )

ax.set_xticks(x)
ax.set_xticklabels(locations)
ax.set_xlabel('Location', fontsize=FS_LABEL)
ax.set_ylabel('Fraction of Entities with Gaps', fontsize=FS_LABEL)
ax.set_title('Frequency of Entities with Gaps\n(Bike vs. Vehicle)', fontsize=FS_TITLE)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
ax.tick_params(axis='both', labelsize=FS_TICK)
ax.legend(fontsize=FS_LEGEND)
ax.grid(axis='y', alpha=0.3)

fig.tight_layout()
fig.savefig("../figures/Gaps_Distribution.pdf", dpi=300, bbox_inches='tight')
fig.savefig("../figures/Gaps_Distribution.png", dpi=300, bbox_inches='tight')
plt.show()


# #############################################################################
# FIGURE 2: EKF results representative case
# #############################################################################
# Specify Trajectory File
date = BikeZ_Config.avail_dates[1]
campaign = f"Zurich_2025{date[5:7]}" # June or September
mode = BikeZ_Config.avail_modes[0] # Bike
data_root = BikeZ_Config.data_root[campaign][mode]

intersection, code = BikeZ_Config.avail_intersections[date][0]
# all_timeslots = BikeZ_Config.avail_timeslots[date][(intersection, code)]
timeslot = BikeZ_Config.avail_timeslots[date][(intersection, code)][2] # 'AM2'

XY_2056_Bounds = BikeZ_Config.XY_2056_Bounds[date][(intersection, code)]
X_2056_offset = XY_2056_Bounds[0][0]
Y_2056_offset = XY_2056_Bounds[1][0]

filename = f"trajectories_{mode}s_{date}_{intersection}_{timeslot}_{code}-1-ekf.csv"
df = pd.read_csv(data_root + f"{date}/{intersection}/{filename}")

# --- Step 1: Get most interesting bike with long gaps ---
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

# --- Step 2: Plotting ---
FS_LEGEND  = 9.5   # legend entries
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset


fig, axes = plt.subplots(1, 2, figsize=(10, 4))
fig.suptitle(
    f'Gap Inference + EKF-RTS Filtering: Bicycle ID {bike_id}\n'
    f'Gap duration = {gaps["gap_duration_s"].item():.2f} s',
    fontsize=FS_TITLE
)

# ── colour palette (consistent with Fig. gaps_dist / pipeline schematic) ──────
COLOR_RAW    = 'gray'
COLOR_EKF    = 'steelblue'
COLOR_GAP    = 'darkorange'
COLOR_ACCEL  = 'firebrick'

# LEFT: XY trajectory
ax = axes[0]
 
obs = veh[~veh['missing']]
ax.scatter(obs['x'], obs['y'], s=10, color=COLOR_RAW, alpha=0.5, zorder=2,
           edgecolors='none', label='Raw observations')
 
ax.plot(veh['x_ekf'], veh['y_ekf'], color=COLOR_EKF, linewidth=1.6,
        alpha=0.9, zorder=3, label='EKF+RTS trajectory')
 
for i, (_, gap) in enumerate(gaps.iterrows()):
    s = int(gap['gap_start_idx'])
    e = int(s + gap['gap_len_frames'])
    gap_rows = veh.iloc[s:e + 1]
    ax.plot(gap_rows['x_ekf'], gap_rows['y_ekf'], color=COLOR_GAP, linewidth=2.5,
            zorder=4, label='Inferred gap' if i == 0 else '_nolegend_')
    ax.scatter(gap_rows['x_ekf'].iloc[0], gap_rows['y_ekf'].iloc[0],
               color=COLOR_GAP, s=70, marker='^', zorder=5,
               edgecolors='black', linewidths=0.6)
    ax.scatter(gap_rows['x_ekf'].iloc[-1], gap_rows['y_ekf'].iloc[-1],
               color=COLOR_GAP, s=70, marker='v', zorder=5,
               edgecolors='black', linewidths=0.6)
 
# ── Inset zoom: first gap + context window ─────────────────────────────────
if len(gaps) > 0:
    first_gap = gaps.iloc[0]
    s = int(first_gap['gap_start_idx'])
    e = int(s + first_gap['gap_len_frames'])
    context = 50  # frames before/after gap to include in zoom
 
    s_ctx = max(0, s - context)
    e_ctx = min(len(veh) - 1, e + context)
    zoom_rows = veh.iloc[s_ctx:e_ctx + 1]
    gap_rows  = veh.iloc[s:e + 1]
    obs_zoom  = zoom_rows[~zoom_rows['missing']]
 
    axins = inset_axes(ax, width='42%', height='36%', loc='lower left',
                       # bbox_to_anchor=(0.12, -0.05, 1, 1),  # (x0, y0, width, height) in axes-fraction coords
                       # bbox_transform=ax.transAxes)
                       borderpad=1.2)
 
    axins.plot(zoom_rows['x_ekf'], zoom_rows['y_ekf'],
               color=COLOR_EKF, linewidth=1.6, alpha=0.9)
    axins.scatter(obs_zoom['x'], obs_zoom['y'],
                  s=14, color=COLOR_RAW, alpha=0.6, zorder=2, edgecolors='none')
    axins.plot(gap_rows['x_ekf'], gap_rows['y_ekf'],
               color=COLOR_GAP, linewidth=2.5, zorder=4)
    axins.scatter(gap_rows['x_ekf'].iloc[0], gap_rows['y_ekf'].iloc[0],
                  color=COLOR_GAP, s=70, marker='^', zorder=5,
                  edgecolors='black', linewidths=0.6)
    axins.scatter(gap_rows['x_ekf'].iloc[-1], gap_rows['y_ekf'].iloc[-1],
                  color=COLOR_GAP, s=70, marker='v', zorder=5,
                  edgecolors='black', linewidths=0.6)
 
    margin = 1.0  # metres
    x_min, x_max = zoom_rows['x_ekf'].min() - margin, zoom_rows['x_ekf'].max() + margin
    y_min, y_max = zoom_rows['y_ekf'].min() - margin, zoom_rows['y_ekf'].max() + margin
    axins.set_xlim(x_min, x_max)
    axins.set_ylim(y_min, y_max)
    axins.set_aspect('equal')
    axins.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    axins.grid(True, alpha=0.3)
    for spine in axins.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(0.8)
 
    mark_inset(ax, axins, loc1=2, loc2=4, fc='none', ec='black', linewidth=0.8)
 
ax.set_xlabel('x [m]', fontsize=FS_LABEL)
ax.set_ylabel('y [m]', fontsize=FS_LABEL)
ax.tick_params(axis='both', labelsize=FS_TICK)
ax.grid(True, alpha=0.3)
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.set_ylim([59, 95])
 
# ── Trajectory start / end markers ─────────────────────────────────────────
ax.scatter(veh['x_ekf'].iloc[0], veh['y_ekf'].iloc[0],
           color='black', marker='*', s=160, zorder=6,
           label='Trajectory start')
ax.scatter(veh['x_ekf'].iloc[-1], veh['y_ekf'].iloc[-1],
           color='black', marker='X', s=110, zorder=6,
           label='Trajectory end')
 
ax.legend(fontsize=FS_LEGEND, frameon=True, loc='upper right', ncols=1)


# RIGHT: time series — speed and acceleration
ax2 = axes[1]
 
t = veh['time'].to_numpy()
ax2.plot(t, veh['speed'] / 3.6,     color=COLOR_RAW, linewidth=1.0, alpha=0.6,
          label='Speed (raw)')
ax2.plot(t, veh['speed_ekf'] / 3.6, color=COLOR_EKF,  linewidth=1.8,
          label='Speed (EKF+RTS)')
 
ax2b = ax2.twinx()
ax2b.plot(t, veh['a_ekf'], color=COLOR_ACCEL, linewidth=1.3, linestyle='--',
          label='Acceleration (EKF+RTS)')
ax2b.axhline(0, color='black', linewidth=0.6, alpha=0.6)
ax2b.set_ylabel(r'Acceleration [m/s$^2$]', fontsize=FS_LABEL)
ax2b.tick_params(axis='y', labelsize=FS_TICK)
# ax2b.spines['top'].set_visible(False)
ax2b.set_ylim([-1, 1])
 
for i, (_, gap) in enumerate(gaps.iterrows()):
    s = int(gap['gap_start_idx'])
    e = int(s + gap['gap_len_frames'])
    t_start = veh['time'].iloc[min(s, len(veh) - 1)]
    t_end   = veh['time'].iloc[min(e, len(veh) - 1)]
    ax2.axvspan(t_start, t_end, alpha=0.15, color=COLOR_GAP,
                label='Gap region' if i == 0 else '_nolegend_')
 
ax2.set_xlabel('Time [s]', fontsize=FS_LABEL)
ax2.set_ylabel('Speed [m/s]', fontsize=FS_LABEL)
ax2.tick_params(axis='both', labelsize=FS_TICK)
ax2.grid(True, alpha=0.3)
# ax2.spines['top'].set_visible(False)
ax2.set_ylim([0, 10])
 
lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax2b.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=FS_LEGEND,
           frameon=True, loc='upper left')

fig.tight_layout()
# fig.savefig(f"../figures/EKF_{mode}_{date}_{intersection}_{code}_{timeslot}_veh{bike_id}.pdf", dpi=300, bbox_inches='tight')
# fig.savefig(f"../figures/EKF_{mode}_{date}_{intersection}_{code}_{timeslot}_veh{bike_id}.png", dpi=300, bbox_inches='tight')
plt.show()


