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
import gc
import sys
import shutil
# import warnings
# warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.backends.backend_pdf import PdfPages

from _constants import BikeZ_Config
from tools_utils import extract_all_gaps

# #############################################################################
# CONSTANTS
# #############################################################################
# Configuration
BikeZ_Config = BikeZ_Config()
all_dates_list = BikeZ_Config.avail_dates
all_modes = BikeZ_Config.avail_modes
# data_root = BikeZ_Config.data_root[campaign][mode]

# ── global font sizes ────────────────────────────────────────────────────
FS_TITLE   = 14   # subplot titles
FS_LABEL   = 12   # axis labels
FS_TICK    = 9   # axis tick labels
FS_LEGEND  = 9   # legend entries

# ── colour palette ───────────────────────────────────────────────────────
COLOR_RAW    = 'gray'
COLOR_EKF    = 'steelblue'
COLOR_GAP    = 'darkorange'
COLOR_ACCEL  = 'firebrick'

# #############################################################################
# FUNCTIONS
# #############################################################################
# def extract_all_gaps(veh_df):
#     df = veh_df.sort_values(['veh_id', 'time']).reset_index(drop=True)

#     # new block whenever veh_id changes OR missing flips
#     new_block = (df['veh_id'] != df['veh_id'].shift()) | (df['missing'] != df['missing'].shift())
#     block_id = new_block.cumsum()

#     mask = df['missing']
#     gaps = (
#         df[mask]
#         .groupby(block_id[mask])
#         .agg(
#             veh_id=('veh_id', 'first'),
#             start_time=('time', 'first'),
#             end_time=('time', 'last'),
#             n_points=('time', 'size'),
#         )
#         .reset_index(drop=True)
#     )
#     gaps['duration'] = gaps['end_time'] - gaps['start_time']
#     return gaps


def plot_ekf(veh_df, veh_gaps_df, veh_id, title):
    
    # ── Plotting ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(f'{title} ID = {veh_id}', fontsize=FS_TITLE)
    
    # LEFT: XY trajectory
    ax = axes[0]
     
    obs = veh_df[~veh_df['missing']]
    ax.scatter(obs['x'], obs['y'], s=10, color=COLOR_RAW, alpha=0.7, zorder=2,
               edgecolors='none', label='Raw observations')
     
    ax.plot(veh_df['x_ekf'], veh_df['y_ekf'], color=COLOR_EKF, linewidth=1.6,
            alpha=0.9, zorder=3, label='EKF+RTS trajectory')
     
    for i, (_, gap) in enumerate(veh_gaps_df.iterrows()):
        st = gap['start_time']
        et = gap['end_time']
        gap_rows = veh_df.loc[(veh_df['time'] >= st) & (veh_df['time'] <= et)]
        ax.plot(gap_rows['x_ekf'], gap_rows['y_ekf'], color=COLOR_GAP, linewidth=2.5,
                zorder=4, label='Inferred gap' if i == 0 else '_nolegend_')
        # ax.scatter(gap_rows['x_ekf'].iloc[0], gap_rows['y_ekf'].iloc[0],
        #            color=COLOR_GAP, s=70, marker='^', zorder=5,
        #            edgecolors='black', linewidths=0.6)
        # ax.scatter(gap_rows['x_ekf'].iloc[-1], gap_rows['y_ekf'].iloc[-1],
        #            color=COLOR_GAP, s=70, marker='v', zorder=5,
        #            edgecolors='black', linewidths=0.6)
     
    # ── Trajectory start / end markers ─────────────────────────────────────────
    ax.scatter(veh_df['x_ekf'].iloc[0], veh_df['y_ekf'].iloc[0],
               color='black', marker='*', s=160, zorder=6,
               label='Trajectory start')
    ax.scatter(veh_df['x_ekf'].iloc[-1], veh_df['y_ekf'].iloc[-1],
               color='black', marker='X', s=110, zorder=6,
               label='Trajectory end')
    
    ax.set_xlabel('x [m]', fontsize=FS_LABEL)
    ax.set_ylabel('y [m]', fontsize=FS_LABEL)
    ax.legend(fontsize=FS_LEGEND, frameon=True, loc='best', ncols=1)
    
    
    # RIGHT: time series — speed and acceleration
    ax2 = axes[1]
     
    t = veh_df['time'].to_numpy()
    ax2.plot(t, veh_df['speed'] / 3.6,     color=COLOR_RAW, linewidth=1.0, alpha=0.6,
              label='Speed (raw)')
    ax2.plot(t, veh_df['speed_ekf'] / 3.6, color=COLOR_EKF,  linewidth=1.8,
              label='Speed (EKF+RTS)')
     
    ax2b = ax2.twinx()
    ax2b.plot(t, veh_df['a_ekf'], color=COLOR_ACCEL, linewidth=1.3, linestyle='--',
              label='Acceleration (EKF+RTS)')
    ax2b.axhline(0, color='black', linewidth=0.6, alpha=0.6)
    ax2b.set_ylabel(r'Acceleration [m/s$^2$]', fontsize=FS_LABEL)
    ax2b.tick_params(axis='y', labelsize=FS_TICK)
    ax2b.set_ylim([-3.5, 3.5])
     
    for i, (_, gap) in enumerate(veh_gaps_df.iterrows()):
        t_start = gap['start_time']
        t_end = gap['end_time']
        ax2.axvspan(t_start, t_end, alpha=0.15, color=COLOR_GAP,
                    label='Gap region' if i == 0 else '_nolegend_')
     
    ax2.set_xlabel('Time [s]', fontsize=FS_LABEL)
    ax2.set_ylabel('Speed [m/s]', fontsize=FS_LABEL)
    ax2.tick_params(axis='both', labelsize=FS_TICK)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, int(veh_df['speed_ekf'].max()/3.6)+1])
     
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=FS_LEGEND,
               frameon=True, loc='best')
    
    fig.tight_layout()
    return fig

# # #############################################################################
# # MAIN: Reduce files by parquet saving
# # #############################################################################
# for date in all_dates_list:
#     campaign = f"Zurich_2025{date[5:7]}"
#     for mode in all_modes:
#         data_root = BikeZ_Config.data_root[campaign][mode]
#         all_intersections_list = BikeZ_Config.avail_intersections[date]
#         for intersection, code in all_intersections_list:
#             all_timeslots = BikeZ_Config.avail_timeslots[date][(intersection, code)]
#             for timeslot in all_timeslots:
#                 filename = f"trajectories_{mode}s_{date}_{intersection}_{timeslot}_{code}-1-ekf.csv"
#                 csv_path = data_root + f"{date}/{intersection}/{filename}"
#                 exists = os.path.exists(csv_path)
                
#                 if exists:
#                     # 1. Read CSV and fix datetime and speed_cols
#                     df = pd.read_csv(csv_path)
#                     df['datetime'] = pd.to_datetime(df['datetime'], format='ISO8601')
                    
#                     # 2. Write Parquet
#                     savename = f"trajectories_{mode}s_{date}_{intersection}_{timeslot}_{code}-1-ekf.parquet"
#                     parquet_path = data_root + f"{date}/{intersection}/{savename}"
#                     df.to_parquet(parquet_path, compression='zstd')
                    
#                     # 3. Verify before deleting anything irreversible
#                     df_check = pd.read_parquet(parquet_path)
#                     if not df.equals(df_check):
#                         raise ValueError(f"Round-trip mismatch, NOT deleting CSV: {csv_path}")
                    
#                     # 4. Delete the original CSV only after verification passes
#                     os.remove(csv_path)
                
#                     # sys.exit(1)
#             print(f"Done for {mode}_{date}_{intersection}_{code}")

# #############################################################################
# MAIN: Save vehicles or bikes with gaps
# #############################################################################
import matplotlib
matplotlib.use('Agg')


save_path = os.path.join("../debugging/", 'EKF_All_Gaps.pdf')
plt.ioff()
with PdfPages(save_path) as pdf:
    for date in all_dates_list:
        campaign = f"Zurich_2025{date[5:7]}"
        for mode in all_modes:
            data_root = BikeZ_Config.data_root[campaign][mode]
            all_intersections_list = BikeZ_Config.avail_intersections[date]
            for intersection, code in all_intersections_list:
                all_timeslots = BikeZ_Config.avail_timeslots[date][(intersection, code)]
                for timeslot in all_timeslots:
                    filename = f"trajectories_{mode}s_{date}_{intersection}_{timeslot}_{code}-1-ekf.parquet"
                    parquet_path = data_root + f"{date}/{intersection}/{filename}"
                    df = pd.read_parquet(parquet_path)
                    if not df.missing.any():
                        continue
                    all_gaps = extract_all_gaps(df)
                    
                    title = f'{date}: {intersection}, {code} - {timeslot}\n{mode}'
                    unique_ids = all_gaps['veh_id'].unique()
                    for veh_id in unique_ids:
                        veh_df = df[df['veh_id'] == veh_id].copy()
                        veh_gaps_df = all_gaps[all_gaps['veh_id'] == veh_id].copy()
                        fig = plot_ekf(veh_df, veh_gaps_df, veh_id, title)                        
                        pdf.savefig(fig, dpi=72)   # low dpi for compact file
                        plt.close(fig)
                print(f'Done {date}-{mode}-{intersection}-{code}')
    print(f"Saved PDF: {save_path}")
                
