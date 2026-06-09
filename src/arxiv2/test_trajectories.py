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
date = BikeZ_Config.avail_dates[2]
campaign = f"Zurich_2025{date[5:7]}" # June or September
mode = BikeZ_Config.avail_modes[0] # Bike
data_root = BikeZ_Config.data_root[campaign][mode]

intersection, code = BikeZ_Config.avail_intersections[date][2]
all_timeslots = BikeZ_Config.avail_timeslots[date][(intersection, code)]
# timeslot = BikeZ_Config.avail_timeslots[date][(intersection, code)][0] # 'AM1' or 'PM1

XY_2056_Bounds = BikeZ_Config.XY_2056_Bounds[date][(intersection, code)]

PLOTTING = True


# #############################################################################
# MAIN: Distribution of gaps
# #############################################################################
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
                if mode == "bike":
                    filename = f"trajectories_bikes_{date}_{intersection}_{timeslot}_{code}-1-ekf.csv"
                else:
                    filename = f"trajectories_vehicles_{date}_{intersection}_{timeslot}_{code}-1-ekf.csv"
                df = pd.read_csv(data_root + f"{date}/{intersection}/{filename}")
                df = df.dropna()
                
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

gap_nums_df = pd.DataFrame(gap_nums_df)
print(all_gaps_df['gap_duration_s'].describe())

sys.exit(1)

location_map = {
    ('06', 'D1', 'A'): 4, ('06', 'D1', 'B'): 4,
    ('06', 'D2', 'G'): 5, ('06', 'D2', 'C'): 5,
    ('06', 'D3', 'E'): 2,
    ('06', 'D4', 'F'): 3,

    ('09', 'D1', 'A'): 6,
    ('09', 'D1', 'C'): 8,
    ('09', 'D1', 'G'): 11,
    ('09', 'D1', 'H'): 12,
    ('09', 'D2', 'B'): 7,
    ('09', 'D2', 'E'): 9,
    ('09', 'D2', 'F'): 10,
    ('09', 'D2', 'I'): 13,
}

all_gaps_df['location_num'] = all_gaps_df.apply(
    lambda r: location_map.get((r['date'][5:7], r['intersection'], r['code'])), axis=1
)
gap_nums_df['location_num'] = gap_nums_df.apply(
    lambda r: location_map.get((r['date'][5:7], r['intersection'], r['code'])), axis=1
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

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# =============================================================================
# LEFT: Boxplot — bike gap duration only
# =============================================================================
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
ax.set_xticklabels(locations, fontsize=8)
ax.set_xlabel('Location', fontsize=11)
ax.set_ylabel('Gap Duration [s]', fontsize=11)
ax.set_title('Gap Duration Distribution\n(Bikes only)', fontsize=12)
ax.grid(axis='y', alpha=0.3)

# =============================================================================
# RIGHT: Bar chart — frequency of vehicles with gaps (mean + std), both modes
# =============================================================================
ax = axes[1]

# Compute gap frequency per row, then aggregate
gap_nums_df['gap_freq'] = gap_nums_df['num_gap_vehicles'] / gap_nums_df['total_vehicles']

for i, mode in enumerate(['bike', 'vehicle']):
    agg = (
        gap_nums_df[gap_nums_df['mode'] == mode]
        .groupby('location_num')['gap_freq']
        .agg(['mean', 'std'])
        .reindex(locations)  # keep consistent ordering
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
    ax.errorbar(
        x + offset,
        agg['mean'],
        yerr=agg['std'],
        fmt='none',
        color='black',
        capsize=4,
        linewidth=1.2
    )

ax.set_xticks(x)
ax.set_xticklabels(locations, fontsize=8)
ax.set_xlabel('Location', fontsize=11)
ax.set_ylabel('Fraction of Vehicles with Gaps', fontsize=11)
ax.set_title('Frequency of Vehicles with Gaps\n(Bike vs. Vehicle)', fontsize=12)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
# plt.savefig('gap_analysis.png', dpi=150)
plt.show()
sys.exit(1)

# #############################################################################
# MAIN
# #############################################################################
for timeslot in all_timeslots[-1:]:
    print(timeslot)
    filename = f"trajectories_bikes_{date}_{intersection}_{timeslot}_{code}-1.csv"
    df = pd.read_csv(data_root + f"{date}/{intersection}/{filename}")
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
    df['datetime'] = pd.to_datetime(df['datetime'], format='ISO8601')
    
    # Fix time = -1 issues
    # Find ref. datetime (i.e. datetime when time == 0)
    ref_datetime = df['datetime'].min()
    ref_time = df.loc[(df['datetime'] == ref_datetime) & (df['time'] >= 0), 'time'].unique()[0]
    df['time'] = df['datetime'].apply(lambda x: np.round((x - ref_datetime).total_seconds() + ref_time, decimals=3))
    
    df = df.sort_values(by=['veh_id', 'time'], ascending=True)
    
    
    if PLOTTING:
        plt.figure(timeslot, figsize=(4, 4))
        grouped = df.groupby(by=['veh_id'])
        for (bike_id,), bike_df in grouped:
            bike_df = bike_df[~bike_df['missing']]
            plt.plot(bike_df['x_act'], bike_df['y_act'], color='blue')
        if XY_2056_Bounds is not None and len(XY_2056_Bounds) == 2:
            plt.xlim(XY_2056_Bounds[0])
            plt.ylim(XY_2056_Bounds[1])
        plt.tight_layout()
    
    
    # # Checking Frame Number and Uniform Delta Time
    # grouped = df.groupby(by=['veh_id'])
    # for (bike_id,), bike_df in grouped:
    #     bike_df['frame_nr'] = np.round(bike_df['time'] * BikeZ_Config.fps + 1e-05, decimals=0)
    #     bike_df['frame_nr'] = bike_df['frame_nr'].astype(int)
    #     if not bike_df['frame_nr'].is_monotonic_increasing:
    #         print("Non-montonic frames, ID = ", bike_id)
    #         sys.exit(1)
    #     if bike_df['frame_nr'].duplicated().any():
    #         print("Duplicated frames, ID = ", bike_id)
    #         sys.exit(1)
