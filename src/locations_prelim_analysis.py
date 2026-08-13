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
locations_list = [8, 9] #, 10, 11, 12, 13]
subsampled_data_root = BikeZ_Config.subsampled_data_root

# ── global font sizes ─────────────────────────────────────────────────────────
FS_TITLE   = 13   # subplot titles
FS_LABEL   = 12   # axis labels
FS_ANNOT   = 12   # annotations
FS_TICK    = 10   # axis tick labels
FS_LEGEND  = 10   # legend entries
plt.rcParams.update({
    'axes.labelsize': FS_LABEL, 
    'axes.titlesize': FS_TITLE,
    'legend.fontsize': FS_LEGEND, 
    'xtick.labelsize': FS_TICK, 
    'ytick.labelsize': FS_TICK,
})

# #############################################################################
# FUNCTIONS
# #############################################################################
def compute_trip_stats(df, id_col='uid', x_col='x_ekf', y_col='y_ekf', time_col='datetime'):
    df = df.sort_values([id_col, time_col])
    
    dx = df.groupby(id_col)[x_col].diff()
    dy = df.groupby(id_col)[y_col].diff()
    df['step_dist'] = np.sqrt(dx**2 + dy**2)
    
    trip_length = df.groupby(id_col)['step_dist'].sum().rename('trip_length')
    
    trip_duration = (
        df.groupby(id_col)[time_col]
        .agg(lambda t: (t.max() - t.min()).total_seconds())
        .rename('trip_duration')
    )
    
    trip_stats = pd.concat([trip_length, trip_duration], axis=1).reset_index()
    trip_stats['trip_mean_speed'] = trip_stats['trip_length'] / trip_stats['trip_duration']
    return trip_stats

# #############################################################################
# MAIN: Obtain Dataset Statistics
# #############################################################################
all_results = []
for loc_num in locations_list:
    avail_dates_timeslots = BikeZ_Config.get_available_dates_and_timeslots(loc_num)
    print(avail_dates_timeslots)
    for date, timeslots in avail_dates_timeslots.items():
        print(f"Date: {date}, timeslots list: {timeslots}")
        for timeslot in timeslots:
            # --- Load Trajectories: Bicycles ---
            mode = 'bike'
            filename = f"location_{loc_num}/{loc_num}_{mode}s_{date}_{timeslot}_lane.csv"
            df_bik = pd.read_csv(subsampled_data_root + filename)
            df_bik['datetime'] = pd.to_datetime(df_bik['datetime'], format='ISO8601')
            df_bik['uid'] = df_bik['veh_type'] + "_" + df_bik['veh_id'].astype(str)
            
            # --- Load Trajectories: Vehicles ---
            mode = 'vehicle'
            filename = f"location_{loc_num}/{loc_num}_{mode}s_{date}_{timeslot}_lane.csv"
            df_veh = pd.read_csv(subsampled_data_root + filename)
            df_veh['datetime'] = pd.to_datetime(df_veh['datetime'], format='ISO8601')
            df_veh['uid'] = df_veh['veh_type'] + "_" + df_veh['veh_id'].astype(str)
            
            # --- Remove standstill vehicles ---
            threshold = 0.5/3.6                 # around 0.1 m/s
            slow_vehicle_ids = (
                df_veh.groupby('veh_id')['speed_ekf']
                .apply(lambda x: (x < threshold).all())
                .pipe(lambda s: s[s].index.tolist())
            )
            df_veh = df_veh[~df_veh['veh_id'].isin(slow_vehicle_ids)]
            
            # --- Compute Trip Length and Duration ---
            bik_trip_stats = compute_trip_stats(df_bik, id_col='uid')
            bik_trip_stats['veh_type'] = bik_trip_stats['uid'].str.split('_').str[0]
            veh_trip_stats = compute_trip_stats(df_veh, id_col='uid')
            veh_trip_stats['veh_type'] = veh_trip_stats['uid'].str.split('_').str[0]
            all_trip_stats = pd.concat([bik_trip_stats, veh_trip_stats], ignore_index=True)
            # Aggregated by veh_type
            type_stats = all_trip_stats.groupby('veh_type').agg(
                n_trips=('uid', 'count'),
                trip_length_mean=('trip_length', 'mean'),
                trip_length_median=('trip_length', 'median'),
                trip_length_std=('trip_length', 'std'),
                trip_duration_mean=('trip_duration', 'mean'),
                trip_duration_median=('trip_duration', 'median'),
                trip_duration_std=('trip_duration', 'std'),
                trip_mean_speed_mean=('trip_mean_speed', 'mean'),
                trip_mean_speed_median=('trip_mean_speed', 'median'),
                trip_mean_speed_std=('trip_mean_speed', 'std'),
            ).reset_index()
            # --- Add mode share ---
            type_stats['mode_share'] = type_stats['n_trips'] / type_stats['n_trips'].sum()
            
            # --- Compute Average Speeds per Mode ---
            avg_speed_by_type_raw = pd.concat([df_bik, df_veh]).groupby('veh_type')['speed_ekf'].mean()

            avg_speed_by_type_per_trip = (
                pd.concat([df_bik, df_veh])
                .groupby(['veh_type', 'uid'])['speed_ekf'].mean()   # per-trip mean speed
                .groupby('veh_type').mean()                          # average across trips
            )
            # Merge speed series into type_stats
            type_stats = type_stats.merge(
                avg_speed_by_type_raw.rename('avg_speed_raw'), on='veh_type', how='left'
            )
            type_stats = type_stats.merge(
                avg_speed_by_type_per_trip.rename('avg_speed_per_trip'), on='veh_type', how='left'
            )
            
            # --- Compute Geographical Density ---
            
            # --- Compute Interaction Scale ---
            
            # --- Tag with location/date/timeslot and store ---
            type_stats['location'] = loc_num
            type_stats['date'] = date
            type_stats['timeslot'] = timeslot
            
            all_results.append(type_stats)
            
            # sys.exit(1)

# --- Combine everything into one master dataframe ---
summary_df = pd.concat(all_results, ignore_index=True)

# #############################################################################
# FIGURE 1: Mode Share, Trip Length, Avg. Speed per Trip
# #############################################################################
def weighted_mean(values, weights):
    return np.average(values, weights=weights)


# --- Aggregate over dates & timeslots & locations ---
agg_by_type = summary_df.groupby('veh_type').apply(
    lambda g: pd.Series({
        'n_trips_total': g['n_trips'].sum(),
        'trip_length_mean': weighted_mean(g['trip_length_mean'], g['n_trips']),
        'avg_speed_per_trip': weighted_mean(g['avg_speed_per_trip'], g['n_trips']),
    })
).reset_index()

# --- Mode share = each type's total trips / grand total trips ---
agg_by_type['mode_share'] = agg_by_type['n_trips_total'] / agg_by_type['n_trips_total'].sum()

# print(agg_by_type)

# --- Plot: 1 row, 3 columns ---
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

axes[0].bar(agg_by_type['veh_type'], agg_by_type['mode_share'])
axes[0].set_title('Mode Share')
axes[0].set_xlabel('Vehicle Type')
axes[0].set_ylabel('Mode Share')

axes[1].bar(agg_by_type['veh_type'], agg_by_type['trip_length_mean'])
axes[1].set_title('Trip Length')
axes[1].set_xlabel('Vehicle Type')
axes[1].set_ylabel('Trip Length (m)')

axes[2].bar(agg_by_type['veh_type'], agg_by_type['avg_speed_per_trip'])
axes[2].set_title('Avg. Speed per Trip')
axes[2].set_xlabel('Vehicle Type')
axes[2].set_ylabel('Speed (m/s)')

fig.tight_layout()


# --- Aggregate over dates & timeslots, but keep location separate ---
agg_by_type_loc = summary_df.groupby(['veh_type', 'location']).apply(
    lambda g: pd.Series({
        'n_trips_total': g['n_trips'].sum(),
        'trip_length_mean': weighted_mean(g['trip_length_mean'], g['n_trips']),
        'avg_speed_per_trip': weighted_mean(g['avg_speed_per_trip'], g['n_trips']),
    })
).reset_index()

# --- Mode share within each location (so bars per location sum to 1) ---
agg_by_type_loc['mode_share'] = agg_by_type_loc.groupby('location')['n_trips_total'].transform(
    lambda x: x / x.sum()
)

# print(agg_by_type_loc)

# --- Plot: 1 row, 3 columns, grouped bars per location ---
veh_types = agg_by_type_loc['veh_type'].unique()
locations = sorted(agg_by_type_loc['location'].unique())

x = np.arange(len(veh_types))          # base positions for veh_type groups
n_loc = len(locations)
bar_width = 0.8 / n_loc                # total group width = 0.8

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
metrics = ['mode_share', 'trip_length_mean', 'avg_speed_per_trip']
titles = ['Mode Share', 'Trip Length', 'Avg. Speed per Trip']
ylabels = ['Mode Share', 'Trip Length (m)', 'Speed (m/s)']

for ax, metric, title, ylabel in zip(axes, metrics, titles, ylabels):
    for i, loc in enumerate(locations):
        sub = agg_by_type_loc[agg_by_type_loc['location'] == loc].set_index('veh_type').reindex(veh_types)
        offset = (i - (n_loc - 1) / 2) * bar_width
        ax.bar(x + offset, sub[metric], width=bar_width, label=f'Loc {loc}')
    ax.set_title(title)
    ax.set_xlabel('Vehicle Type')
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(veh_types)

axes[0].legend(title='Location', bbox_to_anchor=(1.02, 1), loc='upper left')

fig.tight_layout()
plt.show()

