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
filename = f"trajectories_bikes_{date}_{intersection}_{timeslot}_{code}-1-ekf-lane.csv"
mod_df   = pd.read_csv(data_root + f"{date}/{intersection}/{filename}")

# Load the registry (needed for plotting only)
registry_path  = f"../data/registry_{date}_{intersection}_{code}.pkl"
registry       = pickle.load(open(registry_path, 'rb'))
geometry_store    = registry['geometry_store']
segment_registry  = registry['segment_registry']
movement_registry = registry['movement_registry']
XY_2056_Bounds = BikeZ_Config.XY_2056_Bounds[date][(intersection, code)]


# #############################################################################
# 1. BASIC OVERVIEW
# #############################################################################

# How many vehicles, how many matched?
total_vehs   = mod_df['veh_id'].nunique()
matched_vehs = mod_df[mod_df['segment_id'].notna()]['veh_id'].nunique()
print(f"Vehicles total:   {total_vehs}")
print(f"Vehicles matched: {matched_vehs}  ({100*matched_vehs/total_vehs:.1f}%)")

# Match rate per movement
match_rate = (
    mod_df.groupby('movement_key')['veh_id']
    .nunique()
    .sort_values(ascending=False)
)
print("\nVehicles per movement:")
print(match_rate.to_string())

# How many rows are unmatched?
unmatched_frac = mod_df['segment_id'].isna().mean()
print(f"\nUnmatched rows: {100*unmatched_frac:.1f}%")


# #############################################################################
# 2. FILTER TO MATCHED ROWS ONLY
# #############################################################################
df_matched = mod_df[mod_df['segment_id'].notna()].copy()

# Convenience: separate approach / turn / departure
df_approach   = df_matched[df_matched['segment_role'] == 'approach']
df_turn       = df_matched[df_matched['segment_role'] == 'turn']
df_departure  = df_matched[df_matched['segment_role'] == 'departure']


# #############################################################################
# 3. SPEED ANALYSIS
# #############################################################################

# Mean longitudinal speed per movement role
speed_by_role = (
    df_matched
    .groupby('segment_role')[['s_dot', 'd_dot', 'speed_ekf']]
    .agg(['mean', 'std'])
    .round(3)
)
print("\nSpeed by role:")
print(speed_by_role.to_string())

# Speed profile along s for approach segments
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

for seg_key in df_approach['segment_id'].unique():
    seg_df = df_approach[df_approach['segment_id'] == seg_key]
    # Bin by s and take median s_dot per bin
    seg_df = seg_df.copy()
    seg_df['s_bin'] = pd.cut(seg_df['s'], bins=20)
    profile = seg_df.groupby('s_bin', observed=True)['s_dot'].median()
    s_mid   = [interval.mid for interval in profile.index]
    axes[0].plot(s_mid, profile.values, label=seg_key, alpha=0.8)

axes[0].axhline(0, color='gray', linewidth=0.5)
axes[0].set_xlabel('s [m]')
axes[0].set_ylabel('median s_dot [m/s]')
axes[0].set_title('Longitudinal speed profile — approach segments')
axes[0].legend(fontsize=7)
axes[0].grid(True, alpha=0.3)

# Lateral speed distribution per role
for role, grp in df_matched.groupby('segment_role'):
    d_dot_clean = grp['d_dot'].dropna()
    axes[1].hist(d_dot_clean, bins=40, alpha=0.5, label=role, density=True)

axes[1].set_xlabel('d_dot [m/s]')
axes[1].set_ylabel('density')
axes[1].set_title('Lateral speed distribution by role')
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
# plt.savefig('../debugging/speed_profiles.png', dpi=150)
plt.show()


# #############################################################################
# 4. LATERAL POSITION ANALYSIS
# #############################################################################

# d distribution per segment — shows lane keeping behaviour
fig, ax = plt.subplots(figsize=(10, 4))

for seg_key in sorted(df_matched['segment_id'].unique()):
    d_vals = df_matched[df_matched['segment_id'] == seg_key]['d'].dropna()
    ax.hist(d_vals, bins=40, alpha=0.4, label=seg_key, density=True)

ax.axvline(0, color='black', linewidth=1, linestyle='--', label='centerline')
ax.set_xlabel('d [m]  (+ = left of travel direction)')
ax.set_ylabel('density')
ax.set_title('Lateral offset distribution per segment')
ax.legend(fontsize=6, ncol=2)
ax.grid(True, alpha=0.3)
plt.tight_layout()
# plt.savefig('../debugging/lateral_offset_dist.png', dpi=150)
plt.show()


# #############################################################################
# 5. BIKE LANE USAGE
# #############################################################################

# Only rows where in_bike_lane is not NaN (i.e. bike lane geometry exists)
df_bl = df_matched[df_matched['in_bike_lane'].notna()].copy()

if len(df_bl) > 0:
    bl_usage = (
        df_bl.groupby('segment_id')['in_bike_lane']
        .agg(
            n_rows='count',
            pct_in_lane=lambda x: 100 * x.mean()
        )
        .round(1)
    )
    print("\nBike lane usage per segment:")
    print(bl_usage.to_string())

    # Per-vehicle bike lane usage rate
    veh_bl = (
        df_bl.groupby('veh_id')['in_bike_lane']
        .mean()
        .mul(100)
        .rename('pct_in_bike_lane')
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(veh_bl, bins=20, edgecolor='white')
    ax.set_xlabel('% of time in bike lane')
    ax.set_ylabel('# vehicles')
    ax.set_title('Per-vehicle bike lane usage rate')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    # plt.savefig('../debugging/bike_lane_usage.png', dpi=150)
    plt.show()
else:
    print("\nNo bike lane geometry available for this intersection.")


# #############################################################################
# 6. U-TURN / REVERSING DETECTION
# #############################################################################

# s_decreasing flags backward motion
df_reversing = df_matched[df_matched['s_decreasing'] == True]

reversing_vehs = df_reversing['veh_id'].nunique()
print(f"\nVehicles with s_decreasing events: {reversing_vehs}")

# Which movements see most reversing?
if len(df_reversing) > 0:
    reversing_by_mov = (
        df_reversing.groupby('movement_key')
        .size()
        .sort_values(ascending=False)
        .head(5)
    )
    print("Top movements with reversing frames:")
    print(reversing_by_mov.to_string())


# #############################################################################
# 7. TRAJECTORY-LEVEL SUMMARY TABLE
# #############################################################################

def summarise_vehicle(grp):
    """Per-vehicle summary statistics."""
    matched = grp[grp['segment_id'].notna()]
    return pd.Series({
        'movement_key':      grp['movement_key'].dropna().iloc[0]
                             if grp['movement_key'].notna().any() else None,
        'n_frames':          len(grp),
        'n_matched':         len(matched),
        'match_rate':        len(matched) / len(grp),
        'mean_speed':        grp['speed_ekf'].mean(),
        'mean_s_dot':        matched['s_dot'].mean(),
        'mean_d':            matched['d'].mean(),
        'std_d':             matched['d'].std(),
        'pct_in_bike_lane':  matched['in_bike_lane'].mean()
                             if matched['in_bike_lane'].notna().any() else np.nan,
        'any_reversing':     (matched['s_decreasing'] == True).any(),
    })

summary_df = (
    mod_df
    .groupby('veh_id')
    .apply(summarise_vehicle)
    .reset_index()
)

print("\nPer-vehicle summary (first 10):")
print(summary_df.head(10).to_string(index=False))

# # Save for downstream use
# summary_df.to_csv(
#     data_root + f"{date}/{intersection}/"
#     f"summary_bikes_{date}_{intersection}_{timeslot}_{code}.csv",
#     index=False
# )
# print("\nSummary saved.")


# #############################################################################
# 8. DEBUG PLOT FOR A SINGLE VEHICLE
# #############################################################################
from tools_plotting import plot_lane_coord_debug, build_lane_color_map

lane_color_map = build_lane_color_map(geometry_store)

bike_id = 24 # summary_df.sort_values('match_rate', ascending=False).iloc[0]['veh_id']
bike_df = mod_df[mod_df['veh_id'] == bike_id].copy()

print(f"\nDebug plot for veh_id={bike_id}  "
      f"(movement: {bike_df['movement_key'].dropna().iloc[0]})")

plot_lane_coord_debug(
    bike_df,
    segment_registry, geometry_store,
    XY_2056_Bounds, bike_id,
    lane_color_map=lane_color_map,
    save_path=None
)


# #############################################################################
# 9. BATCH DEBUG PLOTS
# #############################################################################

# Uncomment to generate a plot for every matched vehicle
# save_dir = f"../debugging/{date}-{intersection}/"
# import os; os.makedirs(save_dir, exist_ok=True)
# plt.ioff()
#
# for bike_id in tqdm(mod_df['veh_id'].unique()):
#     bike_df = mod_df[mod_df['veh_id'] == bike_id].copy()
#     plot_lane_coord_debug(
#         bike_df,
#         segment_registry, geometry_store,
#         XY_2056_Bounds, bike_id,
#         lane_color_map=lane_color_map,
#         save_path=os.path.join(
#             save_dir,
#             f"{timeslot}_{code}_lane_{bike_id}.png"
#         )
#     )
#     plt.close('all')