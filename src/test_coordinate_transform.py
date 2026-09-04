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

from _constants import BikeZ_Config

# #############################################################################
# CONSTANTS
# #############################################################################
# Configuration
BikeZ_Config = BikeZ_Config()

# Specify Trajectory File
date = BikeZ_Config.avail_dates[-2]
campaign = f"Zurich_2025{date[5:7]}" # June or September
mode = BikeZ_Config.avail_modes[1] # 0: Bike, 1: Vehicle
data_root = BikeZ_Config.data_root[campaign][mode]

intersection, code = BikeZ_Config.avail_intersections[date][-2]
timeslot = BikeZ_Config.avail_timeslots[date][(intersection, code)][0] # 'AM1'

XY_2056_Bounds = BikeZ_Config.XY_2056_Bounds[date][(intersection, code)]
X_2056_offset = XY_2056_Bounds[0][0]
Y_2056_offset = XY_2056_Bounds[1][0]
# sys.exit(1)

# #############################################################################
# MAIN: Load data
# #############################################################################
# trajectories after EKF
if mode == "bike":
    filename = f"trajectories_bikes_{date}_{intersection}_{timeslot}_{code}-1-ekf"
else:
    filename = f"trajectories_vehicles_{date}_{intersection}_{timeslot}_{code}-1-ekf"
# df = pd.read_csv(data_root + f"{date}/{intersection}/{filename}.csv")
df = pd.read_parquet(data_root + f"{date}/{intersection}/{filename}.parquet")
df = df.dropna()
df['x_act_ekf'] = df['x_ekf'] + X_2056_offset
df['y_act_ekf'] = df['y_ekf'] + Y_2056_offset

# Load geometry, segment, and movement registries
registry_path = f"../data/registry_{date}_{intersection}_{code}.pkl"
registry = pickle.load(open(registry_path, 'rb'))
geometry_store    = registry['geometry_store']
segment_registry  = registry['segment_registry']
movement_registry = registry['movement_registry']
max_chain_length  = registry['metadata'].get('max_chain_length', 3)

# #############################################################################
# MAIN: Coordinate Transform (SINGLE Bike)
# #############################################################################
from tools_lane_coords_V5 import setup_site, to_lane_coordinates

site = setup_site(
    geometry_store, segment_registry, movement_registry, mode=mode,
    allow_reverse_turns=True # for location 7 only!
)
# i = 1
# seg_i = list(_idx)[i]
# transitions = tau_class[i]
# for j in range(len(transitions)):
#     seg_j = list(_idx)[j]
#     print(f"from {seg_i} to {seg_j}, transition = {transitions[j]}")

bike_id = 242 # 242
bike_df = df[(df["veh_id"] == bike_id)].copy()

from tools_lane_coords_V5 import (
    prepare_trajectory, project_all, propose_breakpoints,
    build_prefix_sums, build_tau_tables, build_runcost_cache,
    segmental_dp, assign_roles, W_MOV_MIN, W_LAT, W_COR,
    W_CLAMP, W_PRG, W_PSI
)

# ── rebuild everything to_lane_coordinates computes internally ──────────────
traj = prepare_trajectory(bike_df)
proj = project_all(traj, site['geom_keys'], geometry_store,
                   site['poly_keys'], segment_registry,
                   site['geom_to_polycols'], site['buffers'])

all_states = site['states']
active = [k for k, st in enumerate(all_states)
          if st.type == 'off' or proj.inside[:, st.poly_col].any()]
states = [all_states[k] for k in active]
tau_c  = site['tau_class'][np.ix_(active, active)]

B, prov = propose_breakpoints(traj, proj, site['gates'],
                              site['geom_keys'], site['poly_keys'])
pre = build_prefix_sums(traj, proj, states)
base, cont, bnd = build_tau_tables(B, states, tau_c, site['tau_vals'],
                                   proj, geometry_store)
rc = build_runcost_cache(B, states, pre, traj, proj)
runs, total = segmental_dp(B, states, pre, traj, proj, tau_c,
                           site['tau_vals'], geometry_store)
runs = assign_roles(runs)

bidx = {int(v): i for i, v in enumerate(B)}
kidx = {s.key: i for i, s in enumerate(states)}

# ── what the DP decided ────────────────────────────────────────────────────
print(f"veh {bike_id}  n={traj.n}  arc={traj.arc[-1]:.1f} m  "
      f"|B|={len(B)}  S={len(states)}/{len(all_states)}\n")
print("breakpoints:", [(int(n), p) for n, p in zip(B, prov)], "\n")
for r in runs:
    print(f"{r.a:5d}-{r.b:5d}  {r.state.key:38s} {r.cost_per_m:7.3f}  "
          f"{r.trans_class}")
# sys.exit(1)

# Run full transform
import time
start = time.perf_counter()
bike_df = to_lane_coordinates(
    bike_df, movement_registry,
    segment_registry, geometry_store,
    agent_mode=mode, site=site,
    verbose=True
)
end = time.perf_counter()
print(f"Elapsed time: {end - start:.6f} seconds")

from tools_plot_lane_results import plot_debug_panel, plot_trajectory_map
fig = plot_debug_panel(
    bike_df,
    geometry_store,
    segment_registry,
    time_col='time',
    xy_offset=True,     # set False to use raw EPSG:2056 coords
    save_path=None #'../plots/debug_veh42.png',
)
fig.tight_layout()
plt.show()


# plot_trajectory_map(
#     bike_df, geometry_store, segment_registry,
#     color_by='segment_id', show_validity_polygons=True,
#     show_s_change=True, zoom_start=19, 
#     save_path=f"../maps/debug_maps_{date}_{intersection}_{code}_{timeslot}_{mode}{bike_id}.html"
# )


sys.exit(1)

# -----------------------------------------------------------------------------
# from tools_lane_coords_V4 import to_lane_coordinates, setup_registry
# from tools_plot_lane_results import plot_debug_panel, plot_trajectory_map

# setup_registry(geometry_store, segment_registry)

# bike_id = 242
# bike_df = df[(df["veh_id"] == bike_id)].copy()

# # Run full transform
# import time
# start = time.perf_counter()
# bike_df = to_lane_coordinates(
#     bike_df, movement_registry,
#     segment_registry, geometry_store,
#     max_chain_length=max_chain_length,
#     agent_mode=mode,
#     verbose=True
# )
# end = time.perf_counter()
# print(f"Elapsed time: {end - start:.6f} seconds")

# # # Force transform
# # from tools_lane_coords_V4 import to_lane_coordinates_forced
# # bike_df = to_lane_coordinates_forced(
# #     bike_df,
# #     forced_chain=['BaslerstrW_WB', 'BaslerstrW_EB', 'turn_BaslerstrW_EB_2_BaslerstrE_EB', 'BaslerstrE_EB'],
# #     # forced_chain=['FreihofstrN_SB', 'turn_FreihofstrN_SB_2_BaslerstrE_EB', 'BaslerstrE_EB'],
# #     # forced_chain=['BaslerstrE_WB', 'turn_BaslerstrE_WB_2_BaslerstrW_WB', 'BaslerstrW_WB'],
# #     segment_registry=segment_registry,
# #     geometry_store=geometry_store,
# #     movement_registry=movement_registry,
# #     verbose=True,
# # )

# fig = plot_debug_panel(
#     bike_df,
#     geometry_store,
#     segment_registry,
#     time_col='time',
#     xy_offset=True,     # set False to use raw EPSG:2056 coords
#     save_path=None #'../plots/debug_veh42.png',
# )
# fig.tight_layout()
# plt.show()

# # plot_trajectory_map(
# #     bike_df, geometry_store, segment_registry,
# #     color_by='segment_id', show_validity_polygons=True,
# #     show_s_change=True, zoom_start=19, 
# #     save_path=f"../maps/debug_maps_{date}_{intersection}_{code}_{timeslot}.html"
# # )

# # from generate_debug_viz import generate_bikelane_debug_map

# # generate_bikelane_debug_map(
# #     bike_df,
# #     segment_registry,
# #     geometry_store,
# #     output_path=f"../maps/debug_maps_{date}_{intersection}_{code}_{timeslot}_{mode}{bike_id}.html",
# #     n_spline_pts=300,        # optional, spline sampling resolution
# # )


