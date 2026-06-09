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

from tqdm import tqdm
from scipy.interpolate import splprep, splev

from _constants import BikeZ_Config
# from tools_coordinate_transform import match_bicycle_to_centerline
# from tools_coordinate_transform import match_bicycle_to_centerline_v1
# from tools_coordinate_transform import match_bicycle_to_centerline_with_curvature
# from tools_coordinate_transform import convert_xy2056_to_roadway_coordinates
from tools_plotting import plot_lane_coord_debug

# #############################################################################
# CONSTANTS
# #############################################################################
# Configuration
BikeZ_Config = BikeZ_Config()

# Specify Trajectory File
date = BikeZ_Config.avail_dates[0]
campaign = f"Zurich_2025{date[5:7]}" # June or September
mode = BikeZ_Config.avail_modes[0] # 0: Bike, 1: Vehicle
data_root = BikeZ_Config.data_root[campaign][mode]

intersection, code = BikeZ_Config.avail_intersections[date][0]
timeslot = BikeZ_Config.avail_timeslots[date][(intersection, code)][0] # 'AM1'
# PM1 133 centerline_id=nan

XY_2056_Bounds = BikeZ_Config.XY_2056_Bounds[date][(intersection, code)]
X_2056_offset = XY_2056_Bounds[0][0]
Y_2056_offset = XY_2056_Bounds[1][0]

OPP_DIRECTIONS = {"N": "S", "S": "N", "W": "E", "E": "W"}

# bike_lane_tol = 0.4

# #############################################################################
# MAIN: Load data
# #############################################################################
# trajectories after EKF
if mode == "bike":
    filename = f"trajectories_bikes_{date}_{intersection}_{timeslot}_{code}-1-ekf.csv"
else:
    filename = f"trajectories_vehicles_{date}_{intersection}_{timeslot}_{code}-1-ekf.csv"
df = pd.read_csv(data_root + f"{date}/{intersection}/{filename}")
df = df.dropna()
df['x_act_ekf'] = df['x_ekf'] + X_2056_offset
df['y_act_ekf'] = df['y_ekf'] + Y_2056_offset
center_lat, center_lon = df.loc[~df['missing'], "lat"].mean(), df.loc[~df['missing'], "lon"].mean()

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

# from tools_lane_coords_V1 import to_lane_coordinates, assign_segments, score_segment
from tools_lane_coords_V3 import to_lane_coordinates, build_registry_luts
# from tools_lane_coords_V2 import _build_segment_bboxes

# Check: 2, 5, 11, 12, 18
# u-turn: 24
bike_id = 27
bike_df = df[(df["veh_id"] == bike_id)].copy()

# One-time setup — do this once before your vehicle loop
build_registry_luts(geometry_store)
# seg_bboxes = _build_segment_bboxes(segment_registry, geometry_store)

from tools_lane_coords_V3 import POLYGON_ENTRY_TOLERANCE
# Pre-expand validity polygons once — avoids calling .buffer() inside the loop
for entry in segment_registry.values():
    poly = entry.get('validity_polygon')
    if poly is not None and not poly.is_empty:
        entry['_validity_polygon_expanded'] = poly.buffer(POLYGON_ENTRY_TOLERANCE)


# Run full transform
import time
start = time.perf_counter()
bike_df = to_lane_coordinates(
    bike_df, movement_registry,
    segment_registry, geometry_store,
    max_chain_length=max_chain_length,
    # seg_bboxes=seg_bboxes,
    verbose=True
)
end = time.perf_counter()
print(f"Elapsed time: {end - start:.6f} seconds")

from tools_plotting import build_lane_color_map
# Build once per registry — consistent colors across all plots
lane_color_map = build_lane_color_map(geometry_store)


# Debug plot
plot_lane_coord_debug(
    bike_df, segment_registry, geometry_store,
    XY_2056_Bounds, bike_id,
    lane_color_map=lane_color_map,
    save_path=None #f'../debugging/lane_debug_{veh_id}.png'
)

# # Debug map
# from generate_debug_viz import generate_bikelane_debug_map
# generate_bikelane_debug_map(
#     bike_df, segment_registry, geometry_store,
#     output_path='../maps/debug_bikelane_langstr.html'
# )

sys.exit(1)

# print()
# x_offset = geometry_store['x_offset']
# y_offset = geometry_store['y_offset']
# sample_idx = np.linspace(0, len(bike_df)-1, min(20, len(bike_df)), dtype=int)
# xy_local   = bike_df[['x_act_ekf','y_act_ekf']].to_numpy()[sample_idx] \
#              - np.array([x_offset, y_offset])
# psi_sample = bike_df['angle_ekf'].to_numpy()[sample_idx]
# for seg_key in ['LangstrN_SB', 'turn_LangstrN_SB_2_LangstrS_SB']:
#     entry = segment_registry[seg_key]
#     geom_key = entry['geometry_key']
#     tck, unew, cum_dist = geometry_store[geom_key]['spline']

#     # t_vals = []
#     # for pt in xy_local:
#     #     from tools_lane_coords import project_point_warm
#     #     t_star, _ = project_point_warm(pt, tck, t_init=None)
#     #     t_vals.append(t_star)

#     # t_vals = np.array(t_vals)
#     # print(f"{seg_key}: t_vals = {t_vals.round(3)}")
#     # print(f"  clamped at t=0: {(t_vals < 0.001).sum()}  "
#     #       f"clamped at t=1: {(t_vals > 0.999).sum()}  "
#     #       f"free: {((t_vals >= 0.001) & (t_vals <= 0.999)).sum()}")
    
#     score_segment(
#         xy_local, psi_sample, seg_key,
#         segment_registry, geometry_store,
#         verbose=True
#     )
    
# print()
# from tools_lane_coords import project_point_full

# for seg_key in ['LangstrN_SB', 'LangstrS_SB']:
#     entry            = segment_registry[seg_key]
#     geom_key         = entry['geometry_key']
#     tck, unew, cum_dist = geometry_store[geom_key]['spline']
#     is_forward       = entry['is_forward']
#     app_min, app_max = entry['approach_native']
#     dep_min, dep_max = entry['departure_native']
#     L                = geometry_store[geom_key]['total_length']

#     print(f"\n{'='*60}")
#     print(f"{seg_key}")
#     print(f"  is_forward      : {is_forward}")
#     print(f"  total_length    : {L:.2f} m")
#     print(f"  approach_native : s∈[{app_min:.2f}, {app_max:.2f}]")
#     print(f"  departure_native: s∈[{dep_min:.2f}, {dep_max:.2f}]")
#     print(f"  {'pt':>3}  {'t':>6}  {'s':>8}  {'d':>8}  "
#           f"{'in_app':>7}  {'in_dep':>7}  {'clamped':>8}")
#     print(f"  {'-'*3}  {'-'*6}  {'-'*8}  {'-'*8}  "
#           f"{'-'*7}  {'-'*7}  {'-'*8}")

#     for i, pt in enumerate(xy_local):
#         t_star, _, _, s_i, d_i = project_point_full(
#             pt, tck, unew, cum_dist, t_init=None
#         )
#         in_app     = app_min <= s_i <= app_max
#         in_dep     = dep_min <= s_i <= dep_max
#         clamped    = t_star < 0.001 or t_star > 0.999
#         clamp_str  = f't={t_star:.3f} !' if clamped else ''
#         print(f"  {i:>3}  {t_star:>6.3f}  {s_i:>8.2f}  {d_i:>8.2f}  "
#               f"{str(in_app):>7}  {str(in_dep):>7}  {clamp_str:>8}")