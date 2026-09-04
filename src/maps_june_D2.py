"""
TITLE OF PAPAER
-------------------------------------------
Authors:        Shaimaa El-Baklish
Organization:   ETH Zürich, Switzerland, IVT - Institute for Transportation Planning and Systems
Development:    2025-2026
Submitted to:   JOURNAL
-------------------------------------------

maps_june_D2.py
-------------------------------------
Site definition — Ackerstrasse / Zollstrasse / Mattengasse / Neugasse
Zürich, Switzerland — June 2025 campaign (D2 location)
 
Two intersections:
  MainInt  — 4-way: Ackerstrasse × Zollstrasse × Mattengasse
  MattInt  — T-junction: Mattengasse joins Zollstrasse
 
This file owns ALL geometry sourcing and calls the four builder phases.
Each phase takes plain dicts — customise here without touching the builder.
"""

# #############################################################################
# IMPORTS
# #############################################################################
import sys
import folium
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import osmnx as ox
import geopandas as gpd
import matplotlib.pyplot as plt

from shapely.geometry import box
from shapely.geometry import Point
from shapely.plotting import plot_points
from shapely.plotting import plot_line

from _constants import BikeZ_Config
from tools_utils import _PROJ_2056_TO_LONLAT
from tools_coordinate_transform import cut_line_at_stop
from tools_coordinate_transform import densify_linestring
from tools_site_builder import (
    fit_spline_from_osmnx,
    merge_osmnx_edges,
    fit_spline_from_shapely,
    register_geometries,
    build_segment_registry,
    add_bike_lane_boundaries,
    add_car_lane_boundaries,
    build_turns,
    build_intersection_polygon,
    build_movement_registry,
    serialize_registry,
)

# #############################################################################
# CONSTANTS
# #############################################################################
# Configuration
BikeZ_Config = BikeZ_Config()

# Specify Trajectory File
date = BikeZ_Config.avail_dates[1]
campaign = f"Zurich_2025{date[5:7]}" # June or September
mode = BikeZ_Config.avail_modes[0] # Bike
data_root = BikeZ_Config.data_root[campaign][mode]

intersection, code = BikeZ_Config.avail_intersections[date][2]
timeslot = BikeZ_Config.avail_timeslots[date][(intersection, code)][0] # 'AM1'

XY_2056_Bounds = BikeZ_Config.XY_2056_Bounds[date][(intersection, code)]
X_2056_offset = XY_2056_Bounds[0][0]
Y_2056_offset = XY_2056_Bounds[1][0]

# Site constants
kml_path       = '../maps/from_swisstopo/June_D2.kml'
kml_path_lanes = '../maps/from_swisstopo/June_D2_CarLanes.kml'
save_path      = f'../data/registry_{date}_{intersection}_{code}.pkl'
max_chain_len  = 5    # 3 for standard movements + 2 for Mattengasse chain

# Edit Link: https://s.geo.admin.ch/fidy00bie4q1
# Share Link: https://s.geo.admin.ch/alg9cocb8c92

# Car Lanes:
# Share Link: https://s.geo.admin.ch/dwfapw7b1s4d
# Edit Link: https://s.geo.admin.ch/bmbmch1gxyxg

# #############################################################################
# MAIN
# #############################################################################


# # 25 fps
# filename = f"trajectories_{mode}s_{date}_{intersection}_{timeslot}_{code}-1-ekf"
# # df = pd.read_csv(data_root + f"{date}/{intersection}/{filename}.csv")
# df_25fps = pd.read_parquet(data_root + f"{date}/{intersection}/{filename}.parquet")

# fig, ax = plt.subplots(1, 1, figsize=(6, 4))
# for veh_id, df_veh in df_25fps.groupby('veh_id'):
#     ax.plot(df_veh['x_ekf'], df_veh['y_ekf'], color='black', linewidth=1)
# fig.tight_layout()
# plt.show()

# =============================================================================
# STEP 0: load external data sources
print("Loading OSMnx features...")
lonlat = _PROJ_2056_TO_LONLAT.transform(
    np.asarray(XY_2056_Bounds[0]) + np.asarray([-25, 25]),
    np.asarray(XY_2056_Bounds[1]) + np.asarray([-25, 25]),
)
bbox_geom = box(lonlat[0][0], lonlat[1][0], lonlat[0][1], lonlat[1][1])
 
gdf_main   = ox.features.features_from_place('Zürich, Switzerland',
                                              tags={'highway': True})
road_types = ['primary', 'secondary', 'tertiary',
              'residential', 'unclassified', 'cycleway']
gdf = gdf_main[
    gdf_main['name'].isin(
        ['Zollstrasse', 'Ackerstrasse', 'Mattengasse', 'Neugasse']
    )
]
gdf = gdf[
    (gdf.geometry.type == 'LineString') &
    (gdf['highway'].isin(road_types))
]
gdf['geometry'] = gdf.geometry.intersection(bbox_geom)
gdf = gdf[~gdf.is_empty]

# fig, ax = plt.subplots(1, 1)
# gdf.plot(ax=ax, column='name', legend=True)
# fig.tight_layout()


# from tools_utils import _local_to_latlon

# fig, ax = plt.subplots(1, 1, figsize=(6, 4))

# date = BikeZ_Config.avail_dates[0]
# intersection, code = BikeZ_Config.avail_intersections[date][1]
# for timeslot in BikeZ_Config.avail_timeslots[date][(intersection, code)]:
#     filename = f"trajectories_{mode}s_{date}_{intersection}_{timeslot}_{code}-1-ekf"
#     df = pd.read_parquet(data_root + f"{date}/{intersection}/{filename}.parquet")
#     for veh_id, df_veh in df.groupby('veh_id'):
#         latlon = _local_to_latlon(df_veh['x_ekf'].to_numpy(), df_veh['y_ekf'].to_numpy(), X_2056_offset, Y_2056_offset)
#         latlon = np.asarray(latlon)
#         ax.plot(latlon[:, 1], latlon[:, 0], color='black', linewidth=1)
      

# date = BikeZ_Config.avail_dates[0]
# intersection, code = BikeZ_Config.avail_intersections[date][2]
# for timeslot in BikeZ_Config.avail_timeslots[date][(intersection, code)]:
#     filename = f"trajectories_{mode}s_{date}_{intersection}_{timeslot}_{code}-1-ekf"
#     df = pd.read_parquet(data_root + f"{date}/{intersection}/{filename}.parquet")
#     for veh_id, df_veh in df.groupby('veh_id'):
#         latlon = _local_to_latlon(df_veh['x_ekf'].to_numpy(), df_veh['y_ekf'].to_numpy(), X_2056_offset, Y_2056_offset)
#         latlon = np.asarray(latlon)
#         ax.plot(latlon[:, 1], latlon[:, 0], color='black', linewidth=1)


# date = BikeZ_Config.avail_dates[1]
# intersection, code = BikeZ_Config.avail_intersections[date][2]
# for timeslot in BikeZ_Config.avail_timeslots[date][(intersection, code)]:
#     filename = f"trajectories_{mode}s_{date}_{intersection}_{timeslot}_{code}-1-ekf"
#     df = pd.read_parquet(data_root + f"{date}/{intersection}/{filename}.parquet")
#     for veh_id, df_veh in df.groupby('veh_id'):
#         latlon = _local_to_latlon(df_veh['x_ekf'].to_numpy(), df_veh['y_ekf'].to_numpy(), X_2056_offset, Y_2056_offset)
#         latlon = np.asarray(latlon)
#         ax.plot(latlon[:, 1], latlon[:, 0], color='black', linewidth=1)

# gdf.plot(ax=ax, column='name', legend=True)
# fig.tight_layout()
# plt.show()



print("Loading SwissTopo KML...")
gdf_swisstopo = gpd.read_file(kml_path, driver='KML')

# STEP 1: fit splines  (geometry sourcing, customise per road as needed)
print("\nFitting splines...")
 
# Zollstr: OSMnx, shared centerline
line_Z = merge_osmnx_edges(gdf, 'Zollstrasse')
line_Z = densify_linestring(line_Z, num_segments=40)
tck_Z, unew_Z, cum_Z, len_Z = fit_spline_from_shapely(
    line_Z, x_offset=X_2056_offset, y_offset=Y_2056_offset,
)

# Ackerstrasse: split at Neugasse crossline into north and south branches
acker_full  = merge_osmnx_edges(gdf, 'Ackerstrasse')
neu_line  = merge_osmnx_edges(gdf, 'Neugasse')
acker_north = cut_line_at_stop(acker_full, neu_line, choose='last', plotting=False)
acker_north = densify_linestring(acker_north, num_segments=10)
acker_south = cut_line_at_stop(acker_full, neu_line, choose='first', plotting=False)
acker_south = densify_linestring(acker_south, num_segments=10)

tck_AN, unew_AN, cum_AN, len_AN = fit_spline_from_shapely(
    acker_north, x_offset=X_2056_offset, y_offset=Y_2056_offset,
)
tck_AS, unew_AS, cum_AS, len_AS = fit_spline_from_shapely(
    acker_south, x_offset=X_2056_offset, y_offset=Y_2056_offset,
)

# Neugasse: split at Ackerstrasse crossline into north and south branches
neu_west = cut_line_at_stop(neu_line, acker_full, choose='last', plotting=False)
neu_west = densify_linestring(neu_west, num_segments=10)
neu_east = cut_line_at_stop(neu_line, acker_full, choose='first', plotting=False)
neu_east = densify_linestring(neu_east, num_segments=10)

tck_NeuW, unew_NeuW, cum_NeuW, len_NeuW = fit_spline_from_shapely(
    neu_west, x_offset=X_2056_offset, y_offset=Y_2056_offset,
)
tck_NeuE, unew_NeuE, cum_NeuE, len_NeuE = fit_spline_from_shapely(
    neu_east, x_offset=X_2056_offset, y_offset=Y_2056_offset,
)

print(f"  Zollstr     : {len_Z:.1f} m")
print(f"  AckerstrN   : {len_AN:.1f} m")
print(f"  AckerstrS   : {len_AS:.1f} m")
print(f"  NeugW       : {len_NeuW:.1f} m")
print(f"  NeugE       : {len_NeuE:.1f} m")

fig, ax = plt.subplots(1, 1)
# gdf.plot(ax=ax, column='name', legend=True)
plot_line(line_Z, ax=ax, add_points=False, color='tab:orange', label='Zollstr')
plot_line(acker_north, ax=ax, add_points=False, color='tab:green', label='AckerstrN')
plot_line(acker_south, ax=ax, add_points=False, color='tab:purple', label='AckerstrS')
plot_line(neu_west, ax=ax, add_points=False, color='tab:brown', label='NeugW')
plot_line(neu_east, ax=ax, add_points=False, color='tab:blue', label='NeugE')
plot_points(Point(line_Z.coords[0]), color='red', marker='o',) 
plot_points(Point(line_Z.coords[-1]), color='black', marker='x',) 
plot_points(Point(acker_south.coords[1]), color='red', marker='o',) 
plot_points(Point(acker_south.coords[-2]), color='black', marker='x',) 
plot_points(Point(acker_north.coords[1]), color='red', marker='o',) 
plot_points(Point(acker_north.coords[-1]), color='black', marker='x',) 
plot_points(Point(neu_west.coords[1]), color='red', marker='o',) 
plot_points(Point(neu_west.coords[-1]), color='black', marker='x',) 
plot_points(Point(neu_east.coords[0]), color='red', marker='o',) 
plot_points(Point(neu_east.coords[-2]), color='black', marker='x',) 
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncols=5)
fig.suptitle(
    'Postive Direction Validation\n'
    'Red o marker = Start  |  Black x marker = End',
    fontsize=11
)
fig.tight_layout()

# =============================================================================
# PHASE 1: register_geometries
# stop_line_id / yield_line_id must match the Description field in the KML.
# positive_dir: verified from Phase A plot (red=start, x=end).
#   Verified visually from the Phase A validation plot (red = start, x = end).
#   If wrong: reverse the source polyline in Step 1 above and re-run.
 
print("\n--- Phase 1: register geometries ---")
 
RAW_AXES = [
    {
        'name':          'AckerstrN',
        'positive_dir':  'NB',
        'spline':        (tck_AN, unew_AN, cum_AN),
        'total_length':  len_AN,
        'line_wgs84':    acker_north,
        'stop_line_id':  'AckerstrN_Stop',
        'yield_line_id': 'AckerstrN_Yield',
        'change_ratio':   0.9,   # optional — defaults to 0.6 if omitted; 0 at stop and 1 at yield
    },
    {
        # Stub south of MainInt — dead-ends into Zollstr (T-junction).
        'name':          'AckerstrS',
        'positive_dir':  'NB',
        'spline':        (tck_AS, unew_AS, cum_AS),
        'total_length':  len_AS,
        'line_wgs84':    acker_south,
        'stop_line_id':  'AckerstrS_Stop',
        'yield_line_id': 'AckerstrS_Yield',
        'change_ratio':   0.9,   # optional — defaults to 0.6 if omitted; 0 at stop and 1 at yield
        'extra_changes': [
            {'key': 's_ackerstr_south_stop',
             'stop_line_id': 'AckerstrS_T1_Stop'},
            {'key': 's_ackerstr_south_yield',
             'stop_line_id': 'AckerstrS_T1_Yield'},
        ],
    },
    {
        'name':          'NeugW',
        'positive_dir':  'WB',
        'spline':        (tck_NeuW, unew_NeuW, cum_NeuW),
        'total_length':  len_NeuW,
        'line_wgs84':    neu_west,
        'stop_line_id':  'NeugW_Stop',
        'yield_line_id': 'NeugW_Yield',
        'change_ratio':   0.9,   # optional — defaults to 0.6 if omitted; 0 at stop and 1 at yield
    },
    {
        # Stub east/south of MainInt — dead-ends into Zollstr (T-junction).
        'name':          'NeugE',
        'positive_dir':  'WB',
        'spline':        (tck_NeuE, unew_NeuE, cum_NeuE),
        'total_length':  len_NeuE,
        'line_wgs84':    neu_east,
        'stop_line_id':  'NeugE_Stop',
        'yield_line_id': 'NeugE_Yield',
        'change_ratio':   0.9,   # optional — defaults to 0.6 if omitted; 0 at stop and 1 at yield
        'extra_changes': [
            {'key':          's_neug_east_stop',
             'stop_line_id':  'NeugE_T2_Stop'},
        ],
    },
    {
        'name':          'Zollstr',
        'positive_dir':  'EB',
        'spline':        (tck_Z, unew_Z, cum_Z),
        'total_length':  len_Z,
        'line_wgs84':    line_Z,
        # Primary (for intersection with Langstr and Roentgenstr, not included here)
        'stop_line_id':  'Zollstr_Stop',
        'yield_line_id': 'Zollstr_Yield',
        # Extra = T2 boundary (Neugasse T-junction), further along.
        'extra_changes': [
            {'key': 's_T1_zollstr_west_stop',
             'stop_line_id': 'ZollstrW_T1_Stop'},
            {'key': 's_T1_zollstr_east_yield',
             'stop_line_id': 'ZollstrE_T1_Yield'},
            
            {'key': 's_T2_zollstr_west_stop',
             'stop_line_id': 'ZollstrW_T2_Stop'},
            {'key': 's_T2_zollstr_east_yield',
             'stop_line_id': 'ZollstrE_T2_Yield'},
        ],
    },
]


geometry_store = register_geometries(
    RAW_AXES, gdf_swisstopo, X_2056_offset, Y_2056_offset,
)

# =============================================================================
# PHASE 2: build_segment_registry
# d_left  = tolerance LEFT  of travel (GPS margin, median-strip side) [m]
# d_right = full usable carriageway RIGHT of travel                   [m]
#           includes the bike stripe width if present
#
# For shared centerlines both opposing segments' polygons must together
# cover the full road width without overlapping each other.

print("--- Phase 2: build segment registry ---")
 
SEG_DEFS = [
    # ── Zollstrasse ─────────────────────────────────────────────────────────
    {'seg_key': 'Zollstr_EB', 'geometry_key': 'Zollstr',
     'direction': 'EB', 'mode': 'shared', 'bike_lane': None,
     'd_left': 1.5, 'd_right': 14.0,},
    {'seg_key': 'Zollstr_WB', 'geometry_key': 'Zollstr',
     'direction': 'WB', 'mode': 'bike', 'bike_lane': {'w_bike': 2.0},
     'd_left': 2.0, 'd_right': 8.0},
 
    # ── Ackerstrasse North ──────────────────────────────────────────────────
    {'seg_key': 'AckerstrN_NB', 'geometry_key': 'AckerstrN',
     'direction': 'NB', 'mode': 'shared', 'bike_lane': None,
     'd_left': 3.5, 'd_right': 9.0},
    {'seg_key': 'AckerstrN_SB', 'geometry_key': 'AckerstrN',
     'direction': 'SB', 'mode': 'bike', 'bike_lane': None,
     'd_left': 1.5, 'd_right': 8.0},

    # ── Ackerstrasse South ──────────────────────────────────────────────────
    {'seg_key': 'AckerstrS_NB', 'geometry_key': 'AckerstrS',
     'direction': 'NB', 'mode': 'shared', 'bike_lane': None,
     'd_left': 3.5, 'd_right': 9.0},
    {'seg_key': 'AckerstrS_SB', 'geometry_key': 'AckerstrS',
     'direction': 'SB', 'mode': 'bike', 'bike_lane': None,
     'd_left': 1.5, 'd_right': 8.0},

    # ── Neugasse West ───────────────────────────────────────────────────────
    {'seg_key': 'NeugW_WB', 'geometry_key': 'NeugW',
     'direction': 'WB', 'mode': 'shared', 'bike_lane': None,
     'd_left': 3.5, 'd_right': 8.0},
    {'seg_key': 'NeugW_EB', 'geometry_key': 'NeugW',
     'direction': 'EB', 'mode': 'bike', 'bike_lane': None,
     'd_left': 1.5, 'd_right': 8.0},

    # ── Neugasse East ───────────────────────────────────────────────────────
    # NeugE_EB has a drawn bike-lane boundary in the KML — matched by
    # add_bike_lane_boundaries in Step 2b below.
    {'seg_key': 'NeugE_WB', 'geometry_key': 'NeugE',
     'direction': 'WB', 'mode': 'shared', 'bike_lane': None,
     'd_left': 3.5, 'd_right': 9.0},
    {'seg_key': 'NeugE_EB', 'geometry_key': 'NeugE',
     'direction': 'EB', 'mode': 'bike', 'bike_lane': {'w_bike': 2.0},
     'd_left': 1.5, 'd_right': 8.0},
]
 
segment_registry = build_segment_registry(geometry_store, SEG_DEFS)

print("--- Step 2b: project bike lane boundaries ---")
gdf_bike_boundaries = gdf_swisstopo[
    gdf_swisstopo['Description'].str.endswith(
        ('_NB', '_SB', '_EB', '_WB')
    )
].copy()
add_bike_lane_boundaries(segment_registry, geometry_store, gdf_bike_boundaries)


print("--- Step 2c: project car lane polygons and boundaries ---")
gdf_car_lane_polygons = gpd.read_file(kml_path_lanes, driver='KML')
add_car_lane_boundaries(segment_registry, geometry_store, gdf_car_lane_polygons)

# =============================================================================
# PHASE 3: build_turns
# MainInt (Ackerstr × Neugasse): standard 4-way, all arms use their own
#   primary 's_change' — no explicit s_change_key needed.
# T1 (AckerstrS × Zollstr) / T2 (NeugE × Zollstr): 
#   AckerstrS/NeugE/Zollstr use their own extra_changes keys.

print("\n--- Phase 3: build turn splines ---")

TURN_DEFS = [
    # ── MainInt: from AckerstrN (approach heading south) ───────────────────────
    {'approach_seg': 'AckerstrN_SB', 'departure_seg': 'AckerstrS_SB',
     'd_left': 8.0, 'd_right': 8.0},
    {'approach_seg': 'AckerstrN_SB', 'departure_seg': 'NeugW_WB',
     'd_left': 8.0, 'd_right': 8.0},
    {'approach_seg': 'AckerstrN_SB', 'departure_seg': 'NeugE_EB',
     'd_left': 8.0, 'd_right': 8.0},

    # ── MainInt: from AckerstrS (approach heading north, from T1) ──────────────
    {'approach_seg': 'AckerstrS_NB', 'departure_seg': 'AckerstrN_NB',
     'd_left': 8.0, 'd_right': 8.0},
    {'approach_seg': 'AckerstrS_NB', 'departure_seg': 'NeugW_WB',
     'd_left': 8.0, 'd_right': 8.0},
    {'approach_seg': 'AckerstrS_NB', 'departure_seg': 'NeugE_EB',
     'd_left': 8.0, 'd_right': 8.0},

    # ── MainInt: from NeugW (approach heading east) ─────────────────────────────
    {'approach_seg': 'NeugW_EB', 'departure_seg': 'NeugE_EB',
     'd_left': 8.0, 'd_right': 8.0},
    {'approach_seg': 'NeugW_EB', 'departure_seg': 'AckerstrN_NB',
     'd_left': 8.0, 'd_right': 8.0},
    {'approach_seg': 'NeugW_EB', 'departure_seg': 'AckerstrS_SB',
     'd_left': 8.0, 'd_right': 8.0},

    # ── MainInt: from NeugE (approach heading west, from T2) ────────────────────
    {'approach_seg': 'NeugE_WB', 'departure_seg': 'NeugW_WB',
     'd_left': 8.0, 'd_right': 8.0},
    {'approach_seg': 'NeugE_WB', 'departure_seg': 'AckerstrN_NB',
     'd_left': 8.0, 'd_right': 8.0},
    {'approach_seg': 'NeugE_WB', 'departure_seg': 'AckerstrS_SB',
     'd_left': 8.0, 'd_right': 8.0},

    # ── T1: AckerstrS ↔ Zollstr ──────────────────────────────────────────────
    # AckerstrS approach/departure use its own stop/yield extras.
    # Zollstr approach/departure use its T1 west/east extras — EB travel
    # arrives at the west line, departs past the east line; WB is mirrored.
    {'approach_seg': 'AckerstrS_SB', 'departure_seg': 'Zollstr_EB',
     'approach_s_change_key':  's_ackerstr_south_stop',
     'departure_s_change_key': 's_T1_zollstr_east_yield',
     'd_left': 8.0, 'd_right': 8.0},
    {'approach_seg': 'AckerstrS_SB', 'departure_seg': 'Zollstr_WB',
     'approach_s_change_key':  's_ackerstr_south_stop',
     'departure_s_change_key': 's_T1_zollstr_west_stop',
     'd_left': 8.0, 'd_right': 8.0},
    {'approach_seg': 'Zollstr_EB', 'departure_seg': 'AckerstrS_NB',
     'approach_s_change_key':  's_T1_zollstr_west_stop',
     'departure_s_change_key': 's_ackerstr_south_yield',
     'd_left': 8.0, 'd_right': 8.0},
    {'approach_seg': 'Zollstr_WB', 'departure_seg': 'AckerstrS_NB',
     'approach_s_change_key':  's_T1_zollstr_east_yield',
     'departure_s_change_key': 's_ackerstr_south_yield',
     'd_left': 8.0, 'd_right': 8.0},

    # ── T2: NeugE ↔ Zollstr ───────────────────────────────────────────────────
    {'approach_seg': 'NeugE_EB', 'departure_seg': 'Zollstr_EB',
     'approach_s_change_key':  's_neug_east_stop',
     'departure_s_change_key': 's_T2_zollstr_east_yield',
     'd_left': 8.0, 'd_right': 8.0},
    {'approach_seg': 'NeugE_EB', 'departure_seg': 'Zollstr_WB',
     'approach_s_change_key':  's_neug_east_stop',
     'departure_s_change_key': 's_T2_zollstr_west_stop',
     'd_left': 8.0, 'd_right': 8.0},
    {'approach_seg': 'Zollstr_EB', 'departure_seg': 'NeugE_WB',
     'approach_s_change_key':  's_T2_zollstr_west_stop',
     'departure_s_change_key': 's_neug_east_stop',
     'd_left': 8.0, 'd_right': 8.0},
    {'approach_seg': 'Zollstr_WB', 'departure_seg': 'NeugE_WB',
     'approach_s_change_key':  's_T2_zollstr_east_yield',
     'departure_s_change_key': 's_neug_east_stop',
     'd_left': 8.0, 'd_right': 8.0},
]

turn_keys = build_turns(
    geometry_store, segment_registry, TURN_DEFS,
    n_pts=10, n_connector=100, angle_threshold_deg=5,
)

# =============================================================================
# STEP 3b — build intersection area polygons
# Each arm entry: one road axis entering the intersection.
# Width at normal line = d_right(pos_dir seg) LEFT + d_right(opp_dir seg) RIGHT.
# Stored in geometry_store under 'intersection_area_*' keys.
# Used in scoring to exclude lane segment candidates whose points fall inside
# the intersection box.
print("--- Step 3b: build intersection area polygons ---")
 
geometry_store['intersection_area_MainInt'] = build_intersection_polygon(
    arm_defs = [
        {'geom_key': 'NeugW', 's_change_key': 's_change',
         'pos_seg_key': 'NeugW_WB', 'opp_seg_key': 'NeugW_EB',
         'approach_seg_key': 'NeugW_EB'},
        {'geom_key': 'AckerstrN',    's_change_key': 's_change',
         'pos_seg_key': 'AckerstrN_NB',    'opp_seg_key': 'AckerstrN_SB',
         'approach_seg_key': 'AckerstrN_SB'},
        {'geom_key': 'NeugE',     's_change_key': 's_change',
         'pos_seg_key': 'NeugE_WB',     'opp_seg_key': 'NeugE_EB',
         'approach_seg_key': 'NeugE_WB'},
        {'geom_key': 'AckerstrS',    's_change_key': 's_change',
         'pos_seg_key': 'AckerstrS_NB',    'opp_seg_key': 'AckerstrS_SB',
         'approach_seg_key': 'AckerstrS_NB'},
    ],
    geometry_store   = geometry_store,
    segment_registry = segment_registry,
)
geometry_store['intersection_type_MainInt'] = 'standard'
 
geometry_store['intersection_area_T1_ZollAckerInt'] = build_intersection_polygon(
    arm_defs = [
        {'geom_key': 'Zollstr', 's_change_key': 's_T1_zollstr_west_stop',
         'pos_seg_key': 'Zollstr_EB', 'opp_seg_key': 'Zollstr_WB',
         'approach_seg_key': 'Zollstr_EB'},
        {'geom_key': 'Zollstr', 's_change_key': 's_T1_zollstr_east_yield',
         'pos_seg_key': 'Zollstr_EB', 'opp_seg_key': 'Zollstr_WB',
         'approach_seg_key': 'Zollstr_WB'},
        {'geom_key': 'AckerstrS', 's_change_key': 's_ackerstr_south_yield',
         'pos_seg_key': 'AckerstrS_NB', 'opp_seg_key': 'AckerstrS_SB',
         'approach_seg_key': 'AckerstrS_SB'},
    ],
    geometry_store   = geometry_store,
    segment_registry = segment_registry,
)
geometry_store['intersection_type_T1_ZollAckerInt'] = 'T-junction'

geometry_store['intersection_area_T2_ZollNeuInt'] = build_intersection_polygon(
    arm_defs = [
        {'geom_key': 'Zollstr', 's_change_key': 's_T2_zollstr_west_stop',
         'pos_seg_key': 'Zollstr_EB', 'opp_seg_key': 'Zollstr_WB',
         'approach_seg_key': 'Zollstr_EB'},
        {'geom_key': 'Zollstr', 's_change_key': 's_T2_zollstr_east_yield',
         'pos_seg_key': 'Zollstr_EB', 'opp_seg_key': 'Zollstr_WB',
         'approach_seg_key': 'Zollstr_WB'},
        {'geom_key': 'NeugE', 's_change_key': 's_neug_east_stop',
         'pos_seg_key': 'NeugE_WB', 'opp_seg_key': 'NeugE_EB',
         'approach_seg_key': 'NeugE_EB'},
    ],
    geometry_store   = geometry_store,
    segment_registry = segment_registry,
)
geometry_store['intersection_type_T2_ZollNeuInt'] = 'T-junction'

from tools_plot_registry import plot_geometry_store, plot_segment_registry

plot_geometry_store(geometry_store, gdf_swisstopo, offset_m=3.0)
plot_segment_registry(geometry_store, segment_registry, gdf_swisstopo)

# =============================================================================
# PHASE 4: build_movement_registry

print("--- Phase 4: build movement registry ---")
 
MOVEMENT_DEFS = [
    # ── MainInt: from AckerstrN ─────────────────────────────────────────────
    {'key': 'AckerstrN_SB_2_AckerstrS_SB',
     'sequence': [('AckerstrN_SB',                          'approach'),
                  ('turn_AckerstrN_SB_2_AckerstrS_SB',      'turn'),
                  ('AckerstrS_SB',                          'departure')]},
    {'key': 'AckerstrN_SB_2_NeugW_WB',
     'sequence': [('AckerstrN_SB',                          'approach'),
                  ('turn_AckerstrN_SB_2_NeugW_WB',          'turn'),
                  ('NeugW_WB',                               'departure')]},
    {'key': 'AckerstrN_SB_2_NeugE_EB',
     'sequence': [('AckerstrN_SB',                          'approach'),
                  ('turn_AckerstrN_SB_2_NeugE_EB',          'turn'),
                  ('NeugE_EB',                               'departure')]},

    # ── MainInt: from AckerstrS ─────────────────────────────────────────────
    {'key': 'AckerstrS_NB_2_AckerstrN_NB',
     'sequence': [('AckerstrS_NB',                          'approach'),
                  ('turn_AckerstrS_NB_2_AckerstrN_NB',      'turn'),
                  ('AckerstrN_NB',                          'departure')]},
    {'key': 'AckerstrS_NB_2_NeugW_WB',
     'sequence': [('AckerstrS_NB',                          'approach'),
                  ('turn_AckerstrS_NB_2_NeugW_WB',          'turn'),
                  ('NeugW_WB',                               'departure')]},
    {'key': 'AckerstrS_NB_2_NeugE_EB',
     'sequence': [('AckerstrS_NB',                          'approach'),
                  ('turn_AckerstrS_NB_2_NeugE_EB',          'turn'),
                  ('NeugE_EB',                               'departure')]},

    # ── MainInt: from NeugW ──────────────────────────────────────────────────
    {'key': 'NeugW_EB_2_NeugE_EB',
     'sequence': [('NeugW_EB',                              'approach'),
                  ('turn_NeugW_EB_2_NeugE_EB',              'turn'),
                  ('NeugE_EB',                               'departure')]},
    {'key': 'NeugW_EB_2_AckerstrN_NB',
     'sequence': [('NeugW_EB',                              'approach'),
                  ('turn_NeugW_EB_2_AckerstrN_NB',          'turn'),
                  ('AckerstrN_NB',                          'departure')]},
    {'key': 'NeugW_EB_2_AckerstrS_SB',
     'sequence': [('NeugW_EB',                              'approach'),
                  ('turn_NeugW_EB_2_AckerstrS_SB',          'turn'),
                  ('AckerstrS_SB',                          'departure')]},

    # ── MainInt: from NeugE ──────────────────────────────────────────────────
    {'key': 'NeugE_WB_2_NeugW_WB',
     'sequence': [('NeugE_WB',                              'approach'),
                  ('turn_NeugE_WB_2_NeugW_WB',              'turn'),
                  ('NeugW_WB',                               'departure')]},
    {'key': 'NeugE_WB_2_AckerstrN_NB',
     'sequence': [('NeugE_WB',                              'approach'),
                  ('turn_NeugE_WB_2_AckerstrN_NB',          'turn'),
                  ('AckerstrN_NB',                          'departure')]},
    {'key': 'NeugE_WB_2_AckerstrS_SB',
     'sequence': [('NeugE_WB',                              'approach'),
                  ('turn_NeugE_WB_2_AckerstrS_SB',          'turn'),
                  ('AckerstrS_SB',                          'departure')]},

    # ── T1: AckerstrS ↔ Zollstr ──────────────────────────────────────────────
    {'key': 'AckerstrS_SB_2_Zollstr_EB',
     'sequence': [('AckerstrS_SB',                          'approach'),
                  ('turn_AckerstrS_SB_2_Zollstr_EB',        'turn'),
                  ('Zollstr_EB',                             'departure')]},
    {'key': 'AckerstrS_SB_2_Zollstr_WB',
     'sequence': [('AckerstrS_SB',                          'approach'),
                  ('turn_AckerstrS_SB_2_Zollstr_WB',        'turn'),
                  ('Zollstr_WB',                             'departure')]},
    {'key': 'Zollstr_EB_2_AckerstrS_NB',
     'sequence': [('Zollstr_EB',                            'approach'),
                  ('turn_Zollstr_EB_2_AckerstrS_NB',        'turn'),
                  ('AckerstrS_NB',                          'departure')]},
    {'key': 'Zollstr_WB_2_AckerstrS_NB',
     'sequence': [('Zollstr_WB',                            'approach'),
                  ('turn_Zollstr_WB_2_AckerstrS_NB',        'turn'),
                  ('AckerstrS_NB',                          'departure')]},

    # ── T2: NeugE ↔ Zollstr ──────────────────────────────────────────────────
    {'key': 'NeugE_EB_2_Zollstr_EB',
     'sequence': [('NeugE_EB',                              'approach'),
                  ('turn_NeugE_EB_2_Zollstr_EB',            'turn'),
                  ('Zollstr_EB',                             'departure')]},
    {'key': 'NeugE_EB_2_Zollstr_WB',
     'sequence': [('NeugE_EB',                              'approach'),
                  ('turn_NeugE_EB_2_Zollstr_WB',            'turn'),
                  ('Zollstr_WB',                             'departure')]},
    {'key': 'Zollstr_EB_2_NeugE_WB',
     'sequence': [('Zollstr_EB',                            'approach'),
                  ('turn_Zollstr_EB_2_NeugE_WB',            'turn'),
                  ('NeugE_WB',                               'departure')]},
    {'key': 'Zollstr_WB_2_NeugE_WB',
     'sequence': [('Zollstr_WB',                            'approach'),
                  ('turn_Zollstr_WB_2_NeugE_WB',            'turn'),
                  ('NeugE_WB',                               'departure')]},
]
 
movement_registry = build_movement_registry(
    geometry_store, segment_registry, MOVEMENT_DEFS,
)

# =============================================================================
# SERIALIZE

serialize_registry(
    geometry_store, segment_registry, movement_registry,
    max_chain_length = max_chain_len,
    intersection     = f'{intersection}_{code}',
    date             = date,
    save_path        = save_path,
)

import shutil

dest_path = '../data/registry_2025-06-16_D2_G.pkl'
shutil.copy(save_path, dest_path)
dest_path = '../data/registry_2025-06-16_D2_C.pkl'
shutil.copy(save_path, dest_path)

loc_num = BikeZ_Config.location_map[(date[5:7], intersection, code)]
dest_path = f'../data/registry_location{loc_num}.pkl'
shutil.copy(save_path, dest_path)




from tools_map_visualization import create_registry_map

m = create_registry_map(
    geometry_store, segment_registry, movement_registry,
    gdf_swisstopo,
    save_path=f'../maps/registry_{date}_{intersection}_{code}.html',
)

m = create_registry_map(
    geometry_store, segment_registry, movement_registry,
    gdf_swisstopo,
    base_map_src='gis-zh',
    save_path=f'../maps/registry_location{loc_num}.html',
)
