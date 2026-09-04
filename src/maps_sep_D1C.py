"""
TITLE OF PAPAER
-------------------------------------------
Authors:        Shaimaa El-Baklish
Organization:   ETH Zürich, Switzerland, IVT - Institute for Transportation Planning and Systems
Development:    2025-2026
Submitted to:   JOURNAL
-------------------------------------------

maps_sep_D1C.py
-------------------------------------
Site definition — Duttweilerbrücke / Herdernstrasse / Hohlstrasse
Zürich, Switzerland — September 2025 campaign (D1, C location)
 
Two intersections:
  MainInt  — 4-way: Duttweilerbrücke × Herdernstrasse × Hohlstrasse
 
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
    serialize_registry
)

# #############################################################################
# CONSTANTS
# #############################################################################
# Configuration
BikeZ_Config = BikeZ_Config()

# Specify Trajectory File
date = BikeZ_Config.avail_dates[-2]
campaign = f"Zurich_2025{date[5:7]}" # June or September
mode = BikeZ_Config.avail_modes[0] # Bike
data_root = BikeZ_Config.data_root[campaign][mode]

intersection, code = BikeZ_Config.avail_intersections[date][1]
timeslot = BikeZ_Config.avail_timeslots[date][(intersection, code)][0] # 'AM1'

XY_2056_Bounds = BikeZ_Config.XY_2056_Bounds[date][(intersection, code)]
X_2056_offset = XY_2056_Bounds[0][0]
Y_2056_offset = XY_2056_Bounds[1][0]

# Site constants
kml_path       = '../maps/from_swisstopo/September_D1C.kml'
kml_path_lanes = '../maps/from_swisstopo/September_D1C_CarLanes.kml'
save_path      = f'../data/registry_{date}_{intersection}_{code}.pkl'
max_chain_len  = 3


# Share Link: https://s.geo.admin.ch/0jw7ek5rlie5
# Edit Link: https://s.geo.admin.ch/hf588n9dj44o

# Car Lanes:
# Share Link: https://s.geo.admin.ch/wydfgq7c1quy
# Edit Link: https://s.geo.admin.ch/mv9d4ieu21k6


# #############################################################################
# MAIN
# #############################################################################

# =============================================================================
# STEP 0: load external data sources
print("Loading OSMnx features...")
lonlat = _PROJ_2056_TO_LONLAT.transform(
    np.asarray(XY_2056_Bounds[0]) + np.asarray([-50, 50]),
    np.asarray(XY_2056_Bounds[1]) + np.asarray([-50, 50]),
)
bbox_geom = box(lonlat[0][0], lonlat[1][0], lonlat[0][1], lonlat[1][1])
 
gdf_main   = ox.features.features_from_place('Zürich, Switzerland',
                                              tags={'highway': True})
road_types = ['primary', 'secondary', 'tertiary',
              'residential', 'unclassified', 'cycleway']
gdf = gdf_main[
    gdf_main['name'].isin(
        ['Duttweilerstrasse', 'Duttweilerbrücke', 'Herdernstrasse', 'Hohlstrasse']
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
# sys.exit(1)

print("Loading SwissTopo KML...")
gdf_swisstopo = gpd.read_file(kml_path, driver='KML')


# STEP 1: fit splines  (geometry sourcing, customise per road as needed)
print("\nFitting splines...")

# Herdernstr: OSMnx, shared centerline
tck_He, unew_He, cum_He, len_He = fit_spline_from_osmnx(
    gdf, 'Herdernstrasse', x_offset=X_2056_offset, y_offset=Y_2056_offset,
)
line_He = merge_osmnx_edges(gdf, 'Herdernstrasse')

# Duttweilerbr: OSMnx, shared centerline
line_D = merge_osmnx_edges(gdf, 'Duttweilerbrücke')
line_D = densify_linestring(line=line_D, num_segments=10)
tck_D, unew_D, cum_D, len_D = fit_spline_from_shapely(
    line_D, x_offset=X_2056_offset, y_offset=Y_2056_offset,
)


# Hohlstrasse: split at Duttweilerbrücke crossline into east and west branches
hohl_full = merge_osmnx_edges(gdf, 'Hohlstrasse')
dutt_line = merge_osmnx_edges(gdf, 'Duttweilerbrücke')
hohl_east = cut_line_at_stop(hohl_full, dutt_line, choose='last',  plotting=False)
hohl_west = cut_line_at_stop(hohl_full, dutt_line, choose='first', plotting=False)

tck_HoE, unew_HoE, cum_HoE, len_HoE = fit_spline_from_shapely(
    hohl_east, x_offset=X_2056_offset, y_offset=Y_2056_offset,
)
tck_HoW, unew_HoW, cum_HoW, len_HoW = fit_spline_from_shapely(
    hohl_west, x_offset=X_2056_offset, y_offset=Y_2056_offset,
)

print(f"  HohlstrW     : {len_HoW:.1f} m")
print(f"  HohlstrE     : {len_HoE:.1f} m")
print(f"  Duttweilerbr : {len_D:.1f} m")
print(f"  Herdernstr   : {len_He:.1f} m")

fig, ax = plt.subplots(1, 1)
# gdf.plot(ax=ax, column='name', legend=True)
plot_line(hohl_west, ax=ax, add_points=False, color='tab:blue', label='HohlstrW')
plot_line(hohl_east, ax=ax, add_points=False, color='tab:orange', label='HohlstrE')
plot_line(line_D, ax=ax, add_points=False, color='tab:green', label='Duttweilerbr')
plot_line(line_He, ax=ax, add_points=False, color='tab:purple', label='Herdernstr')
plot_points(Point(hohl_west.coords[0]), color='red', marker='o',)
plot_points(Point(hohl_west.coords[-2]), color='black', marker='x',)
plot_points(Point(hohl_east.coords[1]), color='red', marker='o',)
plot_points(Point(hohl_east.coords[-1]), color='black', marker='x',)
plot_points(Point(line_D.coords[0]), color='red', marker='o',)
plot_points(Point(line_D.coords[-2]), color='black', marker='x',)
plot_points(Point(line_He.coords[1]), color='red', marker='o',)
plot_points(Point(line_He.coords[-1]), color='black', marker='x',)
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
        'name':          'HohlstrW',
        'positive_dir':  'EB',
        'spline':        (tck_HoW, unew_HoW, cum_HoW),
        'total_length':  len_HoW,
        'line_wgs84':    hohl_west,
        'stop_line_id':  'HohlstrW_Stop', 
        'yield_line_id': 'HohlstrW_Yield', 
    },
    {
        'name':          'HohlstrE',
        'positive_dir':  'EB',
        'spline':        (tck_HoE, unew_HoE, cum_HoE),
        'total_length':  len_HoE,
        'line_wgs84':    hohl_east,
        'stop_line_id':  'HohlstrE_Stop',
        'yield_line_id': 'HohlstrE_Yield', 
    },
    {
        'name':          'Duttweilerbr',
        'positive_dir':  'SB',
        'spline':        (tck_D, unew_D, cum_D),
        'total_length':  len_D,
        'line_wgs84':    line_D,
        'stop_line_id':  'Duttweilerbr_Stop',  
        'yield_line_id': 'Duttweilerbr_Yield', 
    },
    {
        'name':          'Herdernstr',
        'positive_dir':  'SB',
        'spline':        (tck_He, unew_He, cum_He),
        'total_length':  len_He,
        'line_wgs84':    line_He,
        'stop_line_id':  'Herdernstr_Stop',  
        'yield_line_id': 'Herdernstr_Yield', 
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
    # ── HohlstrW ────────────────────────────────────────────────────
    {'seg_key': 'HohlstrW_EB', 'geometry_key': 'HohlstrW',
     'direction': 'EB', 'mode': 'shared', 'bike_lane': {'w_bike': 1.5},
     'd_left': 2.0, 'd_right': 11.0},
    {'seg_key': 'HohlstrW_WB', 'geometry_key': 'HohlstrW',
     'direction': 'WB', 'mode': 'shared', 'bike_lane': None,
     'd_left': 1.0, 'd_right': 14.0},

    # ── HohlstrE ────────────────────────────────────────────────────
    {'seg_key': 'HohlstrE_EB', 'geometry_key': 'HohlstrE',
     'direction': 'EB', 'mode': 'shared', 'bike_lane': None,  
     'd_left': 1.0, 'd_right': 15.0},   
    {'seg_key': 'HohlstrE_WB', 'geometry_key': 'HohlstrE',
     'direction': 'WB', 'mode': 'shared', 'bike_lane': {'w_bike': 1.5}, 
     'd_left': 2.0, 'd_right': 18.0},   

    # ── Duttweilerbr ──────────────────────────────────────────────
    {'seg_key': 'Duttweilerbr_SB', 'geometry_key': 'Duttweilerbr',
     'direction': 'SB', 'mode': 'shared', 'bike_lane': {'w_bike': 1.5},
     'd_left': 2.5, 'd_right': 14.0},   
    {'seg_key': 'Duttweilerbr_NB', 'geometry_key': 'Duttweilerbr',
     'direction': 'NB', 'mode': 'shared', 'bike_lane': {'w_bike': 1.5},
     'd_left': 1.0, 'd_right': 16.0},   

    # ── Herdernstr ────────────────────────────────────────────────
    {'seg_key': 'Herdernstr_SB', 'geometry_key': 'Herdernstr',
     'direction': 'SB', 'mode': 'shared', 'bike_lane': None,
     'd_left': 1.0, 'd_right': 10.0},   
    {'seg_key': 'Herdernstr_NB', 'geometry_key': 'Herdernstr',
     'direction': 'NB', 'mode': 'shared', 'bike_lane': {'w_bike': 1.5}, 
     'd_left': 1.0, 'd_right': 13.0},   
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

print("\n--- Phase 3: build turn splines ---")
 
TURN_DEFS = [
    # ── Intersection: from HohlstrW_EB (approach from west) ────────────────
    {'approach_seg': 'HohlstrW_EB', 'departure_seg': 'Duttweilerbr_NB',
     'd_left': 15.0, 'd_right': 15.0},
    {'approach_seg': 'HohlstrW_EB', 'departure_seg': 'HohlstrE_EB',
     'd_left': 15.0, 'd_right': 15.0},
    {'approach_seg': 'HohlstrW_EB', 'departure_seg': 'Herdernstr_SB',
     'd_left': 15.0, 'd_right': 15.0},

    # ── Intersection: from HohlstrE_WB (approach from east) ────────────────
    {'approach_seg': 'HohlstrE_WB', 'departure_seg': 'Duttweilerbr_NB',
     'd_left': 15.0, 'd_right': 15.0},
    {'approach_seg': 'HohlstrE_WB', 'departure_seg': 'HohlstrW_WB',
     'd_left': 15.0, 'd_right': 15.0},
    {'approach_seg': 'HohlstrE_WB', 'departure_seg': 'Herdernstr_SB',
     'd_left': 15.0, 'd_right': 15.0},

    # ── Intersection: from Duttweilerbr_SB (approach from north) ───────────
    {'approach_seg': 'Duttweilerbr_SB', 'departure_seg': 'HohlstrW_WB',
     'd_left': 15.0, 'd_right': 15.0},
    {'approach_seg': 'Duttweilerbr_SB', 'departure_seg': 'HohlstrE_EB',
     'd_left': 15.0, 'd_right': 15.0},
    {'approach_seg': 'Duttweilerbr_SB', 'departure_seg': 'Herdernstr_SB',
     'd_left': 15.0, 'd_right': 15.0},

    # ── Intersection: from Herdernstr_NB (approach from south) ─────────────
    {'approach_seg': 'Herdernstr_NB', 'departure_seg': 'HohlstrW_WB',
     'd_left': 15.0, 'd_right': 15.0},
    {'approach_seg': 'Herdernstr_NB', 'departure_seg': 'HohlstrE_EB',
     'd_left': 15.0, 'd_right': 15.0},
    {'approach_seg': 'Herdernstr_NB', 'departure_seg': 'Duttweilerbr_NB',
     'd_left': 15.0, 'd_right': 15.0},
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
        {'geom_key': 'HohlstrW', 's_change_key': 's_change',
         'pos_seg_key': 'HohlstrW_EB', 'opp_seg_key': 'HohlstrW_WB',
         'approach_seg_key': 'HohlstrW_EB'},
        {'geom_key': 'HohlstrE',    's_change_key': 's_change',
         'pos_seg_key': 'HohlstrE_EB',    'opp_seg_key': 'HohlstrE_WB',
         'approach_seg_key': 'HohlstrE_WB'},
        {'geom_key': 'Duttweilerbr',     's_change_key': 's_change',
         'pos_seg_key': 'Duttweilerbr_SB',     'opp_seg_key': 'Duttweilerbr_NB',
         'approach_seg_key': 'Duttweilerbr_SB'},
        {'geom_key': 'Herdernstr',    's_change_key': 's_change',
         'pos_seg_key': 'Herdernstr_SB',    'opp_seg_key': 'Herdernstr_NB',
         'approach_seg_key': 'Herdernstr_NB'},
    ],
    geometry_store   = geometry_store,
    segment_registry = segment_registry,
)
geometry_store['intersection_type_MainInt'] = 'standard'

from tools_plot_registry import plot_geometry_store, plot_segment_registry

plot_geometry_store(geometry_store, gdf_swisstopo, offset_m=3.0)
plot_segment_registry(geometry_store, segment_registry, gdf_swisstopo)

# =============================================================================
# PHASE 4: build_movement_registry

print("--- Phase 4: build movement registry ---")
 
MOVEMENT_DEFS = [
    # ── Intersection: from HohlstrW_EB ──────────────────────────────────────
    {'key': 'HohlstrW_EB_2_Duttweilerbr_NB',
     'sequence': [('HohlstrW_EB',                          'approach'),
                  ('turn_HohlstrW_EB_2_Duttweilerbr_NB',    'turn'),
                  ('Duttweilerbr_NB',                       'departure')]},
    {'key': 'HohlstrW_EB_2_HohlstrE_EB',
     'sequence': [('HohlstrW_EB',                          'approach'),
                  ('turn_HohlstrW_EB_2_HohlstrE_EB',        'turn'),
                  ('HohlstrE_EB',                           'departure')]},
    {'key': 'HohlstrW_EB_2_Herdernstr_SB',
     'sequence': [('HohlstrW_EB',                          'approach'),
                  ('turn_HohlstrW_EB_2_Herdernstr_SB',      'turn'),
                  ('Herdernstr_SB',                         'departure')]},

    # ── Intersection: from HohlstrE_WB ──────────────────────────────────────
    {'key': 'HohlstrE_WB_2_Duttweilerbr_NB',
     'sequence': [('HohlstrE_WB',                          'approach'),
                  ('turn_HohlstrE_WB_2_Duttweilerbr_NB',    'turn'),
                  ('Duttweilerbr_NB',                       'departure')]},
    {'key': 'HohlstrE_WB_2_HohlstrW_WB',
     'sequence': [('HohlstrE_WB',                          'approach'),
                  ('turn_HohlstrE_WB_2_HohlstrW_WB',        'turn'),
                  ('HohlstrW_WB',                           'departure')]},
    {'key': 'HohlstrE_WB_2_Herdernstr_SB',
     'sequence': [('HohlstrE_WB',                          'approach'),
                  ('turn_HohlstrE_WB_2_Herdernstr_SB',      'turn'),
                  ('Herdernstr_SB',                         'departure')]},

    # ── Intersection: from Duttweilerbr_SB ──────────────────────────────────
    {'key': 'Duttweilerbr_SB_2_HohlstrW_WB',
     'sequence': [('Duttweilerbr_SB',                      'approach'),
                  ('turn_Duttweilerbr_SB_2_HohlstrW_WB',    'turn'),
                  ('HohlstrW_WB',                           'departure')]},
    {'key': 'Duttweilerbr_SB_2_HohlstrE_EB',
     'sequence': [('Duttweilerbr_SB',                      'approach'),
                  ('turn_Duttweilerbr_SB_2_HohlstrE_EB',    'turn'),
                  ('HohlstrE_EB',                           'departure')]},
    {'key': 'Duttweilerbr_SB_2_Herdernstr_SB',
     'sequence': [('Duttweilerbr_SB',                      'approach'),
                  ('turn_Duttweilerbr_SB_2_Herdernstr_SB',  'turn'),
                  ('Herdernstr_SB',                         'departure')]},

    # ── Intersection: from Herdernstr_NB ────────────────────────────────────
    {'key': 'Herdernstr_NB_2_HohlstrW_WB',
     'sequence': [('Herdernstr_NB',                        'approach'),
                  ('turn_Herdernstr_NB_2_HohlstrW_WB',      'turn'),
                  ('HohlstrW_WB',                           'departure')]},
    {'key': 'Herdernstr_NB_2_HohlstrE_EB',
     'sequence': [('Herdernstr_NB',                        'approach'),
                  ('turn_Herdernstr_NB_2_HohlstrE_EB',      'turn'),
                  ('HohlstrE_EB',                           'departure')]},
    {'key': 'Herdernstr_NB_2_Duttweilerbr_NB',
     'sequence': [('Herdernstr_NB',                        'approach'),
                  ('turn_Herdernstr_NB_2_Duttweilerbr_NB',  'turn'),
                  ('Duttweilerbr_NB',                       'departure')]},
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