"""
TITLE OF PAPAER
-------------------------------------------
Authors:        Shaimaa El-Baklish
Organization:   ETH Zürich, Switzerland, IVT - Institute for Transportation Planning and Systems
Development:    2025
Submitted to:   JOURNAL
-------------------------------------------

maps_sep_D2I.py
-------------------------------------
Site definition — Baslerstrasse / Flurstrasse
Zürich, Switzerland — September 2025 campaign (D2, I location)
 
Two intersections:
  MainInt  — 4-way: Baslerstrasse × Flurstrasse
 
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

from pyproj import Transformer
from shapely.geometry import box
from shapely.geometry import Point
from shapely.plotting import plot_points
from shapely.plotting import plot_line

from _constants import BikeZ_Config
from tools_coordinate_transform import cut_line_at_stop
from tools_site_builder import (
    fit_spline_from_osmnx,
    merge_osmnx_edges,
    fit_spline_from_shapely,
    register_geometries,
    build_segment_registry,
    add_bike_lane_boundaries,
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
date = BikeZ_Config.avail_dates[-1]
campaign = f"Zurich_2025{date[5:7]}" # June or September
mode = BikeZ_Config.avail_modes[0] # Bike
data_root = BikeZ_Config.data_root[campaign][mode]

intersection, code = BikeZ_Config.avail_intersections[date][-1]
timeslot = BikeZ_Config.avail_timeslots[date][(intersection, code)][0] # 'AM1'

XY_2056_Bounds = BikeZ_Config.XY_2056_Bounds[date][(intersection, code)]
X_2056_offset = XY_2056_Bounds[0][0]
Y_2056_offset = XY_2056_Bounds[1][0]

# Site constants
kml_path      = '../maps/from_swisstopo/September_D2I.kml'
save_path     = f'../data/registry_{date}_{intersection}_{code}.pkl'
max_chain_len = 3


# Share Link: https://s.geo.admin.ch/ijzem4yn8y7c
# Edit Link: https://s.geo.admin.ch/8dguwea3mh0r


# #############################################################################
# MAIN
# #############################################################################

# =============================================================================
# STEP 0: load external data sources
print("Loading OSMnx features...")
transformer = Transformer.from_crs('EPSG:2056', 'EPSG:4326', always_xy=True)
lonlat      = transformer.transform(
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
        ['Baslerstrasse', 'Flurstrasse']
    )
]
gdf = gdf[
    (gdf.geometry.type == 'LineString') &
    (gdf['highway'].isin(road_types))
]
gdf['geometry'] = gdf.geometry.intersection(bbox_geom)
gdf = gdf[~gdf.is_empty]

print("Loading SwissTopo KML...")
gdf_swisstopo = gpd.read_file(kml_path, driver='KML')


# STEP 1: fit splines  (geometry sourcing, customise per road as needed)
print("\nFitting splines...")

# Baslerstrasse: split at Flurstrasse crossline into north and south branches
basler_full  = merge_osmnx_edges(gdf, 'Baslerstrasse')
flur_line = merge_osmnx_edges(gdf, 'Flurstrasse')
basler_west  = cut_line_at_stop(basler_full, flur_line, choose='last',  plotting=False)
basler_east  = cut_line_at_stop(basler_full, flur_line, choose='first', plotting=False)
 
tck_BW, unew_BW, cum_BW, len_BW = fit_spline_from_shapely(
    basler_west, x_offset=X_2056_offset, y_offset=Y_2056_offset,
)
tck_BE, unew_BE, cum_BE, len_BE = fit_spline_from_shapely(
    basler_east, x_offset=X_2056_offset, y_offset=Y_2056_offset,
)

# Flurstrasse: split at Baslerstrasse crossline into north and south branches
flur_full   = merge_osmnx_edges(gdf, 'Flurstrasse')
basler_line    = merge_osmnx_edges(gdf, 'Baslerstrasse')
flur_north  = cut_line_at_stop(flur_full, basler_line, choose='last',  plotting=False)
flur_south  = cut_line_at_stop(flur_full, basler_line, choose='first', plotting=False)
 
tck_FN, unew_FN, cum_FN, len_FN = fit_spline_from_shapely(
    flur_north, x_offset=X_2056_offset, y_offset=Y_2056_offset,
)
tck_FS, unew_FS, cum_FS, len_FS = fit_spline_from_shapely(
    flur_south, x_offset=X_2056_offset, y_offset=Y_2056_offset,
)

print(f"  BaslerstrW  : {len_BW:.1f} m")
print(f"  BaslerstrE  : {len_BE:.1f} m")
print(f"  FlurstrN : {len_FN:.1f} m")
print(f"  FlurstrS : {len_FS:.1f} m")

fig, ax = plt.subplots(1, 1)
# gdf.plot(ax=ax, column='name', legend=True)
plot_line(basler_west, ax=ax, add_points=False, color='tab:blue', label='BaslerstrW')
plot_line(basler_east, ax=ax, add_points=False, color='tab:orange', label='BaslerstrE')
plot_line(flur_north, ax=ax, add_points=False, color='tab:green', label='FlurstrN')
plot_line(flur_south, ax=ax, add_points=False, color='tab:purple', label='FlurstrS')
plot_points(Point(basler_west.coords[1]), color='red', marker='o',)
plot_points(Point(basler_west.coords[-1]), color='black', marker='x',)
plot_points(Point(basler_east.coords[0]), color='red', marker='o',)
plot_points(Point(basler_east.coords[-1]), color='black', marker='x',)
plot_points(Point(flur_north.coords[1]), color='red', marker='o',)
plot_points(Point(flur_north.coords[-1]), color='black', marker='x',)
plot_points(Point(flur_south.coords[0]), color='red', marker='o',)
plot_points(Point(flur_south.coords[-1]), color='black', marker='x',)
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
        'name':          'BaslerstrW',
        'positive_dir':  'WB',
        'spline':        (tck_BW, unew_BW, cum_BW),
        'total_length':  len_BW,
        'line_wgs84':    basler_west,
        'stop_line_id':  'BaslerstrW_Stop',
        'yield_line_id': 'BaslerstrW_Yield',
    },
    {
        'name':          'BaslerstrE',
        'positive_dir':  'WB',
        'spline':        (tck_BE, unew_BE, cum_BE),
        'total_length':  len_BE,
        'line_wgs84':    basler_east,
        'stop_line_id':  'BaslerstrE_Stop',
        'yield_line_id': 'BaslerstrE_Yield',
    },
    {
        'name':          'FlurstrN',
        'positive_dir':  'NB',
        'spline':        (tck_FN, unew_FN, cum_FN),
        'total_length':  len_FN,
        'line_wgs84':    flur_north,
        'stop_line_id':  'FlurstrN_Stop',
        'yield_line_id': 'FlurstrN_Yield',
    },
    {
        'name':          'FlurstrS',
        'positive_dir':  'NB',
        'spline':        (tck_FS, unew_FS, cum_FS),
        'total_length':  len_FS,
        'line_wgs84':    flur_south,
        'stop_line_id':  'FlurstrS_Stop',
        'yield_line_id': 'FlurstrS_Yield',
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
    # ── Baslerstrasse West ────────────────────────────────────────────────────
    {'seg_key': 'BaslerstrW_EB', 'geometry_key': 'BaslerstrW',
     'direction': 'EB', 'mode': 'shared', 'bike_lane': {'w_bike': 3.0},
     'd_left': 1.5, 'd_right': 12.0},
    {'seg_key': 'BaslerstrW_WB', 'geometry_key': 'BaslerstrW',
     'direction': 'WB', 'mode': 'shared', 'bike_lane': {'w_bike': 3.0},
     'd_left': 1.5, 'd_right': 12.0},
    
    # ── Baslerstrasse East ────────────────────────────────────────────────────
    {'seg_key': 'BaslerstrE_EB', 'geometry_key': 'BaslerstrE',
     'direction': 'EB', 'mode': 'bike', 'bike_lane': {'w_bike': 4.0},
     'd_left': 1.5, 'd_right': 12.0},
    {'seg_key': 'BaslerstrE_WB', 'geometry_key': 'BaslerstrE',
     'direction': 'WB', 'mode': 'shared', 'bike_lane': {'w_bike': 2.5},
     'd_left': 2.0, 'd_right': 12.0},
    
    # ── Flurstrasse North ─────────────────────────────────────────────────
    {'seg_key': 'FlurstrN_NB', 'geometry_key': 'FlurstrN',
     'direction': 'NB', 'mode': 'shared', 'bike_lane': None,
     'd_left': 1.5, 'd_right': 12.0},
    {'seg_key': 'FlurstrN_SB', 'geometry_key': 'FlurstrN',
     'direction': 'SB', 'mode': 'shared', 'bike_lane': None,
     'd_left': 2.0, 'd_right': 12.0},
    
    # ── Freihoffstrasse South ─────────────────────────────────────────────────
    {'seg_key': 'FlurstrS_NB', 'geometry_key': 'FlurstrS',
     'direction': 'NB', 'mode': 'shared', 'bike_lane': None,
     'd_left': 1.5, 'd_right': 12.0},
    {'seg_key': 'FlurstrS_SB', 'geometry_key': 'FlurstrS',
     'direction': 'SB', 'mode': 'shared', 'bike_lane': None,
     'd_left': 1.0, 'd_right': 12.0},
]

segment_registry = build_segment_registry(geometry_store, SEG_DEFS)

print("--- Step 2b: project bike lane boundaries ---")
gdf_bike_boundaries = gdf_swisstopo[
    gdf_swisstopo['Description'].str.endswith(
        ('_NB', '_SB', '_EB', '_WB')
    )
].copy()
add_bike_lane_boundaries(segment_registry, geometry_store, gdf_bike_boundaries)


# =============================================================================
# PHASE 3: build_turns

print("\n--- Phase 3: build turn splines ---")
 
TURN_DEFS = [
    # ── MainInt: from BaslerstrW_EB ───────────────────────────────────────────
    {'approach_seg': 'BaslerstrW_EB', 'departure_seg': 'FlurstrN_NB',
     'd_left': 15.0, 'd_right': 15.0},
    {'approach_seg': 'BaslerstrW_EB', 'departure_seg': 'BaslerstrE_EB',
     'd_left': 15.0, 'd_right': 15.0},
    {'approach_seg': 'BaslerstrW_EB', 'departure_seg': 'FlurstrS_SB',
     'd_left': 15.0, 'd_right': 15.0},
    
    # ── MainInt: from BaslerstrE_WB ───────────────────────────────────────────
    {'approach_seg': 'BaslerstrE_WB', 'departure_seg': 'FlurstrN_NB',
     'd_left': 15.0, 'd_right': 15.0},
    {'approach_seg': 'BaslerstrE_WB', 'departure_seg': 'BaslerstrW_WB',
     'd_left': 15.0, 'd_right': 15.0},
    {'approach_seg': 'BaslerstrE_WB', 'departure_seg': 'FlurstrS_SB',
     'd_left': 15.0, 'd_right': 15.0},
    
    # ── MainInt: from FlurstrN_SB ──────────────────────────────────────────
    {'approach_seg': 'FlurstrN_SB', 'departure_seg': 'BaslerstrW_WB',
     'd_left': 15.0, 'd_right': 15.0},
    {'approach_seg': 'FlurstrN_SB', 'departure_seg': 'BaslerstrE_EB',
     'd_left': 15.0, 'd_right': 15.0},
    {'approach_seg': 'FlurstrN_SB', 'departure_seg': 'FlurstrS_SB',
     'd_left': 15.0, 'd_right': 15.0},
    
    # ── MainInt: from FlurstrS_NB ──────────────────────────────────────────
    {'approach_seg': 'FlurstrS_NB', 'departure_seg': 'BaslerstrW_WB',
     'd_left': 15.0, 'd_right': 15.0},
    {'approach_seg': 'FlurstrS_NB', 'departure_seg': 'BaslerstrE_EB',
     'd_left': 15.0, 'd_right': 15.0},
    {'approach_seg': 'FlurstrS_NB', 'departure_seg': 'FlurstrN_NB',
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
        {'geom_key': 'BaslerstrW', 's_change_key': 's_change',
         'pos_seg_key': 'BaslerstrW_WB', 'opp_seg_key': 'BaslerstrW_EB',
         'approach_seg_key': 'BaslerstrW_EB'},
        {'geom_key': 'BaslerstrE',    's_change_key': 's_change',
         'pos_seg_key': 'BaslerstrE_WB',    'opp_seg_key': 'BaslerstrE_EB',
         'approach_seg_key': 'BaslerstrE_WB'},
        {'geom_key': 'FlurstrN',     's_change_key': 's_change',
         'pos_seg_key': 'FlurstrN_NB',     'opp_seg_key': 'FlurstrN_SB',
         'approach_seg_key': 'FlurstrN_SB'},
        {'geom_key': 'FlurstrS',    's_change_key': 's_change',
         'pos_seg_key': 'FlurstrS_NB',    'opp_seg_key': 'FlurstrS_SB',
         'approach_seg_key': 'FlurstrS_NB'},
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
    # ── MainInt: from BaslerstrW_EB ───────────────────────────────────────────
    {'key': 'BaslerstrW_EB_2_FlurstrN_NB',
     'sequence': [('BaslerstrW_EB',                    'approach'),
                  ('turn_BaslerstrW_EB_2_FlurstrN_NB', 'turn'),
                  ('FlurstrN_NB',                   'departure')]},
    {'key': 'BaslerstrW_EB_2_BaslerstrE_EB',
     'sequence': [('BaslerstrW_EB',                    'approach'),
                  ('turn_BaslerstrW_EB_2_BaslerstrE_EB',  'turn'),
                  ('BaslerstrE_EB',                    'departure')]},
    {'key': 'BaslerstrW_EB_2_FlurstrS_SB',
     'sequence': [('BaslerstrW_EB',                    'approach'),
                  ('turn_BaslerstrW_EB_2_FlurstrS_SB', 'turn'),
                  ('FlurstrS_SB',                   'departure')]},
    
    # ── MainInt: from BaslerstrE_WB ───────────────────────────────────────────
    {'key': 'BaslerstrE_WB_2_FlurstrN_NB',
     'sequence': [('BaslerstrE_WB',                    'approach'),
                  ('turn_BaslerstrE_WB_2_FlurstrN_NB', 'turn'),
                  ('FlurstrN_NB',                   'departure')]},
    {'key': 'BaslerstrE_WB_2_BaslerstrW_WB',
     'sequence': [('BaslerstrE_WB',                    'approach'),
                  ('turn_BaslerstrE_WB_2_BaslerstrW_WB',  'turn'),
                  ('BaslerstrW_WB',                    'departure')]},
    {'key': 'BaslerstrE_WB_2_FlurstrS_SB',
     'sequence': [('BaslerstrE_WB',                    'approach'),
                  ('turn_BaslerstrE_WB_2_FlurstrS_SB', 'turn'),
                  ('FlurstrS_SB',                   'departure')]},
    
    # ── MainInt: from FlurstrN_SB ──────────────────────────────────────────
    {'key': 'FlurstrN_SB_2_BaslerstrW_WB',
     'sequence': [('FlurstrN_SB',                       'approach'),
                  ('turn_FlurstrN_SB_2_BaslerstrW_WB', 'turn'),
                  ('BaslerstrW_WB',                    'departure')]},
    {'key': 'FlurstrN_SB_2_BaslerstrE_EB',
     'sequence': [('FlurstrN_SB',                       'approach'),
                  ('turn_FlurstrN_SB_2_BaslerstrE_EB',     'turn'),
                  ('BaslerstrE_EB',                        'departure')]},
    {'key': 'FlurstrN_SB_2_FlurstrS_SB',
     'sequence': [('FlurstrN_SB',                       'approach'),
                  ('turn_FlurstrN_SB_2_FlurstrS_SB',    'turn'),
                  ('FlurstrS_SB',                       'departure')]},
    
    # ── MainInt: from FlurstrS_NB ──────────────────────────────────────────
    {'key': 'FlurstrS_NB_2_BaslerstrW_WB',
     'sequence': [('FlurstrS_NB',                       'approach'),
                  ('turn_FlurstrS_NB_2_BaslerstrW_WB', 'turn'),
                  ('BaslerstrW_WB',                    'departure')]},
    {'key': 'FlurstrS_NB_2_BaslerstrE_EB',
     'sequence': [('FlurstrS_NB',                       'approach'),
                  ('turn_FlurstrS_NB_2_BaslerstrE_EB',     'turn'),
                  ('BaslerstrE_EB',                        'departure')]},
    {'key': 'FlurstrS_NB_2_FlurstrN_NB',
     'sequence': [('FlurstrS_NB',                       'approach'),
                  ('turn_FlurstrS_NB_2_FlurstrN_NB',    'turn'),
                  ('FlurstrN_NB',                       'departure')]},
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


from tools_map_visualization import create_registry_map

m = create_registry_map(
    geometry_store, segment_registry, movement_registry,
    gdf_swisstopo,
    save_path=f'../maps/registry_{date}_{intersection}_{code}.html',
)