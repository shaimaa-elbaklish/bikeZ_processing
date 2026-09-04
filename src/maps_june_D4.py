"""
TITLE OF PAPAER
-------------------------------------------
Authors:        Shaimaa El-Baklish
Organization:   ETH Zürich, Switzerland, IVT - Institute for Transportation Planning and Systems
Development:    2025-2026
Submitted to:   JOURNAL
-------------------------------------------

maps_june_D4.py
-------------------------------------
Site definition — Gessnerbrucke / Gessnerallee / Usteristrasse
Zürich, Switzerland — June 2025 campaign (D4, F location)
 
Two intersections:
  MainInt  — 4-way: Gessnerbrucke × Gessnerallee × Usteristrasse
 
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
date = BikeZ_Config.avail_dates[0]
campaign = f"Zurich_2025{date[5:7]}" # June or September
mode = BikeZ_Config.avail_modes[0] # Bike
data_root = BikeZ_Config.data_root[campaign][mode]

intersection, code = BikeZ_Config.avail_intersections[date][4]
timeslot = BikeZ_Config.avail_timeslots[date][(intersection, code)][0] # 'AM1'

XY_2056_Bounds = BikeZ_Config.XY_2056_Bounds[date][(intersection, code)]
X_2056_offset = XY_2056_Bounds[0][0]
Y_2056_offset = XY_2056_Bounds[1][0]

# Site constants
kml_path       = '../maps/from_swisstopo/June_D4.kml'
kml_path_lanes = '../maps/from_swisstopo/June_D4_CarLanes.kml'
kml_path_gis   = '../maps/from_swisstopo/June_D3_D4_GIS.kml'
save_path      = f'../data/registry_{date}_{intersection}_{code}.pkl'
max_chain_len  = 3

# OLD
# Share Link: https://s.geo.admin.ch/ge8picdgx322
# Edit Link: https://s.geo.admin.ch/tkhkli2704t4

# NEW
# Share Link: https://s.geo.admin.ch/rk5mnnonw0nb
# Edit Link: https://s.geo.admin.ch/n7j61z3xguv3

# GIS:
# Share Link: https://geo.zh.ch/s/bf55b3ae-d28b-4558-a2dd-561a6fdbb716

# Car Lanes
# Share Link: https://s.geo.admin.ch/7v0s81st1ljb
# Edit Link: https://s.geo.admin.ch/p9qxhbi3bwj7

# #############################################################################
# HELPER FUNCTIONS
# #############################################################################
def merge_replace(gdf_base, gdf_override, key='description'):
    """
    Replace rows in gdf_base whose `key` value also appears in gdf_override
    with the gdf_override version. Anything in gdf_override not present in
    gdf_base gets appended as new.
    """
    override_names = set(gdf_override[key].dropna())

    # Keep only base rows NOT being replaced
    gdf_base_filtered = gdf_base[~gdf_base[key].isin(override_names)]

    # Combine: kept base rows + all override rows
    merged = gpd.GeoDataFrame(
        pd.concat([gdf_base_filtered, gdf_override], ignore_index=True),
        crs=gdf_base.crs
    )
    return merged

# #############################################################################
# MAIN
# #############################################################################

# =============================================================================
# STEP 0: load external data sources
print("Loading OSMnx features...")
lonlat = _PROJ_2056_TO_LONLAT.transform(
    np.asarray(XY_2056_Bounds[0]) + np.asarray([-30, 30]),
    np.asarray(XY_2056_Bounds[1]) + np.asarray([-30, 30]),
)
bbox_geom = box(lonlat[0][0], lonlat[1][0], lonlat[0][1], lonlat[1][1])
 
gdf_main   = ox.features.features_from_place('Zürich, Switzerland',
                                              tags={'highway': True})
road_types = ['primary', 'secondary', 'tertiary',
              'residential', 'unclassified', 'cycleway']
gdf = gdf_main[
    gdf_main['name'].isin(
        ['Usteristrasse', 'Gessnerallee', 'Gessnerbrücke']
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

print("Loading GIS ZH KML...")
from tools_site_builder import read_gis_kml

color_to_name = {
    '#ff0000': 'Gessnerbr_EB',            # red
    '#00ff00': 'Gessnerbr_Usteristr_CL',  # green
    '#0000ff': 'KasernenstrN_NB',         # blue
}
gdf_gis = read_gis_kml(kml_path_gis, color_to_name)
# merge them
gdf_swisstopo = merge_replace(gdf_swisstopo, gdf_gis, key='Description')

# STEP 1: fit splines  (geometry sourcing, customise per road as needed)
print("\nFitting splines...")

# Gessnerallee: split at Gessnerbrücke crossline into north and south branches
gessnerall_full  = merge_osmnx_edges(gdf, 'Gessnerallee')
gessnerbr_line   = merge_osmnx_edges(gdf, 'Gessnerbrücke')
gessnerall_north = cut_line_at_stop(gessnerall_full, gessnerbr_line, choose='first',  plotting=False)
gessnerall_south = cut_line_at_stop(gessnerall_full, gessnerbr_line, choose='last', plotting=False)

tck_GAN, unew_GAN, cum_GAN, len_GAN = fit_spline_from_shapely(
    gessnerall_north, x_offset=X_2056_offset, y_offset=Y_2056_offset,
)
tck_GAS, unew_GAS, cum_GAS, len_GAS = fit_spline_from_shapely(
    gessnerall_south, x_offset=X_2056_offset, y_offset=Y_2056_offset,
)

# print("Fitting Gessnerbr spline...")
# tck_GEB, unew_GEB, cum_GEB, len_GEB = fit_spline_from_osmnx(
#     gdf, 'Gessnerbrücke', x_offset=X_2056_offset, y_offset=Y_2056_offset
# )
# line_GEB = merge_osmnx_edges(gdf, 'Gessnerbrücke')

# print("Fitting Gessnerbr sidewalk (WB) spline from SwissTopo...")
# line_GWB = gdf_swisstopo[gdf_swisstopo['Description'] == 'Gessnerbr_WB'].geometry.item()
# tck_GWB, unew_GWB, cum_GWB, len_GWB = fit_spline_from_shapely(
#     line_GWB, x_offset=X_2056_offset, y_offset=Y_2056_offset
# )

# NEW: Get it from CL extracted with GIS-ZH
print("Fitting Gessnerbr spline...")
gessner_usteri_line = gdf_swisstopo[gdf_swisstopo['Description'] == 'Gessnerbr_Usteristr_CL'].geometry.item()
gessnerall_line  = merge_osmnx_edges(gdf, 'Gessnerallee')
gessnerbr_line = cut_line_at_stop(gessner_usteri_line, gessnerall_line, choose='first',  plotting=False)
tck_GB, unew_GB, cum_GB, len_GB = fit_spline_from_shapely(
    gessnerbr_line, x_offset=X_2056_offset, y_offset=Y_2056_offset,
)

# Usteristrasse
print("Fitting Usteristr spline...")
usteri_line = cut_line_at_stop(gessner_usteri_line, gessnerall_line, choose='last',  plotting=False)

tck_U, unew_U, cum_U, len_U = fit_spline_from_shapely(
    usteri_line, x_offset=X_2056_offset, y_offset=Y_2056_offset,
)

print(f"  GessnerallN  : {len_GAN:.1f} m")
print(f"  GessnerallS  : {len_GAS:.1f} m")
# print(f"  Gessnerbr_EB : {len_GEB:.1f} m")
# print(f"  Gessnerbr_WB : {len_GWB:.1f} m")
print(f"  Gessnerbr    : {len_GB:.1f} m")
print(f"  Usteristr    : {len_U:.1f} m")


fig, ax = plt.subplots(1, 1)
# gdf.plot(ax=ax, column='name', legend=True)
plot_line(gessnerall_north, ax=ax, add_points=False, color='tab:blue', label='GessnerallN')
plot_line(gessnerall_south, ax=ax, add_points=False, color='tab:orange', label='GessnerallS')
plot_line(gessnerbr_line, ax=ax, add_points=False, color='tab:green', label='Gessnerbr')
plot_line(usteri_line, ax=ax, add_points=False, color='tab:purple', label='Usteristr')
plot_points(Point(gessnerall_north.coords[0]), color='red', marker='o',)
plot_points(Point(gessnerall_north.coords[-2]), color='black', marker='x',)
plot_points(Point(gessnerall_south.coords[1]), color='red', marker='o',)
plot_points(Point(gessnerall_south.coords[-1]), color='black', marker='x',)
plot_points(Point(gessnerbr_line.coords[0]), color='red', marker='o',)
plot_points(Point(gessnerbr_line.coords[-2]), color='black', marker='x',)
plot_points(Point(usteri_line.coords[1]), color='red', marker='o',)
plot_points(Point(usteri_line.coords[-1]), color='black', marker='x',)
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
        'name':          'Gessnerbr',
        'positive_dir':  'EB',
        'spline':        (tck_GB, unew_GB, cum_GB),
        'total_length':  len_GB,
        'line_wgs84':    gessnerbr_line,
        'stop_line_id':  'Gessnerbr_Stop',
        'yield_line_id': 'Gessnerbr_Yield',
    },
    {
        'name':          'Usteristr',
        'positive_dir':  'EB',
        'spline':        (tck_U, unew_U, cum_U),
        'total_length':  len_U,
        'line_wgs84':    usteri_line,
        'stop_line_id':  'Usteristr_Stop',
        'yield_line_id': 'Usteristr_Yield',
    },
    {
        'name':          'GessnerallN',
        'positive_dir':  'SB',
        'spline':        (tck_GAN, unew_GAN, cum_GAN),
        'total_length':  len_GAN,
        'line_wgs84':    gessnerall_north,
        'stop_line_id':  'GessnerallN_Stop',
        'yield_line_id': 'GessnerallN_Yield',
    },
    {
        'name':          'GessnerallS',
        'positive_dir':  'SB',
        'spline':        (tck_GAS, unew_GAS, cum_GAS),
        'total_length':  len_GAS,
        'line_wgs84':    gessnerall_south,
        'stop_line_id':  'GessnerallS_Stop',
        'yield_line_id': 'GessnerallS_Yield',
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
    # ── Gessnerbrücke (bridge, two-way, shared centerline) ──────────────────
    {'seg_key': 'Gessnerbr_EB', 'geometry_key': 'Gessnerbr',
     'direction': 'EB', 'mode': 'shared', 'bike_lane': {'w_bike': 2.5},
     'd_left': 1.5, 'd_right': 13.0},  
    {'seg_key': 'Gessnerbr_WB', 'geometry_key': 'Gessnerbr',
     'direction': 'WB', 'mode': 'bike', 'bike_lane': {'w_bike': 4.5},
     'd_left': 1.5, 'd_right': 13.0},

    # ── Usteristrasse (one-way EB) ──────────────────────────────────────────
    {'seg_key': 'Usteristr_EB', 'geometry_key': 'Usteristr',
     'direction': 'EB', 'mode': 'shared', 'bike_lane': None,  
     'd_left': 1.5, 'd_right': 12.0},
    # add this for bikes only!
    {'seg_key': 'Usteristr_WB', 'geometry_key': 'Usteristr',
     'direction': 'WB', 'mode': 'bike', 'bike_lane': None, 
     'd_left': 1.5, 'd_right': 14.0},  

    # ── Gessnerallee North (one-way SB) ─────────────────────────────────────
    {'seg_key': 'GessnerallN_SB', 'geometry_key': 'GessnerallN',
     'direction': 'SB', 'mode': 'shared', 'bike_lane': {'w_bike': 4.5},  
     'd_left': 15.0, 'd_right': 10.0},
    # add this for bikes only!
    {'seg_key': 'GessnerallN_NB', 'geometry_key': 'GessnerallN',
     'direction': 'NB', 'mode': 'bike', 'bike_lane': None,  
     'd_left': 1.5, 'd_right': 15.0},

    # ── Gessnerallee South (one-way SB) ─────────────────────────────────────
    {'seg_key': 'GessnerallS_SB', 'geometry_key': 'GessnerallS',
     'direction': 'SB', 'mode': 'shared', 'bike_lane': {'w_bike': 8.5},
     'd_left': 14.0, 'd_right': 16.0},
    # add this for bikes only!
    {'seg_key': 'GessnerallS_NB', 'geometry_key': 'GessnerallS',
     'direction': 'NB', 'mode': 'bike', 'bike_lane': None,
     'd_left': 1.5, 'd_right': 14.0},
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
    # ── MainInt: from Gessnerbr_EB ────────────────────────────────────────
    {'approach_seg': 'Gessnerbr_EB', 'departure_seg': 'Usteristr_EB',
     'd_left': 15.0, 'd_right': 15.0},
    {'approach_seg': 'Gessnerbr_EB', 'departure_seg': 'GessnerallN_NB',
     'd_left': 15.0, 'd_right': 15.0},
    {'approach_seg': 'Gessnerbr_EB', 'departure_seg': 'GessnerallS_SB',
     'd_left': 15.0, 'd_right': 15.0},

    # ── MainInt: from Usteristr_WB ────────────────────────────────────────
    {'approach_seg': 'Usteristr_WB', 'departure_seg': 'Gessnerbr_WB',
     'd_left': 15.0, 'd_right': 15.0},
    {'approach_seg': 'Usteristr_WB', 'departure_seg': 'GessnerallN_NB',
     'd_left': 15.0, 'd_right': 15.0},
    {'approach_seg': 'Usteristr_WB', 'departure_seg': 'GessnerallS_SB',
     'd_left': 15.0, 'd_right': 15.0},

    # ── MainInt: from GessnerallN_SB ──────────────────────────────────────
    {'approach_seg': 'GessnerallN_SB', 'departure_seg': 'Gessnerbr_WB',
     'd_left': 15.0, 'd_right': 15.0},
    {'approach_seg': 'GessnerallN_SB', 'departure_seg': 'Usteristr_EB',
     'd_left': 15.0, 'd_right': 15.0},
    {'approach_seg': 'GessnerallN_SB', 'departure_seg': 'GessnerallS_SB',
     'd_left': 15.0, 'd_right': 15.0},

    # ── MainInt: from GessnerallS_NB ──────────────────────────────────────
    {'approach_seg': 'GessnerallS_NB', 'departure_seg': 'Gessnerbr_WB',
     'd_left': 15.0, 'd_right': 15.0},
    {'approach_seg': 'GessnerallS_NB', 'departure_seg': 'Usteristr_EB',
     'd_left': 15.0, 'd_right': 15.0},
    {'approach_seg': 'GessnerallS_NB', 'departure_seg': 'GessnerallN_NB',
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
        {'geom_key': 'Gessnerbr', 's_change_key': 's_change',
         'pos_seg_key': 'Gessnerbr_EB', 'opp_seg_key': 'Gessnerbr_WB',
         'approach_seg_key': 'Gessnerbr_EB'},
        {'geom_key': 'Usteristr',    's_change_key': 's_change',
         'pos_seg_key': 'Usteristr_EB',    'opp_seg_key': 'Usteristr_WB',
         'approach_seg_key': 'Usteristr_WB'},
        {'geom_key': 'GessnerallN',     's_change_key': 's_change',
         'pos_seg_key': 'GessnerallN_SB',     'opp_seg_key': 'GessnerallN_NB',
         'approach_seg_key': 'GessnerallN_SB'},
        {'geom_key': 'GessnerallS',    's_change_key': 's_change',
         'pos_seg_key': 'GessnerallS_SB',    'opp_seg_key': 'GessnerallS_NB',
         'approach_seg_key': 'GessnerallS_NB'},
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
    # ── MainInt: from Gessnerbr_EB ──────────────────────────────────────────
    {'key': 'Gessnerbr_EB_2_Usteristr_EB',
     'sequence': [('Gessnerbr_EB',                         'approach'),
                  ('turn_Gessnerbr_EB_2_Usteristr_EB',      'turn'),
                  ('Usteristr_EB',                          'departure')]},
    {'key': 'Gessnerbr_EB_2_GessnerallN_NB',
     'sequence': [('Gessnerbr_EB',                          'approach'),
                  ('turn_Gessnerbr_EB_2_GessnerallN_NB',     'turn'),
                  ('GessnerallN_NB',                        'departure')]},
    {'key': 'Gessnerbr_EB_2_GessnerallS_SB',
     'sequence': [('Gessnerbr_EB',                          'approach'),
                  ('turn_Gessnerbr_EB_2_GessnerallS_SB',     'turn'),
                  ('GessnerallS_SB',                        'departure')]},

    # ── MainInt: from Usteristr_WB ──────────────────────────────────────────
    {'key': 'Usteristr_WB_2_Gessnerbr_WB',
     'sequence': [('Usteristr_WB',                          'approach'),
                  ('turn_Usteristr_WB_2_Gessnerbr_WB',       'turn'),
                  ('Gessnerbr_WB',                          'departure')]},
    {'key': 'Usteristr_WB_2_GessnerallN_NB',
     'sequence': [('Usteristr_WB',                          'approach'),
                  ('turn_Usteristr_WB_2_GessnerallN_NB',     'turn'),
                  ('GessnerallN_NB',                        'departure')]},
    {'key': 'Usteristr_WB_2_GessnerallS_SB',
     'sequence': [('Usteristr_WB',                          'approach'),
                  ('turn_Usteristr_WB_2_GessnerallS_SB',     'turn'),
                  ('GessnerallS_SB',                        'departure')]},

    # ── MainInt: from GessnerallN_SB ────────────────────────────────────────
    {'key': 'GessnerallN_SB_2_Gessnerbr_WB',
     'sequence': [('GessnerallN_SB',                        'approach'),
                  ('turn_GessnerallN_SB_2_Gessnerbr_WB',     'turn'),
                  ('Gessnerbr_WB',                          'departure')]},
    {'key': 'GessnerallN_SB_2_Usteristr_EB',
     'sequence': [('GessnerallN_SB',                        'approach'),
                  ('turn_GessnerallN_SB_2_Usteristr_EB',     'turn'),
                  ('Usteristr_EB',                          'departure')]},
    {'key': 'GessnerallN_SB_2_GessnerallS_SB',
     'sequence': [('GessnerallN_SB',                        'approach'),
                  ('turn_GessnerallN_SB_2_GessnerallS_SB',   'turn'),
                  ('GessnerallS_SB',                        'departure')]},

    # ── MainInt: from GessnerallS_NB ────────────────────────────────────────
    {'key': 'GessnerallS_NB_2_Gessnerbr_WB',
     'sequence': [('GessnerallS_NB',                        'approach'),
                  ('turn_GessnerallS_NB_2_Gessnerbr_WB',     'turn'),
                  ('Gessnerbr_WB',                          'departure')]},
    {'key': 'GessnerallS_NB_2_Usteristr_EB',
     'sequence': [('GessnerallS_NB',                        'approach'),
                  ('turn_GessnerallS_NB_2_Usteristr_EB',     'turn'),
                  ('Usteristr_EB',                          'departure')]},
    {'key': 'GessnerallS_NB_2_GessnerallN_NB',
     'sequence': [('GessnerallS_NB',                        'approach'),
                  ('turn_GessnerallS_NB_2_GessnerallN_NB',   'turn'),
                  ('GessnerallN_NB',                        'departure')]},
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

dest_path = '../data/registry_2025-06-17_D4_F.pkl'
shutil.copy(save_path, dest_path)

loc_num = BikeZ_Config.location_map[(date[5:7], intersection, code)]
dest_path = f'../data/registry_location{loc_num}.pkl'
shutil.copy(save_path, dest_path)


from tools_map_visualization import create_registry_map

m = create_registry_map(
    geometry_store, segment_registry, movement_registry,
    gdf_swisstopo,
    save_path=f'../maps/registry_{date}_{intersection}_{code}.html',
) # uses base_map_src='swisstopo' by default

m = create_registry_map(
    geometry_store, segment_registry, movement_registry,
    gdf_swisstopo,
    base_map_src='gis-zh',
    save_path=f'../maps/registry_location{loc_num}.html',
)
