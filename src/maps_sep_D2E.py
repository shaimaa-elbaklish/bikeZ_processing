"""
TITLE OF PAPAER
-------------------------------------------
Authors:        Shaimaa El-Baklish
Organization:   ETH Zürich, Switzerland, IVT - Institute for Transportation Planning and Systems
Development:    2025
Submitted to:   JOURNAL
-------------------------------------------

maps_sep_D2E.py
-------------------------------------
Site definition — Baslerstrasse / Bullingerstrasse / Herdernstrasse
Zürich, Switzerland — September 2025 campaign (D2, E location)
 
Two intersections:
  MainInt  — 4-way: Bullingerstrasse × Herdernstrasse
 
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
from tools_coordinate_transform import densify_linestring
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
date = BikeZ_Config.avail_dates[-2]
campaign = f"Zurich_2025{date[5:7]}" # June or September
mode = BikeZ_Config.avail_modes[0] # Bike
data_root = BikeZ_Config.data_root[campaign][mode]

intersection, code = BikeZ_Config.avail_intersections[date][-1]
timeslot = BikeZ_Config.avail_timeslots[date][(intersection, code)][0] # 'AM1'

XY_2056_Bounds = BikeZ_Config.XY_2056_Bounds[date][(intersection, code)]
X_2056_offset = XY_2056_Bounds[0][0]
Y_2056_offset = XY_2056_Bounds[1][0]

# Site constants
kml_path      = '../maps/from_swisstopo/September_D2E.kml'
save_path     = f'../data/registry_{date}_{intersection}_{code}.pkl'
max_chain_len = 3


# Share Link: https://s.geo.admin.ch/v6yysdm0qxh2
# Edit Link: https://s.geo.admin.ch/7aqsyld1sgfe

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
        ['Baslerstrasse', 'Bullingerstrasse', 'Herdernstrasse']
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

# Herdernstrasse: split at Baslerstrasse crossline into north and south branches
herdern_full  = merge_osmnx_edges(gdf, 'Herdernstrasse')
basler_line = merge_osmnx_edges(gdf, 'Baslerstrasse')
herdern_north  = cut_line_at_stop(herdern_full, basler_line, choose='first',  plotting=False)
herdern_north  = densify_linestring(line=herdern_north, num_segments=10)
herdern_south  = cut_line_at_stop(herdern_full, basler_line, choose='last', plotting=False)
herdern_south  = densify_linestring(line=herdern_south, num_segments=10)

tck_HN, unew_HN, cum_HN, len_HN = fit_spline_from_shapely(
    herdern_north, x_offset=X_2056_offset, y_offset=Y_2056_offset,
)
tck_HS, unew_HS, cum_HS, len_HS = fit_spline_from_shapely(
    herdern_south, x_offset=X_2056_offset, y_offset=Y_2056_offset,
)

# Baslerstrasse: OSMnx, shared centerline
tck_Ba, unew_Ba, cum_Ba, len_Ba = fit_spline_from_osmnx(
    gdf, 'Baslerstrasse', x_offset=X_2056_offset, y_offset=Y_2056_offset,
)
line_Ba = merge_osmnx_edges(gdf, 'Baslerstrasse')

# Bullingerstrasse: OSMnx, shared centerline
tck_Bu, unew_Bu, cum_Bu, len_Bu = fit_spline_from_osmnx(
    gdf, 'Bullingerstrasse', x_offset=X_2056_offset, y_offset=Y_2056_offset,
)
line_Bu = merge_osmnx_edges(gdf, 'Bullingerstrasse')


print(f"  HerdernstrN  : {len_HN:.1f} m")
print(f"  HerdernstrS  : {len_HS:.1f} m")
print(f"  Baslerstr    : {len_Ba:.1f} m")
print(f"  Bullingerstr : {len_Bu:.1f} m")

fig, ax = plt.subplots(1, 1)
# gdf.plot(ax=ax, column='name', legend=True)
plot_line(herdern_north, ax=ax, add_points=False, color='tab:blue', label='HerdernstrN')
plot_line(herdern_south, ax=ax, add_points=False, color='tab:orange', label='HerdernstrS')
plot_line(line_Ba, ax=ax, add_points=False, color='tab:green', label='Baslerstr')
plot_line(line_Bu, ax=ax, add_points=False, color='tab:purple', label='Bullingerstr')
plot_points(Point(herdern_north.coords[0]), color='red', marker='o',)
plot_points(Point(herdern_north.coords[-2]), color='black', marker='x',)
plot_points(Point(herdern_south.coords[1]), color='red', marker='o',)
plot_points(Point(herdern_south.coords[-1]), color='black', marker='x',)
plot_points(Point(line_Ba.coords[0]), color='red', marker='o',)
plot_points(Point(line_Ba.coords[-2]), color='black', marker='x',)
plot_points(Point(line_Bu.coords[1]), color='red', marker='o',)
plot_points(Point(line_Bu.coords[-1]), color='black', marker='x',)
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
        'name':          'Baslerstr',
        'positive_dir':  'EB',
        'spline':        (tck_Ba, unew_Ba, cum_Ba),
        'total_length':  len_Ba,
        'line_wgs84':    line_Ba,
        'stop_line_id':  'Baslerstr_Stop',
        'yield_line_id': 'Baslerstr_Yield',
        'change_ratio':   1.0,   # optional — defaults to 0.6 if omitted; 0 at stop and 1 at yield
    },
    {
        'name':          'Bullingerstr',
        'positive_dir':  'EB',
        'spline':        (tck_Bu, unew_Bu, cum_Bu),
        'total_length':  len_Bu,
        'line_wgs84':    line_Bu,
        'stop_line_id':  'Bullingerstr_Stop',
        'yield_line_id': 'Bullingerstr_Yield',
        'change_ratio':   1.0,   # optional — defaults to 0.6 if omitted; 0 at stop and 1 at yield
    },
    {
        'name':          'HerdernstrN',
        'positive_dir':  'SB',
        'spline':        (tck_HN, unew_HN, cum_HN),
        'total_length':  len_HN,
        'line_wgs84':    herdern_north,
        'stop_line_id':  'HerdernstrN_Stop',
        'yield_line_id': 'HerdernstrN_Yield',
        'change_ratio':   1.0,   # optional — defaults to 0.6 if omitted; 0 at stop and 1 at yield
    },
    {
        'name':          'HerdernstrS',
        'positive_dir':  'SB',
        'spline':        (tck_HS, unew_HS, cum_HS),
        'total_length':  len_HS,
        'line_wgs84':    herdern_south,
        'stop_line_id':  'HerdernstrS_Stop',
        'yield_line_id': 'HerdernstrS_Yield',
        'change_ratio':   1.0,   # optional — defaults to 0.6 if omitted; 0 at stop and 1 at yield
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
    # ── Baslerstrasse ────────────────────────────────────────────────────────
    {'seg_key': 'Baslerstr_EB', 'geometry_key': 'Baslerstr',
     'direction': 'EB', 'mode': 'shared', 'bike_lane': {'w_bike': 1.8},
     'd_left': 1.5, 'd_right': 14.0},
    {'seg_key': 'Baslerstr_WB', 'geometry_key': 'Baslerstr',
     'direction': 'WB', 'mode': 'shared', 'bike_lane': {'w_bike': 1.8},
     'd_left': 1.5, 'd_right': 13.0},
    
    # ── Bullingerstrasse ─────────────────────────────────────────────────────
    {'seg_key': 'Bullingerstr_EB', 'geometry_key': 'Bullingerstr',
     'direction': 'EB', 'mode': 'bike', 'bike_lane': {'w_bike': 3.5},
     'd_left': 1.0, 'd_right': 11.0},
    {'seg_key': 'Bullingerstr_WB', 'geometry_key': 'Bullingerstr',
     'direction': 'WB', 'mode': 'bike', 'bike_lane': {'w_bike': 3.0},
     'd_left': 0.5, 'd_right': 14.0},
    
    # ── Herdernstrasse North ─────────────────────────────────────────────────
    {'seg_key': 'HerdernstrN_NB', 'geometry_key': 'HerdernstrN',
     'direction': 'NB', 'mode': 'shared', 'bike_lane': {'w_bike': 1.5},
     'd_left': 1.0, 'd_right': 11.0},
    {'seg_key': 'HerdernstrN_SB', 'geometry_key': 'HerdernstrN',
     'direction': 'SB', 'mode': 'shared', 'bike_lane': {'w_bike': 1.5},
     'd_left': 1.0, 'd_right': 11.0},
    
    # ── Herdernstrasse South ─────────────────────────────────────────────────
    {'seg_key': 'HerdernstrS_NB', 'geometry_key': 'HerdernstrS',
     'direction': 'NB', 'mode': 'shared', 'bike_lane': {'w_bike': 1.2},
     'd_left': 1.0, 'd_right': 11.0},
    {'seg_key': 'HerdernstrS_SB', 'geometry_key': 'HerdernstrS',
     'direction': 'SB', 'mode': 'shared', 'bike_lane': {'w_bike': 1.2},
     'd_left': 1.0, 'd_right': 10.0},
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
    # ── MainInt: from Baslerstr_EB ───────────────────────────────────────────
    {'approach_seg': 'Baslerstr_EB', 'departure_seg': 'HerdernstrN_NB',
     'd_left': 15.0, 'd_right': 15.0},
    {'approach_seg': 'Baslerstr_EB', 'departure_seg': 'Bullingerstr_EB',
     'd_left': 15.0, 'd_right': 15.0},
    {'approach_seg': 'Baslerstr_EB', 'departure_seg': 'HerdernstrS_SB',
     'd_left': 15.0, 'd_right': 15.0},
    
    # ── MainInt: from Bullingerstr_WB ───────────────────────────────────────────
    {'approach_seg': 'Bullingerstr_WB', 'departure_seg': 'HerdernstrN_NB',
     'd_left': 15.0, 'd_right': 15.0},
    {'approach_seg': 'Bullingerstr_WB', 'departure_seg': 'Baslerstr_WB',
     'd_left': 15.0, 'd_right': 15.0},
    {'approach_seg': 'Bullingerstr_WB', 'departure_seg': 'HerdernstrS_SB',
     'd_left': 15.0, 'd_right': 15.0},
    
    # ── MainInt: from HerdernstrN_SB ──────────────────────────────────────────
    {'approach_seg': 'HerdernstrN_SB', 'departure_seg': 'Baslerstr_WB',
     'd_left': 15.0, 'd_right': 15.0},
    {'approach_seg': 'HerdernstrN_SB', 'departure_seg': 'Bullingerstr_EB',
     'd_left': 15.0, 'd_right': 15.0},
    {'approach_seg': 'HerdernstrN_SB', 'departure_seg': 'HerdernstrS_SB',
     'd_left': 15.0, 'd_right': 15.0},
    
    # ── MainInt: from HerdernstrS_NB ──────────────────────────────────────────
    {'approach_seg': 'HerdernstrS_NB', 'departure_seg': 'Baslerstr_WB',
     'd_left': 15.0, 'd_right': 15.0},
    {'approach_seg': 'HerdernstrS_NB', 'departure_seg': 'Bullingerstr_EB',
     'd_left': 15.0, 'd_right': 15.0},
    {'approach_seg': 'HerdernstrS_NB', 'departure_seg': 'HerdernstrN_NB',
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
        {'geom_key': 'Baslerstr', 's_change_key': 's_change',
         'pos_seg_key': 'Baslerstr_EB', 'opp_seg_key': 'Baslerstr_WB',
         'approach_seg_key': 'Baslerstr_EB'},
        {'geom_key': 'Bullingerstr',    's_change_key': 's_change',
         'pos_seg_key': 'Bullingerstr_EB',    'opp_seg_key': 'Bullingerstr_WB',
         'approach_seg_key': 'Bullingerstr_WB'},
        {'geom_key': 'HerdernstrN',     's_change_key': 's_change',
         'pos_seg_key': 'HerdernstrN_SB',     'opp_seg_key': 'HerdernstrN_NB',
         'approach_seg_key': 'HerdernstrN_SB'},
        {'geom_key': 'HerdernstrS',    's_change_key': 's_change',
         'pos_seg_key': 'HerdernstrS_SB',    'opp_seg_key': 'HerdernstrS_NB',
         'approach_seg_key': 'HerdernstrS_NB'},
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
    # ── MainInt: from Baslerstr_EB ───────────────────────────────────────────
    {'key': 'Baslerstr_EB_2_HerdernstrN_NB',
     'sequence': [('Baslerstr_EB',                    'approach'),
                  ('turn_Baslerstr_EB_2_HerdernstrN_NB', 'turn'),
                  ('HerdernstrN_NB',                   'departure')]},
    {'key': 'Baslerstr_EB_2_Bullingerstr_EB',
     'sequence': [('Baslerstr_EB',                    'approach'),
                  ('turn_Baslerstr_EB_2_Bullingerstr_EB',  'turn'),
                  ('Bullingerstr_EB',                    'departure')]},
    {'key': 'Baslerstr_EB_2_HerdernstrS_SB',
     'sequence': [('Baslerstr_EB',                    'approach'),
                  ('turn_Baslerstr_EB_2_HerdernstrS_SB', 'turn'),
                  ('HerdernstrS_SB',                   'departure')]},
    
    # ── MainInt: from Bullingerstr_WB ───────────────────────────────────────────
    {'key': 'Bullingerstr_WB_2_HerdernstrN_NB',
     'sequence': [('Bullingerstr_WB',                    'approach'),
                  ('turn_Bullingerstr_WB_2_HerdernstrN_NB', 'turn'),
                  ('HerdernstrN_NB',                   'departure')]},
    {'key': 'Bullingerstr_WB_2_Baslerstr_WB',
     'sequence': [('Bullingerstr_WB',                    'approach'),
                  ('turn_Bullingerstr_WB_2_Baslerstr_WB',  'turn'),
                  ('Baslerstr_WB',                    'departure')]},
    {'key': 'Bullingerstr_WB_2_HerdernstrS_SB',
     'sequence': [('Bullingerstr_WB',                    'approach'),
                  ('turn_Bullingerstr_WB_2_HerdernstrS_SB', 'turn'),
                  ('HerdernstrS_SB',                   'departure')]},
    
    # ── MainInt: from HerdernstrN_SB ──────────────────────────────────────────
    {'key': 'HerdernstrN_SB_2_Baslerstr_WB',
     'sequence': [('HerdernstrN_SB',                       'approach'),
                  ('turn_HerdernstrN_SB_2_Baslerstr_WB', 'turn'),
                  ('Baslerstr_WB',                    'departure')]},
    {'key': 'HerdernstrN_SB_2_Bullingerstr_EB',
     'sequence': [('HerdernstrN_SB',                       'approach'),
                  ('turn_HerdernstrN_SB_2_Bullingerstr_EB',     'turn'),
                  ('Bullingerstr_EB',                        'departure')]},
    {'key': 'HerdernstrN_SB_2_HerdernstrS_SB',
     'sequence': [('HerdernstrN_SB',                       'approach'),
                  ('turn_HerdernstrN_SB_2_HerdernstrS_SB',    'turn'),
                  ('HerdernstrS_SB',                       'departure')]},
    
    # ── MainInt: from HerdernstrS_NB ──────────────────────────────────────────
    {'key': 'HerdernstrS_NB_2_Baslerstr_WB',
     'sequence': [('HerdernstrS_NB',                       'approach'),
                  ('turn_HerdernstrS_NB_2_Baslerstr_WB', 'turn'),
                  ('Baslerstr_WB',                    'departure')]},
    {'key': 'HerdernstrS_NB_2_Bullingerstr_EB',
     'sequence': [('HerdernstrS_NB',                       'approach'),
                  ('turn_HerdernstrS_NB_2_Bullingerstr_EB',     'turn'),
                  ('Bullingerstr_EB',                        'departure')]},
    {'key': 'HerdernstrS_NB_2_HerdernstrN_NB',
     'sequence': [('HerdernstrS_NB',                       'approach'),
                  ('turn_HerdernstrS_NB_2_HerdernstrN_NB',    'turn'),
                  ('HerdernstrN_NB',                       'departure')]},
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