"""
TITLE OF PAPAER
-------------------------------------------
Authors:        Shaimaa El-Baklish
Organization:   ETH Zürich, Switzerland, IVT - Institute for Transportation Planning and Systems
Development:    2025
Submitted to:   JOURNAL
-------------------------------------------

maps_june_D1.py
-------------------------------------
Site definition — Langstrasse / Zollstrasse / Röntgenstrasse / Mattengasse
Zürich, Switzerland — June 2025 campaign (D1 location)
 
Two intersections:
  MainInt  — 4-way: Langstrasse × Zollstrasse × Röntgenstrasse
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

intersection, code = BikeZ_Config.avail_intersections[date][1]
timeslot = BikeZ_Config.avail_timeslots[date][(intersection, code)][0] # 'AM1'

XY_2056_Bounds = BikeZ_Config.XY_2056_Bounds[date][(intersection, code)]
X_2056_offset = XY_2056_Bounds[0][0]
Y_2056_offset = XY_2056_Bounds[1][0]

# Site constants
kml_path      = '../maps/from_swisstopo/June_D1.kml'
save_path     = f'../data/registry_{date}_{intersection}_{code}.pkl'
max_chain_len = 5    # 3 for standard movements + 2 for Mattengasse chain

# Link to edit drawing: https://s.geo.admin.ch/an2gmd9mh9zf
# Share Link: https://s.geo.admin.ch/h2rb3k3hqdqb
# GIS Share Link: https://geo.zh.ch/s/d4e68882-4d3a-4946-b208-6eb95c78f4da

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
        ['Zollstrasse', 'Langstrasse', 'Röntgenstrasse', 'Mattengasse']
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
 
# Roentgenstr: OSMnx, shared centerline
tck_R, unew_R, cum_R, len_R = fit_spline_from_osmnx(
    gdf, 'Röntgenstrasse', x_offset=X_2056_offset, y_offset=Y_2056_offset,
)
line_R = merge_osmnx_edges(gdf, 'Röntgenstrasse')

# Zollstr: OSMnx, shared centerline
tck_Z, unew_Z, cum_Z, len_Z = fit_spline_from_osmnx(
    gdf, 'Zollstrasse', x_offset=X_2056_offset, y_offset=Y_2056_offset,
)
line_Z = merge_osmnx_edges(gdf, 'Zollstrasse')

# Langstrasse: split at Zollstrasse crossline into north and south branches
lang_full  = merge_osmnx_edges(gdf, 'Langstrasse')
zoll_line  = merge_osmnx_edges(gdf, 'Zollstrasse')
lang_north = cut_line_at_stop(lang_full, zoll_line, choose='last',  plotting=False)
lang_south = cut_line_at_stop(lang_full, zoll_line, choose='first', plotting=False)
 
tck_LN, unew_LN, cum_LN, len_LN = fit_spline_from_shapely(
    lang_north, x_offset=X_2056_offset, y_offset=Y_2056_offset,
)
tck_LS, unew_LS, cum_LS, len_LS = fit_spline_from_shapely(
    lang_south, x_offset=X_2056_offset, y_offset=Y_2056_offset,
)

# Mattengasse: OSMnx, shared centerline
tck_M, unew_M, cum_M, len_M = fit_spline_from_osmnx(
    gdf, 'Mattengasse', x_offset=X_2056_offset, y_offset=Y_2056_offset,
)
line_M = merge_osmnx_edges(gdf, 'Mattengasse')

print(f"  Roentgenstr : {len_R:.1f} m")
print(f"  Zollstr     : {len_Z:.1f} m")
print(f"  LangstrN    : {len_LN:.1f} m")
print(f"  LangstrS    : {len_LS:.1f} m")
print(f"  Mattengasse : {len_M:.1f} m")

fig, ax = plt.subplots(1, 1)
# gdf.plot(ax=ax, column='name', legend=True)
plot_line(line_R, ax=ax, add_points=False, color='tab:blue', label='Roentgenstr')
plot_line(line_Z, ax=ax, add_points=False, color='tab:orange', label='Zollstr')
plot_line(lang_north, ax=ax, add_points=False, color='tab:green', label='LangstrN')
plot_line(lang_south, ax=ax, add_points=False, color='tab:purple', label='LangstrS')
plot_line(line_M, ax=ax, add_points=False, color='tab:brown', label='Matteng')
plot_points(Point(line_R.coords[1]), color='red', marker='o',) # label='Start Roentgenstr')
plot_points(Point(line_R.coords[-1]), color='black', marker='x',) # label='End Roentgenstr')
plot_points(Point(line_Z.coords[1]), color='red', marker='o',) # label='Start Zollstr')
plot_points(Point(line_Z.coords[-1]), color='black', marker='x',) # label='End Zollstr')
plot_points(Point(lang_south.coords[0]), color='red', marker='o',) # label='Start LangstrS')
plot_points(Point(lang_south.coords[-1]), color='black', marker='x',) # label='End LangstrS')
plot_points(Point(lang_north.coords[1]), color='red', marker='o',) # label='Start LangstrN')
plot_points(Point(lang_north.coords[-1]), color='black', marker='x',) # label='End LangstrN')
plot_points(Point(line_M.coords[0]), color='red', marker='o',) # label='Start Matteng')
plot_points(Point(line_M.coords[-1]), color='black', marker='x',) # label='End Matteng')
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
        'name':          'Roentgenstr',
        'positive_dir':  'WB',
        'spline':        (tck_R, unew_R, cum_R),
        'total_length':  len_R,
        'line_wgs84':    line_R,
        'stop_line_id':  'Roentgenstr_Stop',
        'yield_line_id': 'Roentgenstr_Yield',
    },
    {
        'name':          'Zollstr',
        'positive_dir':  'EB',
        'spline':        (tck_Z, unew_Z, cum_Z),
        'total_length':  len_Z,
        'line_wgs84':    line_Z,
        'stop_line_id':  'Zollstr_Stop',
        'yield_line_id': 'Zollstr_Yield',
        # Secondary junction: Mattengasse T-junction on Zollstr.
        # Two individual boundary lines — stored directly, NOT as midpoint.
        # s_zollstr_east_stop  = eastern red line (ZollstrE_Stop)
        # s_zollstr_west_yield = western red line (ZollstrW_Yield)
        # Each is used by exactly the turns where it is the relevant boundary.
        'extra_changes': [
            {'key': 's_zollstr_east_stop',
             'stop_line_id': 'ZollstrE_Stop'},
            {'key': 's_zollstr_west_yield',
             'stop_line_id': 'ZollstrW_Yield'},
        ],
    },
    {
        'name':          'LangstrN',
        'positive_dir':  'NB',
        'spline':        (tck_LN, unew_LN, cum_LN),
        'total_length':  len_LN,
        'line_wgs84':    lang_north,
        'stop_line_id':  'LangstrN_Stop',
        'yield_line_id': 'LangstrN_Yield',
    },
    {
        'name':          'LangstrS',
        'positive_dir':  'NB',
        'spline':        (tck_LS, unew_LS, cum_LS),
        'total_length':  len_LS,
        'line_wgs84':    lang_south,
        'stop_line_id':  'LangstrS_Stop',
        'yield_line_id': 'LangstrS_Yield',
    },
    {
        # positive_dir='SB': s increases southbound toward the T-junction.
        # Verify: red marker should be at the northern (far-from-Zollstr) end.
        'name':          'Matteng',
        'positive_dir':  'SB',
        'spline':        (tck_M, unew_M, cum_M),
        'total_length':  len_M,
        'line_wgs84':    line_M,
        'stop_line_id':  'Matteng_Stop',
        'yield_line_id': 'Matteng_Yield',
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
    # ── Röntgenstrasse ───────────────────────────────────────────────────────
    {'seg_key': 'Roentgenstr_EB', 'geometry_key': 'Roentgenstr',
     'direction': 'EB', 'mode': 'shared', 'bike_lane': {'w_bike': 1.5},
     'd_left': 1.0, 'd_right': 18.0},
    {'seg_key': 'Roentgenstr_WB', 'geometry_key': 'Roentgenstr',
     'direction': 'WB', 'mode': 'shared', 'bike_lane': {'w_bike': 1.5},
     'd_left': 1.0, 'd_right': 12.0},
 
    # ── Zollstrasse ─────────────────────────────────────────────────────────
    {'seg_key': 'Zollstr_EB', 'geometry_key': 'Zollstr',
     'direction': 'EB', 'mode': 'shared', 'bike_lane': None,
     'd_left': 1.0, 'd_right': 18.0},
    {'seg_key': 'Zollstr_WB', 'geometry_key': 'Zollstr',
     'direction': 'WB', 'mode': 'bike', 'bike_lane': {'w_bike': 2.0},
     'd_left': 2.0, 'd_right': 8.0},
 
    # ── Langstrasse North ────────────────────────────────────────────────────
    {'seg_key': 'LangstrN_NB', 'geometry_key': 'LangstrN',
     'direction': 'NB', 'mode': 'shared', 'bike_lane': None,
     'd_left': 1.0, 'd_right': 8.0},
    {'seg_key': 'LangstrN_SB', 'geometry_key': 'LangstrN',
     'direction': 'SB', 'mode': 'shared', 'bike_lane': None,
     'd_left': 3.0, 'd_right': 8.0},
 
    # ── Langstrasse South ────────────────────────────────────────────────────
    {'seg_key': 'LangstrS_NB', 'geometry_key': 'LangstrS',
     'direction': 'NB', 'mode': 'shared', 'bike_lane': {'w_bike': 2.5},
     'd_left': 3.0, 'd_right': 11.0},
    {'seg_key': 'LangstrS_SB', 'geometry_key': 'LangstrS',
     'direction': 'SB', 'mode': 'shared', 'bike_lane': {'w_bike': 2.5},
     'd_left': 1.0, 'd_right': 12.0},
 
    # ── Mattengasse ──────────────────────────────────────────────────────────
    {'seg_key': 'Matteng_SB', 'geometry_key': 'Matteng',
     'direction': 'SB', 'mode': 'shared', 'bike_lane': None,
     'd_left': 2.5, 'd_right': 12.0},
    {'seg_key': 'Matteng_NB', 'geometry_key': 'Matteng',
     'direction': 'NB', 'mode': 'bike', 'bike_lane': {'w_bike': 1.2},
     'd_left': 2.5, 'd_right': 12.0},
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
# MattInt s_change_key convention:
#   Matteng_SB approach  → uses 's_change'           (Matteng primary boundary)
#   Zollstr_EB departure → uses 's_zollstr_east_stop' (eastern line, near for EB)
#   Zollstr_WB departure → uses 's_zollstr_west_yield'(western line, near for WB)
#   Zollstr_EB approach  → uses 's_zollstr_east_stop' (leaving EB at eastern line)
#   Zollstr_WB approach  → uses 's_zollstr_west_yield'(leaving WB at western line)
#   Matteng_NB departure → uses 's_change'           (Matteng primary boundary)

print("\n--- Phase 3: build turn splines ---")
 
TURN_DEFS = [
    # ── MainInt: from Röntgenstr EB ──────────────────────────────────────────
    {'approach_seg': 'Roentgenstr_EB', 'departure_seg': 'LangstrN_NB',
     'd_left': 15.0, 'd_right': 15.0},
    {'approach_seg': 'Roentgenstr_EB', 'departure_seg': 'Zollstr_EB',
     'd_left': 17.0, 'd_right': 17.0},
    {'approach_seg': 'Roentgenstr_EB', 'departure_seg': 'LangstrS_SB',
     'd_left': 15.0, 'd_right': 15.0},
 
    # ── MainInt: from Zollstr WB ─────────────────────────────────────────────
    {'approach_seg': 'Zollstr_WB', 'departure_seg': 'LangstrN_NB',
     'd_left': 15.0, 'd_right': 15.0},
    {'approach_seg': 'Zollstr_WB', 'departure_seg': 'Roentgenstr_WB',
     'd_left': 17.0, 'd_right': 17.0},
    {'approach_seg': 'Zollstr_WB', 'departure_seg': 'LangstrS_SB',
     'd_left': 15.0, 'd_right': 15.0},
 
    # ── MainInt: from LangstrN SB ────────────────────────────────────────────
    {'approach_seg': 'LangstrN_SB', 'departure_seg': 'Roentgenstr_WB',
     'd_left': 15.0, 'd_right': 15.0},
    {'approach_seg': 'LangstrN_SB', 'departure_seg': 'Zollstr_EB',
     'd_left': 15.0, 'd_right': 15.0},
    {'approach_seg': 'LangstrN_SB', 'departure_seg': 'LangstrS_SB',
     'd_left': 17.0, 'd_right': 17.0},
 
    # ── MainInt: from LangstrS NB ────────────────────────────────────────────
    {'approach_seg': 'LangstrS_NB', 'departure_seg': 'Roentgenstr_WB',
     'd_left': 15.0, 'd_right': 15.0},
    {'approach_seg': 'LangstrS_NB', 'departure_seg': 'Zollstr_EB',
     'd_left': 15.0, 'd_right': 15.0},
    {'approach_seg': 'LangstrS_NB', 'departure_seg': 'LangstrN_NB',
     'd_left': 17.0, 'd_right': 17.0},
 
    # ── MattInt: Matteng_SB → Zollstr ────────────────────────────────────────
    # Matteng_SB approaches from north, exits onto Zollstr.
    # → right turn onto Zollstr_EB: departure samples from eastern stop line
    {'approach_seg': 'Matteng_SB', 'departure_seg': 'Zollstr_EB',
     'approach_s_change_key':  's_change',
     'departure_s_change_key': 's_zollstr_east_stop',
     'd_left': 12.0, 'd_right': 12.0},
    # → left turn onto Zollstr_WB: departure samples from western yield line
    {'approach_seg': 'Matteng_SB', 'departure_seg': 'Zollstr_WB',
     'approach_s_change_key':  's_change',
     'departure_s_change_key': 's_zollstr_west_yield',
     'd_left': 12.0, 'd_right': 12.0},
 
    # ── MattInt: Zollstr → Matteng_NB ────────────────────────────────────────
    # Zollstr_EB heading east, turns left into Mattengasse (going NB).
    # → approach samples up to eastern stop line
    {'approach_seg': 'Zollstr_EB', 'departure_seg': 'Matteng_NB',
     'approach_s_change_key':  's_zollstr_west_yield',
     'departure_s_change_key': 's_change',
     'd_left': 12.0, 'd_right': 12.0},
    # Zollstr_WB heading west, turns right into Mattengasse (going NB).
    # → approach samples up to western yield line
    {'approach_seg': 'Zollstr_WB', 'departure_seg': 'Matteng_NB',
     'approach_s_change_key':  's_zollstr_east_stop',
     'departure_s_change_key': 's_change',
     'd_left': 12.0, 'd_right': 12.0},
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
        {'geom_key': 'Roentgenstr', 's_change_key': 's_change',
         'pos_seg_key': 'Roentgenstr_WB', 'opp_seg_key': 'Roentgenstr_EB',
         'approach_seg_key': 'Roentgenstr_EB'},
        {'geom_key': 'LangstrN',    's_change_key': 's_change',
         'pos_seg_key': 'LangstrN_NB',    'opp_seg_key': 'LangstrN_SB',
         'approach_seg_key': 'LangstrN_SB'},
        {'geom_key': 'Zollstr',     's_change_key': 's_change',
         'pos_seg_key': 'Zollstr_EB',     'opp_seg_key': 'Zollstr_WB',
         'approach_seg_key': 'Zollstr_WB'},
        {'geom_key': 'LangstrS',    's_change_key': 's_change',
         'pos_seg_key': 'LangstrS_NB',    'opp_seg_key': 'LangstrS_SB',
         'approach_seg_key': 'LangstrS_NB'},
    ],
    geometry_store   = geometry_store,
    segment_registry = segment_registry,
)
geometry_store['intersection_type_MainInt'] = 'standard'
 
geometry_store['intersection_area_MattInt'] = build_intersection_polygon(
    arm_defs = [
        # Zollstr at the Mattengasse T-junction uses the individual stop lines
        {'geom_key': 'Zollstr',  's_change_key': 's_zollstr_east_stop',
         'pos_seg_key': 'Zollstr_EB',  'opp_seg_key': 'Zollstr_WB',
         'approach_seg_key': 'Zollstr_WB'},
        {'geom_key': 'Zollstr',  's_change_key': 's_zollstr_west_yield',
         'pos_seg_key': 'Zollstr_EB',  'opp_seg_key': 'Zollstr_WB',
         'approach_seg_key': 'Zollstr_EB'},
        {'geom_key': 'Matteng',  's_change_key': 's_change',
         'pos_seg_key': 'Matteng_SB',  'opp_seg_key': 'Matteng_NB',
         'approach_seg_key': 'Matteng_SB'},
    ],
    geometry_store   = geometry_store,
    segment_registry = segment_registry,
)
geometry_store['intersection_type_MattInt'] = 'T-junction'

from tools_plot_registry import plot_geometry_store, plot_segment_registry

plot_geometry_store(geometry_store, gdf_swisstopo, offset_m=3.0)
plot_segment_registry(geometry_store, segment_registry, gdf_swisstopo)

# =============================================================================
# PHASE 4: build_movement_registry

print("--- Phase 4: build movement registry ---")
 
MOVEMENT_DEFS = [
    # ── MainInt: from Röntgenstr EB ──────────────────────────────────────────
    {'key': 'Roentgenstr_EB_2_LangstrN_NB',
     'sequence': [('Roentgenstr_EB',                    'approach'),
                  ('turn_Roentgenstr_EB_2_LangstrN_NB', 'turn'),
                  ('LangstrN_NB',                       'departure')]},
    {'key': 'Roentgenstr_EB_2_Zollstr_EB',
     'sequence': [('Roentgenstr_EB',                    'approach'),
                  ('turn_Roentgenstr_EB_2_Zollstr_EB',  'turn'),
                  ('Zollstr_EB',                        'departure')]},
    {'key': 'Roentgenstr_EB_2_LangstrS_SB',
     'sequence': [('Roentgenstr_EB',                    'approach'),
                  ('turn_Roentgenstr_EB_2_LangstrS_SB', 'turn'),
                  ('LangstrS_SB',                       'departure')]},
 
    # ── MainInt: from Zollstr WB ─────────────────────────────────────────────
    {'key': 'Zollstr_WB_2_LangstrN_NB',
     'sequence': [('Zollstr_WB',                        'approach'),
                  ('turn_Zollstr_WB_2_LangstrN_NB',     'turn'),
                  ('LangstrN_NB',                       'departure')]},
    {'key': 'Zollstr_WB_2_Roentgenstr_WB',
     'sequence': [('Zollstr_WB',                        'approach'),
                  ('turn_Zollstr_WB_2_Roentgenstr_WB',  'turn'),
                  ('Roentgenstr_WB',                    'departure')]},
    {'key': 'Zollstr_WB_2_LangstrS_SB',
     'sequence': [('Zollstr_WB',                        'approach'),
                  ('turn_Zollstr_WB_2_LangstrS_SB',     'turn'),
                  ('LangstrS_SB',                       'departure')]},
 
    # ── MainInt: from LangstrN SB ────────────────────────────────────────────
    {'key': 'LangstrN_SB_2_Roentgenstr_WB',
     'sequence': [('LangstrN_SB',                       'approach'),
                  ('turn_LangstrN_SB_2_Roentgenstr_WB', 'turn'),
                  ('Roentgenstr_WB',                    'departure')]},
    {'key': 'LangstrN_SB_2_Zollstr_EB',
     'sequence': [('LangstrN_SB',                       'approach'),
                  ('turn_LangstrN_SB_2_Zollstr_EB',     'turn'),
                  ('Zollstr_EB',                        'departure')]},
    {'key': 'LangstrN_SB_2_LangstrS_SB',
     'sequence': [('LangstrN_SB',                       'approach'),
                  ('turn_LangstrN_SB_2_LangstrS_SB',    'turn'),
                  ('LangstrS_SB',                       'departure')]},
 
    # ── MainInt: from LangstrS NB ────────────────────────────────────────────
    {'key': 'LangstrS_NB_2_Roentgenstr_WB',
     'sequence': [('LangstrS_NB',                       'approach'),
                  ('turn_LangstrS_NB_2_Roentgenstr_WB', 'turn'),
                  ('Roentgenstr_WB',                    'departure')]},
    {'key': 'LangstrS_NB_2_Zollstr_EB',
     'sequence': [('LangstrS_NB',                       'approach'),
                  ('turn_LangstrS_NB_2_Zollstr_EB',     'turn'),
                  ('Zollstr_EB',                        'departure')]},
    {'key': 'LangstrS_NB_2_LangstrN_NB',
     'sequence': [('LangstrS_NB',                       'approach'),
                  ('turn_LangstrS_NB_2_LangstrN_NB',    'turn'),
                  ('LangstrN_NB',                       'departure')]},
 
    # ── MattInt: Matteng_SB → Zollstr ────────────────────────────────────────
    {'key': 'Matteng_SB_2_Zollstr_EB',
     'sequence': [('Matteng_SB',                        'approach'),
                  ('turn_Matteng_SB_2_Zollstr_EB',      'turn'),
                  ('Zollstr_EB',                        'departure')]},
    {'key': 'Matteng_SB_2_Zollstr_WB',
     'sequence': [('Matteng_SB',                        'approach'),
                  ('turn_Matteng_SB_2_Zollstr_WB',      'turn'),
                  ('Zollstr_WB',                        'departure')]},
 
    # ── MattInt: Zollstr → Matteng_NB ────────────────────────────────────────
    {'key': 'Zollstr_EB_2_Matteng_NB',
     'sequence': [('Zollstr_EB',                        'approach'),
                  ('turn_Zollstr_EB_2_Matteng_NB',      'turn'),
                  ('Matteng_NB',                        'departure')]},
    {'key': 'Zollstr_WB_2_Matteng_NB',
     'sequence': [('Zollstr_WB',                        'approach'),
                  ('turn_Zollstr_WB_2_Matteng_NB',      'turn'),
                  ('Matteng_NB',                        'departure')]},
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

dest_path = '../data/registry_2025-06-17_D1_A.pkl'
shutil.copy(save_path, dest_path)
dest_path = '../data/registry_2025-06-16_D1_A.pkl'
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
