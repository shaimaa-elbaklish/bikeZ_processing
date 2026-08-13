"""
TITLE OF PAPAER
-------------------------------------------
Authors:        Shaimaa El-Baklish
Organization:   ETH Zürich, Switzerland, IVT - Institute for Transportation Planning and Systems
Development:    2025
Submitted to:   JOURNAL
-------------------------------------------

maps_sep_D1C.py
-------------------------------------
Site definition — Birmensdorferstrasse / Gutstrasse / Talwiesenstrasse
Zürich, Switzerland — September 2025 campaign (D2, F location)
 
Two intersections:
  MainInt  — 4-way: Birmensdorferstrasse × Gutstrasse × Talwiesenstrasse
 
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
date = BikeZ_Config.avail_dates[-1]
campaign = f"Zurich_2025{date[5:7]}" # June or September
mode = BikeZ_Config.avail_modes[0] # Bike
data_root = BikeZ_Config.data_root[campaign][mode]

intersection, code = BikeZ_Config.avail_intersections[date][-2]
timeslot = BikeZ_Config.avail_timeslots[date][(intersection, code)][0] # 'AM1'

XY_2056_Bounds = BikeZ_Config.XY_2056_Bounds[date][(intersection, code)]
X_2056_offset = XY_2056_Bounds[0][0]
Y_2056_offset = XY_2056_Bounds[1][0]

# Site constants
kml_path       = '../maps/from_swisstopo/September_D2F.kml'
kml_path_gis   = '../maps/from_swisstopo/September_D2F_GIS.kml'
kml_path_lanes = '../maps/from_swisstopo/September_D2F_CarLanes.kml'
save_path      = f'../data/registry_{date}_{intersection}_{code}.pkl'
max_chain_len  = 3


# Share Link: https://s.geo.admin.ch/h6qpz0aodn2c
# Edit Link: https://s.geo.admin.ch/xm9ci79rgw5j

# GIS
# Share link: https://geo.zh.ch/s/8e72863b-b537-4b82-b440-f21982785336
# NEW Share Link: https://geo.zh.ch/s/31044a9d-a9f5-4790-9edc-f22cfe201d1d

# Car Lanes:
# Share Link: https://s.geo.admin.ch/skmqc35shdkr
# Edit Link: https://s.geo.admin.ch/1z38mo7bk7k8

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
transformer = Transformer.from_crs('EPSG:2056', 'EPSG:4326', always_xy=True)
lonlat      = transformer.transform(
    np.asarray(XY_2056_Bounds[0]) + np.asarray([-15, 15]),
    np.asarray(XY_2056_Bounds[1]) + np.asarray([-15, 15]),
)
bbox_geom = box(lonlat[0][0], lonlat[1][0], lonlat[0][1], lonlat[1][1])
 
gdf_main   = ox.features.features_from_place('Zürich, Switzerland',
                                              tags={'highway': True})
road_types = ['primary', 'secondary', 'tertiary',
              'residential', 'unclassified', 'cycleway']
gdf = gdf_main[
    gdf_main['name'].isin(
        ['Birmensdorferstrasse', 'Gutstrasse', 'Talwiesenstrasse']
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
    '#ff0000': 'Birmensdorferstr_CL',      # red
    '#00ff00': 'Gutstr_CL',                # green
    '#0000ff': 'Talwiesenstr_CL',          # blue
    '#ffff00': 'BirmensdorferstrE_WB',     # yellow
    '#ff00ff': 'BirmensdorferstrE_EB',     # pink
    '#000000': 'BirmensdorferstrW_EB',     # black
}
gdf_gis = read_gis_kml(kml_path_gis, color_to_name)
# merge them
gdf_swisstopo = merge_replace(gdf_swisstopo, gdf_gis, key='Description')

# STEP 1: fit splines  (geometry sourcing, customise per road as needed)
print("\nFitting splines...")

# Talwiesenstr: OSMnx, shared centerline
line_T = merge_osmnx_edges(gdf, 'Talwiesenstrasse')
line_T = densify_linestring(line=line_T, num_segments=10)
tck_T, unew_T, cum_T, len_T = fit_spline_from_shapely(
    line_T, x_offset=X_2056_offset, y_offset=Y_2056_offset,
)

# Birmensdorferstr: Split into west and east sides
birm_line = gdf_swisstopo[gdf_swisstopo['Description'] == 'Birmensdorferstr_CL'].geometry.item()
birm_west_line = cut_line_at_stop(birm_line, line_T, choose='first', plotting=False)
tck_BW, unew_BW, cum_BW, len_BW = fit_spline_from_shapely(
    birm_west_line, x_offset=X_2056_offset, y_offset=Y_2056_offset,
)

birm_east_line = cut_line_at_stop(birm_line, line_T, choose='last', plotting=False)
tck_BE, unew_BE, cum_BE, len_BE = fit_spline_from_shapely(
    birm_east_line, x_offset=X_2056_offset, y_offset=Y_2056_offset,
)

# Gutstr: from swisstopo/GIS-ZH, shared centerline
line_G = gdf_swisstopo[gdf_swisstopo['Description'] == 'Gutstr_CL'].geometry.item()
line_G = cut_line_at_stop(line_G, birm_line, choose='first', plotting=False)
tck_G, unew_G, cum_G, len_G = fit_spline_from_shapely(
    line_G, x_offset=X_2056_offset, y_offset=Y_2056_offset,
)

print(f"  Talwiesenstr      : {len_T:.1f} m")
print(f"  Gutstr            : {len_G:.1f} m")
print(f"  BirmensdorferstrE : {len_BE:.1f} m")
print(f"  BirmensdorferstrW : {len_BW:.1f} m")


fig, ax = plt.subplots(1, 1)
# gdf.plot(ax=ax, column='name', legend=True)
plot_line(line_T, ax=ax, add_points=False, color='tab:blue', label='Talwiesenstr')
plot_line(line_G, ax=ax, add_points=False, color='tab:orange', label='Gutstr')
plot_line(birm_east_line, ax=ax, add_points=False, color='tab:green', label='BirmensdorferstrE')
plot_line(birm_west_line, ax=ax, add_points=False, color='tab:purple', label='BirmensdorferstrW')
plot_points(Point(line_T.coords[1]), color='red', marker='o',)
plot_points(Point(line_T.coords[-1]), color='black', marker='x',)
plot_points(Point(line_G.coords[0]), color='red', marker='o',)
plot_points(Point(line_G.coords[-2]), color='black', marker='x',)
plot_points(Point(birm_east_line.coords[2]), color='red', marker='o',)
plot_points(Point(birm_east_line.coords[-1]), color='black', marker='x',)
plot_points(Point(birm_west_line.coords[0]), color='red', marker='o',)
plot_points(Point(birm_west_line.coords[-2]), color='black', marker='x',)
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncols=3)
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
        'name':          'Gutstr',
        'positive_dir':  'SB',
        'spline':        (tck_G, unew_G, cum_G),
        'total_length':  len_G,
        'line_wgs84':    line_G,
        'stop_line_id':  'Gutstr_Stop',
        'yield_line_id': 'Gutstr_Yield',
    },
    {
        'name':          'Talwiesenstr',
        'positive_dir':  'SB',
        'spline':        (tck_T, unew_T, cum_T),
        'total_length':  len_T,
        'line_wgs84':    line_T,
        'stop_line_id':  'Talwiesenstr_Stop',
        'yield_line_id': 'Talwiesenstr_Yield',
    },
    {
        'name':          'BirmensdorferstrE',
        'positive_dir':  'EB',
        'spline':        (tck_BE, unew_BE, cum_BE),
        'total_length':  len_BE,
        'line_wgs84':    birm_east_line,
        'stop_line_id':  'BirmensdorferstrE_Stop',   
        'yield_line_id': 'BirmensdorferstrE_Yield',
    },
    {
        'name':          'BirmensdorferstrW',
        'positive_dir':  'EB',
        'spline':        (tck_BW, unew_BW, cum_BW),
        'total_length':  len_BW,
        'line_wgs84':    birm_west_line,
        'stop_line_id':  'BirmensdorferstrW_Stop',
        'yield_line_id': 'BirmensdorferstrW_Yield',
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
    # ── Gutstrasse ────────────────────────────────────────────────────────
    {'seg_key': 'Gutstr_NB', 'geometry_key': 'Gutstr',
     'direction': 'NB', 'mode': 'shared', 'bike_lane': {'w_bike': 1.5},
     'd_left': -2.5, 'd_right': 15.0,},
    {'seg_key': 'Gutstr_SB', 'geometry_key': 'Gutstr',
     'direction': 'SB', 'mode': 'shared', 'bike_lane': None,
     'd_left': 4.0, 'd_right': 20.0,},

    # ── Talwiesenstrasse ─────────────────────────────────────────────────
    {'seg_key': 'Talwiesenstr_NB', 'geometry_key': 'Talwiesenstr',
     'direction': 'NB', 'mode': 'bike', 'bike_lane': {'w_bike': 1.5},
     'd_left': 0.5, 'd_right': 12.0,},
    {'seg_key': 'Talwiesenstr_SB', 'geometry_key': 'Talwiesenstr',
     'direction': 'SB', 'mode': 'shared', 'bike_lane': None,
     'd_left': 4.0, 'd_right': 14.0,},

    # ── Birmensdorferstrasse East ────────────────────────────────────────
    {'seg_key': 'BirmensdorferstrE_WB', 'geometry_key': 'BirmensdorferstrE',
     'direction': 'WB', 'mode': 'shared', 'bike_lane': {'w_bike': 1.5},
     'd_left': 1.5, 'd_right': 20.0,},
    {'seg_key': 'BirmensdorferstrE_EB', 'geometry_key': 'BirmensdorferstrE',
     'direction': 'EB', 'mode': 'shared', 'bike_lane': {'w_bike': 1.5},
     'd_left': 1.5, 'd_right': 14.0,},

    # ── Birmensdorferstrasse West ────────────────────────────────────────
    {'seg_key': 'BirmensdorferstrW_WB', 'geometry_key': 'BirmensdorferstrW',
     'direction': 'WB', 'mode': 'shared', 'bike_lane': {'w_bike': 1.5},
     'd_left': 1.5, 'd_right': 20.0,},
    {'seg_key': 'BirmensdorferstrW_EB', 'geometry_key': 'BirmensdorferstrW',
     'direction': 'EB', 'mode': 'shared', 'bike_lane': {'w_bike': 1.5},
     'd_left': 1.5, 'd_right': 16.0,},
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
    # ── from BirmensdorferstrE_WB (entering from east, heading west) ────────
    {'approach_seg': 'BirmensdorferstrE_WB', 'departure_seg': 'BirmensdorferstrW_WB',
     'd_left': 17.0, 'd_right': 17.0},   # through
    {'approach_seg': 'BirmensdorferstrE_WB', 'departure_seg': 'Talwiesenstr_SB',
     'd_left': 15.0, 'd_right': 15.0},
    {'approach_seg': 'BirmensdorferstrE_WB', 'departure_seg': 'Gutstr_NB',
     'd_left': 15.0, 'd_right': 15.0},

    # ── from BirmensdorferstrW_EB (entering from west, heading east) ────────
    {'approach_seg': 'BirmensdorferstrW_EB', 'departure_seg': 'BirmensdorferstrE_EB',
     'd_left': 17.0, 'd_right': 17.0},   # through
    {'approach_seg': 'BirmensdorferstrW_EB', 'departure_seg': 'Talwiesenstr_SB',
     'd_left': 15.0, 'd_right': 15.0},
    {'approach_seg': 'BirmensdorferstrW_EB', 'departure_seg': 'Gutstr_NB',
     'd_left': 15.0, 'd_right': 15.0},

    # ── from Gutstr_SB (entering from north, heading south) ─────────────────
    {'approach_seg': 'Gutstr_SB', 'departure_seg': 'Talwiesenstr_SB',
     'd_left': 17.0, 'd_right': 17.0},   # through
    {'approach_seg': 'Gutstr_SB', 'departure_seg': 'BirmensdorferstrW_WB',
     'd_left': 15.0, 'd_right': 15.0},
    {'approach_seg': 'Gutstr_SB', 'departure_seg': 'BirmensdorferstrE_EB',
     'd_left': 15.0, 'd_right': 15.0},

    # ── from Talwiesenstr_NB (entering from south, heading north) ───────────
    {'approach_seg': 'Talwiesenstr_NB', 'departure_seg': 'Gutstr_NB',
     'd_left': 17.0, 'd_right': 17.0},   # through
    {'approach_seg': 'Talwiesenstr_NB', 'departure_seg': 'BirmensdorferstrW_WB',
     'd_left': 15.0, 'd_right': 15.0},
    {'approach_seg': 'Talwiesenstr_NB', 'departure_seg': 'BirmensdorferstrE_EB',
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
        {'geom_key': 'Gutstr', 's_change_key': 's_change',
         'pos_seg_key': 'Gutstr_SB', 'opp_seg_key': 'Gutstr_NB',
         'approach_seg_key': 'Gutstr_SB'},
        {'geom_key': 'Talwiesenstr',    's_change_key': 's_change',
         'pos_seg_key': 'Talwiesenstr_SB',    'opp_seg_key': 'Talwiesenstr_NB',
         'approach_seg_key': 'Talwiesenstr_NB'},
        {'geom_key': 'BirmensdorferstrW',     's_change_key': 's_change',
         'pos_seg_key': 'BirmensdorferstrW_EB',     'opp_seg_key': 'BirmensdorferstrW_WB',
         'approach_seg_key': 'BirmensdorferstrW_EB'},
        {'geom_key': 'BirmensdorferstrE',     's_change_key': 's_change',
         'pos_seg_key': 'BirmensdorferstrE_EB',     'opp_seg_key': 'BirmensdorferstrE_WB',
         'approach_seg_key': 'BirmensdorferstrE_WB'},
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
    # ── MainInt: from BirmensdorferstrE_WB ───────────────────────────────────
    {'key': 'BirmensdorferstrE_WB_2_BirmensdorferstrW_WB',
     'sequence': [('BirmensdorferstrE_WB',                                  'approach'),
                  ('turn_BirmensdorferstrE_WB_2_BirmensdorferstrW_WB',      'turn'),
                  ('BirmensdorferstrW_WB',                                  'departure')]},
    {'key': 'BirmensdorferstrE_WB_2_Talwiesenstr_SB',
     'sequence': [('BirmensdorferstrE_WB',                                  'approach'),
                  ('turn_BirmensdorferstrE_WB_2_Talwiesenstr_SB',           'turn'),
                  ('Talwiesenstr_SB',                                       'departure')]},
    {'key': 'BirmensdorferstrE_WB_2_Gutstr_NB',
     'sequence': [('BirmensdorferstrE_WB',                                  'approach'),
                  ('turn_BirmensdorferstrE_WB_2_Gutstr_NB',                 'turn'),
                  ('Gutstr_NB',                                             'departure')]},

    # ── MainInt: from BirmensdorferstrW_EB ───────────────────────────────────
    {'key': 'BirmensdorferstrW_EB_2_BirmensdorferstrE_EB',
     'sequence': [('BirmensdorferstrW_EB',                                  'approach'),
                  ('turn_BirmensdorferstrW_EB_2_BirmensdorferstrE_EB',      'turn'),
                  ('BirmensdorferstrE_EB',                                  'departure')]},
    {'key': 'BirmensdorferstrW_EB_2_Talwiesenstr_SB',
     'sequence': [('BirmensdorferstrW_EB',                                  'approach'),
                  ('turn_BirmensdorferstrW_EB_2_Talwiesenstr_SB',           'turn'),
                  ('Talwiesenstr_SB',                                       'departure')]},
    {'key': 'BirmensdorferstrW_EB_2_Gutstr_NB',
     'sequence': [('BirmensdorferstrW_EB',                                  'approach'),
                  ('turn_BirmensdorferstrW_EB_2_Gutstr_NB',                 'turn'),
                  ('Gutstr_NB',                                             'departure')]},

    # ── MainInt: from Gutstr_SB ───────────────────────────────────────────────
    {'key': 'Gutstr_SB_2_Talwiesenstr_SB',
     'sequence': [('Gutstr_SB',                                             'approach'),
                  ('turn_Gutstr_SB_2_Talwiesenstr_SB',                      'turn'),
                  ('Talwiesenstr_SB',                                       'departure')]},
    {'key': 'Gutstr_SB_2_BirmensdorferstrW_WB',
     'sequence': [('Gutstr_SB',                                             'approach'),
                  ('turn_Gutstr_SB_2_BirmensdorferstrW_WB',                 'turn'),
                  ('BirmensdorferstrW_WB',                                  'departure')]},
    {'key': 'Gutstr_SB_2_BirmensdorferstrE_EB',
     'sequence': [('Gutstr_SB',                                             'approach'),
                  ('turn_Gutstr_SB_2_BirmensdorferstrE_EB',                 'turn'),
                  ('BirmensdorferstrE_EB',                                  'departure')]},

    # ── MainInt: from Talwiesenstr_NB ─────────────────────────────────────────
    {'key': 'Talwiesenstr_NB_2_Gutstr_NB',
     'sequence': [('Talwiesenstr_NB',                                       'approach'),
                  ('turn_Talwiesenstr_NB_2_Gutstr_NB',                      'turn'),
                  ('Gutstr_NB',                                             'departure')]},
    {'key': 'Talwiesenstr_NB_2_BirmensdorferstrW_WB',
     'sequence': [('Talwiesenstr_NB',                                       'approach'),
                  ('turn_Talwiesenstr_NB_2_BirmensdorferstrW_WB',           'turn'),
                  ('BirmensdorferstrW_WB',                                  'departure')]},
    {'key': 'Talwiesenstr_NB_2_BirmensdorferstrE_EB',
     'sequence': [('Talwiesenstr_NB',                                       'approach'),
                  ('turn_Talwiesenstr_NB_2_BirmensdorferstrE_EB',           'turn'),
                  ('BirmensdorferstrE_EB',                                  'departure')]},
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

