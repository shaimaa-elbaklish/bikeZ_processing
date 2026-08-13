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
Site definition — Birmensdorferstrasse / Schweighofstrasse / Schaufelbergerstrasse
Zürich, Switzerland — September 2025 campaign (D1, G location)
 
Two intersections:
  MainInt  — 4-way: Birmensdorferstrasse × Schweighofstrasse × Schaufelbergerstrasse
 
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

intersection, code = BikeZ_Config.avail_intersections[date][0]
timeslot = BikeZ_Config.avail_timeslots[date][(intersection, code)][0] # 'AM1'

XY_2056_Bounds = BikeZ_Config.XY_2056_Bounds[date][(intersection, code)]
X_2056_offset = XY_2056_Bounds[0][0]
Y_2056_offset = XY_2056_Bounds[1][0]

# Site constants
kml_path       = '../maps/from_swisstopo/September_D1G.kml'
kml_path_lanes = '../maps/from_swisstopo/September_D1G_CarLanes.kml'
kml_path_gis   = '../maps/from_swisstopo/September_D1G_GIS.kml'
save_path      = f'../data/registry_{date}_{intersection}_{code}.pkl'
max_chain_len  = 3


# Share Link: https://s.geo.admin.ch/1ssj1wf1tkvy
# Edit Link: https://s.geo.admin.ch/zf0fmp83szbr

# GIS-ZH 
# Share Link: https://geo.zh.ch/s/7f08db85-491d-4a44-b606-7ce0b074c20b

# Car Lanes:
# Share Link: https://s.geo.admin.ch/uviulhctrb2r
# Edit Link: https://s.geo.admin.ch/9ng0dlvnr01n

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
        ['Birmensdorferstrasse', 'Schweighofstrasse', 'Schaufelbergerstrasse']
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
    '#ff0000': 'Birmensdorferstr_CL',                         # red
    '#0000ff': 'Schweighofstr_Schaufelbergerstr_CutLine',     # blue
}
gdf_gis = read_gis_kml(kml_path_gis, color_to_name)
# merge them
gdf_swisstopo = merge_replace(gdf_swisstopo, gdf_gis, key='Description')


# STEP 1: fit splines  (geometry sourcing, customise per road as needed)
print("\nFitting splines...")

# Schweighofstr: OSMnx, shared centerline
line_SchwH = merge_osmnx_edges(gdf, 'Schweighofstrasse')
line_SchwH = densify_linestring(line=line_SchwH, num_segments=10)
tck_SchwH, unew_SchwH, cum_SchwH, len_SchwH = fit_spline_from_shapely(
    line_SchwH, x_offset=X_2056_offset, y_offset=Y_2056_offset,
)

# Schaufelbergerstr: OSMnx, shared centerline
line_SchaufB = merge_osmnx_edges(gdf, 'Schaufelbergerstrasse')
line_SchaufB = densify_linestring(line=line_SchaufB, num_segments=10)
tck_SchaufB, unew_SchaufB, cum_SchaufB, len_SchaufB = fit_spline_from_shapely(
    line_SchaufB, x_offset=X_2056_offset, y_offset=Y_2056_offset,
)

# Birmensdorferstr: Split into west and east sides
birm_line = gdf_swisstopo[gdf_swisstopo['Description'] == 'Birmensdorferstr_CL'].geometry.item()
cut_line = gdf_swisstopo[gdf_swisstopo['Description'] == 'Schweighofstr_Schaufelbergerstr_CutLine'].geometry.item()
birm_west_line = cut_line_at_stop(birm_line, cut_line, choose='first', plotting=False)
tck_BW, unew_BW, cum_BW, len_BW = fit_spline_from_shapely(
    birm_west_line, x_offset=X_2056_offset, y_offset=Y_2056_offset,
)

birm_east_line = cut_line_at_stop(birm_line, cut_line, choose='last', plotting=False)
tck_BE, unew_BE, cum_BE, len_BE = fit_spline_from_shapely(
    birm_east_line, x_offset=X_2056_offset, y_offset=Y_2056_offset,
)


print(f"  Schweighofstr     : {len_SchwH:.1f} m")
print(f"  Schaufelbergerstr : {len_SchaufB:.1f} m")
print(f"  BirmensdorferstrE : {len_BE:.1f} m")
print(f"  BirmensdorferstrW : {len_BW:.1f} m")


fig, ax = plt.subplots(1, 1)
# gdf.plot(ax=ax, column='name', legend=True)
plot_line(line_SchwH, ax=ax, add_points=False, color='tab:blue', label='Schweighofstr')
plot_line(line_SchaufB, ax=ax, add_points=False, color='tab:orange', label='Schaufelbergerstr')
plot_line(birm_east_line, ax=ax, add_points=False, color='tab:green', label='BirmensdorferstrE')
plot_line(birm_west_line, ax=ax, add_points=False, color='tab:purple', label='BirmensdorferstrW')
plot_points(Point(line_SchwH.coords[0]), color='red', marker='o',)
plot_points(Point(line_SchwH.coords[-1]), color='black', marker='x',)
plot_points(Point(line_SchaufB.coords[0]), color='red', marker='o',)
plot_points(Point(line_SchaufB.coords[-1]), color='black', marker='x',)
plot_points(Point(birm_east_line.coords[1]), color='red', marker='o',)
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
        'name':          'Schweighofstr',
        'positive_dir':  'NB',
        'spline':        (tck_SchwH, unew_SchwH, cum_SchwH),
        'total_length':  len_SchwH,
        'line_wgs84':    line_SchwH,
        'stop_line_id':  'Schweighofstr_Stop',
        'yield_line_id': 'Schweighofstr_Yield',
    },
    {
        'name':          'Schaufelbergerstr',
        'positive_dir':  'NB',
        'spline':        (tck_SchaufB, unew_SchaufB, cum_SchaufB),
        'total_length':  len_SchaufB,
        'line_wgs84':    line_SchaufB,
        'stop_line_id':  'Schaufelbergerstr_Stop',
        'yield_line_id': 'Schaufelbergerstr_Yield',
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
    # ── Schweighofstrasse ────────────────────────────────────────────────────
    {'seg_key': 'Schweighofstr_NB', 'geometry_key': 'Schweighofstr',
     'direction': 'NB', 'mode': 'shared', 'bike_lane': {'w_bike': 1.5},
     'd_left': 1.0, 'd_right': 13.0,},
    {'seg_key': 'Schweighofstr_SB', 'geometry_key': 'Schweighofstr',
     'direction': 'SB', 'mode': 'shared', 'bike_lane': None,
     'd_left': 1.0, 'd_right': 13.0,},

    # ── Schaufelbergerstrasse ────────────────────────────────────────────────
    {'seg_key': 'Schaufelbergerstr_NB', 'geometry_key': 'Schaufelbergerstr',
     'direction': 'NB', 'mode': 'shared', 'bike_lane': None,
     'd_left': 1.5, 'd_right': 12.0,},
    {'seg_key': 'Schaufelbergerstr_SB', 'geometry_key': 'Schaufelbergerstr',
     'direction': 'SB', 'mode': 'shared', 'bike_lane': None,
     'd_left': 1.5, 'd_right': 12.0,},

    # ── Birmensdorferstrasse East  ───────────────────────────────────────────
    {'seg_key': 'BirmensdorferstrE_WB', 'geometry_key': 'BirmensdorferstrE',
     'direction': 'WB', 'mode': 'shared', 'bike_lane': {'w_bike': 1.5},
     'd_left': 2.0, 'd_right': 20.0,},
    {'seg_key': 'BirmensdorferstrE_EB', 'geometry_key': 'BirmensdorferstrE',
     'direction': 'EB', 'mode': 'shared', 'bike_lane': {'w_bike': 1.5},
     'd_left': 2.0, 'd_right': 18.0,},

    # ── Birmensdorferstrasse West  ───────────────────────────────────────────
    {'seg_key': 'BirmensdorferstrW_WB', 'geometry_key': 'BirmensdorferstrW',
     'direction': 'WB', 'mode': 'shared', 'bike_lane': {'w_bike': 1.5},
     'd_left': 2.0, 'd_right': 20.0,},
    {'seg_key': 'BirmensdorferstrW_EB', 'geometry_key': 'BirmensdorferstrW',
     'direction': 'EB', 'mode': 'shared', 'bike_lane': {'w_bike': 1.5},
     'd_left': 2.0, 'd_right': 20.0,},
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
    {'approach_seg': 'BirmensdorferstrE_WB', 'departure_seg': 'Schweighofstr_SB',
     'd_left': 15.0, 'd_right': 15.0},
    {'approach_seg': 'BirmensdorferstrE_WB', 'departure_seg': 'Schaufelbergerstr_NB',
     'd_left': 15.0, 'd_right': 15.0},

    # ── from BirmensdorferstrW_EB (entering from west, heading east) ────────
    {'approach_seg': 'BirmensdorferstrW_EB', 'departure_seg': 'BirmensdorferstrE_EB',
     'd_left': 17.0, 'd_right': 17.0},   # through
    {'approach_seg': 'BirmensdorferstrW_EB', 'departure_seg': 'Schweighofstr_SB',
     'd_left': 15.0, 'd_right': 15.0},
    {'approach_seg': 'BirmensdorferstrW_EB', 'departure_seg': 'Schaufelbergerstr_NB',
     'd_left': 15.0, 'd_right': 15.0},

    # ── from Schweighofstr_NB (entering from south, heading north) ──────────
    {'approach_seg': 'Schweighofstr_NB', 'departure_seg': 'Schaufelbergerstr_NB',
     'd_left': 17.0, 'd_right': 17.0},   # through
    {'approach_seg': 'Schweighofstr_NB', 'departure_seg': 'BirmensdorferstrW_WB',
     'd_left': 15.0, 'd_right': 15.0},
    {'approach_seg': 'Schweighofstr_NB', 'departure_seg': 'BirmensdorferstrE_EB',
     'd_left': 15.0, 'd_right': 15.0},

    # ── from Schaufelbergerstr_SB (entering from north, heading south) ──────
    {'approach_seg': 'Schaufelbergerstr_SB', 'departure_seg': 'Schweighofstr_SB',
     'd_left': 17.0, 'd_right': 17.0},   # through
    {'approach_seg': 'Schaufelbergerstr_SB', 'departure_seg': 'BirmensdorferstrW_WB',
     'd_left': 15.0, 'd_right': 15.0},
    {'approach_seg': 'Schaufelbergerstr_SB', 'departure_seg': 'BirmensdorferstrE_EB',
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
        {'geom_key': 'Schweighofstr', 's_change_key': 's_change',
         'pos_seg_key': 'Schweighofstr_NB', 'opp_seg_key': 'Schweighofstr_SB',
         'approach_seg_key': 'Schweighofstr_NB'},
        {'geom_key': 'Schaufelbergerstr',    's_change_key': 's_change',
         'pos_seg_key': 'Schaufelbergerstr_NB',    'opp_seg_key': 'Schaufelbergerstr_SB',
         'approach_seg_key': 'Schaufelbergerstr_SB'},
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
    # ── BirmInt: from BirmensdorferstrE_WB ───────────────────────────────────
    {'key': 'BirmensdorferstrE_WB_2_BirmensdorferstrW_WB',
     'sequence': [('BirmensdorferstrE_WB',                                  'approach'),
                  ('turn_BirmensdorferstrE_WB_2_BirmensdorferstrW_WB',      'turn'),
                  ('BirmensdorferstrW_WB',                                  'departure')]},
    {'key': 'BirmensdorferstrE_WB_2_Schweighofstr_SB',
     'sequence': [('BirmensdorferstrE_WB',                                  'approach'),
                  ('turn_BirmensdorferstrE_WB_2_Schweighofstr_SB',          'turn'),
                  ('Schweighofstr_SB',                                      'departure')]},
    {'key': 'BirmensdorferstrE_WB_2_Schaufelbergerstr_NB',
     'sequence': [('BirmensdorferstrE_WB',                                  'approach'),
                  ('turn_BirmensdorferstrE_WB_2_Schaufelbergerstr_NB',      'turn'),
                  ('Schaufelbergerstr_NB',                                  'departure')]},

    # ── BirmInt: from BirmensdorferstrW_EB ───────────────────────────────────
    {'key': 'BirmensdorferstrW_EB_2_BirmensdorferstrE_EB',
     'sequence': [('BirmensdorferstrW_EB',                                  'approach'),
                  ('turn_BirmensdorferstrW_EB_2_BirmensdorferstrE_EB',      'turn'),
                  ('BirmensdorferstrE_EB',                                  'departure')]},
    {'key': 'BirmensdorferstrW_EB_2_Schweighofstr_SB',
     'sequence': [('BirmensdorferstrW_EB',                                  'approach'),
                  ('turn_BirmensdorferstrW_EB_2_Schweighofstr_SB',          'turn'),
                  ('Schweighofstr_SB',                                      'departure')]},
    {'key': 'BirmensdorferstrW_EB_2_Schaufelbergerstr_NB',
     'sequence': [('BirmensdorferstrW_EB',                                  'approach'),
                  ('turn_BirmensdorferstrW_EB_2_Schaufelbergerstr_NB',      'turn'),
                  ('Schaufelbergerstr_NB',                                  'departure')]},

    # ── BirmInt: from Schweighofstr_NB ───────────────────────────────────────
    {'key': 'Schweighofstr_NB_2_Schaufelbergerstr_NB',
     'sequence': [('Schweighofstr_NB',                                     'approach'),
                  ('turn_Schweighofstr_NB_2_Schaufelbergerstr_NB',         'turn'),
                  ('Schaufelbergerstr_NB',                                 'departure')]},
    {'key': 'Schweighofstr_NB_2_BirmensdorferstrW_WB',
     'sequence': [('Schweighofstr_NB',                                     'approach'),
                  ('turn_Schweighofstr_NB_2_BirmensdorferstrW_WB',         'turn'),
                  ('BirmensdorferstrW_WB',                                 'departure')]},
    {'key': 'Schweighofstr_NB_2_BirmensdorferstrE_EB',
     'sequence': [('Schweighofstr_NB',                                     'approach'),
                  ('turn_Schweighofstr_NB_2_BirmensdorferstrE_EB',         'turn'),
                  ('BirmensdorferstrE_EB',                                 'departure')]},

    # ── BirmInt: from Schaufelbergerstr_SB ───────────────────────────────────
    {'key': 'Schaufelbergerstr_SB_2_Schweighofstr_SB',
     'sequence': [('Schaufelbergerstr_SB',                                 'approach'),
                  ('turn_Schaufelbergerstr_SB_2_Schweighofstr_SB',         'turn'),
                  ('Schweighofstr_SB',                                     'departure')]},
    {'key': 'Schaufelbergerstr_SB_2_BirmensdorferstrW_WB',
     'sequence': [('Schaufelbergerstr_SB',                                 'approach'),
                  ('turn_Schaufelbergerstr_SB_2_BirmensdorferstrW_WB',     'turn'),
                  ('BirmensdorferstrW_WB',                                 'departure')]},
    {'key': 'Schaufelbergerstr_SB_2_BirmensdorferstrE_EB',
     'sequence': [('Schaufelbergerstr_SB',                                 'approach'),
                  ('turn_Schaufelbergerstr_SB_2_BirmensdorferstrE_EB',     'turn'),
                  ('BirmensdorferstrE_EB',                                 'departure')]},
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