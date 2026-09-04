"""
TITLE OF PAPAER
-------------------------------------------
Authors:        Shaimaa El-Baklish
Organization:   ETH Zürich, Switzerland, IVT - Institute for Transportation Planning and Systems
Development:    2025-2026
Submitted to:   JOURNAL
-------------------------------------------

maps_sep_D2B.py
-------------------------------------
Site definition — Bullingerplatz
Zürich, Switzerland — September 2025 campaign (D2,B location)
 
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
    add_bike_lane_width_profile,
    add_car_lane_boundaries,
    build_turns,
    build_intersection_polygon,
    build_movement_registry,
    serialize_registry,
    read_ring_kml,
    register_ring_geometry,
    build_ring_segment,
)

# #############################################################################
# CONSTANTS
# #############################################################################
# Configuration
BikeZ_Config = BikeZ_Config()

# Specify Trajectory File
date = BikeZ_Config.avail_dates[2]
campaign = f"Zurich_2025{date[5:7]}" # June or September
mode = BikeZ_Config.avail_modes[0] # Bike
data_root = BikeZ_Config.data_root[campaign][mode]

intersection, code = BikeZ_Config.avail_intersections[date][2]
timeslot = BikeZ_Config.avail_timeslots[date][(intersection, code)][0] # 'AM1'

XY_2056_Bounds = BikeZ_Config.XY_2056_Bounds[date][(intersection, code)]
X_2056_offset = XY_2056_Bounds[0][0]
Y_2056_offset = XY_2056_Bounds[1][0]

# Site constants
kml_path       = '../maps/from_swisstopo/September_D2B.kml'
kml_path_lanes = '../maps/from_swisstopo/September_D2B_CarLanes.kml'
save_path      = f'../data/registry_{date}_{intersection}_{code}.pkl'
max_chain_len  = 5

# Edit Link: https://s.geo.admin.ch/dkcg5dm3unwa
# Share Link: https://s.geo.admin.ch/g6raoefxogyx

# Car Lanes:
# Share Link: 
# Edit Link: 

# #############################################################################
# FUNTIONS
# #############################################################################
from shapely.geometry import LineString, MultiLineString
from shapely.ops import linemerge

def merge_edges_by_ids(gdf, osmids):
    """Merge specific OSMnx edges by osmid into a single ordered LineString."""
    rows = gdf[gdf.index.get_level_values('id').isin(osmids)]
    merged = linemerge(MultiLineString(list(rows.geometry)))
    if isinstance(merged, MultiLineString):
        raise ValueError(f"Edges did not merge into a single LineString — "
                         f"check connectivity of osmids: {osmids}")
    return merged


def concat_linestrings(line1, line2, gap_warn_m=5.0):
    """
    Join two LineStrings end-to-end into one, auto-orienting them so
    line1's nearest endpoint connects to line2's nearest endpoint.
    Falls back to a straight bridge across any gap between them.
    """
    c1, c2 = list(line1.coords), list(line2.coords)
    endpoints = {
        'ee': (Point(c1[-1]).distance(Point(c2[0])),  c1,               c2),
        'es': (Point(c1[-1]).distance(Point(c2[-1])), c1,               c2[::-1]),
        'se': (Point(c1[0]).distance(Point(c2[0])),   c1[::-1],         c2),
        'ss': (Point(c1[0]).distance(Point(c2[-1])),  c1[::-1],         c2[::-1]),
    }
    key = min(endpoints, key=lambda k: endpoints[k][0])
    gap, ordered1, ordered2 = endpoints[key]

    if gap > gap_warn_m:
        print(f"[concat_linestrings] warning: {gap:.1f} m gap between the two lines "
              f"— bridging with a straight segment. Check this is expected.")

    combined = ordered1 + ordered2  # drop nothing; if they truly touch, coords will just be near-duplicate
    return LineString(combined)

# #############################################################################
# MAIN
# #############################################################################

# =============================================================================
# STEP 0: load external data sources
print("Loading OSMnx features...")
lonlat = _PROJ_2056_TO_LONLAT.transform(
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
        ['Bullingerplatz', 'Bullingerstrasse', 'Zypressenstrasse', 
         'Sihlfeldstrasse', 'Stauffacherstrasse']
    )
]
gdf = gdf[
    (gdf.geometry.type == 'LineString') #& (gdf['highway'].isin(road_types))
]
gdf['geometry'] = gdf.geometry.intersection(bbox_geom)
gdf = gdf[~gdf.is_empty]

# import contextily as cx

# fig, ax = plt.subplots(1, 1)
# gdf_3857 = gdf.to_crs(epsg=3857)
# gdf_3857.plot(ax=ax, column="name", legend=True)
# cx.add_basemap(
#     ax,
#     source="https://wmts.geo.admin.ch/1.0.0/ch.swisstopo.swissimage/"
#            "default/current/3857/{z}/{x}/{y}.jpeg",
#     attribution="© swisstopo",
# )
# fig.tight_layout()


# # --- Plot GDF with OSM IDs ---
# lvl = 'osmid' if 'osmid' in (gdf.index.names or []) else 'id'
# osm_ids = gdf.index.get_level_values(lvl)

# fig, ax = plt.subplots(1, 1)
# gdf.plot(ax=ax, column='name', legend=True)

# def _start_point(geom):
#     """First coordinate of a LineString, or of the first part of a multipart."""
#     g = geom.geoms[0] if geom.geom_type.startswith('Multi') else geom
#     return g.coords[0]

# for oid, geom, gname in zip(osm_ids, gdf.geometry, gdf.name):
#     if geom.is_empty:
#         continue
#     if gname not in ('Zypressenstrasse', 'Sihlfeldstrasse'):
#         continue
#     x0, y0 = _start_point(geom)
#     ax.annotate(str(oid), xy=(x0, y0), xytext=(3, 3),
#                 textcoords='offset points',
#                 fontsize=7, color='black',
#                 bbox=dict(boxstyle='round,pad=0.15', fc='white',
#                           ec='0.6', lw=0.5, alpha=0.85))

# fig.tight_layout()
# # ZypressenstrN ---> OSMIDs = [4527091, 152001420]
# # ZypressenstrS ---> OSMIDs = [5007325, 152001419]
# # SihlfeldstrN ---> OSMIDs = [71160124, 152001415]
# # SihlfeldstrS ---> OSMIDs = [1324076751, 152001416]


print("Loading SwissTopo KML...")
gdf_swisstopo = gpd.read_file(kml_path, driver='KML')
# row = gdf_swisstopo[gdf_swisstopo['Description'] == 'Bullingerpl_RA_Rinner']

# Zypressenstr: Split by OSMIDs
line_ZypN = merge_edges_by_ids(gdf, [4527091, 152001420])
line_ZypN = densify_linestring(line_ZypN, num_segments=20)
tck_ZypN, unew_ZypN, cum_ZypN, len_ZypN = fit_spline_from_shapely(
    line_ZypN, x_offset=X_2056_offset, y_offset=Y_2056_offset,
)

line_ZypS = merge_edges_by_ids(gdf, [5007325, 152001419])
line_ZypS = densify_linestring(line_ZypS, num_segments=20)
tck_ZypS, unew_ZypS, cum_ZypS, len_ZypS = fit_spline_from_shapely(
    line_ZypS, x_offset=X_2056_offset, y_offset=Y_2056_offset,

)
# Sihlfeldstr: Split by OSMIDs
line_SihlN = merge_edges_by_ids(gdf, [71160124, 152001415])
line_SihlN = densify_linestring(line_SihlN, num_segments=10)
tck_SihlN, unew_SihlN, cum_SihlN, len_SihlN = fit_spline_from_shapely(
    line_SihlN, x_offset=X_2056_offset, y_offset=Y_2056_offset,
)

line_SihlS = merge_edges_by_ids(gdf, [1324076751, 152001416])
line_SihlS = densify_linestring(line_SihlS, num_segments=10)
tck_SihlS, unew_SihlS, cum_SihlS, len_SihlS = fit_spline_from_shapely(
    line_SihlS, x_offset=X_2056_offset, y_offset=Y_2056_offset,
)

# Bullingerstr
line_B = merge_osmnx_edges(gdf, 'Bullingerstrasse')
line_B = densify_linestring(line_B, num_segments=10)
tck_B, unew_B, cum_B, len_B = fit_spline_from_shapely(
    line_B, x_offset=X_2056_offset, y_offset=Y_2056_offset,
)

# Stauffacherstr
line_S = merge_osmnx_edges(gdf, 'Stauffacherstrasse')
line_S = densify_linestring(line_S, num_segments=10)
tck_S, unew_S, cum_S, len_S = fit_spline_from_shapely(
    line_S, x_offset=X_2056_offset, y_offset=Y_2056_offset,
)

fig, ax = plt.subplots(1, 1)
# gdf.plot(ax=ax, column='name', legend=True)
plot_line(line_ZypN, ax=ax, add_points=False, color='tab:blue', label='ZypressenstrN')
plot_line(line_ZypS, ax=ax, add_points=False, color='tab:orange', label='ZypressenstrS')
plot_line(line_SihlN, ax=ax, add_points=False, color='tab:green', label='SihlfeldstrN')
plot_line(line_SihlS, ax=ax, add_points=False, color='tab:brown', label='SihlfeldstrS')
plot_line(line_B, ax=ax, add_points=False, color='tab:purple', label='Bullingerstr')
plot_line(line_S, ax=ax, add_points=False, color='tab:pink', label='Stauffacherstr')
plot_points(Point(line_ZypN.coords[1]), color='red', marker='o',) 
plot_points(Point(line_ZypN.coords[-1]), color='black', marker='x',) 
plot_points(Point(line_ZypS.coords[0]), color='red', marker='o',) 
plot_points(Point(line_ZypS.coords[-1]), color='black', marker='x',) 
plot_points(Point(line_SihlN.coords[0]), color='red', marker='o',) 
plot_points(Point(line_SihlN.coords[-1]), color='black', marker='x',) 
plot_points(Point(line_SihlS.coords[0]), color='red', marker='o',) 
plot_points(Point(line_SihlS.coords[-1]), color='black', marker='x',)
plot_points(Point(line_B.coords[0]), color='red', marker='o',) 
plot_points(Point(line_B.coords[-1]), color='black', marker='x',) 
plot_points(Point(line_S.coords[0]), color='red', marker='o',) 
plot_points(Point(line_S.coords[-2]), color='black', marker='x',) 
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
        'name':          'ZypressenstrN',
        'positive_dir':  'NB',
        'spline':        (tck_ZypN, unew_ZypN, cum_ZypN),
        'total_length':  len_ZypN,
        'line_wgs84':    line_ZypN,
        'stop_line_id':  'ZypressenstrN_Stop',
        'yield_line_id': 'ZypressenstrN_Yield',
        'change_ratio':   0.8,
    },
    {
        'name':          'ZypressenstrS',
        'positive_dir':  'SB',
        'spline':        (tck_ZypS, unew_ZypS, cum_ZypS),
        'total_length':  len_ZypS,
        'line_wgs84':    line_ZypS,
        'stop_line_id':  'ZypressenstrS_Stop',
        'yield_line_id': 'ZypressenstrS_Yield',
        'change_ratio':   0.8,
    },
    {
        'name':          'SihlfeldstrN',
        'positive_dir':  'SB',
        'spline':        (tck_SihlN, unew_SihlN, cum_SihlN),
        'total_length':  len_SihlN,
        'line_wgs84':    line_SihlN,
        'stop_line_id':  'SihlfeldstrN_Stop',
        'yield_line_id': 'SihlfeldstrN_Yield',
        'change_ratio':   0.8,
    },
    {
        'name':          'SihlfeldstrS',
        'positive_dir':  'NB',
        'spline':        (tck_SihlS, unew_SihlS, cum_SihlS),
        'total_length':  len_SihlS,
        'line_wgs84':    line_SihlS,
        'stop_line_id':  'SihlfeldstrS_Stop',
        'yield_line_id': 'SihlfeldstrS_Yield',
        'change_ratio':   0.8,
    },
    {
        'name':          'Bullingerstr',
        'positive_dir':  'EB',
        'spline':        (tck_B, unew_B, cum_B),
        'total_length':  len_B,
        'line_wgs84':    line_B,
        'stop_line_id':  'Bullingerstr_Stop',
        'yield_line_id': 'Bullingerstr_Yield',
        'change_ratio':   0.8,
    },
    {
        'name':          'Stauffacherstr',
        'positive_dir':  'WB',
        'spline':        (tck_S, unew_S, cum_S),
        'total_length':  len_S,
        'line_wgs84':    line_S,
        'stop_line_id':  'Stauffacherstr_Stop',
        'yield_line_id': 'Stauffacherstr_Yield',
        'change_ratio':   0.8,
    },
]


geometry_store = register_geometries(
    RAW_AXES, gdf_swisstopo, X_2056_offset, Y_2056_offset,
)

ring = read_ring_kml(gdf_swisstopo, prefix='Bullingerpl_RA', verbose=True)
ring['r_inner'] = ring['r_inner'] - 1.0
ring['r_outer'] = ring['r_outer'] + 2.0
geometry_store['Bullingerpl_RA_Ring'] = register_ring_geometry(ring, X_2056_offset, Y_2056_offset)

# =============================================================================
# PHASE 2: build_segment_registry
# d_left  = tolerance LEFT  of travel (GPS margin, median-strip side) [m]
# d_right = full usable carriageway RIGHT of travel                   [m]
#           includes the bike stripe width if present

print("--- Phase 2: build segment registry ---")
 
SEG_DEFS = [
    # --- Zypressenstrasse, north branch (spline runs OUTWARD) ---------------
    {'seg_key': 'ZypressenstrN_SB', 'geometry_key': 'ZypressenstrN',   
     'direction': 'SB', 'mode': 'bike', 'bike_lane': None,
     'd_left': 1.0, 'd_right': 8.0},
    {'seg_key': 'ZypressenstrN_NB', 'geometry_key': 'ZypressenstrN',   
     'direction': 'NB', 'mode': 'shared', 'bike_lane': None,
     'd_left': 5.0, 'd_right': 11.0},

    # --- Zypressenstrasse, south branch (spline runs INWARD) ----------------
    {'seg_key': 'ZypressenstrS_NB', 'geometry_key': 'ZypressenstrS',   
     'direction': 'NB', 'mode': 'shared', 'bike_lane': None,
     'd_left': 1.0, 'd_right': 10.0},
    {'seg_key': 'ZypressenstrS_SB', 'geometry_key': 'ZypressenstrS',   
     'direction': 'SB', 'mode': 'shared', 'bike_lane': None,
     'd_left': 1.0, 'd_right': 10.0},

    # --- Sihlfeldstrasse, north branch --------------------------------------
    {'seg_key': 'SihlfeldstrN_SB', 'geometry_key': 'SihlfeldstrN',    
     'direction': 'SB', 'mode': 'shared', 'bike_lane': None,
     'd_left': 1.0, 'd_right': 10.0},
    {'seg_key': 'SihlfeldstrN_NB', 'geometry_key': 'SihlfeldstrN',     
     'direction': 'NB', 'mode': 'shared', 'bike_lane': None,
     'd_left': 2.0, 'd_right': 13.0},

    # --- Sihlfeldstrasse, south branch --------------------------------------
    {'seg_key': 'SihlfeldstrS_NB', 'geometry_key': 'SihlfeldstrS',     
     'direction': 'NB', 'mode': 'shared', 'bike_lane': None,
     'd_left': 1.0, 'd_right': 11.0},
    {'seg_key': 'SihlfeldstrS_SB', 'geometry_key': 'SihlfeldstrS',    
     'direction': 'SB', 'mode': 'shared', 'bike_lane': None,
     'd_left': 2.0, 'd_right': 11.0},

    # --- Bullingerstrasse (west leg) ----------------------------------------
    {'seg_key': 'Bullingerstr_EB', 'geometry_key': 'Bullingerstr',    
     'direction': 'EB', 'mode': 'shared', 'bike_lane': None,
     'd_left': 1.0, 'd_right': 15.0},
    {'seg_key': 'Bullingerstr_WB', 'geometry_key': 'Bullingerstr',     
     'direction': 'WB', 'mode': 'shared', 'bike_lane': None,
     'd_left': 1.0, 'd_right': 13.0},

    # --- Stauffacherstrasse (east leg) --------------------------------------
    {'seg_key': 'Stauffacherstr_WB', 'geometry_key': 'Stauffacherstr', 
     'direction': 'WB', 'mode': 'shared', 'bike_lane': None,
     'd_left': 1.5, 'd_right': 13.0},
    {'seg_key': 'Stauffacherstr_EB', 'geometry_key': 'Stauffacherstr', 
     'direction': 'EB', 'mode': 'shared', 'bike_lane': None,
     'd_left': 1.5, 'd_right': 11.0},
]
 
segment_registry = build_segment_registry(geometry_store, SEG_DEFS)

build_ring_segment(geometry_store, segment_registry, {
    'seg_key':      'Bullingerpl_RA_CCW',
    'geometry_key': 'Bullingerpl_RA_Ring',
    'mode':         'shared',
})

# No bike lane boundaries here

# No car lane markings here (all bi-directional, except ZypressenstrN)

# =============================================================================
# PHASE 3: build_turns

print("\n--- Phase 3: build turn splines ---")

RING_SEG = 'Bullingerpl_RA_CCW'

TURN_DEFS = [
    # ---- Stauffacherstr (east) --------------------------------------------
    {'approach_seg': 'Stauffacherstr_WB', 'departure_seg': RING_SEG,
     'departure_s_change_key': 's_entry_Stauffacherstr',
     'departure_window_m': 8.0, 'd_left': 6.0, 'd_right': 8.0},
    {'approach_seg': RING_SEG, 'departure_seg': 'Stauffacherstr_EB',
     'approach_s_change_key': 's_exit_Stauffacherstr',
     'approach_window_m': 8.0, 'd_left': 6.0, 'd_right': 8.0},

    # ---- ZypressenstrN (north-east) ---------------------------------------
    {'approach_seg': 'ZypressenstrN_SB', 'departure_seg': RING_SEG,
     'departure_s_change_key': 's_entry_ZypressenstrN',
     'departure_window_m': 8.0, 'd_left': 6.0, 'd_right': 8.0},
    {'approach_seg': RING_SEG, 'departure_seg': 'ZypressenstrN_NB',
     'approach_s_change_key': 's_exit_ZypressenstrN',
     'approach_window_m': 8.0, 'd_left': 6.0, 'd_right': 8.0},

    # ---- SihlfeldstrN (north) ---------------------------------------------
    {'approach_seg': 'SihlfeldstrN_SB', 'departure_seg': RING_SEG,
     'departure_s_change_key': 's_entry_SihlfeldstrN',
     'departure_window_m': 8.0, 'd_left': 6.0, 'd_right': 8.0},
    {'approach_seg': RING_SEG, 'departure_seg': 'SihlfeldstrN_NB',
     'approach_s_change_key': 's_exit_SihlfeldstrN',
     'approach_window_m': 8.0, 'd_left': 6.0, 'd_right': 8.0},

    # ---- Bullingerstr (west) ----------------------------------------------
    {'approach_seg': 'Bullingerstr_EB', 'departure_seg': RING_SEG,
     'departure_s_change_key': 's_entry_Bullingerstr',
     'departure_window_m': 8.0, 'd_left': 6.0, 'd_right': 8.0},
    {'approach_seg': RING_SEG, 'departure_seg': 'Bullingerstr_WB',
     'approach_s_change_key': 's_exit_Bullingerstr',
     'approach_window_m': 8.0, 'd_left': 6.0, 'd_right': 8.0},

    # ---- ZypressenstrS (south-west) ---------------------------------------
    {'approach_seg': 'ZypressenstrS_NB', 'departure_seg': RING_SEG,
     'departure_s_change_key': 's_entry_ZypressenstrS',
     'departure_window_m': 8.0, 'd_left': 6.0, 'd_right': 8.0},
    {'approach_seg': RING_SEG, 'departure_seg': 'ZypressenstrS_SB',
     'approach_s_change_key': 's_exit_ZypressenstrS',
     'approach_window_m': 8.0, 'd_left': 6.0, 'd_right': 8.0},

    # ---- SihlfeldstrS (south) ---------------------------------------------
    {'approach_seg': 'SihlfeldstrS_NB', 'departure_seg': RING_SEG,
     'departure_s_change_key': 's_entry_SihlfeldstrS',
     'departure_window_m': 8.0, 'd_left': 6.0, 'd_right': 8.0},
    {'approach_seg': RING_SEG, 'departure_seg': 'SihlfeldstrS_SB',
     'approach_s_change_key': 's_exit_SihlfeldstrS',
     'approach_window_m': 8.0, 'd_left': 6.0, 'd_right': 8.0},

    # ---- direct turns between the close leg pairs (no ring) ----------------
    {'approach_seg': 'Stauffacherstr_WB', 'departure_seg': 'ZypressenstrN_NB',
     'd_left': 6.0, 'd_right': 8.0},
    {'approach_seg': 'Bullingerstr_EB', 'departure_seg': 'ZypressenstrS_SB',
     'd_left': 6.0, 'd_right': 8.0},
    
    {'approach_seg': 'ZypressenstrN_SB', 'departure_seg': 'Stauffacherstr_EB',
     'd_left': 6.0, 'd_right': 8.0},
    {'approach_seg': 'ZypressenstrS_NB', 'departure_seg': 'Bullingerstr_WB',
     'd_left': 6.0, 'd_right': 8.0},
]

turn_keys = build_turns(geometry_store, segment_registry, TURN_DEFS)

from tools_plot_registry import plot_geometry_store, plot_segment_registry

plot_geometry_store(geometry_store, gdf_swisstopo, offset_m=3.0)
plot_segment_registry(geometry_store, segment_registry, gdf_swisstopo, turn_offset_m=0.8)

# =============================================================================
# PHASE 4: build_movement_registry
# 36 movements: 6 legs x 6, including the six U-turns.
#
#   through-ring   [approach, entry_turn, ring, exit_turn, departure]  (5)
#   direct         [approach, turn, departure]                        (3)

LEG_SEGS = {                       # leg: (approach_seg, departure_seg)
    'Stauffacherstr': ('Stauffacherstr_WB', 'Stauffacherstr_EB'),
    'ZypressenstrN':  ('ZypressenstrN_SB',  'ZypressenstrN_NB'),
    'SihlfeldstrN':   ('SihlfeldstrN_SB',   'SihlfeldstrN_NB'),
    'Bullingerstr':   ('Bullingerstr_EB',   'Bullingerstr_WB'),
    'ZypressenstrS':  ('ZypressenstrS_NB',  'ZypressenstrS_SB'),
    'SihlfeldstrS':   ('SihlfeldstrS_NB',   'SihlfeldstrS_SB'),
}

# (entry_leg, exit_leg) pairs handled by a direct turn instead of the ring.
# Ordered, not symmetric — only the short direction bypasses the ring.
DIRECT_MOVEMENTS = {
    ('Stauffacherstr', 'ZypressenstrN'),
    ('Bullingerstr',   'ZypressenstrS'),
    ('ZypressenstrN',  'Stauffacherstr'),
    ('ZypressenstrS',   'Bullingerstr'),
}

MOVEMENT_DEFS = []
for a, (app_a, _) in LEG_SEGS.items():
    for b, (_, dep_b) in LEG_SEGS.items():
        key = f'{a}_2_{b}'
        if (a, b) in DIRECT_MOVEMENTS:
            MOVEMENT_DEFS.append({'key': key, 'sequence': [
                (app_a,                          'approach'),
                (f'turn_{app_a}_2_{dep_b}',      'turn'),
                (dep_b,                          'departure'),
            ]})
        else:
            MOVEMENT_DEFS.append({'key': key, 'sequence': [
                (app_a,                              'approach'),
                (f'turn_{app_a}_2_{RING_SEG}',       'turn'),
                (RING_SEG,                           'ring'),
                (f'turn_{RING_SEG}_2_{dep_b}',       'turn'),
                (dep_b,                              'departure'),
            ]})

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