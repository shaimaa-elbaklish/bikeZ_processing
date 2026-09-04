"""
TITLE OF PAPAER
-------------------------------------------
Authors:        Shaimaa El-Baklish
Organization:   ETH Zürich, Switzerland, IVT - Institute for Transportation Planning and Systems
Development:    2025-2026
Submitted to:   JOURNAL
-------------------------------------------

maps_sep_D1A.py
-------------------------------------
Site definition — Quaibrücke / Stadthausquai / Bürkliplatz / Fraumünsterstrasse
Zürich, Switzerland — September 2025 campaign (D1,A location)
 
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

intersection, code = BikeZ_Config.avail_intersections[date][0]
timeslot = BikeZ_Config.avail_timeslots[date][(intersection, code)][0] # 'AM1'

XY_2056_Bounds = BikeZ_Config.XY_2056_Bounds[date][(intersection, code)]
X_2056_offset = XY_2056_Bounds[0][0]
Y_2056_offset = XY_2056_Bounds[1][0]

# Site constants
kml_path       = '../maps/from_swisstopo/September_D1A.kml'
kml_path_lanes = '../maps/from_swisstopo/September_D1A_CarLanes.kml'
save_path      = f'../data/registry_{date}_{intersection}_{code}.pkl'
max_chain_len  = 5    # 3 for standard movements + 2 for Fraumunsterstr chain

# Edit Link: https://s.geo.admin.ch/x73p6kkmkkx5
# Share Link: https://s.geo.admin.ch/jns6pcct7j02

# Car Lanes:
# Share Link: https://s.geo.admin.ch/5vzqc34a8bzn
# Edit Link: https://s.geo.admin.ch/ntbnygtd1vjf

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
    np.asarray(XY_2056_Bounds[0]) + np.asarray([-40, 15]),
    np.asarray(XY_2056_Bounds[1]) + np.asarray([-40, 15]),
)
bbox_geom = box(lonlat[0][0], lonlat[1][0], lonlat[0][1], lonlat[1][1])
 
gdf_main   = ox.features.features_from_place('Zürich, Switzerland',
                                              tags={'highway': True})
road_types = ['primary', 'secondary', 'tertiary',
              'residential', 'unclassified', 'cycleway']
gdf = gdf_main[
    gdf_main['name'].isin(
        ['Quaibrücke', 'Stadthausquai', 'Bürkliplatz', 'Fraumünsterstrasse']
    )
]
gdf = gdf[
    (gdf.geometry.type == 'LineString') #& (gdf['highway'].isin(road_types))
]
gdf['geometry'] = gdf.geometry.intersection(bbox_geom)
gdf = gdf[~gdf.is_empty]

fig, ax = plt.subplots(1, 1)
gdf.plot(ax=ax, column='name', legend=True)
# fig.tight_layout()

print("Loading SwissTopo KML...")
gdf_swisstopo = gpd.read_file(kml_path, driver='KML')

# 4th arm of intersection (from Burkliplatz to Stadthausquai) ---> OSMID = 150132676
# Burkliplatz EB ---> OSMID = 60447876, 1169386963
# Burkliplatz WB ---> OSMID = 60447877, 1169386967

# Quaibrucke EB ---> OSMID = 60447874, 27361257, 374433860
# Quaibrucke WB ---> OSMID = 67571252, 569781024, 27361247, 15801814

# Burklipl/Quaibr EB
line_Burklipl_EB = merge_edges_by_ids(gdf, [60447876, 1169386963])
line_Burklipl_EB = densify_linestring(line_Burklipl_EB, num_segments=16)
line_Quaibr_EB = merge_edges_by_ids(gdf, [60447874, 27361257, 374433860])
line_Quaibr_EB = densify_linestring(line_Quaibr_EB, num_segments=28)
line_Burkli_Quai_EB = concat_linestrings(line_Burklipl_EB, line_Quaibr_EB)
line_Burkli_Quai_EB = densify_linestring(line_Burkli_Quai_EB, num_segments=60)
tck_EB, unew_EB, cum_EB, len_EB = fit_spline_from_shapely(
    line_Burkli_Quai_EB, x_offset=X_2056_offset, y_offset=Y_2056_offset,
)

# Burklipl/Quaibr WB
line_Burklipl_WB = merge_edges_by_ids(gdf, [60447877, 1169386967, 374522039])
line_Burklipl_WB = densify_linestring(line_Burklipl_WB, num_segments=22)
line_Quaibr_WB = merge_edges_by_ids(gdf, [67571252, 569781024, 27361247, 15801814])
line_Quaibr_WB = densify_linestring(line_Quaibr_WB, num_segments=18)
line_Burkli_Quai_WB = concat_linestrings(line_Quaibr_WB, line_Burklipl_WB)
line_Burkli_Quai_WB = densify_linestring(line_Burkli_Quai_WB, num_segments=60)
tck_WB, unew_WB, cum_WB, len_WB = fit_spline_from_shapely(
    line_Burkli_Quai_WB, x_offset=X_2056_offset, y_offset=Y_2056_offset,
)

# Stadthausquai
line_S = merge_osmnx_edges(gdf, 'Stadthausquai')
line_S = densify_linestring(line_S, num_segments=10)
tck_S, unew_S, cum_S, len_S = fit_spline_from_shapely(
    line_S, x_offset=X_2056_offset, y_offset=Y_2056_offset,
)

# Fraumunsterstr
line_F = merge_osmnx_edges(gdf, 'Fraumünsterstrasse')
line_F = densify_linestring(line_F, num_segments=10)
tck_F, unew_F, cum_F, len_F = fit_spline_from_shapely(
    line_F, x_offset=X_2056_offset, y_offset=Y_2056_offset,
)

print(f"  Fraumunsterstr      : {len_F:.1f} m")
print(f"  Stadthausquai       : {len_S:.1f} m")
print(f"  Burklipl_Quaibr_EB  : {len_EB:.1f} m")
print(f"  Burklipl_Quaibr_WB  : {len_WB:.1f} m")


fig, ax = plt.subplots(1, 1)
# gdf.plot(ax=ax, column='name', legend=True)
plot_line(line_F, ax=ax, add_points=False, color='tab:orange', label='Fraumunsterstr')
plot_line(line_S, ax=ax, add_points=False, color='tab:green', label='Stadthausquai')
plot_line(line_Burkli_Quai_EB, ax=ax, add_points=False, color='tab:purple', label='Burklipl_Quaibr_EB')
plot_line(line_Burkli_Quai_WB, ax=ax, add_points=False, color='tab:pink', label='Burklipl_Quaibr_WB')
plot_points(Point(line_F.coords[0]), color='red', marker='o',) 
plot_points(Point(line_F.coords[-2]), color='black', marker='x',) 
plot_points(Point(line_S.coords[1]), color='red', marker='o',) 
plot_points(Point(line_S.coords[-1]), color='black', marker='x',) 
plot_points(Point(line_Burkli_Quai_EB.coords[0]), color='red', marker='o',) 
plot_points(Point(line_Burkli_Quai_EB.coords[-1]), color='black', marker='x',) 
plot_points(Point(line_Burkli_Quai_WB.coords[0]), color='red', marker='o',) 
plot_points(Point(line_Burkli_Quai_WB.coords[-1]), color='black', marker='x',) 
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
        'name':          'Burklipl_Quaibr_EB',
        'positive_dir':  'EB',
        'spline':        (tck_EB, unew_EB, cum_EB),
        'total_length':  len_EB,
        'line_wgs84':    line_Burkli_Quai_EB,
        'stop_line_id':  'Burklipl_Quaibr_EB_Stop',
        'yield_line_id': 'Burklipl_Quaibr_EB_Stop',
        'extra_changes': [
            {'key': 's_burklipl_east_T2_stop',
             'stop_line_id': 'Burklipl_Quaibr_EB_T2_Stop'},
            {'key': 's_quaibr_stop',
             'stop_line_id': 'Burklipl_Quaibr_EB_B_Stop'},
        ],
    },
    {
        'name':          'Burklipl_Quaibr_WB',
        'positive_dir':  'WB',
        'spline':        (tck_WB, unew_WB, cum_WB),
        'total_length':  len_WB,
        'line_wgs84':    line_Burkli_Quai_WB,
        'stop_line_id':  'Burklipl_Quaibr_WB_Stop',
        'yield_line_id': 'Burklipl_Quaibr_WB_Stop',
        'extra_changes': [
            {'key': 's_quaibr_west_T2_stop',
             'stop_line_id': 'Burklipl_Quaibr_WB_T2_Stop'},
            {'key': 's_burklipl_west_T2_yield',
             'stop_line_id': 'Burklipl_Quaibr_WB_T2_Yield'},
            {'key': 's_burklipl_west_T1_stop',
             'stop_line_id': 'Burklipl_Quaibr_WB_T1_Stop'},
            {'key': 's_burklipl_west_T1_yield',
             'stop_line_id': 'Burklipl_Quaibr_WB_T1_Yield'},
            {'key': 'w_quaibr_end_2p5',
             'stop_line_id': 'Burklipl_Quaibr_WB_End2.5Width'},
            {'key': 'w_quaibr_start_1p5',
             'stop_line_id': 'Burklipl_Quaibr_WB_Start1.5Width'},
        ],
    },
    {
        # Stub south of MainInt — dead-ends into Zollstr (T-junction).
        'name':          'Stadthausquai',
        'positive_dir':  'NB',
        'spline':        (tck_S, unew_S, cum_S),
        'total_length':  len_S,
        'line_wgs84':    line_S,
        'stop_line_id':  'Stadthausquai_Stop',
        'yield_line_id': 'Stadthausquai_Yield',
        'change_ratio':   0.9,   # optional — defaults to 0.6 if omitted; 0 at stop and 1 at yield
    },
    {
        'name':          'Fraumunsterstr',
        'positive_dir':  'SB',
        'spline':        (tck_F, unew_F, cum_F),
        'total_length':  len_F,
        'line_wgs84':    line_F,
        'stop_line_id':  'Fraumunsterstr_Stop',
        'yield_line_id': 'Fraumunsterstr_Yield',
        'change_ratio':   0.9,   # optional — defaults to 0.6 if omitted; 0 at stop and 1 at yield
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

print("--- Phase 2: build segment registry ---")
 
SEG_DEFS = [
    # ── Burklipl_Quaibr_EB ──────────────────────────────────────────────────
    {'seg_key': 'Burklipl_Quaibr_EB', 'geometry_key': 'Burklipl_Quaibr_EB',
     'direction': 'EB', 'mode': 'shared', 'bike_lane': {'w_bike': 2.5},
     'd_left': 12.5, 'd_right': 14.0,},
    
    # ── Burklipl_Quaibr_WB ──────────────────────────────────────────────────
    {'seg_key': 'Burklipl_Quaibr_WB', 'geometry_key': 'Burklipl_Quaibr_WB',
     'direction': 'WB', 'mode': 'shared', 
     'bike_lane': {
         'w_bike': 2.5,
         'w_taper': {'from_key': 'w_quaibr_end_2p5',   'w_from': 2.5,
                     'to_key':   'w_quaibr_start_1p5', 'w_to':   1.5}
      },
     'd_left': 11.5, 'd_right': 11.0,},
 
    # ── Stadthausquai ───────────────────────────────────────────────────────
    {'seg_key': 'Stadthausquai_NB', 'geometry_key': 'Stadthausquai',
     'direction': 'NB', 'mode': 'shared', 'bike_lane': None,
     'd_left': 4.5, 'd_right': 9.0},
    {'seg_key': 'Stadthausquai_SB', 'geometry_key': 'Stadthausquai',
     'direction': 'SB', 'mode': 'bike', 'bike_lane': None,
     'd_left': 1.5, 'd_right': 12.0},

    # ── Fraumunsterstrasse ──────────────────────────────────────────────────
    {'seg_key': 'Fraumunsterstr_NB', 'geometry_key': 'Fraumunsterstr',
     'direction': 'NB', 'mode': 'bike', 'bike_lane': None,
     'd_left': 1.5, 'd_right': 9.5},
    {'seg_key': 'Fraumunsterstr_SB', 'geometry_key': 'Fraumunsterstr',
     'direction': 'SB', 'mode': 'shared', 'bike_lane': None,
     'd_left': 4.0, 'd_right': 8.0},
]
 
segment_registry = build_segment_registry(geometry_store, SEG_DEFS)

print("--- Step 2b: project bike lane boundaries ---")
gdf_bike_boundaries = gdf_swisstopo[
    gdf_swisstopo['Description'].str.endswith(
        ('_NB', '_SB', '_EB', '_WB')
    )
].copy()
add_bike_lane_boundaries(segment_registry, geometry_store, gdf_bike_boundaries)
add_bike_lane_width_profile(segment_registry, geometry_store)

print("--- Step 2c: project car lane polygons and boundaries ---")
gdf_car_lane_polygons = gpd.read_file(kml_path_lanes, driver='KML')
add_car_lane_boundaries(segment_registry, geometry_store, gdf_car_lane_polygons)

# =============================================================================
# PHASE 3: build_turns

print("\n--- Phase 3: build turn splines ---")

TURN_DEFS = [
    # ── T1: Fraumunsterstr ↔ Burklipl/Quaibr ─────────────────────────────────
    # WB through traffic turning right into Fraumunsterstr (heading NB, away from junction)
    {'approach_seg': 'Burklipl_Quaibr_WB', 'departure_seg': 'Fraumunsterstr_NB',
     'approach_s_change_key':  's_burklipl_west_T1_stop',
     'departure_s_change_key': 's_change',
     'd_left': 8.0, 'd_right': 8.0},
    # Fraumunsterstr traffic (heading SB, arriving) merging onto Burklipl_Quaibr_WB
    {'approach_seg': 'Fraumunsterstr_SB', 'departure_seg': 'Burklipl_Quaibr_WB',
     'approach_s_change_key':  's_change',
     'departure_s_change_key': 's_burklipl_west_T1_yield',
     'd_left': 8.0, 'd_right': 8.0},
    
    # ── T2: Stadthausquai ↔ Burklipl/Quaibr ──────────────────────────────────
    # Stadthausquai_SB (arriving) → right turn onto Burklipl_Quaibr_EB (continuing east)
    {'approach_seg': 'Stadthausquai_SB', 'departure_seg': 'Burklipl_Quaibr_EB',
     'approach_s_change_key':  's_change',
     'departure_s_change_key': 's_quaibr_stop',
     'd_left': 8.0, 'd_right': 8.0},
    # Stadthausquai_SB (arriving) → left turn onto Burklipl_Quaibr_WB (continuing west)
    {'approach_seg': 'Stadthausquai_SB', 'departure_seg': 'Burklipl_Quaibr_WB',
     'approach_s_change_key':  's_change',
     'departure_s_change_key': 's_burklipl_west_T2_yield',
     'd_left': 10.0, 'd_right': 10.0},
    # Burklipl_Quaibr_EB → left turn into Stadthausquai_NB
    # (this is the "4th arm" channelized movement — synthesized clothoid here;
    #  see note below re: the unused Burklipl_Quaibr_Crossing digitization)
    {'approach_seg': 'Burklipl_Quaibr_EB', 'departure_seg': 'Stadthausquai_NB',
     'approach_s_change_key':  's_burklipl_east_T2_stop',
     'departure_s_change_key': 's_change',
     'd_left': 8.0, 'd_right': 8.0},
    # Burklipl_Quaibr_WB → right turn into Stadthausquai_NB
    {'approach_seg': 'Burklipl_Quaibr_WB', 'departure_seg': 'Stadthausquai_NB',
     'approach_s_change_key':  's_quaibr_west_T2_stop',
     'departure_s_change_key': 's_change',
     'd_left': 10.0, 'd_right': 10.0},
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

geometry_store['intersection_area_T1_Fraumunsterstr'] = build_intersection_polygon(
    arm_defs = [
        {'geom_key': 'Burklipl_Quaibr_WB', 's_change_key': 's_burklipl_west_T1_stop',
         'pos_seg_key': 'Burklipl_Quaibr_WB', 'opp_seg_key': 'Burklipl_Quaibr_EB', 
         'approach_seg_key': 'Burklipl_Quaibr_WB'},
        {'geom_key': 'Burklipl_Quaibr_WB', 's_change_key': 's_burklipl_west_T1_yield',
         'pos_seg_key': 'Burklipl_Quaibr_WB', 'opp_seg_key': 'Burklipl_Quaibr_EB', 
         'approach_seg_key': 'Burklipl_Quaibr_EB'},
        {'geom_key': 'Fraumunsterstr', 's_change_key': 's_change',
         'pos_seg_key': 'Fraumunsterstr_SB', 'opp_seg_key': 'Fraumunsterstr_NB',
         'approach_seg_key': 'Fraumunsterstr_SB'},
    ],
    geometry_store   = geometry_store,
    segment_registry = segment_registry,
)
geometry_store['intersection_type_T1_Fraumunsterstr'] = 'T-junction'

geometry_store['intersection_area_T2_Stadthausquai'] = build_intersection_polygon(
    arm_defs = [
        {'geom_key': 'Burklipl_Quaibr_WB', 's_change_key': 's_quaibr_west_T2_stop',
         'pos_seg_key': 'Burklipl_Quaibr_WB', 'opp_seg_key': 'Burklipl_Quaibr_EB', 
         'approach_seg_key': 'Burklipl_Quaibr_WB'},
        {'geom_key': 'Burklipl_Quaibr_WB', 's_change_key': 's_burklipl_west_T2_yield',
         'pos_seg_key': 'Burklipl_Quaibr_WB', 'opp_seg_key': 'Burklipl_Quaibr_EB', 
         'approach_seg_key': 'Burklipl_Quaibr_EB'},           
        {'geom_key': 'Stadthausquai', 's_change_key': 's_change',
         'pos_seg_key': 'Stadthausquai_NB', 'opp_seg_key': 'Stadthausquai_SB',
         'approach_seg_key': 'Stadthausquai_SB'},
        {'geom_key': 'Burklipl_Quaibr_EB', 's_change_key': 's_burklipl_east_T2_stop',
         'pos_seg_key': 'Burklipl_Quaibr_EB', 'opp_seg_key': 'Burklipl_Quaibr_WB', 
         'approach_seg_key': 'Burklipl_Quaibr_EB'},
    ],
    geometry_store   = geometry_store,
    segment_registry = segment_registry,
)
geometry_store['intersection_type_T2_Stadthausquai'] = 'T-junction'

from tools_plot_registry import plot_geometry_store, plot_segment_registry

plot_geometry_store(geometry_store, gdf_swisstopo, offset_m=3.0)
plot_segment_registry(geometry_store, segment_registry, gdf_swisstopo)

# =============================================================================
# PHASE 4: build_movement_registry

print("--- Phase 4: build movement registry ---")

MOVEMENT_DEFS = [
    # ── T1: Fraumunsterstr ⊥ Burklipl_Quaibr_WB ─────────────────────────────
    {'key': 'Burklipl_Quaibr_WB_2_Fraumunsterstr_NB',
     'sequence': [('Burklipl_Quaibr_WB',                          'approach'),
                  ('turn_Burklipl_Quaibr_WB_2_Fraumunsterstr_NB', 'turn'),
                  ('Fraumunsterstr_NB',                            'departure')]},
    {'key': 'Fraumunsterstr_SB_2_Burklipl_Quaibr_WB',
     'sequence': [('Fraumunsterstr_SB',                            'approach'),
                  ('turn_Fraumunsterstr_SB_2_Burklipl_Quaibr_WB', 'turn'),
                  ('Burklipl_Quaibr_WB',                           'departure')]},

    # ── T2: Stadthausquai ⊥ Burklipl_Quaibr_EB/WB ───────────────────────────
    {'key': 'Stadthausquai_SB_2_Burklipl_Quaibr_EB',
     'sequence': [('Stadthausquai_SB',                             'approach'),
                  ('turn_Stadthausquai_SB_2_Burklipl_Quaibr_EB',  'turn'),
                  ('Burklipl_Quaibr_EB',                           'departure')]},
    {'key': 'Stadthausquai_SB_2_Burklipl_Quaibr_WB',
     'sequence': [('Stadthausquai_SB',                             'approach'),
                  ('turn_Stadthausquai_SB_2_Burklipl_Quaibr_WB',  'turn'),
                  ('Burklipl_Quaibr_WB',                           'departure')]},
    {'key': 'Burklipl_Quaibr_EB_2_Stadthausquai_NB',
     'sequence': [('Burklipl_Quaibr_EB',                           'approach'),
                  ('turn_Burklipl_Quaibr_EB_2_Stadthausquai_NB',  'turn'),
                  ('Stadthausquai_NB',                              'departure')]},
    {'key': 'Burklipl_Quaibr_WB_2_Stadthausquai_NB',
     'sequence': [('Burklipl_Quaibr_WB',                           'approach'),
                  ('turn_Burklipl_Quaibr_WB_2_Stadthausquai_NB',  'turn'),
                  ('Stadthausquai_NB',                              'departure')]},
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
