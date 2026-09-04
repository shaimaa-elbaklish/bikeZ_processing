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
kml_path      = '../maps/from_swisstopo/September_D1G.kml'
save_path     = f'../data/registry_{date}_{intersection}_{code}.pkl'
max_chain_len = 3


# Share Link: https://s.geo.admin.ch/1ssj1wf1tkvy
# Edit Link: https://s.geo.admin.ch/zf0fmp83szbr


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

# Birmensdorferstr: Split by IDs into the forked branches
gdf_birm = gdf[gdf['name'] == 'Birmensdorferstrasse']
gdf_birm = gdf_birm[gdf_birm.geometry.type == "LineString"]
# for idx, row in gdf_birm.iterrows():
#     osmid = idx[1] if isinstance(idx, tuple) else idx  # ('way', id) MultiIndex
#     c = list(row.geometry.coords)
#     print(f"osmid={osmid:>12}  n_pts={len(c):3d}  "
#           f"start=({c[0][0]:.6f},{c[0][1]:.6f})  "
#           f"end=({c[-1][0]:.6f},{c[-1][1]:.6f})  "
#           f"oneway={row.get('oneway')}  len={row.geometry.length:.6f}")

# # Plot each fragment separately, colored + labeled by osmid
# import matplotlib.pyplot as plt
# import matplotlib.cm as cm
# fig, ax = plt.subplots(1, 1, figsize=(10, 8))
# colors = cm.tab20(np.linspace(0, 1, len(gdf_birm)))
# for (idx, row), c in zip(gdf_birm.iterrows(), colors):
#     osmid = idx[1] if isinstance(idx, tuple) else idx
#     xs, ys = row.geometry.xy
#     ax.plot(xs, ys, color=c, linewidth=2, label=f"{osmid}")
#     mx, my = row.geometry.interpolate(0.5, normalized=True).coords[0]
#     ax.annotate(str(osmid), (mx, my), fontsize=7)
# ax.legend(fontsize=6, ncol=2, loc='best')
# ax.set_title("Birmensdorferstrasse — edges by osmid")
# plt.show()
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


BIRMENSDORFER_WB_IDS = [546118531, 546118521, 546118524, 422926694]
BIRMENSDORFER_EB_IDS = [546118531, 546118517, 14943840]

line_Birm_WB = merge_edges_by_ids(gdf_birm, BIRMENSDORFER_WB_IDS)
line_Birm_EB = merge_edges_by_ids(gdf_birm, BIRMENSDORFER_EB_IDS)

# Birmensdorferstr_WB: Split into west and east sides
line_BirmE_WB = cut_line_at_stop(line_Birm_WB, line_SchaufB, choose='first', plotting=False)
line_BirmE_WB = densify_linestring(line_BirmE_WB, num_segments=20)
tck_BE_WB, unew_BE_WB, cum_BE_WB, len_BE_WB = fit_spline_from_shapely(
    line_BirmE_WB, x_offset=X_2056_offset, y_offset=Y_2056_offset
)

line_BirmW_WB = cut_line_at_stop(line_Birm_WB, line_SchaufB, choose='last', plotting=False)
line_BirmW_WB = densify_linestring(line_BirmW_WB, num_segments=20)
tck_BW_WB, unew_BW_WB, cum_BW_WB, len_BW_WB = fit_spline_from_shapely(
    line_BirmW_WB, x_offset=X_2056_offset, y_offset=Y_2056_offset
)

# Birmensdorferstr_EB: Split into west and east sides
line_BirmE_EB = cut_line_at_stop(line_Birm_EB, line_SchwH, choose='last', plotting=False)
line_BirmE_EB = densify_linestring(line_BirmE_EB, num_segments=20)
tck_BE_EB, unew_BE_EB, cum_BE_EB, len_BE_EB = fit_spline_from_shapely(
    line_BirmE_EB, x_offset=X_2056_offset, y_offset=Y_2056_offset
)

line_BirmW_EB = cut_line_at_stop(line_Birm_EB, line_SchwH, choose='first', plotting=False)
line_BirmW_EB = densify_linestring(line_BirmW_EB, num_segments=20)
tck_BW_EB, unew_BW_EB, cum_BW_EB, len_BW_EB = fit_spline_from_shapely(
    line_BirmW_EB, x_offset=X_2056_offset, y_offset=Y_2056_offset
)


print(f"  Schweighofstr        : {len_SchwH:.1f} m")
print(f"  Schaufelbergerstr    : {len_SchaufB:.1f} m")
print(f"  BirmensdorferstrE_WB : {len_BE_WB:.1f} m")
print(f"  BirmensdorferstrW_WB : {len_BW_WB:.1f} m")
print(f"  BirmensdorferstrE_EB : {len_BE_EB:.1f} m")
print(f"  BirmensdorferstrW_EB : {len_BW_EB:.1f} m")


fig, ax = plt.subplots(1, 1)
# gdf.plot(ax=ax, column='name', legend=True)
plot_line(line_SchwH, ax=ax, add_points=False, color='tab:blue', label='Schweighofstr')
plot_line(line_SchaufB, ax=ax, add_points=False, color='tab:orange', label='Schaufelbergerstr')
plot_line(line_BirmE_WB, ax=ax, add_points=False, color='tab:green', label='BirmensdorferstrE_WB')
plot_line(line_BirmW_WB, ax=ax, add_points=False, color='tab:purple', label='BirmensdorferstrW_WB')
plot_line(line_BirmE_EB, ax=ax, add_points=False, color='tab:brown', label='BirmensdorferstrE_EB')
plot_line(line_BirmW_EB, ax=ax, add_points=False, color='tab:pink', label='BirmensdorferstrW_EB')
plot_points(Point(line_SchwH.coords[0]), color='red', marker='o',)
plot_points(Point(line_SchwH.coords[-2]), color='black', marker='x',)
plot_points(Point(line_SchaufB.coords[1]), color='red', marker='o',)
plot_points(Point(line_SchaufB.coords[-1]), color='black', marker='x',)
plot_points(Point(line_BirmE_WB.coords[0]), color='red', marker='o',)
plot_points(Point(line_BirmE_WB.coords[-2]), color='black', marker='x',)
plot_points(Point(line_BirmW_WB.coords[1]), color='red', marker='o',)
plot_points(Point(line_BirmW_WB.coords[-1]), color='black', marker='x',)
plot_points(Point(line_BirmE_EB.coords[1]), color='red', marker='o',)
plot_points(Point(line_BirmE_EB.coords[-1]), color='black', marker='x',)
plot_points(Point(line_BirmW_EB.coords[1]), color='red', marker='o',)
plot_points(Point(line_BirmW_EB.coords[-2]), color='black', marker='x',)
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
        'name':          'BirmensdorferstrE_WB',
        'positive_dir':  'WB',
        'spline':        (tck_BE_WB, unew_BE_WB, cum_BE_WB),
        'total_length':  len_BE_WB,
        'line_wgs84':    line_BirmE_WB,
        'stop_line_id':  'BirmensdorferstrE_Stop',   
        'yield_line_id': 'BirmensdorferstrE_Yield',
    },
    {
        'name':          'BirmensdorferstrE_EB',
        'positive_dir':  'EB',
        'spline':        (tck_BE_EB, unew_BE_EB, cum_BE_EB),
        'total_length':  len_BE_EB,
        'line_wgs84':    line_BirmE_EB,
        'stop_line_id':  'BirmensdorferstrE_Stop',
        'yield_line_id': 'BirmensdorferstrE_Yield',
    },
    
    {
        'name':          'BirmensdorferstrW_WB',
        'positive_dir':  'WB',
        'spline':        (tck_BW_WB, unew_BW_WB, cum_BW_WB),
        'total_length':  len_BW_WB,
        'line_wgs84':    line_BirmW_WB,
        'stop_line_id':  'BirmensdorferstrW_Stop',
        'yield_line_id': 'BirmensdorferstrW_Yield',
    },
    {
        'name':          'BirmensdorferstrW_EB',
        'positive_dir':  'EB',
        'spline':        (tck_BW_EB, unew_BW_EB, cum_BW_EB),
        'total_length':  len_BW_EB,
        'line_wgs84':    line_BirmW_EB,
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

    # ── Birmensdorferstrasse East, WB carriageway ────────────────────────────
    {'seg_key': 'BirmensdorferstrE_WB', 'geometry_key': 'BirmensdorferstrE_WB',
     'direction': 'WB', 'mode': 'shared', 'bike_lane': {'w_bike': 1.5},
     'd_left': 6.0, 'd_right': 10.0,},

    # ── Birmensdorferstrasse East, EB carriageway ────────────────────────────
    {'seg_key': 'BirmensdorferstrE_EB', 'geometry_key': 'BirmensdorferstrE_EB',
     'direction': 'EB', 'mode': 'shared', 'bike_lane': {'w_bike': 1.5},
     'd_left': 6.0, 'd_right': 10.0,},

    # ── Birmensdorferstrasse West, WB carriageway ────────────────────────────
    {'seg_key': 'BirmensdorferstrW_WB', 'geometry_key': 'BirmensdorferstrW_WB',
     'direction': 'WB', 'mode': 'shared', 'bike_lane': {'w_bike': 1.5},
     'd_left': 7.0, 'd_right': 13.0,},

    # ── Birmensdorferstrasse West, EB carriageway ────────────────────────────
    {'seg_key': 'BirmensdorferstrW_EB', 'geometry_key': 'BirmensdorferstrW_EB',
     'direction': 'EB', 'mode': 'shared', 'bike_lane': {'w_bike': 1.5},
     'd_left': 7.0, 'd_right': 13.0,},
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
        
        {'geom_key': 'BirmensdorferstrW_EB',     's_change_key': 's_change',
         'pos_seg_key': 'BirmensdorferstrW_EB',     'opp_seg_key': 'BirmensdorferstrW_WB',
         'approach_seg_key': 'BirmensdorferstrW_EB'},
        {'geom_key': 'BirmensdorferstrW_WB',     's_change_key': 's_change',
         'pos_seg_key': 'BirmensdorferstrW_WB',     'opp_seg_key': 'BirmensdorferstrW_EB',
         'approach_seg_key': 'BirmensdorferstrW_EB'},
        
        {'geom_key': 'BirmensdorferstrE_EB',     's_change_key': 's_change',
         'pos_seg_key': 'BirmensdorferstrE_EB',     'opp_seg_key': 'BirmensdorferstrE_WB',
         'approach_seg_key': 'BirmensdorferstrE_WB'},
        {'geom_key': 'BirmensdorferstrE_WB',     's_change_key': 's_change',
         'pos_seg_key': 'BirmensdorferstrE_WB',     'opp_seg_key': 'BirmensdorferstrE_EB',
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