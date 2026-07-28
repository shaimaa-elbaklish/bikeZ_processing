"""
TITLE OF PAPAER
-------------------------------------------
Authors:        Shaimaa El-Baklish
Organization:   ETH Zürich, Switzerland, IVT - Institute for Transportation Planning and Systems
Development:    2025
Submitted to:   JOURNAL
-------------------------------------------

maps_june_D3.py
-------------------------------------
Site definition — Gessnerbrucke / Kasernenstrasse / Lagerstrasse
Zürich, Switzerland — June 2025 campaign (D3, E location)
 
Two intersections:
  MainInt  — 4-way: Gessnerbrucke × Kasernenstrasse × Lagerstrasse
 
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
date = BikeZ_Config.avail_dates[0]
campaign = f"Zurich_2025{date[5:7]}" # June or September
mode = BikeZ_Config.avail_modes[0] # Bike
data_root = BikeZ_Config.data_root[campaign][mode]

intersection, code = BikeZ_Config.avail_intersections[date][3]
timeslot = BikeZ_Config.avail_timeslots[date][(intersection, code)][0] # 'AM1'

XY_2056_Bounds = BikeZ_Config.XY_2056_Bounds[date][(intersection, code)]
X_2056_offset = XY_2056_Bounds[0][0]
Y_2056_offset = XY_2056_Bounds[1][0]

# Site constants
kml_path      = '../maps/from_swisstopo/June_D3.kml'
save_path     = f'../data/registry_{date}_{intersection}_{code}.pkl'
max_chain_len = 3


# Share Link: https://s.geo.admin.ch/m7vfc4w337ds
# Edit Link: https://s.geo.admin.ch/9m04oebq3knc

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
        ['Kasernenstrasse', 'Lagerstrasse', 'Gessnerbrücke', 'Stadttunnel']
    )
]
gdf = gdf[
    (gdf.geometry.type == 'LineString') &
    (gdf['highway'].isin(road_types))
]
gdf['geometry'] = gdf.geometry.intersection(bbox_geom)
gdf = gdf[~gdf.is_empty]

fig, ax = plt.subplots(1, 1)
gdf.plot(ax=ax, column='name', legend=True)
# sys.exit(1)

print("Loading SwissTopo KML...")
gdf_swisstopo = gpd.read_file(kml_path, driver='KML')

# STEP 1: fit splines  (geometry sourcing, customise per road as needed)
print("Fitting Lagerstr spline...")
tck_L, unew_L, cum_L, len_L = fit_spline_from_osmnx(
    gdf, 'Lagerstrasse', x_offset=X_2056_offset, y_offset=Y_2056_offset
)
line_L = merge_osmnx_edges(gdf, 'Lagerstrasse')

print("Fitting Kasernenstr spline...")
gdf_kasern = gdf[gdf['name'] == 'Kasernenstrasse']
gdf_kasern = gdf_kasern[gdf_kasern.geometry.type == "LineString"]
KASERN_SB_IDS = [146977154, 1162383511, 25957947, 217451773, 536842076, 536842077]
KASERN_NB_IDS = [378128430]

from shapely.geometry import LineString, MultiLineString
from shapely.ops import linemerge, snap
def merge_edges_by_ids(gdf, osmids):
    """Merge specific OSMnx edges by osmid into a single ordered LineString."""
    rows = gdf[gdf.index.get_level_values('id').isin(osmids)]
    merged = linemerge(MultiLineString(list(rows.geometry)))
    if isinstance(merged, MultiLineString):
        raise ValueError(f"Edges did not merge into a single LineString — "
                         f"check connectivity of osmids: {osmids}")
    return merged

# Raw Shapely LineStrings (equivalent to line_R, line_Z in D1)
line_KSB = merge_edges_by_ids(gdf_kasern, KASERN_SB_IDS)
KasernenstrSB_Stop = merge_osmnx_edges(gdf, 'Lagerstrasse')
line_KSB_south = cut_line_at_stop(line_KSB, KasernenstrSB_Stop, choose='last', plotting=False)
line_KSB_south = densify_linestring(line_KSB_south, num_segments=20)
line_KSB_south = LineString(list(line_KSB_south.coords)[::-1])
line_KSB_north = cut_line_at_stop(line_KSB, KasernenstrSB_Stop, choose='first', plotting=False)
line_KSB_north = LineString(list(line_KSB_north.coords)[::-1])
line_KSB_north = densify_linestring(line_KSB_north, num_segments=10)


line_KNB_south = merge_edges_by_ids(gdf_kasern, KASERN_NB_IDS)
line_KNB_south = densify_linestring(line_KNB_south, num_segments=20)

# Spline fits
tck_KSB_s, unew_KSB_s, cum_KSB_s, len_KSB_s = fit_spline_from_shapely(
    line_KSB_south, x_offset=X_2056_offset, y_offset=Y_2056_offset
)
tck_KSB_n, unew_KSB_n, cum_KSB_n, len_KSB_n = fit_spline_from_shapely(
    line_KSB_north, x_offset=X_2056_offset, y_offset=Y_2056_offset
)
tck_KNB_s, unew_KNB_s, cum_KNB_s, len_KNB_s = fit_spline_from_shapely(
    line_KNB_south, x_offset=X_2056_offset, y_offset=Y_2056_offset, smoothing=0
)

print("Fitting Gessnerbr spline...")
tck_GEB, unew_GEB, cum_GEB, len_GEB = fit_spline_from_osmnx(
    gdf, 'Gessnerbrücke', x_offset=X_2056_offset, y_offset=Y_2056_offset
)
line_GEB = merge_osmnx_edges(gdf, 'Gessnerbrücke')


print("Fitting Gessnerbr sidewalk (WB) spline from SwissTopo...")
line_GWB = gdf_swisstopo[gdf_swisstopo['Description'] == 'Gessnerbr_WB'].geometry.item()
tck_GWB, unew_GWB, cum_GWB, len_GWB = fit_spline_from_shapely(
    line_GWB, x_offset=X_2056_offset, y_offset=Y_2056_offset
)
print(f"  Gessnerbr sidewalk length: {len_GWB:.1f} m")
print("Gessnerbr (WB) start:", line_GWB.coords[0], "end:", line_GWB.coords[-1])

print("Fitting KasernenstrN NB continuation spline from SwissTopo...")
line_KNB_st = gdf_swisstopo[gdf_swisstopo['Description'] == 'KasernenstrN_NB'].geometry.item()
# Reverse the SwissTopo line to flow NB (south→north)
line_KNB_st_reversed = LineString(list(line_KNB_st.coords)[::-1])
STADTTUNNEL_IDS = [824512237, 526215130]
line_stadttunnel = merge_edges_by_ids(gdf, STADTTUNNEL_IDS)

line_KNB_st_snapped = snap(line_KNB_st_reversed, line_stadttunnel, tolerance=0.0001)
line_KNB_full = linemerge(MultiLineString([
    line_KNB_st_snapped,
    line_stadttunnel
]))
if isinstance(line_KNB_full, MultiLineString):
    # Try manual join by forcing endpoint connection
    if line_KNB_st_reversed.has_z:
        coords_combined = [(x, y) for x, y, *rest in line_KNB_st_reversed.coords] + \
            list(line_stadttunnel.coords)
    else:
        coords_combined = list(line_KNB_st_reversed.coords) + list(line_stadttunnel.coords)
    line_KNB_full = LineString(coords_combined)
line_KNB_full = LineString(list(line_KNB_full.coords)[::-1])
line_KNB_full = densify_linestring(line_KNB_full, num_segments=20)
line_KNB_full = LineString(list(line_KNB_full.coords)[::-1])

tck_KNB_n, unew_KNB_n, cum_KNB_n, len_KNB_n = fit_spline_from_shapely(
    line_KNB_full, x_offset=X_2056_offset, y_offset=Y_2056_offset
)
line_KNB_n = line_KNB_full
print(f"  KasernenstrN length: {len_KNB_n:.1f} m")


fig, ax = plt.subplots(1, 1, figsize=(6, 5))
# gdf.plot(ax=ax, column='name', legend=True)
# KasernenstrN — NB (positive) and SB (opposite)
plot_line(line_KNB_n, ax=ax, add_points=False, color='tab:blue', label='KasernenstrN_NB')
plot_points(Point(line_KNB_n.coords[0]),    color='red',   marker='o',)
plot_points(Point(line_KNB_n.coords[-1]),   color='black', marker='x',)
plot_line(line_KSB_north, ax=ax, add_points=False, color='tab:orange', label='KasernenstrN_SB')
plot_points(Point(line_KSB_north.coords[1]),  color='red',   marker='o',)
plot_points(Point(line_KSB_north.coords[-1]), color='black', marker='x',)


# KasernenstrS — NB (positive) and SB (opposite)
plot_line(line_KNB_south, ax=ax, add_points=False, color='tab:green', label='KasernenstrS_NB')
plot_points(Point(line_KNB_south.coords[1]),  color='red',   marker='o',)
plot_points(Point(line_KNB_south.coords[-1]), color='black', marker='x',)
plot_line(line_KSB_south, ax=ax, add_points=False, color='tab:red', label='KasernenstrS_SB')
plot_points(Point(line_KSB_south.coords[1]),  color='red',   marker='o',)
plot_points(Point(line_KSB_south.coords[-2]), color='black', marker='x',)

# Lagerstr — EB (positive), WB uses same spline
plot_line(line_L, ax=ax, add_points=False, color='tab:purple', label='Lagerstr')
plot_points(Point(line_L.coords[0]),   color='red',   marker='o',)
plot_points(Point(line_L.coords[-2]),  color='black', marker='x',)

# Gessnerbr — EB (positive) and WB (opposite/sidewalk)
plot_line(line_GEB, ax=ax, add_points=False, color='tab:brown', label='Gessnerbr_EB')
plot_points(Point(line_GEB.coords[1]),  color='red',   marker='o',)
plot_points(Point(line_GEB.coords[-1]), color='black', marker='x',)
plot_line(line_GWB, ax=ax, add_points=False, color='tab:pink', label='Gessnerbr_WB')
plot_points(Point(line_GWB.coords[0]),  color='red',   marker='o',)
plot_points(Point(line_GWB.coords[-1]), color='black', marker='x',)

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
        'name':          'KasernenstrN_NB',
        'positive_dir':  'NB',
        'spline':        (tck_KNB_n, unew_KNB_n, cum_KNB_n),
        'total_length':  len_KNB_n,
        'line_wgs84':    line_KNB_n,
        'stop_line_id':  'KasernenstrN_Stop',
        'yield_line_id': 'KasernenstrN_Yield',
    },
    {
        'name':          'KasernenstrN_SB',
        'positive_dir':  'NB',
        'spline':        (tck_KSB_n, unew_KSB_n, cum_KSB_n),
        'total_length':  len_KSB_n,
        'line_wgs84':    line_KSB_north,
        'stop_line_id':  'KasernenstrN_Stop',
        'yield_line_id': 'KasernenstrN_Yield',
    },
    {
        'name':          'KasernenstrS_NB',
        'positive_dir':  'NB',
        'spline':        (tck_KNB_s, unew_KNB_s, cum_KNB_s),
        'total_length':  len_KNB_s,
        'line_wgs84':    line_KNB_south,
        'stop_line_id':  'KasernenstrS_Stop',
        'yield_line_id': 'KasernenstrS_Yield',
    },
    {
        'name':          'KasernenstrS_SB',
        'positive_dir':  'NB',
        'spline':        (tck_KSB_s, unew_KSB_s, cum_KSB_s),
        'total_length':  len_KSB_s,
        'line_wgs84':    line_KSB_south,
        'stop_line_id':  'KasernenstrS_Stop',
        'yield_line_id': 'KasernenstrS_Yield',
    },
    {
        'name':          'Lagerstr',
        'positive_dir':  'EB',
        'spline':        (tck_L, unew_L, cum_L),
        'total_length':  len_L,
        'line_wgs84':    line_L,
        'stop_line_id':  'Lagerstr_Stop',
        'yield_line_id': 'Lagerstr_Yield',
    },
    # --- Done ---
    {
        'name':          'Gessnerbr_EB',
        'positive_dir':  'EB',
        'spline':        (tck_GEB, unew_GEB, cum_GEB),
        'total_length':  len_GEB,
        'line_wgs84':    line_GEB,
        'stop_line_id':  'Gessnerbr_Stop',
        'yield_line_id': 'Gessnerbr_Yield',
    },
    {
        'name':          'Gessnerbr_WB',
        'positive_dir':  'WB',
        'spline':        (tck_GWB, unew_GWB, cum_GWB),
        'total_length':  len_GWB,
        'line_wgs84':    line_GWB,
        'stop_line_id':  'Gessnerbr_Stop',
        'yield_line_id': 'Gessnerbr_Yield',
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

SEG_DEFS = [   
    # ── Kasernenstrasse North ────────────────────────────────────────────────
    {'seg_key': 'KasernenstrN_NB', 'geometry_key': 'KasernenstrN_NB',
     'direction': 'NB', 'mode': 'bike', 'bike_lane': {'w_bike': 2.75},
     'd_left': 9.0, 'd_right': 8.0},
    {'seg_key': 'KasernenstrN_SB', 'geometry_key': 'KasernenstrN_SB',
     'direction': 'SB', 'mode': 'shared', 'bike_lane': {'w_bike': 1.0},
     'd_left': 5.0, 'd_right': 14.0},
    
    # ── Kasernenstrasse South ────────────────────────────────────────────────
    {'seg_key': 'KasernenstrS_NB', 'geometry_key': 'KasernenstrS_NB',
     'direction': 'NB', 'mode': 'shared', 'bike_lane': {'w_bike': 3.5},
     'd_left': 6.0, 'd_right': 10.0},
    {'seg_key': 'KasernenstrS_SB', 'geometry_key': 'KasernenstrS_SB',
     'direction': 'SB', 'mode': 'shared', 'bike_lane': {'w_bike': 1.5},
     'd_left': 5.0, 'd_right': 8.0},
    
    # ── Lagerstrasse (shared centerline) ─────────────────────────────────────
    {'seg_key': 'Lagerstr_EB', 'geometry_key': 'Lagerstr',
     'direction': 'EB', 'mode': 'shared', 'bike_lane': {'w_bike': 1.5},
     'd_left': 1.5, 'd_right': 15.0},
    {'seg_key': 'Lagerstr_WB', 'geometry_key': 'Lagerstr',
     'direction': 'WB', 'mode': 'shared', 'bike_lane': {'w_bike': 1.5},
     'd_left': 1.5, 'd_right': 15.0},
    
    # ── Gessnerbrücke ─────────────────────────────────────────────────────────
    {'seg_key': 'Gessnerbr_EB', 'geometry_key': 'Gessnerbr_EB',
     'direction': 'EB', 'mode': 'shared', 'bike_lane': {'w_bike': 2.5},
     'd_left': 6.0, 'd_right': 10.0},
    {'seg_key': 'Gessnerbr_WB', 'geometry_key': 'Gessnerbr_WB',
     'direction': 'WB', 'mode': 'bike', 'bike_lane': {'w_bike': 4.5},
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

# =============================================================================
# PHASE 3: build_turns

print("\n--- Phase 3: build turn splines ---")

TURN_DEFS = [
    # ── From north (KasernenstrN_SB approaching southbound) ───────────────────
    {'approach_seg': 'KasernenstrN_SB', 'departure_seg': 'KasernenstrS_SB',
     'd_left': 15.0, 'd_right': 15.0},   # straight south
    {'approach_seg': 'KasernenstrN_SB', 'departure_seg': 'Lagerstr_WB',
     'd_left': 15.0, 'd_right': 15.0},
    {'approach_seg': 'KasernenstrN_SB', 'departure_seg': 'Gessnerbr_EB',
     'd_left': 15.0, 'd_right': 15.0},

    # ── From south (KasernenstrS_NB approaching northbound) ───────────────────
    {'approach_seg': 'KasernenstrS_NB', 'departure_seg': 'KasernenstrN_NB',
     'd_left': 15.0, 'd_right': 15.0},   # straight north
    {'approach_seg': 'KasernenstrS_NB', 'departure_seg': 'Lagerstr_WB',
     'd_left': 15.0, 'd_right': 15.0},
    {'approach_seg': 'KasernenstrS_NB', 'departure_seg': 'Gessnerbr_EB',
     'd_left': 15.0, 'd_right': 15.0},

    # ── From west (Lagerstr_EB approaching eastbound) ─────────────────────────
    {'approach_seg': 'Lagerstr_EB', 'departure_seg': 'Gessnerbr_EB',
     'd_left': 15.0, 'd_right': 15.0},   # straight east
    {'approach_seg': 'Lagerstr_EB', 'departure_seg': 'KasernenstrN_NB',
     'd_left': 15.0, 'd_right': 15.0},
    {'approach_seg': 'Lagerstr_EB', 'departure_seg': 'KasernenstrS_SB',
     'd_left': 15.0, 'd_right': 15.0},

    # ── From east (Gessnerbr_WB approaching westbound) ────────────────────────
    {'approach_seg': 'Gessnerbr_WB', 'departure_seg': 'Lagerstr_WB',
     'd_left': 15.0, 'd_right': 15.0},   # straight west
    {'approach_seg': 'Gessnerbr_WB', 'departure_seg': 'KasernenstrN_NB',
     'd_left': 15.0, 'd_right': 15.0},
    {'approach_seg': 'Gessnerbr_WB', 'departure_seg': 'KasernenstrS_SB',
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
        {'geom_key': 'Lagerstr', 's_change_key': 's_change',
         'pos_seg_key': 'Lagerstr_EB', 'opp_seg_key': 'Lagerstr_WB',
         'approach_seg_key': 'Lagerstr_EB'},
        {'geom_key': 'Gessnerbr_WB',    's_change_key': 's_change',
         'pos_seg_key': 'Gessnerbr_WB',  'opp_seg_key': 'Gessnerbr_EB',
         'approach_seg_key': 'Gessnerbr_WB'},
        {'geom_key': 'Gessnerbr_EB',    's_change_key': 's_change',
         'pos_seg_key': 'Gessnerbr_EB',  'opp_seg_key': 'Gessnerbr_WB',
         'approach_seg_key': 'Gessnerbr_WB'},
        
        {'geom_key': 'KasernenstrS_SB', 's_change_key': 's_change',
         'pos_seg_key': 'KasernenstrS_NB', 'opp_seg_key': 'KasernenstrS_SB',
         'approach_seg_key': 'KasernenstrS_NB'},
        {'geom_key': 'KasernenstrS_NB', 's_change_key': 's_change',
         'pos_seg_key': 'KasernenstrS_NB', 'opp_seg_key': 'KasernenstrS_SB',
         'approach_seg_key': 'KasernenstrS_NB'},
        
        {'geom_key': 'KasernenstrN_NB', 's_change_key': 's_change',
         'pos_seg_key': 'KasernenstrN_NB', 'opp_seg_key': 'KasernenstrN_SB',
         'approach_seg_key': 'KasernenstrN_SB'},
        {'geom_key': 'KasernenstrN_SB', 's_change_key': 's_change',
         'pos_seg_key': 'KasernenstrN_NB', 'opp_seg_key': 'KasernenstrN_SB',
         'approach_seg_key': 'KasernenstrN_SB'},
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
    # ── From north (KasernenstrN_SB) ───────────────────────────────────────────
    {'key': 'KasernenstrN_SB_2_KasernenstrS_SB',
     'sequence': [('KasernenstrN_SB',                              'approach'),
                  ('turn_KasernenstrN_SB_2_KasernenstrS_SB',        'turn'),
                  ('KasernenstrS_SB',                               'departure')]},
    {'key': 'KasernenstrN_SB_2_Lagerstr_WB',
     'sequence': [('KasernenstrN_SB',                              'approach'),
                  ('turn_KasernenstrN_SB_2_Lagerstr_WB',            'turn'),
                  ('Lagerstr_WB',                                   'departure')]},
    {'key': 'KasernenstrN_SB_2_Gessnerbr_EB',
     'sequence': [('KasernenstrN_SB',                              'approach'),
                  ('turn_KasernenstrN_SB_2_Gessnerbr_EB',           'turn'),
                  ('Gessnerbr_EB',                                  'departure')]},

    # ── From south (KasernenstrS_NB) ───────────────────────────────────────────
    {'key': 'KasernenstrS_NB_2_KasernenstrN_NB',
     'sequence': [('KasernenstrS_NB',                              'approach'),
                  ('turn_KasernenstrS_NB_2_KasernenstrN_NB',        'turn'),
                  ('KasernenstrN_NB',                               'departure')]},
    {'key': 'KasernenstrS_NB_2_Lagerstr_WB',
     'sequence': [('KasernenstrS_NB',                              'approach'),
                  ('turn_KasernenstrS_NB_2_Lagerstr_WB',            'turn'),
                  ('Lagerstr_WB',                                   'departure')]},
    {'key': 'KasernenstrS_NB_2_Gessnerbr_EB',
     'sequence': [('KasernenstrS_NB',                              'approach'),
                  ('turn_KasernenstrS_NB_2_Gessnerbr_EB',           'turn'),
                  ('Gessnerbr_EB',                                  'departure')]},

    # ── From west (Lagerstr_EB) ─────────────────────────────────────────────────
    {'key': 'Lagerstr_EB_2_Gessnerbr_EB',
     'sequence': [('Lagerstr_EB',                                  'approach'),
                  ('turn_Lagerstr_EB_2_Gessnerbr_EB',               'turn'),
                  ('Gessnerbr_EB',                                  'departure')]},
    {'key': 'Lagerstr_EB_2_KasernenstrN_NB',
     'sequence': [('Lagerstr_EB',                                  'approach'),
                  ('turn_Lagerstr_EB_2_KasernenstrN_NB',            'turn'),
                  ('KasernenstrN_NB',                               'departure')]},
    {'key': 'Lagerstr_EB_2_KasernenstrS_SB',
     'sequence': [('Lagerstr_EB',                                  'approach'),
                  ('turn_Lagerstr_EB_2_KasernenstrS_SB',            'turn'),
                  ('KasernenstrS_SB',                               'departure')]},

    # ── From east (Gessnerbr_WB) ────────────────────────────────────────────────
    {'key': 'Gessnerbr_WB_2_Lagerstr_WB',
     'sequence': [('Gessnerbr_WB',                                 'approach'),
                  ('turn_Gessnerbr_WB_2_Lagerstr_WB',               'turn'),
                  ('Lagerstr_WB',                                   'departure')]},
    {'key': 'Gessnerbr_WB_2_KasernenstrN_NB',
     'sequence': [('Gessnerbr_WB',                                 'approach'),
                  ('turn_Gessnerbr_WB_2_KasernenstrN_NB',           'turn'),
                  ('KasernenstrN_NB',                               'departure')]},
    {'key': 'Gessnerbr_WB_2_KasernenstrS_SB',
     'sequence': [('Gessnerbr_WB',                                 'approach'),
                  ('turn_Gessnerbr_WB_2_KasernenstrS_SB',           'turn'),
                  ('KasernenstrS_SB',                               'departure')]},
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
