"""
TITLE OF PAPAER
-------------------------------------------
Authors:        Shaimaa El-Baklish
Organization:   ETH Zürich, Switzerland, IVT - Institute for Transportation Planning and Systems
Development:    2025
Submitted to:   JOURNAL
-------------------------------------------
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

from _constants import BikeZ_Config
from tools_coordinate_transform import cut_line_at_stop

from tools_map_visualization import create_registry_map


from tools_osmnx import merge_osmnx_edges
from tools_osmnx import fit_spline_from_osmnx
from tools_osmnx import fit_spline_from_shapely
from tools_infrastructure_geometry import get_s_domain
from tools_infrastructure_geometry import build_segment_registry
from tools_infrastructure_geometry import add_bike_lane_boundaries
from tools_infrastructure_geometry import build_all_turns
from tools_infrastructure_geometry import build_movement_registry
from tools_infrastructure_geometry import restrict_segment_roles
from tools_infrastructure_geometry import rebuild_validity_polygons
from tools_infrastructure_geometry import serialize_registry
# from tools_infrastructure_geometry import build_turn_spline, sample_spline_near_boundary
from tools_plotting import plot_geometry_store
# from tools_plotting import plot_turn_debug
from tools_plotting import plot_turn_splines

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

# #############################################################################
# MAIN: Load Data (Trajectories)
# #############################################################################
filename = f"trajectories_bikes_{date}_{intersection}_{timeslot}_{code}-1-ekf.csv"
df = pd.read_csv(data_root + f"{date}/{intersection}/{filename}")
df = df.dropna()

# Convert from EPSG:2056 to EPSG:4326 (lat, lon)
df['x_act_ekf'] = df['x_ekf'] + X_2056_offset
df['y_act_ekf'] = df['y_ekf'] + Y_2056_offset
transformer = Transformer.from_crs("EPSG:2056", "EPSG:4326", always_xy=True)
df["lon_ekf"], df["lat_ekf"] = transformer.transform(df["x_act_ekf"].values, df["y_act_ekf"].values)

center_lat, center_lon = df["lat_ekf"].mean(), df["lon_ekf"].mean()

# #############################################################################
# MAIN: Create Bounding Box for location
# #############################################################################
lonlat_bounds = transformer.transform(np.asarray(XY_2056_Bounds[0]) + np.asarray([-50, 50]), 
                                      np.asarray(XY_2056_Bounds[1]) + np.asarray([-50, 50]))
bbox = box(lonlat_bounds[0][0], lonlat_bounds[1][0], 
           lonlat_bounds[0][1], lonlat_bounds[1][1])
lonlat_bounds = transformer.transform(XY_2056_Bounds[0], XY_2056_Bounds[1])


# #############################################################################
# MAIN: Extract Available Centerlines from OSMNX
# #############################################################################
# Define your area
place = "Zürich, Switzerland"
tags = {"highway": True} # Download all features with highway tag
gdf_main = ox.features.features_from_place(place, tags=tags)
main_road_types = ['primary', 'secondary', 'tertiary', 'residential', 'unclassified', "cycleway"]
                   # "cycleway", "path", "service", "living_street"]
bikeable = ["yes", "designated", "permissive"]

gdf = gdf_main[gdf_main['name'].isin(['Zollstrasse', 'Langstrasse', 'Röntgenstrasse', 'Mattengasse'])]
gdf = gdf[(gdf.geometry.type == "LineString") & (gdf['highway'].isin(main_road_types))]
gdf["geometry"] = gdf.geometry.intersection(bbox)
gdf = gdf[~gdf.is_empty] # Drop empty geometries
gdf.plot(column='name', legend=True)

# #############################################################################
# MAIN: Extract other features from SwissTopo
# #############################################################################
# Link to edit drawing: https://s.geo.admin.ch/0damnomql60l
# Share Link: https://s.geo.admin.ch/4a4byyeagawv
# GIS Share Link: https://geo.zh.ch/s/d4e68882-4d3a-4946-b208-6eb95c78f4da

kml_path = "../maps/from_swisstopo/June_D1.kml"
gdf_swisstopo = gpd.read_file(kml_path, driver='KML')

row = gdf_swisstopo[gdf_swisstopo['Description'] == 'Intersection_Area'].copy()
intersection_polygon = row.geometry.item()

# #############################################################################
# MAIN: Assemble geometry_store
# #############################################################################
print("Fitting Roentgenstr spline...")
tck_R, unew_R, cum_R, len_R = fit_spline_from_osmnx(
    gdf, 'Röntgenstrasse', x_offset=X_2056_offset, y_offset=Y_2056_offset
)
line_R = merge_osmnx_edges(gdf, 'Röntgenstrasse')

print("Fitting Zollstr spline...")
tck_Z, unew_Z, cum_Z, len_Z = fit_spline_from_osmnx(
    gdf, 'Zollstrasse', x_offset=X_2056_offset, y_offset=Y_2056_offset
)
line_Z = merge_osmnx_edges(gdf, 'Zollstrasse')

print("Fitting Langstrasse splines (North + South)...")
## North branch
lang_line   = merge_osmnx_edges(gdf, 'Langstrasse')
lang_stop = merge_osmnx_edges(gdf, 'Zollstrasse')
lang_north = cut_line_at_stop(lang_line, lang_stop, choose='last', plotting=False)
## South branch
lang_line   = merge_osmnx_edges(gdf, 'Langstrasse')
lang_south = cut_line_at_stop(lang_line, lang_stop, choose='first', plotting=False)
tck_LN, unew_LN, cum_LN, len_LN = fit_spline_from_shapely(
    lang_north, x_offset=X_2056_offset, y_offset=Y_2056_offset
)
tck_LS, unew_LS, cum_LS, len_LS = fit_spline_from_shapely(
    lang_south, x_offset=X_2056_offset, y_offset=Y_2056_offset
)

# print("Fitting Matteng spline...")
# tck_M, unew_M, cum_M, len_M = fit_spline_from_osmnx(
#     gdf, 'Mattengasse', x_offset=X_2056_offset, y_offset=Y_2056_offset
# )
# line_M = merge_osmnx_edges(gdf, 'Mattengasse')

print(f"  LangstrN length: {len_LN:.1f} m")
print(f"  LangstrS length: {len_LS:.1f} m")
print(f"  Roentgenstr length: {len_R:.1f} m")
print(f"  Zollstr length: {len_Z:.1f} m")
# print(f"  Matteng length: {len_M:.1f} m")

# Positive parameterization direction per geometry_key
# (the cardinal direction that aligns with increasing spline parameter t)
POSITIVE_DIR = {
    'Roentgenstr': 'WB',
    'Zollstr':     'EB',
    'LangstrN':    'NB',
    'LangstrS':    'NB',
    # 'Matteng':     'SB',
}

geometry_store = {
    'x_offset': X_2056_offset,
    'y_offset': Y_2056_offset, 
    'Roentgenstr': {
        'spline':        (tck_R, unew_R, cum_R),
        'positive_dir':  POSITIVE_DIR['Roentgenstr'],
        'total_length':  len_R,
        'line_wgs84':    line_R # raw Shapely LineString from OSMnx
    },
    'Zollstr': {
        'spline':        (tck_Z, unew_Z, cum_Z),
        'positive_dir':  POSITIVE_DIR['Zollstr'],
        'total_length':  len_Z,
        'line_wgs84':    line_Z # raw Shapely LineString from OSMnx
    },
    'LangstrN': {
        'spline':        (tck_LN, unew_LN, cum_LN),
        'positive_dir':  POSITIVE_DIR['LangstrN'],
        'total_length':  len_LN,
        'line_wgs84':    lang_north # raw Shapely LineString from OSMnx
    },
    'LangstrS': {
        'spline':        (tck_LS, unew_LS, cum_LS),
        'positive_dir':  POSITIVE_DIR['LangstrS'],
        'total_length':  len_LS,
        'line_wgs84':    lang_south # raw Shapely LineString from OSMnx
    },
    # 'Matteng': {
    #     'spline':        (tck_M, unew_M, cum_M),
    #     'positive_dir':  POSITIVE_DIR['Matteng'],
    #     'total_length':  len_M,
    #     'line_wgs84':    line_M # raw Shapely LineString from OSMnx
    # },
}

print("\ngeometry_store assembled:")
for key, val in geometry_store.items():
    if key == 'x_offset' or key == 'y_offset':
        continue
    print(f"  {key}: {val['total_length']:.1f} m, positive_dir={val['positive_dir']}")

from shapely.geometry import Point
from shapely.plotting import plot_points
plot_points(Point(line_R.coords[1]), color='red', marker='o', label='Start Roentgenstr')
plot_points(Point(line_R.coords[-1]), color='black', marker='x', label='End Roentgenstr')
plot_points(Point(line_Z.coords[1]), color='red', marker='o', label='Start Zollstr')
plot_points(Point(line_Z.coords[-1]), color='black', marker='x', label='End Zollstr')
plot_points(Point(lang_south.coords[0]), color='red', marker='o', label='Start LangstrS')
plot_points(Point(lang_south.coords[-1]), color='black', marker='x', label='End LangstrS')
plot_points(Point(lang_north.coords[1]), color='red', marker='o', label='Start LangstrN')
plot_points(Point(lang_north.coords[-1]), color='black', marker='x', label='End LangstrN')
# plot_points(Point(line_M.coords[0]), color='red', marker='o', label='Start Matteng')
# plot_points(Point(line_M.coords[-1]), color='black', marker='x', label='End Matteng')
plt.title(
    'Postive Direction Validation — Phase A\n'
    'Red o marker = Start  |  Black x marker = End',
    fontsize=11
)
plt.tight_layout()

# Compute s_stop and s_yield per directed segment from stop/yield lines
geom_items = ['Roentgenstr', 'Zollstr', 'LangstrN', 'LangstrS'] #, 'Matteng']
s_boundaries = {}
for geom_key in geom_items:
    get_s_domain(geom_key, geometry_store, gdf_swisstopo)
    sb = geometry_store[geom_key]
    print(f"  {geom_key}: "
          f"s_stop={sb['s_stop']:.2f} m  "
          f"s_yield={sb['s_yield']:.2f} m  "
          f"total={sb['total_length']:.2f} m  "
          f"gap={sb['s_yield']-sb['s_stop']:.2f} m")
    
plot_geometry_store(
    geometry_store, gdf_swisstopo,
    offset_m=3.0,
    save_path= None #'../debugging/geometry_store_inspection.png'
)

# sys.exit(1)

# #############################################################################
# MAIN: Assemble segment_registry
# #############################################################################
# seg_key → {
#     type:             'lane'               # always 'lane' here; turns added later
#     geometry_key:     'Roentgenstr'        # lookup key into geometry_store
#     direction:        'EB'                 # travel direction
#     is_forward:       True/False           # direction == positive_dir
#     mode:             'car'/'bike'/'shared'
#     approach_native:  (s_min, s_max)       # spline-native arc-length
#     departure_native: (s_min, s_max)       # spline-native arc-length
#     bike_lane:        {'w_bike': 1.6,      # None if no bike lane
#                        'side':   -1}
# }


# Bike lane info per directed segment
BIKE_LANE_INFO = {
    'Roentgenstr_EB': {'w_bike': 1.6},   # shared, has dedicated stripe
    'Roentgenstr_WB': {'w_bike': 1.6},   # shared, has dedicated stripe
    'Zollstr_EB':     None,               # shared, no dedicated stripe
    'Zollstr_WB':     {'w_bike': 2.0},   # bike only — but w_bike = full width
    'LangstrN_NB':    None,               # shared, no dedicated stripe
    'LangstrN_SB':    None,               # shared, no dedicated stripe
    'LangstrS_NB':    {'w_bike': 2.75},  # shared, has dedicated stripe
    'LangstrS_SB':    {'w_bike': 2.75},  # shared, has dedicated stripe
}

DIRECTED_SEGMENTS = [
    ('Roentgenstr', 'EB'),
    ('Roentgenstr', 'WB'),
    ('Zollstr',     'EB'),
    ('Zollstr',     'WB'),
    ('LangstrN',    'NB'),
    ('LangstrN',    'SB'),
    ('LangstrS',    'NB'),
    ('LangstrS',    'SB'),
]

# Mode per directed segment
MODE = {
    'Roentgenstr_EB': 'shared',
    'Roentgenstr_WB': 'shared',
    'Zollstr_EB':     'shared',
    'Zollstr_WB':     'bike',    # bike/pedestrian only
    'LangstrN_NB':    'shared',
    'LangstrN_SB':    'shared',
    'LangstrS_NB':    'shared',
    'LangstrS_SB':    'shared',
}


# Lateral validity corridor per directed segment.
# Each value is either:
#   float  → symmetric: d_left = d_right = value
#   {'d_left': x, 'd_right': y} → asymmetric
#
# Convention:
#   d_right : offset to the RIGHT of travel direction [m]
#             → toward the expected carriageway (cyclists keep right)
#             → should cover full lane width + any bike lane
#   d_left  : offset to the LEFT of travel direction [m]
#             → GPS-noise tolerance + a small margin for centerline error
#             → kept narrow to reject cyclists on the opposite carriageway
#             → or on the sidewalk of the opposing direction
#
# These govern both:
#   Step 1: validity polygon construction (spatial pre-filter)
#   Step 2 (V3): hard lateral veto in score_segment
#
# Turns remain symmetric — cyclists can deviate widely in any direction
# through the intersection, so there is no preferred side.
D_MAX = {
    # ── Lane segments ──────────────────────────────────────────────────────
    # Roentgenstr: bidirectional shared road, EB and WB on same centerline.
    # Asymmetric to separate EB carriageway from WB sidewalk confusions.
    'Roentgenstr_EB': {'d_right': 12.0, 'd_left': 1.5},
    'Roentgenstr_WB': {'d_right': 12.0, 'd_left': 1.5},

    # Zollstr: EB is a shared road; WB is narrow bike/pedestrian path.
    'Zollstr_EB':     {'d_right': 12.0, 'd_left': 1.5},
    'Zollstr_WB':     {'d_right': 12.0, 'd_left': 1.5},

    # LangstrN: NB and SB on same centerline, physically close carriageways.
    'LangstrN_NB':    {'d_right': 8.0, 'd_left': 1.5},
    'LangstrN_SB':    {'d_right': 8.0, 'd_left': 1.5},

    # LangstrS: wider road with dedicated bike lanes on both sides.
    'LangstrS_NB':    {'d_right': 12.0, 'd_left': 3.0},
    'LangstrS_SB':    {'d_right': 12.0, 'd_left': 1.0},

    # ── Turn segments (symmetric) ──────────────────────────────────────────
    'turn_Roentgenstr_EB_2_LangstrN_NB':  15.0,
    'turn_Roentgenstr_EB_2_Zollstr_EB':   15.0,
    'turn_Roentgenstr_EB_2_LangstrS_SB':  15.0,
    'turn_Zollstr_WB_2_LangstrN_NB':      15.0,
    'turn_Zollstr_WB_2_Roentgenstr_WB':   15.0,
    'turn_Zollstr_WB_2_LangstrS_SB':      15.0,
    'turn_LangstrN_SB_2_Roentgenstr_WB':  15.0,
    'turn_LangstrN_SB_2_Zollstr_EB':      15.0,
    'turn_LangstrN_SB_2_LangstrS_SB':     15.0,
    'turn_LangstrS_NB_2_Roentgenstr_WB':  15.0,
    'turn_LangstrS_NB_2_Zollstr_EB':      15.0,
    'turn_LangstrS_NB_2_LangstrN_NB':     15.0,
}

segment_registry = build_segment_registry(
    geometry_store, DIRECTED_SEGMENTS, BIKE_LANE_INFO, MODE,
    d_max_map=D_MAX
)

# Print summary
print("\nsegment_registry:")
for seg_key, entry in segment_registry.items():
    app  = entry['approach_native']
    dep  = entry['departure_native']
    fwd  = 'fwd' if entry['is_forward'] else 'rev'
    bl   = f"w={entry['bike_lane']['w_bike']}m" \
           if entry['bike_lane'] else 'None'
    d_left  = entry.get('d_left',  '?')
    d_right = entry.get('d_right', '?')
    poly    = entry.get('validity_polygon')
    poly_ok = '✓' if (poly is not None and not poly.is_empty) else '✗'
    app_str = f"[{app[0]:.1f}, {app[1]:.1f}]" if app else 'None'
    dep_str = f"[{dep[0]:.1f}, {dep[1]:.1f}]" if dep else 'None'
    print(f"  {seg_key} ({fwd}, mode={entry['mode']}, bike_lane={bl}, "
          f"d_left={d_left}m d_right={d_right}m, poly={poly_ok}): "
          f"approach s∈{app_str}  departure s∈{dep_str}")


# Project bike lane boundaries → d_boundary(s) splines
print("\nProjecting bike lane boundaries...")
gdf_bike_boundaries = gdf_swisstopo[
    gdf_swisstopo['Description'].str.endswith(('_NB','_SB','_EB','_WB'))
].copy()

add_bike_lane_boundaries(segment_registry, geometry_store, gdf_bike_boundaries)


# #############################################################################
# MAIN: Build turning movement clothoids
# #############################################################################
# Enumerate all turning movements as (approach_seg, departure_seg)
TURNING_MOVEMENTS = [
    # From Roentgenstr (approaching EB from west)
    ('Roentgenstr_EB', 'LangstrN_NB'),    # right turn → north
    ('Roentgenstr_EB', 'Zollstr_EB'),     # straight  → east
    ('Roentgenstr_EB', 'LangstrS_SB'),    # left turn → south

    # From Zollstr (approaching WB from east), bikes only do that here!
    ('Zollstr_WB',     'LangstrN_NB'),    # left turn → north
    ('Zollstr_WB',     'Roentgenstr_WB'), # straight  → west
    ('Zollstr_WB',     'LangstrS_SB'),    # right turn → south

    # From LangstrN (approaching SB from north)
    ('LangstrN_SB',    'Roentgenstr_WB'), # left turn → west
    ('LangstrN_SB',    'Zollstr_EB'),     # right turn → east
    ('LangstrN_SB',    'LangstrS_SB'),    # straight  → south

    # From LangstrS (approaching NB from south)
    ('LangstrS_NB',    'Roentgenstr_WB'), # right turn → west
    ('LangstrS_NB',    'Zollstr_EB'),     # left turn → east
    ('LangstrS_NB',    'LangstrN_NB'),    # straight  → north
]

# (
#  tck, unew, cum_dist, total_length, 
#  connector_clean, method
# ) = build_turn_spline(
#     'LangstrS_NB', 'Zollstr_EB',
#     segment_registry, geometry_store,
#     n_pts=10, n_connector=100,
#     angle_threshold_deg=5,
#     verbose=True
# )
    
# pts_approach  = sample_spline_near_boundary(
#     'LangstrS_NB',  'approach',  segment_registry, geometry_store, 10
# )
# pts_departure = sample_spline_near_boundary(
#     'Zollstr_EB', 'departure', segment_registry, geometry_store, 10
# )
# plot_turn_debug(
#     'LangstrS_NB', 'Zollstr_EB',
#     pts_approach, pts_departure,
#     connector_clean, method,
#     geometry_store, segment_registry,
#     gdf_swisstopo, save_path=None
# )

# print(f"    connector shape: {connector_clean.shape}")
# print(f"    connector start: {connector_clean[0]}")
# print(f"    connector end:   {connector_clean[-1]}")
# print(f"    connector unique pts: {len(np.unique(connector_clean, axis=0))}")

# # Check for duplicates
# diff = np.diff(connector_clean, axis=0)
# dist = np.linalg.norm(diff, axis=1)
# print(f"    min segment length: {dist.min():.6f} m")
# print(f"    max segment length: {dist.max():.6f} m")
# print(f"    zero-length segments: {(dist < 1e-6).sum()}")
# sys.exit(1)

print("\nA2: Building turning movement splines...")
turn_keys = build_all_turns(
    TURNING_MOVEMENTS, segment_registry, geometry_store,
    n_pts=10, n_connector=100,
    angle_threshold_deg=5,
    d_max_map=D_MAX,
    verbose=False
)
print(f"\nA2 complete: {len(turn_keys)}/{len(TURNING_MOVEMENTS)} "
      f"turning movements built")

plot_turn_splines(
    turn_keys, segment_registry, geometry_store, gdf_swisstopo,
    offset_m=3.0, turn_offset_m=2.0,
    save_path=None #'../debugging/turn_splines_validation.png'
)

# #############################################################################
# MAIN: Build and serialize movement_registry
# #############################################################################
# Each movement is a list of (segment_key, role) tuples
# role: 'approach', 'turn', or 'departure'
# Segment keys must exist in segment_registry
MOVEMENTS = [
    # From Roentgenstr EB (approaching from west)
    ('Roentgenstr_EB_2_LangstrN_NB', [
        ('Roentgenstr_EB',                        'approach'),
        ('turn_Roentgenstr_EB_2_LangstrN_NB',     'turn'),
        ('LangstrN_NB',                           'departure'),
    ]),
    ('Roentgenstr_EB_2_Zollstr_EB', [
        ('Roentgenstr_EB',                        'approach'),
        ('turn_Roentgenstr_EB_2_Zollstr_EB',      'turn'),
        ('Zollstr_EB',                            'departure'),
    ]),
    ('Roentgenstr_EB_2_LangstrS_SB', [
        ('Roentgenstr_EB',                        'approach'),
        ('turn_Roentgenstr_EB_2_LangstrS_SB',     'turn'),
        ('LangstrS_SB',                           'departure'),
    ]),

    # From Zollstr WB (approaching from east)
    ('Zollstr_WB_2_LangstrN_NB', [
        ('Zollstr_WB',                            'approach'),
        ('turn_Zollstr_WB_2_LangstrN_NB',         'turn'),
        ('LangstrN_NB',                           'departure'),
    ]),
    ('Zollstr_WB_2_Roentgenstr_WB', [
        ('Zollstr_WB',                            'approach'),
        ('turn_Zollstr_WB_2_Roentgenstr_WB',      'turn'),
        ('Roentgenstr_WB',                        'departure'),
    ]),
    ('Zollstr_WB_2_LangstrS_SB', [
        ('Zollstr_WB',                            'approach'),
        ('turn_Zollstr_WB_2_LangstrS_SB',         'turn'),
        ('LangstrS_SB',                           'departure'),
    ]),

    # From LangstrN SB (approaching from north)
    ('LangstrN_SB_2_Roentgenstr_WB', [
        ('LangstrN_SB',                           'approach'),
        ('turn_LangstrN_SB_2_Roentgenstr_WB',     'turn'),
        ('Roentgenstr_WB',                        'departure'),
    ]),
    ('LangstrN_SB_2_Zollstr_EB', [
        ('LangstrN_SB',                           'approach'),
        ('turn_LangstrN_SB_2_Zollstr_EB',         'turn'),
        ('Zollstr_EB',                            'departure'),
    ]),
    ('LangstrN_SB_2_LangstrS_SB', [
        ('LangstrN_SB',                           'approach'),
        ('turn_LangstrN_SB_2_LangstrS_SB',        'turn'),
        ('LangstrS_SB',                           'departure'),
    ]),

    # From LangstrS NB (approaching from south)
    ('LangstrS_NB_2_Roentgenstr_WB', [
        ('LangstrS_NB',                           'approach'),
        ('turn_LangstrS_NB_2_Roentgenstr_WB',     'turn'),
        ('Roentgenstr_WB',                        'departure'),
    ]),
    ('LangstrS_NB_2_Zollstr_EB', [
        ('LangstrS_NB',                           'approach'),
        ('turn_LangstrS_NB_2_Zollstr_EB',         'turn'),
        ('Zollstr_EB',                            'departure'),
    ]),
    ('LangstrS_NB_2_LangstrN_NB', [
        ('LangstrS_NB',                           'approach'),
        ('turn_LangstrS_NB_2_LangstrN_NB',        'turn'),
        ('LangstrN_NB',                           'departure'),
    ]),
]

movement_registry = build_movement_registry(MOVEMENTS, segment_registry)
# Restrict each lane segment to only the roles it plays in registered movements.
# Prevents e.g. Roentgenstr_WB approach domain competing with turn centerlines.
restrict_segment_roles(segment_registry, movement_registry)
# Rebuild validity polygons to reflect the final restricted s-domains.
# Must be called after restrict_segment_roles() since that nulls some domains.
rebuild_validity_polygons(segment_registry, geometry_store)

save_path = (f"../data/registry_{date}_{intersection}_{code}.pkl")

serialize_registry(
    geometry_store, segment_registry, movement_registry,
    max_chain_length=3,
    intersection=f'{intersection}_{code}',
    date=date,
    save_path=save_path
)

# #############################################################################
# MAIN: Create Folium map with SwissTopo base image
# #############################################################################
m = create_registry_map(
    geometry_store, segment_registry, movement_registry,
    gdf_swisstopo,
    center_lat=center_lat, center_lon=center_lon,
    save_path=f'../maps/registry_map_{date}_{intersection}.html'
)