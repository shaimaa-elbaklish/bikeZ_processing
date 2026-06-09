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
date = BikeZ_Config.avail_dates[0]
campaign = f"Zurich_2025{date[5:7]}" # June or September
mode = BikeZ_Config.avail_modes[0] # Bike
data_root = BikeZ_Config.data_root[campaign][mode]

intersection, code = BikeZ_Config.avail_intersections[date][3]
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
lonlat_bounds = transformer.transform(np.asarray(XY_2056_Bounds[0]) + np.asarray([-15, 15]), 
                                      np.asarray(XY_2056_Bounds[1]) + np.asarray([-15, 15]))
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

gdf = gdf_main[gdf_main['name'].isin(['Lagerstrasse', 'Gessnerbrücke', 'Kasernenstrasse', 'Stadttunnel'])]
gdf = gdf[(gdf.geometry.type == "LineString") & (gdf['highway'].isin(main_road_types))]
gdf["geometry"] = gdf.geometry.intersection(bbox)
gdf = gdf[~gdf.is_empty] # Drop empty geometries
gdf.plot(column='name', legend=True)
# sys.exit(1)

# #############################################################################
# MAIN: Extract other features from SwissTopo
# #############################################################################
# Share Link: https://s.geo.admin.ch/xael6t1zicil
# Edit drawing link: https://s.geo.admin.ch/pxy0k6lay41g

kml_path = "../maps/from_swisstopo/June_D3.kml"
gdf_swisstopo = gpd.read_file(kml_path, driver='KML')

row = gdf_swisstopo[gdf_swisstopo['Description'] == 'Intersection_Area'].copy()
intersection_polygon = row.geometry.item()

# #############################################################################
# MAIN: Assemble geometry_store
# #############################################################################
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

from tools_coordinate_transform import densify_linestring
# Raw Shapely LineStrings (equivalent to line_R, line_Z in D1)
line_KSB = merge_edges_by_ids(gdf_kasern, KASERN_SB_IDS)
KasernenstrSB_Stop = merge_osmnx_edges(gdf, 'Lagerstrasse')
line_KSB_south = cut_line_at_stop(line_KSB, KasernenstrSB_Stop, choose='last', plotting=False)
line_KSB_south = densify_linestring(line_KSB_south, num_segments=20)
line_KSB_south = LineString(list(line_KSB_south.coords)[::-1])
line_KSB_north = cut_line_at_stop(line_KSB, KasernenstrSB_Stop, choose='first', plotting=False)
line_KSB_north = LineString(list(line_KSB_north.coords)[::-1])


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
    coords_combined = list(line_KNB_st_reversed.coords) + list(line_stadttunnel.coords)
    line_KNB_full = LineString(coords_combined)
line_KNB_full = LineString(list(line_KNB_full.coords)[::-1])

tck_KNB_n, unew_KNB_n, cum_KNB_n, len_KNB_n = fit_spline_from_shapely(
    line_KNB_full, x_offset=X_2056_offset, y_offset=Y_2056_offset
)
line_KNB_n = line_KNB_full
print(f"  KasernenstrN length: {len_KNB_n:.1f} m")


geometry_store = {
    'x_offset': X_2056_offset,
    'y_offset': Y_2056_offset,
    'Lagerstr': {
        'spline':       (tck_L, unew_L, cum_L),
        'positive_dir': 'EB',
        'total_length': len_L,
        'line_wgs84':   line_L,
        # no 'opposite' → WB uses same spline reversed
    },
    'KasernenstrN': {
        'spline':       (tck_KNB_n, unew_KNB_n, cum_KNB_n),
        'positive_dir': 'NB',
        'total_length': len_KNB_n,
        'line_wgs84':   line_KNB_n,
        'opposite': {
            'spline':       (tck_KSB_n, unew_KSB_n, cum_KSB_n),
            'total_length': len_KSB_n,
            'line_wgs84':   line_KSB_north,
        },
    },
    'KasernenstrS': {
        'spline':       (tck_KNB_s, unew_KNB_s, cum_KNB_s),
        'positive_dir': 'NB',
        'total_length': len_KNB_s,
        'line_wgs84':   line_KNB_south,
        'opposite': {
            'spline':       (tck_KSB_s, unew_KSB_s, cum_KSB_s),
            'total_length': len_KSB_s,
            'line_wgs84':   line_KSB_south,
        },
    },
    'Gessnerbr': {
        'spline':       (tck_GEB, unew_GEB, cum_GEB),
        'positive_dir': 'EB',
        'total_length': len_GEB,
        'line_wgs84':   line_GEB,
        'opposite': {
            'spline':       (tck_GWB, unew_GWB, cum_GWB),
            'total_length': len_GWB,
            'line_wgs84':   line_GWB,
        },
    },
}

print("\ngeometry_store assembled:")
for key, val in geometry_store.items():
    if key in ('x_offset', 'y_offset'):
        continue
    print(f"  {key}: {val['total_length']:.1f} m, positive_dir={val['positive_dir']}")


from shapely.geometry import Point
from shapely.plotting import plot_points

# KasernenstrN — NB (positive) and SB (opposite)
plot_points(Point(line_KNB_n.coords[0]),    color='red',   marker='o', label='Start KasernenstrN_NB')
plot_points(Point(line_KNB_n.coords[-1]),   color='black', marker='x', label='End KasernenstrN_NB')
plot_points(Point(line_KSB_north.coords[0]),  color='red',   marker='o', label='Start KasernenstrN_SB')
plot_points(Point(line_KSB_north.coords[-1]), color='black', marker='x', label='End KasernenstrN_SB')

# KasernenstrS — NB (positive) and SB (opposite)
plot_points(Point(line_KNB_south.coords[0]),  color='red',   marker='o', label='Start KasernenstrS_NB')
plot_points(Point(line_KNB_south.coords[-1]), color='black', marker='x', label='End KasernenstrS_NB')
plot_points(Point(line_KSB_south.coords[0]),  color='red',   marker='o', label='Start KasernenstrS_SB')
plot_points(Point(line_KSB_south.coords[-1]), color='black', marker='x', label='End KasernenstrS_SB')

# Lagerstr — EB (positive), WB uses same spline
plot_points(Point(line_L.coords[0]),   color='red',   marker='o', label='Start Lagerstr_EB')
plot_points(Point(line_L.coords[-1]),  color='black', marker='x', label='End Lagerstr_EB')

# Gessnerbr — EB (positive) and WB (opposite/sidewalk)
plot_points(Point(line_GEB.coords[0]),  color='red',   marker='o', label='Start Gessnerbr_EB')
plot_points(Point(line_GEB.coords[-1]), color='black', marker='x', label='End Gessnerbr_EB')
plot_points(Point(line_GWB.coords[0]),  color='red',   marker='o', label='Start Gessnerbr_WB')
plot_points(Point(line_GWB.coords[-1]), color='black', marker='x', label='End Gessnerbr_WB')

plt.title(
    'Positive Direction Validation\n'
    'Red o marker = Start  |  Black x marker = End',
    fontsize=11
)
plt.tight_layout()
plt.show()
# sys.exit(1)

geom_items = ['KasernenstrN', 'KasernenstrS', 'Lagerstr', 'Gessnerbr']
for geom_key in geom_items:
    get_s_domain(geom_key, geometry_store, gdf_swisstopo)
    sb  = geometry_store[geom_key]
    opp = sb.get('opposite')
    print(f"  {geom_key} (positive): "
          f"s_stop={sb['s_stop']:.2f} m  "
          f"s_yield={sb['s_yield']:.2f} m  "
          f"s_change={sb['s_change']:.2f} m  "
          f"total={sb['total_length']:.2f} m")
    if opp:
        print(f"  {geom_key} (opposite): "
              f"s_stop={opp['s_stop']:.2f} m  "
              f"s_yield={opp['s_yield']:.2f} m  "
              f"s_change={opp['s_change']:.2f} m  "
              f"total={opp['total_length']:.2f} m")

plot_geometry_store(
    geometry_store, gdf_swisstopo,
    offset_m=3.0,
    save_path= None #'../debugging/geometry_store_inspection.png'
)

# #############################################################################
# MAIN: Assemble segment_registry
# #############################################################################
DIRECTED_SEGMENTS = [
    ('KasernenstrN', 'NB'),   # positive dir
    ('KasernenstrN', 'SB'),   # opposite
    ('KasernenstrS', 'NB'),   # positive dir
    ('KasernenstrS', 'SB'),   # opposite
    ('Lagerstr',     'EB'),   # positive dir
    ('Lagerstr',     'WB'),   # opposite, same spline
    ('Gessnerbr',    'EB'),   # positive dir
    ('Gessnerbr',    'WB'),   # opposite, sidewalk
]

BIKE_LANE_INFO = {
    'KasernenstrN_NB': None, #{'w_bike': 2.75},  # green, bike-only ramp
    'KasernenstrN_SB': {'w_bike': 1.0},   # red
    'KasernenstrS_NB': {'w_bike': 3.5},   # green
    'KasernenstrS_SB': {'w_bike': 1.5},   # red
    'Lagerstr_EB':     {'w_bike': 1.5},   # red
    'Lagerstr_WB':     {'w_bike': 1.5},   # red
    'Gessnerbr_EB':    None,              # shared, no dedicated stripe
    'Gessnerbr_WB':    None, #{'w_bike': 2.5},   # green, sidewalk
}

MODE = {
    'KasernenstrN_NB': 'bike',    # Stadttunnel ramp, bike only
    'KasernenstrN_SB': 'shared',
    'KasernenstrS_NB': 'shared',
    'KasernenstrS_SB': 'shared',
    'Lagerstr_EB':     'shared',
    'Lagerstr_WB':     'shared',
    'Gessnerbr_EB':    'shared',
    'Gessnerbr_WB':    'bike',    # sidewalk, bike only
}

D_MAX = {
    # ── Turn segments (symmetric) — filled in after turns are built ──────────
    # Lane segments use default 10.0 m for now — tune after seeing polygons
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
    poly = entry.get('validity_polygon')
    poly_ok = '✓' if (poly is not None and not poly.is_empty) else '✗'
    app_str = f"[{app[0]:.1f}, {app[1]:.1f}]" if app else 'None'
    dep_str = f"[{dep[0]:.1f}, {dep[1]:.1f}]" if dep else 'None'
    print(f"  {seg_key} ({fwd}, mode={entry['mode']}, bike_lane={bl}, "
          f"poly={poly_ok}): "
          f"approach s∈{app_str}  departure s∈{dep_str}")
    
print("\nProjecting bike lane boundaries...")
gdf_bike_boundaries = gdf_swisstopo[
    gdf_swisstopo['Description'].str.endswith(('_NB', '_SB', '_EB', '_WB')) &
    ~gdf_swisstopo['Description'].isin(['KasernenstrN_NB', 'Gessnerbr_WB'])
].copy()

add_bike_lane_boundaries(segment_registry, geometry_store, gdf_bike_boundaries)

# #############################################################################
# MAIN: Build turning movement clothoids
# #############################################################################
TURNING_MOVEMENTS = [
    # From KasernenstrN SB (approaching from north)
    # ('KasernenstrN_SB', 'KasernenstrS_SB'),   # straight → south
    # ('KasernenstrN_SB', 'Lagerstr_WB'),        # right turn → west
    # ('KasernenstrN_SB', 'Gessnerbr_WB'),       # left turn → bridge sidewalk

    # # From KasernenstrN NB (exiting Stadttunnel)
    # ('KasernenstrN_NB', 'KasernenstrS_SB'),   # 
    # ('KasernenstrN_NB', 'Lagerstr_WB'),        # 
    # ('KasernenstrN_NB', 'Gessnerbr_WB'),       # 

    # # From KasernenstrS NB (approaching from south)
    ('KasernenstrS_NB', 'KasernenstrN_NB'),    # straight → Stadttunnel
    ('KasernenstrS_NB', 'Lagerstr_WB'),        # left turn → west
    ('KasernenstrS_NB', 'Gessnerbr_EB'),       # right turn → bridge

    # # From Lagerstr EB (approaching from west)
    # ('Lagerstr_EB', 'KasernenstrN_NB'),        # left turn → Stadttunnel
    # ('Lagerstr_EB', 'KasernenstrS_SB'),        # right turn → south
    # ('Lagerstr_EB', 'Gessnerbr_EB'),           # straight → bridge

    # # From Gessnerbr EB (approaching from bridge, main carriageway)
    # ('Gessnerbr_EB', 'KasernenstrN_NB'),       # right turn → Stadttunnel
    # ('Gessnerbr_EB', 'KasernenstrS_SB'),       # left turn → south
    # ('Gessnerbr_EB', 'Lagerstr_WB'),           # straight → west

    # # From Gessnerbr WB (approaching from bridge, sidewalk)
    # ('Gessnerbr_WB', 'KasernenstrN_NB'),       # 
    # ('Gessnerbr_WB', 'KasernenstrS_SB'),       # 
    # ('Gessnerbr_WB', 'Lagerstr_WB'),           # 
]

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
    save_path=None
)



