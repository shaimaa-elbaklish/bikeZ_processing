"""
TITLE OF PAPAER
-------------------------------------------
Authors:        Shaimaa El-Baklish
Organization:   ETH Zürich, Switzerland, IVT - Institute for Transportation Planning and Systems
Development:    2025
Submitted to:   JOURNAL
-------------------------------------------

maps_sep_D1A.py
-------------------------------------
Site definition — Quaibrücke / Stadthausquai / Bürkliplatz / Fraumünsterstrasse
Zürich, Switzerland — September 2025 campaign (D1,A location)
 
Two intersections:
  MainInt  — 4-way: Ackerstrasse × Zollstrasse × Mattengasse
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
max_chain_len  = 5    # 3 for standard movements + 2 for Mattengasse chain

# Edit Link: https://s.geo.admin.ch/2nh5rdrfsvzg
# Share Link: https://s.geo.admin.ch/yyq9zxqzmwwb

# Car Lanes:
# Share Link: 
# Edit Link: 

    
# location 7: https://s.geo.admin.ch/1znaiye58z0k


# #############################################################################
# FUNTIONS
# #############################################################################
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


# # 25 fps
# filename = f"trajectories_{mode}s_{date}_{intersection}_{timeslot}_{code}-1-ekf"
# # df = pd.read_csv(data_root + f"{date}/{intersection}/{filename}.csv")
# df_25fps = pd.read_parquet(data_root + f"{date}/{intersection}/{filename}.parquet")

# fig, ax = plt.subplots(1, 1, figsize=(6, 4))
# for veh_id, df_veh in df_25fps.groupby('veh_id'):
#     ax.plot(df_veh['x_ekf'], df_veh['y_ekf'], color='black', linewidth=1)
# fig.tight_layout()
# plt.show()

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
fig.tight_layout()

# --- Plot with bicycle trajectories ---
from tools_utils import _local_to_latlon

fig, ax = plt.subplots(1, 1, figsize=(6, 4))

for timeslot in BikeZ_Config.avail_timeslots[date][(intersection, code)]:
    filename = f"trajectories_{mode}s_{date}_{intersection}_{timeslot}_{code}-1-ekf"
    df = pd.read_parquet(data_root + f"{date}/{intersection}/{filename}.parquet")
    for veh_id, df_veh in df.groupby('veh_id'):
        latlon = _local_to_latlon(df_veh['x_ekf'].to_numpy(), df_veh['y_ekf'].to_numpy(), X_2056_offset, Y_2056_offset)
        latlon = np.asarray(latlon)
        ax.plot(latlon[:, 1], latlon[:, 0], color='black', linewidth=1)
        
gdf.plot(ax=ax, column='name', legend=True)
fig.tight_layout()
plt.show()
sys.exit(1)

# --- Plot with OSMIDs ---
osmids = [i[1] if isinstance(i, tuple) else i for i in gdf.index]

info = gdf[['name']].copy()
info['osmid']  = osmids
info['n_pts']  = gdf.geometry.apply(lambda g: len(g.coords))
info['start']  = gdf.geometry.apply(lambda g: (round(g.coords[0][0], 6), round(g.coords[0][1], 6)))
info['end']    = gdf.geometry.apply(lambda g: (round(g.coords[-1][0], 6), round(g.coords[-1][1], 6)))
print(info.sort_values('name').to_string())

# plot each way separately, labeled by osmid, so disconnected fragments are obvious
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
gdf.plot(ax=ax, column='name', legend=True)

for (idx, row), osmid in zip(gdf.iterrows(), osmids):
    if row['name'] not in ('Bürkliplatz'): # ('Bürkliplatz', 'Quaibrücke'):
        continue
    line = row.geometry
    coords = list(line.coords)
    mid = coords[len(coords) // 2]
    # ax.annotate(f"{row['name']}\nway {osmid}", mid, fontsize=6,
    #             bbox=dict(boxstyle='round', fc='white', alpha=0.75))
    # mark + label start/end vertices so you can eyeball which ways share an endpoint
    ax.plot(*coords[0], 'go', markersize=4)
    if len(coords) > 2:
        ax.plot(*coords[-2], 'rs', markersize=4)
    else:
        mid = 0.5*np.asarray(coords[0]) + 0.5*np.asarray(coords[1])
        ax.plot(*mid, 'rs', markersize=4)
    ax.annotate(f"{osmid}", coords[-1], fontsize=10, color='green')
    # ax.annotate(f"{osmid}:-1", coords[-1], fontsize=5, color='red')

fig.tight_layout()
plt.show()


# 4th arm of intersection (from Burkliplatz to Stadthausquai) ---> OSMID = 150132676
# Burkliplatz EB ---> OSMID = 60447876, 1169386963
# Burkliplatz WB ---> OSMID = 60447877, 1169386967

# Quaibrucke EB ---> OSMID = 60447874, 27361257, 374433860
# Quaibrucke WB ---> OSMID = 67571252, 569781024, 27361247, 15801814

line_Burklipl_EB_1 = merge_edges_by_ids(gdf, [60447876, 1169386963])
line_Burklipl_EB_1 = densify_linestring(line_Burklipl_EB_1, num_segments=10)
line_Burklipl_EB_2 = merge_edges_by_ids(gdf, [150132676])
line_Burklipl_EB_2 = densify_linestring(line_Burklipl_EB_2, num_segments=10)
line_Burklipl_EB = concat_linestrings(line_Burklipl_EB_1, line_Burklipl_EB_2)
tck_BuE, unew_BuE, cum_BuE, len_BuE = fit_spline_from_shapely(
    line_Burklipl_EB, x_offset=X_2056_offset, y_offset=Y_2056_offset,
)


line_Burklipl_WB = merge_edges_by_ids(gdf, [60447877, 1169386967])
line_Burklipl_WB = densify_linestring(line_Burklipl_WB, num_segments=10)
tck_BuW, unew_BuW, cum_BuW, len_BuW = fit_spline_from_shapely(
    line_Burklipl_WB, x_offset=X_2056_offset, y_offset=Y_2056_offset,
)

line_Quaibr_EB = merge_edges_by_ids(gdf, [60447874, 27361257, 374433860])
line_Quaibr_EB = densify_linestring(line_Quaibr_EB, num_segments=20)
tck_QuE, unew_QuE, cum_QuE, len_QuE = fit_spline_from_shapely(
    line_Quaibr_EB, x_offset=X_2056_offset, y_offset=Y_2056_offset,
)

line_Quaibr_WB = merge_edges_by_ids(gdf, [67571252, 569781024, 27361247, 15801814])
line_Quaibr_WB = densify_linestring(line_Quaibr_WB, num_segments=20)
tck_QuW, unew_QuW, cum_QuW, len_QuW = fit_spline_from_shapely(
    line_Quaibr_WB, x_offset=X_2056_offset, y_offset=Y_2056_offset,
)

line_S = merge_osmnx_edges(gdf, 'Stadthausquai')
line_S = densify_linestring(line_S, num_segments=10)
tck_S, unew_S, cum_S, len_S = fit_spline_from_shapely(
    line_S, x_offset=X_2056_offset, y_offset=Y_2056_offset,
)

line_F = merge_osmnx_edges(gdf, 'Fraumünsterstrasse')
line_F = densify_linestring(line_F, num_segments=10)
tck_F, unew_F, cum_F, len_F = fit_spline_from_shapely(
    line_F, x_offset=X_2056_offset, y_offset=Y_2056_offset,
)

print(f"  Fraumunsterstr : {len_F:.1f} m")
print(f"  Stadthausquai  : {len_S:.1f} m")
print(f"  Quaibr_EB      : {len_QuE:.1f} m")
print(f"  Quaibr_WB      : {len_QuW:.1f} m")
print(f"  Burklipl_EB    : {len_BuE:.1f} m")
print(f"  Burklipl_WB    : {len_BuW:.1f} m")

fig, ax = plt.subplots(1, 1)
# gdf.plot(ax=ax, column='name', legend=True)
plot_line(line_F, ax=ax, add_points=False, color='tab:orange', label='Fraumunsterstr')
plot_line(line_S, ax=ax, add_points=False, color='tab:green', label='Stadthausquai')
plot_line(line_Quaibr_EB, ax=ax, add_points=False, color='tab:purple', label='Quaibr_EB')
plot_line(line_Quaibr_WB, ax=ax, add_points=False, color='tab:pink', label='Quaibr_WB')
plot_line(line_Burklipl_EB, ax=ax, add_points=False, color='tab:brown', label='Burklipl_EB')
plot_line(line_Burklipl_WB, ax=ax, add_points=False, color='tab:blue', label='Burklipl_WB')
plot_points(Point(line_F.coords[0]), color='red', marker='o',) 
plot_points(Point(line_F.coords[-2]), color='black', marker='x',) 
plot_points(Point(line_S.coords[1]), color='red', marker='o',) 
plot_points(Point(line_S.coords[-1]), color='black', marker='x',) 
plot_points(Point(line_Quaibr_EB.coords[1]), color='red', marker='o',) 
plot_points(Point(line_Quaibr_EB.coords[-1]), color='black', marker='x',) 
plot_points(Point(line_Quaibr_WB.coords[0]), color='red', marker='o',) 
plot_points(Point(line_Quaibr_WB.coords[-2]), color='black', marker='x',) 
plot_points(Point(line_Burklipl_EB.coords[0]), color='red', marker='o',) 
plot_points(Point(line_Burklipl_EB.coords[-2]), color='black', marker='x',) 
plot_points(Point(line_Burklipl_WB.coords[1]), color='red', marker='o',)
plot_points(Point(line_Burklipl_WB.coords[-1]), color='black', marker='x',) 
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncols=4)
fig.suptitle(
    'Postive Direction Validation\n'
    'Red o marker = Start  |  Black x marker = End',
    fontsize=11
)
fig.tight_layout()
