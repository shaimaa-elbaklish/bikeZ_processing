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

from pyproj import Transformer
from shapely.geometry import LineString, Point
from scipy.interpolate import splev
from shapely.ops import linemerge, snap
from shapely.geometry import box

from _constants import BikeZ_Config
from tools_coordinate_transform import fit_roadway_centerline_spline
from tools_coordinate_transform import convert_roadway_to_xy2056_coordinates
from tools_coordinate_transform import connect_lines_g2
from tools_coordinate_transform import cut_line_at_stop
from tools_coordinate_transform import densify_linestring

from tools_map_visualization import create_swisstopo_map
from tools_map_visualization import plot_spline_xy_2056
from tools_map_visualization import plot_line_xy_2056
from tools_map_visualization import plot_line_latlon
from tools_map_visualization import plot_all_centerlines_splines_xy_2056
from tools_map_visualization import plot_bicycles_trajectories_xy_2056

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

intersection, code = BikeZ_Config.avail_intersections[date][2]
timeslot = BikeZ_Config.avail_timeslots[date][(intersection, code)][2] # 'AM1'

XY_2056_Bounds = BikeZ_Config.XY_2056_Bounds[date][(intersection, code)]
X_2056_offset = XY_2056_Bounds[0][0]
Y_2056_offset = XY_2056_Bounds[1][0]

# #############################################################################
# MAIN: Load Data (Trajectories)
# #############################################################################
filename = f"trajectories_bikes_{date}_{intersection}_{timeslot}_{code}-1.csv"
df = pd.read_csv(data_root + f"{date}/{intersection}/{filename}")
df = df.dropna()

df['missing'] = (df['speed(km/h)'] == -1)
df = df.rename(columns={
    'speed(km/h)': 'speed',
    'a(m/s2)': 'a',
    'time(s)': 'time',
    'X_2056(m)': 'x_act',
    'Y_2056(m)': 'y_act',
    'longitude': 'lon',
    'latitude': 'lat'
})
df['x'] = df['x_act'] - X_2056_offset
df['y'] = df['y_act'] - Y_2056_offset
df['datetime'] = pd.to_datetime(df['datetime'], format='ISO8601')
ref_datetime = df['datetime'].min()
ref_time = df.loc[(df['datetime'] == ref_datetime) & (df['time'] >= 0), 'time'].unique()[0]
df['time'] = df['datetime'].apply(lambda x: np.round((x - ref_datetime).total_seconds() + ref_time, decimals=3))
df = df.sort_values(by=['veh_id', 'time'], ascending=True)

# Create a folium map
df = df[~df['missing']]
center_lat, center_lon = df["lat"].mean(), df["lon"].mean()

transformer = Transformer.from_crs("EPSG:2056", "EPSG:4326", always_xy=True)
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

# Filter for road name
road_name = "Zollstrasse"
gdf = gdf_main[gdf_main['name'] == road_name]
gdf["geometry"] = gdf.geometry.intersection(bbox) # Clip / intersect
gdf = gdf[~gdf.is_empty] # Drop empty geometries
# Optional: filter to LineStrings only
gdf = gdf[(gdf.geometry.type == "LineString") & (gdf['highway'].isin(main_road_types))]
merged_centerline = linemerge(list(gdf.geometry))
# Extract coordinates (handle both LineString and MultiLineString)
if merged_centerline.geom_type == 'LineString':
    coords = list(merged_centerline.coords)
elif merged_centerline.geom_type == 'MultiLineString':
    coords = []
    for line in merged_centerline.geoms:
        coords.extend(list(line.coords))
# Convert (lon, lat) to (lat, lon) for folium
zollstrasse_branch = [(lat, lon) for lon, lat in coords]


# Filter for road name
road_name = "Ackerstrasse"
gdf = gdf_main[gdf_main['name'] == road_name]
gdf["geometry"] = gdf.geometry.intersection(bbox) # Clip / intersect
gdf = gdf[~gdf.is_empty] # Drop empty geometries
# Optional: filter to LineStrings only
gdf = gdf[(gdf.geometry.type == "LineString") & (gdf['highway'].isin(main_road_types))]
merged_centerline = linemerge(list(gdf.geometry))
# Extract coordinates (handle both LineString and MultiLineString)
if merged_centerline.geom_type == 'LineString':
    coords = list(merged_centerline.coords)
elif merged_centerline.geom_type == 'MultiLineString':
    coords = []
    for line in merged_centerline.geoms:
        coords.extend(list(line.coords))
# Convert (lon, lat) to (lat, lon) for folium
ackerstrasse_branch = [(lat, lon) for lon, lat in coords]


# Filter for road name
road_name = "Neugasse"
gdf = gdf_main[gdf_main['name'] == road_name]
gdf["geometry"] = gdf.geometry.intersection(bbox) # Clip / intersect
gdf = gdf[~gdf.is_empty] # Drop empty geometries
# Optional: filter to LineStrings only
gdf = gdf[(gdf.geometry.type == "LineString") & (gdf['highway'].isin(main_road_types))]
merged_centerline = linemerge(list(gdf.geometry))
# Extract coordinates (handle both LineString and MultiLineString)
if merged_centerline.geom_type == 'LineString':
    coords = list(merged_centerline.coords)
elif merged_centerline.geom_type == 'MultiLineString':
    coords = []
    for line in merged_centerline.geoms:
        coords.extend(list(line.coords))
# Convert (lon, lat) to (lat, lon) for folium
neugasse_branch = [(lat, lon) for lon, lat in coords]


# # Filter for road name
# road_name = "Mattengasse"
# gdf = gdf_main[gdf_main['name'] == road_name]
# # Optional: filter to LineStrings only
# gdf = gdf[(gdf.geometry.type == "LineString") & (gdf['highway'].isin(main_road_types))]
# merged_centerline = linemerge(list(gdf.geometry))
# # Extract coordinates (handle both LineString and MultiLineString)
# if merged_centerline.geom_type == 'LineString':
#     coords = list(merged_centerline.coords)
# elif merged_centerline.geom_type == 'MultiLineString':
#     coords = []
#     for line in merged_centerline.geoms:
#         coords.extend(list(line.coords))
# # Convert (lon, lat) to (lat, lon) for folium
# mattengasse_branch = [(lat, lon) for lon, lat in coords]


# m = create_swisstopo_map(center_lat, center_lon)
# plot_bicycles_trajectories_xy_2056(m, df, linecolor='black', linealpha=0.25, add_layer_control=False, ekf=False)
# m.save(f"../maps/trajectories_map_{date}_{intersection}_{timeslot}_{code}.html")


import matplotlib.pyplot as plt

fig, ax= plt.subplots(1, 1, figsize=(6, 4))
for bike_id in df['veh_id'].unique():
    traj = df[(df["veh_id"] == bike_id)]
    ax.plot(traj['lon'], traj['lat'], color='black', alpha=0.1)
ax.plot(np.asarray(zollstrasse_branch)[:, 1], np.asarray(zollstrasse_branch)[:, 0], linewidth=2, color='red', label='Zollstrasse')
ax.plot(np.asarray(ackerstrasse_branch)[:, 1], np.asarray(ackerstrasse_branch)[:, 0], linewidth=2, color='blue', label='Ackerstrasse')
ax.plot(np.asarray(neugasse_branch)[:, 1], np.asarray(neugasse_branch)[:, 0], linewidth=2, color='green', label='Neugasse')
# ax.plot(np.asarray(mattengasse_branch)[:, 1], np.asarray(mattengasse_branch)[:, 0], linewidth=2, color='magenta', label='Mattengasse')
ax.set_xlim(lonlat_bounds[0])
ax.set_ylim(lonlat_bounds[1])
ax.legend()
fig.tight_layout()


# #############################################################################
# MAIN: Create Folium map with SwissTopo base image
# #############################################################################
# Create a folium map
m = create_swisstopo_map(center_lat, center_lon)

# Create splines dictionary for saving
splines_dict = {}

colors_dict = {
    'north': 'blue',
    'south': 'red',
    'west': 'green',
    'east': 'yellow',
}


# #############################################################################
# MAIN: Get Through East -> West Spline
# #############################################################################
point_zollstrasse = Point(8.532347, 47.380956) # lon, lat
centerline = LineString([(lon, lat) for lat, lon in zollstrasse_branch])
d = centerline.project(point_zollstrasse)
point_zollstrasse = centerline.interpolate(d)
centerline = snap(centerline, point_zollstrasse, 1e-8)
centerline = cut_line_at_stop(centerline, point_zollstrasse, choose='last')
centerline = densify_linestring(line=centerline, num_segments=5)
starting_branch = [(lat, lon) for lon, lat in centerline.coords]

east_west_spl = fit_roadway_centerline_spline(starting_branch[::-1] + neugasse_branch, smoothing=0.01)
plot_spline_xy_2056(m, east_west_spl, label="Neugasse Centerline (E->W)", 
                    linecolor=colors_dict['east'], linedashed=True, start_point=True)
splines_dict['E_2_W_B'] = east_west_spl       # B = Bikes only


east_west_spl = fit_roadway_centerline_spline(neugasse_branch) # tuple (tck, unew, cum_dist)
plot_spline_xy_2056(m, east_west_spl, label="Neugasse Centerline (E->W)", 
                    linecolor='magenta', linedashed=True, start_point=True)
splines_dict['E_2_W_A'] = east_west_spl       # A = Both Cars and Bikes


# #############################################################################
# MAIN: Get Through West -> East Spline
# #############################################################################
west_east_spl = fit_roadway_centerline_spline(neugasse_branch[::-1] + starting_branch, smoothing=0.01)
plot_spline_xy_2056(m, west_east_spl, label="Neugasse Centerline (W->E)", 
                    linecolor='brown', linedashed=True, start_point=True)
splines_dict['W_2_E_B'] = west_east_spl       # B = Bikes only


west_east_spl = fit_roadway_centerline_spline(zollstrasse_branch) # tuple (tck, unew, cum_dist)
plot_spline_xy_2056(m, west_east_spl, label="Zollstrasse Centerline (W->E)", 
                    linecolor=colors_dict['west'], linedashed=True, start_point=True)
splines_dict['W_2_E_A'] = west_east_spl       # A = Both Cars and Bikes



# #############################################################################
# MAIN: Get Through West -> North Spline
# #############################################################################
point_zollstrasse = Point(8.530669, 47.381488) # lon, lat
centerline = LineString([(lon, lat) for lat, lon in zollstrasse_branch])
d = centerline.project(point_zollstrasse)
point_zollstrasse = centerline.interpolate(d)
centerline = snap(centerline, point_zollstrasse, 1e-8)
centerline = cut_line_at_stop(centerline, point_zollstrasse, choose='first')
# centerline = densify_linestring(line=centerline, num_segments=5)
zollstrasse_west_branch = [(lat, lon) for lon, lat in centerline.coords]

# folium.Marker(
#     location=(point_zollstrasse.y, point_zollstrasse.x),
#     icon=folium.Icon(color="blue"),
#     tooltip="West"
# ).add_to(m)

tmp_spl = fit_roadway_centerline_spline(zollstrasse_west_branch)
tck = tmp_spl[0]
x_spline, y_spline = splev(np.linspace(0, 1, 50), tck)
xy_zollstrasse_west = np.column_stack((x_spline, y_spline))


point_ackerstrasse = Point(8.530781, 47.381520) # lon, lat
centerline = LineString([(lon, lat) for lat, lon in ackerstrasse_branch])
d = centerline.project(point_ackerstrasse)
point_ackerstrasse = centerline.interpolate(d)
centerline = snap(centerline, point_ackerstrasse, 1e-8)
centerline = cut_line_at_stop(centerline, point_ackerstrasse, choose='last')
# centerline = densify_linestring(line=centerline, num_segments=5)
ackerstrasse_north_branch = [(lat, lon) for lon, lat in centerline.coords]

# folium.Marker(
#     location=(point_ackerstrasse.y, point_ackerstrasse.x),
#     icon=folium.Icon(color="blue"),
#     tooltip="West"
# ).add_to(m)

tmp_spl = fit_roadway_centerline_spline(ackerstrasse_north_branch)
tck = tmp_spl[0]
x_spline, y_spline = splev(np.linspace(0, 1, 50), tck)
xy_ackerstrasse_north = np.column_stack((x_spline, y_spline))

west_north_merged_coords, _, _= connect_lines_g2(xy_zollstrasse_west, xy_ackerstrasse_north, n_connector=120, verbose=True)
west_north_spl = fit_roadway_centerline_spline(west_north_merged_coords, coordsys='2056')
plot_spline_xy_2056(m, west_north_spl, label="Zollstrasse Centerline (W->N)", 
                    linecolor=colors_dict['west'], linedashed=True, start_point=True)
splines_dict['W_2_N_A'] = west_north_spl       # A = Both Cars and Bikes


# #############################################################################
# MAIN: Get Through East -> North Spline
# #############################################################################
point_neugasse = Point(8.531126, 47.381756) # lon, lat
centerline = LineString([(lon, lat) for lat, lon in neugasse_branch])
d = centerline.project(point_neugasse)
point_neugasse = centerline.interpolate(d)
centerline = snap(centerline, point_neugasse, 1e-8)
centerline = cut_line_at_stop(centerline, point_neugasse, choose='first')
# centerline = densify_linestring(line=centerline, num_segments=5)
neugasse_east_branch = [(lat, lon) for lon, lat in centerline.coords]

# folium.Marker(
#     location=(point_neugasse.y, point_neugasse.x),
#     icon=folium.Icon(color="blue"),
#     tooltip="East"
# ).add_to(m)

tmp_spl = fit_roadway_centerline_spline(neugasse_east_branch)
tck = tmp_spl[0]
x_spline, y_spline = splev(np.linspace(0, 1, 50), tck)
xy_neugasse_east = np.column_stack((x_spline, y_spline))


point_ackerstrasse = Point(8.531091, 47.381848) # lon, lat
centerline = LineString([(lon, lat) for lat, lon in ackerstrasse_branch])
d = centerline.project(point_ackerstrasse)
point_ackerstrasse = centerline.interpolate(d)
centerline = snap(centerline, point_ackerstrasse, 1e-8)
centerline = cut_line_at_stop(centerline, point_ackerstrasse, choose='last')
centerline = densify_linestring(line=centerline, num_segments=5)
ackerstrasse_north_branch = [(lat, lon) for lon, lat in centerline.coords]

# folium.Marker(
#     location=(point_ackerstrasse.y, point_ackerstrasse.x),
#     icon=folium.Icon(color="blue"),
#     tooltip="East"
# ).add_to(m)

tmp_spl = fit_roadway_centerline_spline(ackerstrasse_north_branch)
tck = tmp_spl[0]
x_spline, y_spline = splev(np.linspace(0, 1, 50), tck)
xy_ackerstrasse_north = np.column_stack((x_spline, y_spline))

east_north_merged_coords, _, _= connect_lines_g2(xy_neugasse_east, xy_ackerstrasse_north, n_connector=120, verbose=True)
east_north_spl = fit_roadway_centerline_spline(east_north_merged_coords, coordsys='2056')
plot_spline_xy_2056(m, east_north_spl, label="Neugasse Centerline (E->N)", 
                    linecolor=colors_dict['east'], linedashed=True, start_point=True)
splines_dict['E_2_N_A'] = east_north_spl       # A = Both Cars and Bikes


# #############################################################################
# MAIN: Get Through East -> North Spline
# #############################################################################
point_zollstrasse = Point(8.530847, 47.381434) # lon, lat
centerline = LineString([(lon, lat) for lat, lon in zollstrasse_branch])
d = centerline.project(point_zollstrasse)
point_zollstrasse = centerline.interpolate(d)
centerline = snap(centerline, point_zollstrasse, 1e-8)
centerline = cut_line_at_stop(centerline, point_zollstrasse, choose='last')
# centerline = densify_linestring(line=centerline, num_segments=5)
zollstrasse_east_branch = [(lat, lon) for lon, lat in centerline.coords]

folium.Marker(
    location=(point_zollstrasse.y, point_zollstrasse.x),
    icon=folium.Icon(color="blue"),
    tooltip="East"
).add_to(m)

tmp_spl = fit_roadway_centerline_spline(zollstrasse_east_branch)
tck = tmp_spl[0]
x_spline, y_spline = splev(np.linspace(0, 1, 50), tck)
xy_zollstrasse_east = np.column_stack((x_spline, y_spline))


point_ackerstrasse = Point(8.530781, 47.381520) # lon, lat
centerline = LineString([(lon, lat) for lat, lon in ackerstrasse_branch])
d = centerline.project(point_ackerstrasse)
point_ackerstrasse = centerline.interpolate(d)
centerline = snap(centerline, point_ackerstrasse, 1e-8)
centerline = cut_line_at_stop(centerline, point_ackerstrasse, choose='last')
# centerline = densify_linestring(line=centerline, num_segments=5)
ackerstrasse_north_branch = [(lat, lon) for lon, lat in centerline.coords]

folium.Marker(
    location=(point_ackerstrasse.y, point_ackerstrasse.x),
    icon=folium.Icon(color="blue"),
    tooltip="East"
).add_to(m)

tmp_spl = fit_roadway_centerline_spline(ackerstrasse_north_branch)
tck = tmp_spl[0]
x_spline, y_spline = splev(np.linspace(0, 1, 50), tck)
xy_ackerstrasse_north = np.column_stack((x_spline, y_spline))

east_north_merged_coords, _, _= connect_lines_g2(xy_zollstrasse_east[::-1], xy_ackerstrasse_north, n_connector=120, verbose=True)
east_north_spl = fit_roadway_centerline_spline(east_north_merged_coords, coordsys='2056')
plot_spline_xy_2056(m, east_north_spl, label="Zollstrasse Centerline (W->N)", 
                    linecolor=colors_dict['east'], linedashed=True, start_point=True)
splines_dict['E_2_N_B'] = east_north_spl       # B = Bikes only

# #############################################################################
# MAIN: Get Through North -> East Spline
# #############################################################################

north_east_merged_coords, _, _= connect_lines_g2(xy_ackerstrasse_north[::-1], xy_zollstrasse_east, n_connector=120, verbose=True)
north_east_spl = fit_roadway_centerline_spline(north_east_merged_coords, coordsys='2056')
plot_spline_xy_2056(m, north_east_spl, label="Zollstrasse Centerline (N->E)", 
                    linecolor=colors_dict['north'], linedashed=True, start_point=True)
splines_dict['N_2_E_B'] = north_east_spl       # B = Bikes only

# #############################################################################
# MAIN: Get Through North -> West Spline
# #############################################################################

north_west_merged_coords, _, _= connect_lines_g2(xy_ackerstrasse_north[::-1], xy_zollstrasse_west[::-1], n_connector=120, verbose=True)
north_west_spl = fit_roadway_centerline_spline(north_west_merged_coords, coordsys='2056')
plot_spline_xy_2056(m, north_west_spl, label="Zollstrasse Centerline (N->W)", 
                    linecolor=colors_dict['north'], linedashed=True, start_point=True)
splines_dict['N_2_W_B'] = north_west_spl       # B = Bikes only

# #############################################################################
# MAIN: Saving Map and Splines
# #############################################################################
# with open(f"../data/centerlines_splines_{date}_{intersection}_{code}.pkl", "wb") as f:
#     pickle.dump(splines_dict, f)

m.save(f"../maps/road_centerlines_map_{date}_{intersection}_{code}_debugging.html")


# #############################################################################
# MAIN: Plotting and Saving FINAL Map
# #############################################################################
# Create a folium map
m = create_swisstopo_map(center_lat, center_lon, add_layer_control=False)
plot_all_centerlines_splines_xy_2056(m, splines_dict, add_layer_control=True)

m.save(f"../maps/road_centerlines_map_{date}_{intersection}_{code}.html")



m = create_swisstopo_map(center_lat, center_lon, add_layer_control=False)
plot_all_centerlines_splines_xy_2056(m, splines_dict, add_layer_control=False)
plot_bicycles_trajectories_xy_2056(m, df, linecolor='black', linealpha=0.25, add_layer_control=True, ekf=False)

m.save(f"../maps/trajectories_map_{date}_{intersection}_{timeslot}_{code}.html")

