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
date = BikeZ_Config.avail_dates[2]
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
road_name = "Zypressenstrasse"
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
zypressen_branch = [(lat, lon) for lon, lat in coords]


# Filter for road name
road_name = "Sihlfeldstrasse"
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
sihlfeld_branch = [(lat, lon) for lon, lat in coords]


# Filter for road name
road_name = "Stauffacherstrasse"
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
stauffacher_branch = [(lat, lon) for lon, lat in coords]


# Filter for road name
road_name = "Bullingerstrasse"
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
bullingerstr_branch = [(lat, lon) for lon, lat in coords]


import matplotlib.pyplot as plt

fig, ax= plt.subplots(1, 1, figsize=(6, 4))
for bike_id in df['veh_id'].unique():
    traj = df[(df["veh_id"] == bike_id)]
    ax.plot(traj['lon'], traj['lat'], color='black', alpha=0.1)
ax.plot(np.asarray(zypressen_branch)[:, 1], np.asarray(zypressen_branch)[:, 0], linewidth=2, color='red', label='Bullingerplatz')
ax.plot(np.asarray(stauffacher_branch)[:, 1], np.asarray(stauffacher_branch)[:, 0], linewidth=2, color='blue', label='Stauffacherstrasse')
ax.plot(np.asarray(sihlfeld_branch)[:, 1], np.asarray(sihlfeld_branch)[:, 0], linewidth=2, color='green', label='Sihlfeldstrasse')
ax.plot(np.asarray(bullingerstr_branch)[:, 1], np.asarray(bullingerstr_branch)[:, 0], linewidth=2, color='cyan', label='Bullingerstrasse')
ax.set_xlim(lonlat_bounds[0])
ax.set_ylim(lonlat_bounds[1])
ax.legend()
fig.tight_layout()