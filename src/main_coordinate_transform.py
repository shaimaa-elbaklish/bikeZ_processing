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
import gc
import sys
import pytz
import folium
import warnings
warnings.filterwarnings("ignore")
import simplekml

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from pyproj import Transformer

from _constants import BikeZ_Config
from tools_kalman import calculate_kalman_filtered_trajectory

# #############################################################################
# CONSTANTS
# #############################################################################
date = BikeZ_Config.avail_dates[0]
intersection = BikeZ_Config.avail_intersections[2]
time_slot = 'AM1'
code= 'E'

# #############################################################################
# MAIN
# #############################################################################
filename = f"trajectories_bikes_{date}_{intersection}_{time_slot}_{code}-1-ekf.csv"
df = pd.read_csv(BikeZ_Config.data_root + f"{date}/{intersection}/{filename}")
df = df.dropna()

# # oveview of trajectories
# fig, axs = plt.subplots(1, 2, figsize=(8, 4))
# grouped = df[~df['missing']].groupby(by='veh_id')
# for veh_id, veh_df in grouped:
#     axs[0].plot(veh_df['x_act'], veh_df['y_act'], 'b')
#     axs[1].plot(veh_df['x'], veh_df['y'], 'b')

# axs[0].set_xlabel('X_2056 [m]')
# axs[0].set_ylabel('Y_2056 [m]')
# axs[0].set_xlim(BikeZ_Config.X_2056_Bounds)
# axs[0].set_ylim(BikeZ_Config.Y_2056_Bounds)

# axs[1].set_xlabel('X_2056 - X_ref [m]')
# axs[1].set_ylabel('Y_2056 - Y_ref [m]')

# fig.tight_layout()

# #############################################################################
# MAIN: Visualize in map svia folium
# #############################################################################

# Convert from EPSG:2056 to EPSG:4326 (lat, lon)
df['x_act_ekf'] = df['x_ekf'] + BikeZ_Config.X_2056_Bounds[0]
df['y_act_ekf'] = df['y_ekf'] + BikeZ_Config.Y_2056_Bounds[0]
transformer = Transformer.from_crs("EPSG:2056", "EPSG:4326", always_xy=True)
df["lon_ekf"], df["lat_ekf"] = transformer.transform(df["x_act_ekf"].values, df["y_act_ekf"].values)

# Create a folium map
center_lat, center_lon = df["lat_ekf"].mean(), df["lon_ekf"].mean()
m = folium.Map(location=[center_lat, center_lon], zoom_start=20)

# Plot trajectories on map
# bike_id = 1
for bike_id in df['veh_id'].unique():
    traj = df[(df["veh_id"] == bike_id)]
    folium.PolyLine(
        locations=traj[['lat_ekf', 'lon_ekf']].values.tolist(), 
        color="blue", 
        weight=3, 
        opacity=0.8,
        tooltip=f"Bicycle {bike_id}"
    ).add_to(m)


m.save(f"../maps/trajectories_map_{date}_{intersection}_{time_slot}_{code}.html")

# # #############################################################################
# # MAIN: Visualize in maps via Google Earth
# # #############################################################################
# kml = simplekml.Kml()
# for bike_id in df['veh_id'].unique():
#     traj = df[(df["veh_id"] == bike_id) & (~df['missing'])]
#     coords = list(zip(traj["lon_ekf"], traj["lat_ekf"]))
#     line = kml.newlinestring(name=f"Bicycle {bike_id}", coords=coords)
#     line.altitudemode = simplekml.AltitudeMode.relativetoground   
#     line.style.linestyle.color = simplekml.Color.blue
#     line.style.linestyle.width = 4

# kml.savekmz(f"../maps/trajectories_map_{date}_{intersection}_{time_slot}_{code}.kmz")

# kml = simplekml.Kml()
# for bike_id in df['veh_id'].unique():
#     traj = df[(df["veh_id"] == bike_id) & (~df['missing'])]
#     altitudes = [10]*len(traj) # to handle water area for Gessnerbruecke
#     coords = list(zip(traj["lon_ekf"], traj["lat_ekf"], altitudes))
#     line = kml.newlinestring(name=f"Bicycle {bike_id}", coords=coords)
#     line.altitudemode = simplekml.AltitudeMode.relativetoground   
#     line.style.linestyle.color = simplekml.Color.red
#     line.style.linestyle.width = 4

# kml.savekmz(f"../maps/trajectories_map_{date}_{intersection}_{time_slot}_{code}_elevated.kmz")
# sys.exit(1)

###############################################################################
# MAIN: Extract centerline of Rheinstrasse
###############################################################################
import osmnx as ox
from functools import partial
from shapely.ops import linemerge, transform, unary_union
from scipy.interpolate import splprep, splev, interp1d
from scipy.optimize import minimize_scalar

# Define your area
place = "Zürich, Switzerland"

# Download all features with highway tag
tags = {"highway": True}
gdf_main = ox.features.features_from_place(place, tags=tags)


# Filter for road name
road_name = "Kasernenstrasse"
gdf = gdf_main[gdf_main['name'] == road_name]

# Optional: filter to LineStrings only
main_road_types = ['primary', 'secondary', 'tertiary', 'residential', 'unclassified', "cycleway"]
                   # "cycleway", "path", "service", "living_street"]
bikeable = ["yes", "designated", "permissive"]
gdf = gdf[(gdf.geometry.type == "LineString") & (gdf['highway'].isin(main_road_types))]
          # & (gdf['bicycle'].isin(bikeable))]

# Plot
gdf.plot()
plt.title("Kasernenstrasse centerline")
plt.show()

# Union merges touching geometries into clusters
merged = unary_union(list(gdf.geometry))

# Keep each cluster separately
if merged.geom_type == "MultiLineString":
    branches = list(merged.geoms)
else:
    branches = [merged]

centerline_coords_list = [
    [(lat, lon) for lon, lat in branch.coords]
    for branch in branches if branch.geom_type == "LineString"
]
print(f"Extracted {len(centerline_coords_list)} separate branches.")

for i in range(len(centerline_coords_list)):
    folium.PolyLine(
        locations=centerline_coords_list[i],
        color='black',
        weight=5,
        opacity=0.8,
        tooltip=f"Kasernenstrasse Centerline {i}"
    ).add_to(m)

folium.Marker(
    location=centerline_coords_list[9][0],
    popup="Start C9",
    icon=folium.Icon(color='black', icon='play')
).add_to(m)

folium.Marker(
    location=centerline_coords_list[14][0],
    popup="Start C14",
    icon=folium.Icon(color='black', icon='play')
).add_to(m)

left_branch = centerline_coords_list[14] + centerline_coords_list[0] + centerline_coords_list[3] + centerline_coords_list[9]
right_branch = centerline_coords_list[4]
folium.PolyLine(
    locations=left_branch,
    color='pink',
    weight=5,
    opacity=0.8,
    dash_array="10, 20",
    tooltip="Kasernenstrasse Centerline (LEFT)"
).add_to(m)
folium.PolyLine(
    locations=right_branch,
    color='pink',
    weight=5,
    opacity=0.8,
    dash_array="10, 20",
    tooltip="Kasernenstrasse Centerline (RIGHT)"
).add_to(m)

# Filter for road name
road_name = "Lagerstrasse"
gdf = gdf_main[gdf_main['name'] == road_name]

# Optional: filter to LineStrings only
gdf = gdf[(gdf.geometry.type == "LineString") & (gdf['highway'].isin(main_road_types))]
          # & (gdf['bicycle'].isin(bikeable))]

merged_centerline = linemerge(list(gdf.geometry))

# Extract coordinates (handle both LineString and MultiLineString)
if merged_centerline.geom_type == 'LineString':
    coords = list(merged_centerline.coords)
elif merged_centerline.geom_type == 'MultiLineString':
    coords = []
    for line in merged_centerline.geoms:
        coords.extend(list(line.coords))

# Convert (lon, lat) to (lat, lon) for folium
centerline_coords = [(lat, lon) for lon, lat in coords]

# Add centerline as a blue polyline
folium.PolyLine(
    locations=centerline_coords,
    color='red',
    weight=5,
    opacity=0.8,
    tooltip="Lagerstrasse Centerline"
).add_to(m)


# Filter for road name
road_name = "Gessnerbrücke"
gdf = gdf_main[gdf_main['name'] == road_name]

# Optional: filter to LineStrings only
gdf = gdf[(gdf.geometry.type == "LineString") & (gdf['highway'].isin(main_road_types))]
          # & (gdf['bicycle'].isin(bikeable))]

merged_centerline = linemerge(list(gdf.geometry))

# Extract coordinates (handle both LineString and MultiLineString)
if merged_centerline.geom_type == 'LineString':
    coords = list(merged_centerline.coords)
elif merged_centerline.geom_type == 'MultiLineString':
    coords = []
    for line in merged_centerline.geoms:
        coords.extend(list(line.coords))

# Convert (lon, lat) to (lat, lon) for folium
centerline_coords = [(lat, lon) for lon, lat in coords]

# Add centerline as a blue polyline
folium.PolyLine(
    locations=centerline_coords,
    color='green',
    weight=5,
    opacity=0.8,
    tooltip="Lagerstrasse Centerline"
).add_to(m)

road_name = "Stadttunnel"
gdf = gdf_main[gdf_main['name'] == road_name]

# Optional: filter to LineStrings only
main_road_types = ['primary', 'secondary', 'tertiary', 'residential', 'unclassified', "cycleway"]
                   # "cycleway", "path", "service", "living_street"]
bikeable = ["yes", "designated", "permissive"]
gdf = gdf[(gdf.geometry.type == "LineString") & (gdf['highway'].isin(main_road_types))]
          # & (gdf['bicycle'].isin(bikeable))]

# # Plot
# gdf.plot()
# plt.title("Kasernenstrasse centerline")
# plt.show()

merged_centerline = linemerge(list(gdf.geometry))

# Extract coordinates (handle both LineString and MultiLineString)
if merged_centerline.geom_type == 'LineString':
    coords = list(merged_centerline.coords)
elif merged_centerline.geom_type == 'MultiLineString':
    coords = []
    for line in merged_centerline.geoms:
        coords.extend(list(line.coords))

# Convert (lon, lat) to (lat, lon) for folium
centerline_coords = [(lat, lon) for lon, lat in coords]

# Add centerline as a blue polyline
folium.PolyLine(
    locations=centerline_coords,
    color='orange',
    weight=5,
    opacity=0.8,
    tooltip="Stadttunnel Centerline"
).add_to(m)


m.save(f"../maps/trajectories_map_{date}_{intersection}_{time_slot}_{code}.html")