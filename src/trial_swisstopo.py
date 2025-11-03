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
import fiona
import folium
import warnings
warnings.filterwarnings("ignore")
import simplekml

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

from tqdm import tqdm
from pyproj import Transformer
from shapely.geometry import box

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
# MAIN: Load Data (Trajectories)
# #############################################################################
filename = f"trajectories_bikes_{date}_{intersection}_{time_slot}_{code}-1-ekf.csv"
df = pd.read_csv(BikeZ_Config.data_root + f"{date}/{intersection}/{filename}")
df = df.dropna()

# Convert from EPSG:2056 to EPSG:4326 (lat, lon)
df['x_act_ekf'] = df['x_ekf'] + BikeZ_Config.X_2056_Bounds[0]
df['y_act_ekf'] = df['y_ekf'] + BikeZ_Config.Y_2056_Bounds[0]
transformer = Transformer.from_crs("EPSG:2056", "EPSG:4326", always_xy=True)
df["lon_ekf"], df["lat_ekf"] = transformer.transform(df["x_act_ekf"].values, df["y_act_ekf"].values)

# Create a folium map
center_lat, center_lon = df["lat_ekf"].mean(), df["lon_ekf"].mean()

# #############################################################################
# MAIN: Load Data (swissTLM3D)
# #############################################################################
gpkg_path = 'C:/Users/ShaimaaElBaklish/Documents/Datasets/BikeZ/swisstlm3d_2024-03_2056_5728.gpkg/SWISSTLM3D_2024_LV95_LN02.gpkg'

layers = fiona.listlayers(gpkg_path)
print(layers)

minx, miny, maxx, maxy = 2680000, 1240000, 2690000, 1250000
zurich_bbox = box(minx, miny, maxx, maxy)
bikez_bbox = (BikeZ_Config.X_2056_Bounds[0],
              BikeZ_Config.Y_2056_Bounds[0],
              BikeZ_Config.X_2056_Bounds[1],
              BikeZ_Config.Y_2056_Bounds[1])
geodf = gpd.read_file(gpkg_path, 
                      layer='tlm_strassen_strasse',
                      bbox=bikez_bbox)

print(geodf.crs)      # should be EPSG:2056
print(geodf.head())   # inspect columns

plt.figure()
ax = plt.gca()
grouped = df.groupby(by='veh_id')
for veh_id, veh_df in grouped:
    ax.plot(veh_df['x_act_ekf'], veh_df['y_act_ekf'], 'red')
geodf.plot(ax=ax)
plt.xlim(BikeZ_Config.X_2056_Bounds)
plt.ylim(BikeZ_Config.Y_2056_Bounds)
plt.tight_layout()


# #############################################################################
# MAIN: Try using SwissTopo app (export) + OSMNX
# #############################################################################
import osmnx as ox
from shapely.ops import linemerge, unary_union


# Create a folium map
center_lat, center_lon = df["lat_ekf"].mean(), df["lon_ekf"].mean()
# m = folium.Map(location=[center_lat, center_lon], zoom_start=20)
m = folium.Map(location=[center_lat, center_lon], zoom_start=20, tiles=None, control_scale=True)
# Add swisstopo basemap
folium.TileLayer(
    tiles="https://wmts.geo.admin.ch/1.0.0/ch.swisstopo.pixelkarte-farbe/default/current/3857/{z}/{x}/{y}.jpeg",
    attr="© swisstopo / geo.admin.ch",
    name="swisstopo.pixelkarte-farbe",
    overlay=False,
    control=True,
    max_zoom=25,
    min_zoom=0,
    subdomains=None,
    tms=False
).add_to(m)
# Optional: add orthophoto as another layer
folium.TileLayer(
    tiles="https://wmts.geo.admin.ch/1.0.0/ch.swisstopo.swissimage/default/current/3857/{z}/{x}/{y}.jpeg",
    attr="© swisstopo / geo.admin.ch",
    name="swisstopo.swissimage",
    overlay=True,
    control=True,
    max_zoom=25
).add_to(m)
# Add a layer control so you can toggle
folium.LayerControl().add_to(m)

# Define your area
place = "Zürich, Switzerland"
tags = {"highway": True} # Download all features with highway tag
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

left_branch = []
for i in [14, 0, 3, 9]:
    if len(left_branch) == 0:
        left_branch = centerline_coords_list[i]
        continue
    if left_branch[-1] == centerline_coords_list[i][0]:
        left_branch = left_branch + centerline_coords_list[i][1:]
    else:
        left_branch = left_branch + centerline_coords_list[i]
# left_branch = centerline_coords_list[14] + centerline_coords_list[0] + centerline_coords_list[3] + centerline_coords_list[9]
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
    tooltip="Gessnerbrücke Centerline"
).add_to(m)


# Link to edit drawing: https://map.geo.admin.ch/#/map?lang=en&center=2682815.13,1247903.71&z=13&topic=ech&layers=ch.swisstopo.zeitreihen@year=1864,f;ch.bfs.gebaeude_wohnungs_register,f;ch.bav.haltestellen-oev,f;ch.swisstopo.swisstlm3d-wanderwege,f;ch.vbs.schiessanzeigen,f;ch.astra.wanderland-sperrungen_umleitungen,f;KML%7Chttps://public.geo.admin.ch/api/kml/files/NUSEuRoGT_mOVN1edWFbEw,f;KML%7Chttps://public.geo.admin.ch/api/kml/files/AMdLJu9mRei9FqSDEVx51Q@adminId=vezMDMwQT0axzF76Z2JYjw&bgLayer=ch.swisstopo.swissimage&featureInfo=default

kml_path = "../maps/from_swisstopo/gessnerbrucke.kml"
gdf = gpd.read_file(kml_path, driver='KML')

print(gdf.columns)
print(gdf.crs)
print(gdf.head())

gdf_2056 = gdf.to_crs(epsg=2056)

# Add your drawn lines
for _, row in gdf.iterrows():
    geom_type = row.geometry.geom_type
    if geom_type == "LineString":
        coords = [(lat, lon) for lon, lat in row.geometry.coords]
        folium.PolyLine(
            locations=coords,
            color='orange',
            weight=5,
            opacity=0.8,
            tooltip="From SwissTopo"
        ).add_to(m)
    elif geom_type == "Polygon":
        coords = [(lat, lon) for lon, lat in row.geometry.exterior.coords]
        folium.Polygon(
            locations=coords,
            color='blue',
            fill=True, 
            fill_opacity=0.2,
            tooltip="From SwissTopo"
        ).add_to(m)


m.save(f"../maps/road_centerlines_map_{date}_{intersection}_{time_slot}_{code}.html")