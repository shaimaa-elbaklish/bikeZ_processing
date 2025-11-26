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
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import osmnx as ox
import geopandas as gpd
import matplotlib.pyplot as plt

from pyproj import Transformer
from shapely.geometry import LineString
from scipy.interpolate import splev
from shapely.ops import linemerge, unary_union

from _constants import BikeZ_Config
from tools_coordinate_transform import fit_roadway_centerline_spline
from tools_coordinate_transform import convert_roadway_to_xy2056_coordinates
from tools_coordinate_transform import connect_lines_g2
from tools_coordinate_transform import cut_line_at_stop
from tools_coordinate_transform import densify_linestring
from tools_coordinate_transform import _compute_curvature, _compute_distance_traveled

from tools_map_visualization import create_swisstopo_map
from tools_map_visualization import plot_spline_xy_2056
from tools_map_visualization import plot_line_xy_2056
from tools_map_visualization import plot_line_latlon
from tools_map_visualization import plot_bicycles_trajectories_xy_2056

# #############################################################################
# CONSTANTS
# #############################################################################
date = BikeZ_Config.avail_dates[0]
intersection = BikeZ_Config.avail_intersections[2]
time_slot = 'AM1'
code= 'E'

# conda env export --from-history > *name file*.yml

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
# MAIN: Extract Remaining Centerlines from SwissTopo
# #############################################################################
# Link to edit drawing: https://map.geo.admin.ch/#/map?lang=en&center=2682815.13,1247903.71&z=13&topic=ech&layers=ch.swisstopo.zeitreihen@year=1864,f;ch.bfs.gebaeude_wohnungs_register,f;ch.bav.haltestellen-oev,f;ch.swisstopo.swisstlm3d-wanderwege,f;ch.vbs.schiessanzeigen,f;ch.astra.wanderland-sperrungen_umleitungen,f;KML%7Chttps://public.geo.admin.ch/api/kml/files/NUSEuRoGT_mOVN1edWFbEw,f;KML%7Chttps://public.geo.admin.ch/api/kml/files/AMdLJu9mRei9FqSDEVx51Q@adminId=vezMDMwQT0axzF76Z2JYjw&bgLayer=ch.swisstopo.swissimage&featureInfo=default
# Share Link: https://s.geo.admin.ch/jkfynb8vzf5w


kml_path = "../maps/from_swisstopo/gessnerbrucke.kml"
gdf_swisstopo = gpd.read_file(kml_path, driver='KML')

row = gdf_swisstopo[gdf_swisstopo['Description'] == 'Observed_Area'].copy()
observed_area_polygon = row.geometry.item()

row = gdf_swisstopo[gdf_swisstopo['Description'] == 'Gessnerbrucke_Stopline'].copy()
gessnerbrucke_stopline = row.geometry.item()

row = gdf_swisstopo[gdf_swisstopo['Description'] == 'Gessnerbrucke_Centerline'].copy()
centerline = row.geometry.item()
centerline = centerline.intersection(observed_area_polygon)
gessnerbrucke_EW_branch = cut_line_at_stop(centerline, gessnerbrucke_stopline, choose='last')
gessnerbrucke_EW_branch = [(c[1], c[0]) for c in gessnerbrucke_EW_branch.coords]

row = gdf_swisstopo[gdf_swisstopo['Description'] == 'Stadttunnel_Centerline'].copy()
centerline = row.geometry.item()
# centerline = centerline.intersection(observed_area_polygon)
stadttunnel_branch = [(c[1], c[0]) for c in centerline.coords]

row = gdf_swisstopo[gdf_swisstopo['Description'] == 'Lagerstrasse_Stopline'].copy()
lagerstrasse_stopline = row.geometry.item()

row = gdf_swisstopo[gdf_swisstopo['Description'] == 'Kasernenstrasse_South_Stopline'].copy()
kasernenstrasse_south_stopline = row.geometry.item()

row = gdf_swisstopo[gdf_swisstopo['Description'] == 'Kasernenstrasse_North_Stopline'].copy()
kasernenstrasse_north_stopline = row.geometry.item()


# #############################################################################
# MAIN: Extract Available Centerlines from OSMNX
# #############################################################################
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

kasernenstrasse_NS_branch = []
for i in [14, 0, 3, 9]:
    if len(kasernenstrasse_NS_branch) == 0:
        kasernenstrasse_NS_branch = centerline_coords_list[i]
        continue
    if kasernenstrasse_NS_branch[-1] == centerline_coords_list[i][0]:
        kasernenstrasse_NS_branch = kasernenstrasse_NS_branch + centerline_coords_list[i][1:]
    else:
        kasernenstrasse_NS_branch = kasernenstrasse_NS_branch + centerline_coords_list[i]

kasernenstrasse_SN_branch = centerline_coords_list[4][1:]
kasernenstrasse_SN_branch = densify_linestring(latlon_pts=kasernenstrasse_SN_branch, num_segments=5)
kasernenstrasse_SN_branch = [(lat, lon) for lon, lat in kasernenstrasse_SN_branch.coords]

# Filter for road name
road_name = "Lagerstrasse"
gdf = gdf_main[gdf_main['name'] == road_name]
# Optional: filter to LineStrings only
gdf = gdf[(gdf.geometry.type == "LineString") & (gdf['highway'].isin(main_road_types))]
          # & (gdf['bicycle'].isin(bikeable))]
merged_centerline = linemerge(list(gdf.geometry))
merged_centerline = merged_centerline.intersection(observed_area_polygon)
merged_centerline = cut_line_at_stop(merged_centerline, lagerstrasse_stopline, choose='first')
# Extract coordinates (handle both LineString and MultiLineString)
if merged_centerline.geom_type == 'LineString':
    coords = list(merged_centerline.coords)
elif merged_centerline.geom_type == 'MultiLineString':
    coords = []
    for line in merged_centerline.geoms:
        coords.extend(list(line.coords))
# Convert (lon, lat) to (lat, lon) for folium
lagerstrasse_branch = [(lat, lon) for lon, lat, _ in coords]

# Filter for road name
road_name = "Gessnerbrücke"
gdf = gdf_main[gdf_main['name'] == road_name]
# Optional: filter to LineStrings only
gdf = gdf[(gdf.geometry.type == "LineString") & (gdf['highway'].isin(main_road_types))]
          # & (gdf['bicycle'].isin(bikeable))]
merged_centerline = linemerge(list(gdf.geometry))
merged_centerline = merged_centerline.intersection(observed_area_polygon)
merged_centerline = cut_line_at_stop(merged_centerline, gessnerbrucke_stopline, choose='last')
# Extract coordinates (handle both LineString and MultiLineString)
if merged_centerline.geom_type == 'LineString':
    coords = list(merged_centerline.coords)
elif merged_centerline.geom_type == 'MultiLineString':
    coords = []
    for line in merged_centerline.geoms:
        coords.extend(list(line.coords))
# Convert (lon, lat) to (lat, lon) for folium
gessnerbrucke_WE_branch = [(lat, lon) for lon, lat, _ in coords]


# #############################################################################
# MAIN: Create Folium map with SwissTopo base image
# #############################################################################
# Create a folium map
m = create_swisstopo_map(center_lat=df["lat_ekf"].mean(), center_lon=df["lon_ekf"].mean(), add_layer_control=False)

fg1 = folium.FeatureGroup(name="Split Centerlines", show=False)
fg2 = folium.FeatureGroup(name="Clothoid Centerlines", show=False)
fg3 = folium.FeatureGroup(name="Hermite Centerlines", show=False)

# #############################################################################
# MAIN: Get Through South -> North Spline
# #############################################################################
# south_north_spl = fit_roadway_centerline_spline(kasernenstrasse_SN_branch + stadttunnel_branch, smoothing=0.25)
# plot_spline_xy_2056(m, south_north_spl, label="Through (S->N)", 
#                     linecolor="blue", linedashed=True, start_point=True)


# #############################################################################
# MAIN: Get Turning South -> East Spline
# #############################################################################
# Cut kasernenstrasse_SN_branch at stopline
centerline = LineString([(lon, lat) for lat, lon in kasernenstrasse_SN_branch])
centerline = cut_line_at_stop(centerline, kasernenstrasse_south_stopline, choose='first')
kasernenstrasse_SN_branch = [(lat, lon) for lon, lat in centerline.coords]

tmp_spl = fit_roadway_centerline_spline(kasernenstrasse_SN_branch)
tck = tmp_spl[0]
x_spline, y_spline = splev(np.linspace(0, 1, 50), tck)
xy_kasernenstrasse_south = np.column_stack((x_spline, y_spline))

tmp_spl = fit_roadway_centerline_spline(gessnerbrucke_WE_branch)
tck = tmp_spl[0]
x_spline, y_spline = splev(np.linspace(0, 1, 50), tck)
xy_gessnerbrucke_east = np.column_stack((x_spline, y_spline))

plot_line_xy_2056(fg1, xy_kasernenstrasse_south[:, 0], xy_kasernenstrasse_south[:, 1], 
                  label="Kasernenstrasse Centerline (S->N)", linecolor='red', linedashed=False, start_point=True)
plot_line_xy_2056(fg1, xy_gessnerbrucke_east[:, 0], xy_gessnerbrucke_east[:, 1], 
                  label="Gessnerbrücke Centerline", linecolor='red', linedashed=False, start_point=True)


# Clothoid
south_east_merged_coords, _, method = connect_lines_g2(xy_kasernenstrasse_south, xy_gessnerbrucke_east, n_connector=120, verbose=True)
print(f"S_2_E using clothoid: final method = {method}")
south_east_spl_C = fit_roadway_centerline_spline(south_east_merged_coords, coordsys='2056')
plot_spline_xy_2056(fg2, south_east_spl_C, label="Clothoid (S->E)", 
                    linecolor="blue", linedashed=True, start_point=True)

# Hermite
south_east_merged_coords, _, _= connect_lines_g2(xy_kasernenstrasse_south, xy_gessnerbrucke_east, n_connector=120, verbose=True,
                                                 force_method="hermite", scale=0.6)
south_east_spl_H = fit_roadway_centerline_spline(south_east_merged_coords, coordsys='2056')
plot_spline_xy_2056(fg3, south_east_spl_H, label="Clothoid (S->E)", 
                    linecolor="yellow", linedashed=True, start_point=True)


def _compute_tangents(xy, eps=1e-06):
    dx = np.gradient(xy[:, 0])
    dy = np.gradient(xy[:, 1])
    return dy / (dx + eps), np.rad2deg(np.arctan2(dy, dx))


# Compare Tangents and Curvature
u = np.linspace(0, 1, 1000)
x_spline, y_spline = splev(u, south_east_spl_C[0])
xy_C = np.column_stack((x_spline, y_spline))
curv_C = _compute_curvature(xy_C)
s_C, _ = _compute_distance_traveled(xy_C)
_, tang_C = _compute_tangents(xy_C)
x_spline, y_spline = splev(u, south_east_spl_H[0])
xy_H = np.column_stack((x_spline, y_spline))
curv_H = _compute_curvature(xy_H)
_, tang_H = _compute_tangents(xy_H)
s_H, _ = _compute_distance_traveled(xy_H)

curv_1 = _compute_curvature(xy_kasernenstrasse_south)
_, tang_1 = _compute_tangents(xy_kasernenstrasse_south)
s_1, _ = _compute_distance_traveled(xy_kasernenstrasse_south)
curv_2 = _compute_curvature(xy_gessnerbrucke_east)
_, tang_2 = _compute_tangents(xy_gessnerbrucke_east)
s_2, _ = _compute_distance_traveled(xy_gessnerbrucke_east)
s_2 = s_2  + 98.5

plt.figure("S_2_E", figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.plot(s_C, curv_C, label="Clothoid", alpha=0.75)
plt.plot(s_H, curv_H, label="Hermite", alpha=0.75, linestyle="--")
plt.plot(s_1, curv_1, label="Split C1", alpha=0.75)
plt.plot(s_2, curv_2, label="Split C2", alpha=0.75)
plt.xlabel("Arc Length")
plt.ylabel("Curvature")

plt.subplot(1, 2, 2)
plt.plot(s_C, tang_C, label="Clothoid", alpha=0.75)
plt.plot(s_H, tang_H, label="Hermite", alpha=0.75, linestyle="--")
plt.plot(s_1, tang_1, label="Split C1", alpha=0.75)
plt.plot(s_2, tang_2, label="Split C2", alpha=0.75)
plt.xlabel("Arc Length")
plt.ylabel("Tangent Angle")
plt.legend()
plt.tight_layout()


# #############################################################################
# MAIN: Get Turning South -> West Spline
# #############################################################################
lagerstrasse_branch = densify_linestring(latlon_pts=lagerstrasse_branch, num_segments=5)
lagerstrasse_branch = [(lat, lon) for lon, lat in lagerstrasse_branch.coords]
east_west_spl = fit_roadway_centerline_spline(lagerstrasse_branch)
tck, unew, cum_dist = east_west_spl
all_s = cum_dist
all_d = 3.0 * np.ones_like(all_s)
x_spline = np.zeros_like(all_s)
y_spline = np.zeros_like(all_s)
for i in range(x_spline.shape[0]):
    x_spline[i], y_spline[i] = convert_roadway_to_xy2056_coordinates(all_s[i], all_d[i], tck, unew, cum_dist)

xy_lagerstrasse_west = np.column_stack((x_spline, y_spline))

plot_line_latlon(fg1, kasernenstrasse_SN_branch, label="Kasernenstrasse Centerline (S->N)", linecolor='red', linedashed=False, start_point=True)
plot_line_xy_2056(fg1, xy_lagerstrasse_west[:, 0], xy_lagerstrasse_west[:, 1], label="Lagerstrasse West Centerline", linecolor='red', linedashed=False, start_point=True)

# CLothoid
south_west_merged_coords, _, method = connect_lines_g2(xy_kasernenstrasse_south, xy_lagerstrasse_west[::-1], n_connector=120, verbose=True)
print(f"S_2_W using clothoid: final method = {method}")
south_west_spl_C = fit_roadway_centerline_spline(south_west_merged_coords, coordsys='2056')
plot_spline_xy_2056(fg2, south_west_spl_C, label="Clothoid (S->W)", 
                    linecolor="blue", linedashed=True, start_point=True)

# Hermite
south_west_merged_coords, _, _= connect_lines_g2(xy_kasernenstrasse_south, xy_lagerstrasse_west[::-1], n_connector=120, verbose=True,
                                                 force_method="hermite", scale=0.6)
south_west_spl_H = fit_roadway_centerline_spline(south_west_merged_coords, coordsys='2056')
plot_spline_xy_2056(fg3, south_west_spl_H, label="Hermite (S->W)", 
                    linecolor="yellow", linedashed=True, start_point=True)


# Compare Tangents and Curvature
u = np.linspace(0, 1, 1000)
x_spline, y_spline = splev(u, south_west_spl_C[0])
xy_C = np.column_stack((x_spline, y_spline))
curv_C = _compute_curvature(xy_C)
s_C, _ = _compute_distance_traveled(xy_C)
_, tang_C = _compute_tangents(xy_C)
x_spline, y_spline = splev(u, south_west_spl_H[0])
xy_H = np.column_stack((x_spline, y_spline))
curv_H = _compute_curvature(xy_H)
_, tang_H = _compute_tangents(xy_H)
s_H, _ = _compute_distance_traveled(xy_H)

curv_1 = _compute_curvature(xy_kasernenstrasse_south)
_, tang_1 = _compute_tangents(xy_kasernenstrasse_south)
s_1, _ = _compute_distance_traveled(xy_kasernenstrasse_south)
curv_2 = _compute_curvature(xy_lagerstrasse_west[::-1])
_, tang_2 = _compute_tangents(xy_lagerstrasse_west[::-1])
s_2, _ = _compute_distance_traveled(xy_lagerstrasse_west[::-1])
s_2 = s_2  + 117

plt.figure("S_2_W", figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.plot(s_C, curv_C, label="Clothoid", alpha=0.75)
plt.plot(s_H, curv_H, label="Hermite", alpha=0.75, linestyle="--")
plt.plot(s_1, curv_1, label="Split C1", alpha=0.75)
plt.plot(s_2, curv_2, label="Split C2", alpha=0.75)
plt.xlabel("Arc Length")
plt.ylabel("Curvature")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(s_C, tang_C, label="Clothoid", alpha=0.75)
plt.plot(s_H, tang_H, label="Hermite", alpha=0.75, linestyle="--")
plt.plot(s_1, tang_1, label="Split C1", alpha=0.75)
plt.plot(s_2, tang_2, label="Split C2", alpha=0.75)
plt.xlabel("Arc Length")
plt.ylabel("Tangent Angle")
plt.legend()
plt.tight_layout()


# #############################################################################
# MAIN: Get Turning North -> East Spline
# #############################################################################
# Cut kasernenstrasse_NS_branch at stopline
centerline = LineString([(lon, lat) for lat, lon in kasernenstrasse_NS_branch])
centerline = cut_line_at_stop(centerline, kasernenstrasse_north_stopline, choose='first')
centerline = densify_linestring(centerline, num_segments=5)
kasernenstrasse_NS_branch_1 = [(lat, lon) for lon, lat in centerline.coords]

tmp_spl = fit_roadway_centerline_spline(kasernenstrasse_NS_branch_1)
tck = tmp_spl[0]
x_spline, y_spline = splev(np.linspace(0, 1, 50), tck)
xy_kasernenstrasse_north = np.column_stack((x_spline, y_spline))

plot_line_latlon(fg1, kasernenstrasse_NS_branch_1, label="Kasernenstrasse Centerline (N->S)", 
                 linecolor='red', linedashed=False, start_point=False)
plot_line_xy_2056(fg1, xy_gessnerbrucke_east[:, 0], xy_gessnerbrucke_east[:, 1], 
                  label="Gessnerbrücke East Centerline", linecolor='red', linedashed=False, start_point=False)

# Clothoid
north_east_merged_coords, _, _= connect_lines_g2(xy_kasernenstrasse_north, xy_gessnerbrucke_east, n_connector=120, verbose=True)
north_east_spl = fit_roadway_centerline_spline(north_east_merged_coords, coordsys='2056')
plot_spline_xy_2056(fg2, north_east_spl, label="Clothoid (N->E)", 
                    linecolor="blue", linedashed=True, start_point=True)

# Hermite
north_east_merged_coords, _, _= connect_lines_g2(xy_kasernenstrasse_north, xy_gessnerbrucke_east, n_connector=120, verbose=True,
                                                 force_method="hermite")
north_east_spl = fit_roadway_centerline_spline(north_east_merged_coords, coordsys='2056')
plot_spline_xy_2056(fg3, north_east_spl, label="Hermite (N->E)", 
                    linecolor="yellow", linedashed=True, start_point=True)


# #############################################################################
# MAIN: Get Turning North -> West Spline
# #############################################################################

# Clothoid
north_west_merged_coords, _, _= connect_lines_g2(xy_kasernenstrasse_north, xy_lagerstrasse_west[::-1], n_connector=120, verbose=True)
north_west_spl = fit_roadway_centerline_spline(north_west_merged_coords, coordsys='2056')
plot_spline_xy_2056(fg2, north_west_spl, label="Clothoid (N->W)", 
                    linecolor="blue", linedashed=True, start_point=True)

# Hermite
north_west_merged_coords, _, _= connect_lines_g2(xy_kasernenstrasse_north, xy_lagerstrasse_west[::-1], n_connector=120, verbose=True,
                                                 force_method="hermite")
north_west_spl = fit_roadway_centerline_spline(north_west_merged_coords, coordsys='2056')
plot_spline_xy_2056(fg3, north_west_spl, label="Hermite (N->W)", 
                    linecolor="yellow", linedashed=True, start_point=True)


# Add feature groups and layer control
fg1.add_to(m)
fg2.add_to(m)
fg3.add_to(m)
# folium.LayerControl(collapsed=False).add_to(m)

plot_bicycles_trajectories_xy_2056(m, df, add_layer_control=True)

m.save(f"../maps/road_centerlines_map_{date}_{intersection}_clothoid_vs_hermite.html")
