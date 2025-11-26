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
from shapely.geometry import LineString
from scipy.interpolate import splev
from shapely.ops import linemerge, unary_union

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
m = create_swisstopo_map(center_lat=df["lat_ekf"].mean(), center_lon=df["lon_ekf"].mean())

# Create splines dictionary for saving
splines_dict = {}

colors_dict = {
    'north': 'blue',
    'south': 'red',
    'west': 'green',
    'east': 'yellow',
}

# #############################################################################
# MAIN: Get Through North -> South Spline
# #############################################################################
north_south_spl = fit_roadway_centerline_spline(kasernenstrasse_NS_branch) # tuple (tck, unew, cum_dist)
plot_spline_xy_2056(m, north_south_spl, label="Kasernenstrasse Centerline (N->S)", 
                    linecolor=colors_dict['north'], linedashed=True, start_point=True)

splines_dict['N_2_S'] = north_south_spl

# #############################################################################
# MAIN: Get Through South -> North Spline
# #############################################################################
south_north_spl = fit_roadway_centerline_spline(kasernenstrasse_SN_branch + stadttunnel_branch, smoothing=0.25)
plot_spline_xy_2056(m, south_north_spl, label="Kasernenstrasse Centerline (S->N)", 
                    linecolor=colors_dict['south'], linedashed=True, start_point=True)

splines_dict['S_2_N'] = south_north_spl

# #############################################################################
# MAIN: Get Through East -> West Spline
# #############################################################################
east_west_spl = fit_roadway_centerline_spline(lagerstrasse_branch)
tck, unew, cum_dist = east_west_spl
all_s = cum_dist
all_d = 3.0 * np.ones_like(all_s)
x_spline = np.zeros_like(all_s)
y_spline = np.zeros_like(all_s)
for i in range(x_spline.shape[0]):
    x_spline[i], y_spline[i] = convert_roadway_to_xy2056_coordinates(all_s[i], all_d[i], tck, unew, cum_dist)

# plot_line_xy_2056(m, x_spline, y_spline, label="Lagerstrasse Centerline (UP)", linecolor='red', linedashed=False, start_point=True)
# plot_line_latlon(m, gessnerbrucke_EW_branch, label="@SwissTopo: Gessnerbrücke Centerline", linecolor='red', linedashed=False, start_point=True)

xy_lagerstrasse_west = np.column_stack((x_spline, y_spline))
tmp_spl = fit_roadway_centerline_spline(gessnerbrucke_EW_branch)
tck = tmp_spl[0]
x_spline, y_spline = splev(np.linspace(0, 1, 50), tck)
xy_gessnerbrucke_west = np.column_stack((x_spline, y_spline))
east_west_merged_coords, _, _= connect_lines_g2(xy_gessnerbrucke_west[::-1], xy_lagerstrasse_west[::-1], n_connector=120, verbose=True)
east_west_spl = fit_roadway_centerline_spline(east_west_merged_coords, coordsys='2056')
plot_spline_xy_2056(m, east_west_spl, label="Lagerstrasse Centerline (E->W)", 
                    linecolor=colors_dict['east'], linedashed=True, start_point=True)

splines_dict['E_2_W'] = east_west_spl

# #############################################################################
# MAIN: Get Through West -> East Spline
# #############################################################################
west_east_spl = fit_roadway_centerline_spline(lagerstrasse_branch)
tck, unew, cum_dist = west_east_spl
all_s = cum_dist
all_d = -3.0 * np.ones_like(all_s)
x_spline = np.zeros_like(all_s)
y_spline = np.zeros_like(all_s)
for i in range(x_spline.shape[0]):
    x_spline[i], y_spline[i] = convert_roadway_to_xy2056_coordinates(all_s[i], all_d[i], tck, unew, cum_dist)

# plot_line_xy_2056(m, x_spline, y_spline, label="Lagerstrasse Centerline (DOWN)", linecolor='red', linedashed=False)
# plot_line_latlon(m, gessnerbrucke_WE_branch, label="Gessnerbrücke Centerline", linecolor='red', linedashed=False)

xy_lagerstrasse_east = np.column_stack((x_spline, y_spline))
tmp_spl = fit_roadway_centerline_spline(gessnerbrucke_WE_branch)
tck = tmp_spl[0]
x_spline, y_spline = splev(np.linspace(0, 1, 50), tck)
xy_gessnerbrucke_east = np.column_stack((x_spline, y_spline))
west_east_merged_coords, _, _= connect_lines_g2(xy_lagerstrasse_east, xy_gessnerbrucke_east, n_connector=120, verbose=True)
west_east_spl = fit_roadway_centerline_spline(west_east_merged_coords, coordsys='2056')
plot_spline_xy_2056(m, west_east_spl, label="Lagerstrasse Centerline (W->E)",
                    linecolor=colors_dict['west'], linedashed=True, start_point=True)

splines_dict['W_2_E'] = west_east_spl

# #############################################################################
# MAIN: Get Turning South -> East Spline
# #############################################################################
# Cut kasernenstrasse_SN_branch at stopline
centerline = LineString([(lon, lat) for lat, lon in kasernenstrasse_SN_branch])
centerline = cut_line_at_stop(centerline, kasernenstrasse_south_stopline, choose='first')
kasernenstrasse_SN_branch = [(lat, lon) for lon, lat in centerline.coords]

# plot_line_latlon(m, kasernenstrasse_SN_branch, label="Kasernenstrasse Centerline (S->N)", linecolor='red', linedashed=False)
# plot_line_latlon(m, gessnerbrucke_WE_branch, label="Gessnerbrücke Centerline", linecolor='red', linedashed=False)

tmp_spl = fit_roadway_centerline_spline(kasernenstrasse_SN_branch)
tck = tmp_spl[0]
x_spline, y_spline = splev(np.linspace(0, 1, 50), tck)
xy_kasernenstrasse_south = np.column_stack((x_spline, y_spline))

south_east_merged_coords, _, _= connect_lines_g2(xy_kasernenstrasse_south, xy_gessnerbrucke_east, n_connector=120, verbose=True)
south_east_spl = fit_roadway_centerline_spline(south_east_merged_coords, coordsys='2056')
plot_spline_xy_2056(m, south_east_spl, label="Turning Centerline (S->E)", 
                    linecolor=colors_dict['south'], linedashed=True, start_point=True)

splines_dict['S_2_E'] = south_east_spl

# #############################################################################
# MAIN: Get Turning South -> West Spline
# #############################################################################

# plot_line_latlon(m, kasernenstrasse_SN_branch, label="Kasernenstrasse Centerline (S->N)", linecolor='red', linedashed=False, start_point=True)
# plot_line_xy_2056(m, xy_lagerstrasse_west[:, 0], xy_lagerstrasse_west[:, 1], label="Lagerstrasse West Centerline", linecolor='red', linedashed=False, start_point=True)

south_west_merged_coords, _, _= connect_lines_g2(xy_kasernenstrasse_south, xy_lagerstrasse_west[::-1], n_connector=120, verbose=True)
south_west_spl = fit_roadway_centerline_spline(south_west_merged_coords, coordsys='2056')
plot_spline_xy_2056(m, south_west_spl, label="Turning Centerline (S->W)", 
                    linecolor=colors_dict['south'], linedashed=True, start_point=True)

splines_dict['S_2_W'] = south_west_spl

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

# plot_line_latlon(m, kasernenstrasse_NS_branch_1, label="Kasernenstrasse Centerline (N->S)", linecolor='red', linedashed=False, start_point=True)
# plot_line_xy_2056(m, xy_gessnerbrucke_east[:, 0], xy_gessnerbrucke_east[:, 1], label="Gessnerbrücke East Centerline", linecolor='red', linedashed=False, start_point=True)

north_east_merged_coords, _, _= connect_lines_g2(xy_kasernenstrasse_north, xy_gessnerbrucke_east, n_connector=120, verbose=True)
north_east_spl = fit_roadway_centerline_spline(north_east_merged_coords, coordsys='2056')
plot_spline_xy_2056(m, north_east_spl, label="Turning Centerline (N->E)", 
                    linecolor=colors_dict['north'], linedashed=True, start_point=True)

splines_dict['N_2_E'] = north_east_spl

# #############################################################################
# MAIN: Get Turning North -> West Spline
# #############################################################################

# plot_line_latlon(m, kasernenstrasse_NS_branch_1, label="Kasernenstrasse Centerline (N->S)", linecolor='red', linedashed=False, start_point=True)
# plot_line_xy_2056(m, xy_lagerstrasse_west[:, 0], xy_lagerstrasse_west[:, 1], label="Lagerstrasse West Centerline", linecolor='red', linedashed=False, start_point=True)

north_west_merged_coords, _, _= connect_lines_g2(xy_kasernenstrasse_north, xy_lagerstrasse_west[::-1], n_connector=120, verbose=True)
north_west_spl = fit_roadway_centerline_spline(north_west_merged_coords, coordsys='2056')
plot_spline_xy_2056(m, north_west_spl, label="Turning Centerline (N->W)", 
                    linecolor=colors_dict['north'], linedashed=True, start_point=True)

splines_dict['N_2_W'] = north_west_spl

# #############################################################################
# MAIN: Get Turning West -> North Spline
# #############################################################################
# Cut stadttunnel_branch at stopline
centerline = LineString([(lon, lat) for lat, lon in stadttunnel_branch])
centerline = cut_line_at_stop(centerline, kasernenstrasse_north_stopline, choose='last')
stadttunnel_branch = [(lat, lon) for lon, lat in centerline.coords]

# plot_line_latlon(m, stadttunnel_branch, label="Stadttunnel Centerline", linecolor='red', linedashed=False, start_point=True)
# plot_line_xy_2056(m, xy_lagerstrasse_east[:, 0], xy_lagerstrasse_east[:, 1], label="Lagerstrasse East Centerline", linecolor='red', linedashed=False, start_point=True)

tmp_spl = fit_roadway_centerline_spline(stadttunnel_branch)
tck = tmp_spl[0]
x_spline, y_spline = splev(np.linspace(0, 1, 50), tck)
xy_stadttunnel = np.column_stack((x_spline, y_spline))

west_north_merged_coords, _, _= connect_lines_g2(xy_lagerstrasse_east, xy_stadttunnel, n_connector=120, verbose=True)
west_north_spl = fit_roadway_centerline_spline(west_north_merged_coords, coordsys='2056')
plot_spline_xy_2056(m, west_north_spl, label="Turning Centerline (W->N)",
                    linecolor=colors_dict['west'], linedashed=True, start_point=True)

splines_dict['W_2_N'] = west_north_spl

# #############################################################################
# MAIN: Get Turning West -> South Spline
# #############################################################################
# Cut kasernenstrasse_NS_branch at stopline
centerline = LineString([(lon, lat) for lat, lon in kasernenstrasse_NS_branch])
centerline = cut_line_at_stop(centerline, kasernenstrasse_south_stopline, choose='last')
# centerline = densify_linestring(centerline, num_segments=5)
kasernenstrasse_NS_branch_2 = [(lat, lon) for lon, lat in centerline.coords]

# plot_line_latlon(m, kasernenstrasse_NS_branch_2, label="Kasernenstrasse Centerline (N->S)", linecolor='red', linedashed=False, start_point=True)
# plot_line_xy_2056(m, xy_lagerstrasse_east[:, 0], xy_lagerstrasse_east[:, 1], label="Lagerstrasse East Centerline", linecolor='red', linedashed=False, start_point=True)

tmp_spl = fit_roadway_centerline_spline(kasernenstrasse_NS_branch_2)
tck = tmp_spl[0]
x_spline, y_spline = splev(np.linspace(0, 1, 50), tck)
xy_kasernenstrasse_south = np.column_stack((x_spline, y_spline))

west_south_merged_coords, _, _= connect_lines_g2(xy_lagerstrasse_east, xy_kasernenstrasse_south, n_connector=120, verbose=True)
west_south_spl = fit_roadway_centerline_spline(west_south_merged_coords, coordsys='2056')
plot_spline_xy_2056(m, west_south_spl, label="Turning Centerline (W->S)", 
                    linecolor=colors_dict['west'], linedashed=True, start_point=True)

splines_dict['W_2_S'] = west_south_spl

# #############################################################################
# MAIN: Get Turning East -> North Spline
# #############################################################################

# plot_line_latlon(m, stadttunnel_branch, label="Stadttunnel Centerline", linecolor='red', linedashed=False, start_point=True)
# plot_line_xy_2056(m, xy_gessnerbrucke_west[:, 0], xy_gessnerbrucke_west[:, 1], label="Gessnerbrücke West Centerline", linecolor='red', linedashed=False, start_point=True)

east_north_merged_coords, _, _= connect_lines_g2(xy_gessnerbrucke_west[::-1], xy_stadttunnel, n_connector=120, verbose=True)
east_north_spl = fit_roadway_centerline_spline(east_north_merged_coords, coordsys='2056')
plot_spline_xy_2056(m, east_north_spl, label="Turning Centerline (E->N)", 
                    linecolor=colors_dict['east'], linedashed=True, start_point=True)

splines_dict['E_2_N'] = east_north_spl

# #############################################################################
# MAIN: Get Turning East -> South Spline
# #############################################################################

# plot_line_latlon(m, kasernenstrasse_NS_branch_2, label="Stadttunnel Centerline", linecolor='red', linedashed=False, start_point=True)
# plot_line_xy_2056(m, xy_gessnerbrucke_west[:, 0], xy_gessnerbrucke_west[:, 1], label="Gessnerbrücke West Centerline", linecolor='red', linedashed=False, start_point=True)

east_south_merged_coords, _, _= connect_lines_g2(xy_gessnerbrucke_west[::-1], xy_kasernenstrasse_south, n_connector=120, verbose=True)
east_south_spl = fit_roadway_centerline_spline(east_south_merged_coords, coordsys='2056')
plot_spline_xy_2056(m, east_south_spl, label="Turning Centerline (E->S)", 
                    linecolor=colors_dict['east'], linedashed=True, start_point=True)

splines_dict['E_2_S'] = east_south_spl


# #############################################################################
# MAIN: Saving Map and Splines
# #############################################################################
with open(f"../data/centerlines_splines_{date}_{intersection}.pkl", "wb") as f:
    pickle.dump(splines_dict, f)

m.save(f"../maps/road_centerlines_map_{date}_{intersection}_debugging.html")


# #############################################################################
# MAIN: Plotting and Saving FINAL Map
# #############################################################################
# Create a folium map
m = create_swisstopo_map(center_lat=df["lat_ekf"].mean(), center_lon=df["lon_ekf"].mean(), add_layer_control=False)
plot_all_centerlines_splines_xy_2056(m, splines_dict, add_layer_control=True)

m.save(f"../maps/road_centerlines_map_{date}_{intersection}.html")



m = create_swisstopo_map(center_lat=df["lat_ekf"].mean(), center_lon=df["lon_ekf"].mean(), add_layer_control=False)
plot_all_centerlines_splines_xy_2056(m, splines_dict, add_layer_control=False)
plot_bicycles_trajectories_xy_2056(m, df, linecolor='black', linealpha=0.25, add_layer_control=True)

m.save(f"../maps/trajectories_map_{date}_{intersection}_{timeslot}_{code}.html")


# #############################################################################
# MAIN: Get Lane Boundary Splines
# #############################################################################
lane_boundaries_splines_dict = {}

import matplotlib.pyplot as plt

plt.figure()

# North (Southbound direction)
row = gdf_swisstopo[gdf_swisstopo['Description'] == 'Kasernenstrasse_NSup_Bike_Lane_Boundary'].copy()
bike_lane_boundary = row.geometry.item()
bike_lane_boundary = [(lat, lon) for lon, lat in bike_lane_boundary.coords]

bike_lb_spl = fit_roadway_centerline_spline(bike_lane_boundary[::-1], smoothing=0.1) # tuple (tck, unew, cum_dist)
lane_boundaries_splines_dict['N_SB'] = bike_lb_spl

x_lb, y_lb = splev(np.linspace(0, 1, 50), bike_lb_spl[0])
plt.plot(x_lb, y_lb, label='N_SB')
plt.scatter(x_lb[[0, -1]], y_lb[[0, -1]], c=['black', 'red'])

# South (Southbound direction)
row = gdf_swisstopo[gdf_swisstopo['Description'] == 'Kasernenstrasse_NS_Bike_Lane_Boundary'].copy()
bike_lane_boundary = row.geometry.item()
bike_lane_boundary = [(lat, lon) for lon, lat in bike_lane_boundary.coords]

bike_lb_spl = fit_roadway_centerline_spline(bike_lane_boundary, smoothing=0.1) # tuple (tck, unew, cum_dist)
lane_boundaries_splines_dict['S_SB'] = bike_lb_spl

x_lb, y_lb = splev(np.linspace(0, 1, 50), bike_lb_spl[0])
plt.plot(x_lb, y_lb, label='S_SB')
plt.scatter(x_lb[[0, -1]], y_lb[[0, -1]], c=['black', 'red'])

# South (Northbound direction)
row = gdf_swisstopo[gdf_swisstopo['Description'] == 'Kasernenstrasse_SN_Bike_Lane_Boundary'].copy()
bike_lane_boundary = row.geometry.item()
bike_lane_boundary = [(lat, lon) for lon, lat in bike_lane_boundary.coords]

bike_lb_spl = fit_roadway_centerline_spline(bike_lane_boundary[::-1], smoothing=0.1) # tuple (tck, unew, cum_dist)
lane_boundaries_splines_dict['S_NB'] = bike_lb_spl

x_lb, y_lb = splev(np.linspace(0, 1, 50), bike_lb_spl[0])
plt.plot(x_lb, y_lb, label='S_NB')
plt.scatter(x_lb[[0, -1]], y_lb[[0, -1]], c=['black', 'red'])

# West (Eastbound direction)
row = gdf_swisstopo[gdf_swisstopo['Description'] == 'Lagerstrasse_WE_Bike_Lane_Boundary'].copy()
bike_lane_boundary = row.geometry.item()
bike_lane_boundary = [(lat, lon) for lon, lat in bike_lane_boundary.coords]

bike_lb_spl = fit_roadway_centerline_spline(bike_lane_boundary[::-1], smoothing=0.1) # tuple (tck, unew, cum_dist)
lane_boundaries_splines_dict['W_EB'] = bike_lb_spl

x_lb, y_lb = splev(np.linspace(0, 1, 50), bike_lb_spl[0])
plt.plot(x_lb, y_lb, label='W_EB')
plt.scatter(x_lb[[0, -1]], y_lb[[0, -1]], c=['black', 'red'])

# West (Westbound direction)
row = gdf_swisstopo[gdf_swisstopo['Description'] == 'Lagerstrasse_EW_Bike_Lane_Boundary'].copy()
bike_lane_boundary = row.geometry.item()
bike_lane_boundary = [(lat, lon) for lon, lat in bike_lane_boundary.coords]

bike_lb_spl = fit_roadway_centerline_spline(bike_lane_boundary, smoothing=0.1) # tuple (tck, unew, cum_dist)
lane_boundaries_splines_dict['W_WB'] = bike_lb_spl

x_lb, y_lb = splev(np.linspace(0, 1, 50), bike_lb_spl[0])
plt.plot(x_lb, y_lb, label='W_WB')
plt.scatter(x_lb[[0, -1]], y_lb[[0, -1]], c=['black', 'red'])
plt.legend()
plt.tight_layout()

# Identify the rest (i.e. without boundaries) as "bicycles" or "cars" or "mixed"
lane_boundaries_splines_dict['E_EB'] = "mixed"
lane_boundaries_splines_dict['E_WB'] = "bicycles"
lane_boundaries_splines_dict['N_NB'] = "bicycles"

# Save
with open(f"../data/bike_lane_boundaries_splines_{date}_{intersection}.pkl", "wb") as f:
    pickle.dump(lane_boundaries_splines_dict, f)
