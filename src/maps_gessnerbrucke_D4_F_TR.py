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
import matplotlib.pyplot as plt
from tools_map_visualization import plot_bicycles_trajectories_xy_2056
from tools_map_visualization import plot_all_centerlines_splines_xy_2056
from tools_map_visualization import plot_line_latlon
from tools_map_visualization import plot_line_xy_2056
from tools_map_visualization import plot_spline_xy_2056
from tools_map_visualization import create_swisstopo_map
from tools_coordinate_transform import densify_linestring
from tools_coordinate_transform import cut_line_at_stop
from tools_coordinate_transform import connect_lines_g2
from tools_coordinate_transform import convert_roadway_to_xy2056_coordinates
from tools_coordinate_transform import fit_roadway_centerline_spline
from _constants import BikeZ_Config
from shapely.ops import linemerge, unary_union
from scipy.interpolate import splev
from shapely.geometry import LineString
from pyproj import Transformer
import geopandas as gpd
import osmnx as ox
import pandas as pd
import numpy as np
import sys
import folium
import pickle
import warnings
warnings.filterwarnings("ignore")


def plot_idx_merged(centerline_coords_list):
    """ Plotting if needed the OSM links to know which one we need"""
    plt.figure()
    for idx, i in enumerate(centerline_coords_list):
        arr = np.array(i)
        plt.plot(arr[:, 1], arr[:, 0])
        plt.text(np.mean(arr[:, 1]), np.mean(arr[:, 0]), f"Idx{idx}")
    plt.show()


def get_centerl_from_swisstopo(gdf_swisstopo, name_centerline, num_seg_dens=20):
    row = gdf_swisstopo[gdf_swisstopo['Description']
                        == name_centerline].copy()
    centerline = row.geometry.item()
    centerline = densify_linestring(
        centerline, num_segments=num_seg_dens)
    cent_coord = [(c[1], c[0]) for c in centerline.coords]
    return cent_coord


# #############################################################################
# CONSTANTS
# #############################################################################
# Configuration
BikeZ_Config = BikeZ_Config()

# Specify Trajectory File
date = BikeZ_Config.avail_dates[0]
campaign = f"Zurich_2025{date[5:7]}"  # June or September
mode = BikeZ_Config.avail_modes[0]  # Bike
data_root = BikeZ_Config.data_root[campaign][mode]

intersection, code = BikeZ_Config.avail_intersections[date][4]
timeslot = BikeZ_Config.avail_timeslots[date][(intersection, code)][0]  # 'AM1'

XY_2056_Bounds = BikeZ_Config.XY_2056_Bounds[date][(intersection, code)]
X_2056_offset = XY_2056_Bounds[0][0]
Y_2056_offset = XY_2056_Bounds[1][0]

# #############################################################################
# MAIN: Load Data (Trajectories)
# #############################################################################
<<<<<<< Updated upstream
filename = f"trajectories_bikes_{date}_{
    intersection}_{timeslot}_{code}-1-ekf.csv"
=======
filename = f"trajectories_bikes_{date}_{intersection}_{timeslot}_{code}-1.csv"
>>>>>>> Stashed changes
df = pd.read_csv(data_root + f"{date}/{intersection}/{filename}")
df = df.dropna()

# Convert from EPSG:2056 to EPSG:4326 (lat, lon)
<<<<<<< Updated upstream
df['x_act_ekf'] = df['x_ekf'] + X_2056_offset
df['y_act_ekf'] = df['y_ekf'] + Y_2056_offset
transformer = Transformer.from_crs("EPSG:2056", "EPSG:4326", always_xy=True)
df["lon_ekf"], df["lat_ekf"] = transformer.transform(
    df["x_act_ekf"].values, df["y_act_ekf"].values)

# Create a folium map
center_lat, center_lon = df["lat_ekf"].mean(), df["lon_ekf"].mean()
=======
# df['x_act_ekf'] = df['x_ekf'] + X_2056_offset
# df['y_act_ekf'] = df['y_ekf'] + Y_2056_offset
# transformer = Transformer.from_crs("EPSG:2056", "EPSG:4326", always_xy=True)
# df["lon_ekf"], df["lat_ekf"] = transformer.transform(
#     df["x_act_ekf"].values, df["y_act_ekf"].values)

# Create a folium map
center_lat, center_lon = df.loc[df['latitude'] != -1, "latitude"].mean(), df.loc[df['longitude'] != -1, "longitude"].mean()
>>>>>>> Stashed changes

# #############################################################################
# MAIN: Extract Remaining Centerlines from SwissTopo
# #############################################################################
# Link to edit drawing: https://map.geo.admin.ch/#/map?lang=en&center=2682815.13,1247903.71&z=13&topic=ech&layers=ch.swisstopo.zeitreihen@year=1864,f;ch.bfs.gebaeude_wohnungs_register,f;ch.bav.haltestellen-oev,f;ch.swisstopo.swisstlm3d-wanderwege,f;ch.vbs.schiessanzeigen,f;ch.astra.wanderland-sperrungen_umleitungen,f;KML%7Chttps://public.geo.admin.ch/api/kml/files/NUSEuRoGT_mOVN1edWFbEw,f;KML%7Chttps://public.geo.admin.ch/api/kml/files/AMdLJu9mRei9FqSDEVx51Q@adminId=vezMDMwQT0axzF76Z2JYjw&bgLayer=ch.swisstopo.swissimage&featureInfo=default
# Share Link: https://s.geo.admin.ch/jkfynb8vzf5w


kml_path = "../maps/from_swisstopo/gessnerbrucke_D4_F.kml"
gdf_swisstopo = gpd.read_file(kml_path, driver='KML')

row = gdf_swisstopo[gdf_swisstopo['Description'] == 'Observed_Area'].copy()
observed_area_polygon = row.geometry.item()

# Retrieving all relevant centerlines
centerl_Gessnerallee_NS = get_centerl_from_swisstopo(
    gdf_swisstopo, 'Gessnerallee_NS')
centerl_Gessnerallee_S = get_centerl_from_swisstopo(
    gdf_swisstopo, 'Gessnerallee_S')
centerl_Gessbrucke_EW = get_centerl_from_swisstopo(
    gdf_swisstopo, 'Gessnerbrucke_EW')
centerl_Gessbrucke_WE = get_centerl_from_swisstopo(
    gdf_swisstopo, 'Gessnerbrucke_WE')
centerl_Usteristrasse_WE = get_centerl_from_swisstopo(
    gdf_swisstopo, 'Usteristrasse_WE')
centerl_Usteristrasse_EW = get_centerl_from_swisstopo(
    gdf_swisstopo, 'Usteristrasse_EW')

# centerline = centerline.intersection(observed_area_polygon)

# #############################################################################
# MAIN: Create Folium map with SwissTopo base image
# #############################################################################
# Create a folium map
m = create_swisstopo_map(
<<<<<<< Updated upstream
    center_lat=df["lat_ekf"].mean(), center_lon=df["lon_ekf"].mean())
=======
    center_lat=center_lat, center_lon=center_lon)
>>>>>>> Stashed changes

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
spl = fit_roadway_centerline_spline(
    centerl_Gessnerallee_NS + centerl_Gessnerallee_S)
plot_spline_xy_2056(m, spl, label="Gessnerallee Centerline (N->S)",
                    linecolor=colors_dict['north'], linedashed=True, start_point=True)

splines_dict['N_2_S'] = spl

# #############################################################################
# MAIN: Get Through North -> West Spline
# #############################################################################
spl = fit_roadway_centerline_spline(
    centerl_Gessnerallee_NS + centerl_Gessbrucke_EW)
plot_spline_xy_2056(m, spl, label="Gessnerallee Centerline (N->S)",
                    linecolor=colors_dict['north'], linedashed=True, start_point=True)

splines_dict['N_2_W'] = spl

# #############################################################################
# MAIN: Get Through West -> South Spline
# #############################################################################
spl = fit_roadway_centerline_spline(
    centerl_Gessbrucke_WE + centerl_Gessnerallee_S)
plot_spline_xy_2056(m, spl, label="Gessnerallee Centerline (N->S)",
                    linecolor=colors_dict['north'], linedashed=True, start_point=True)

splines_dict['W_2_S'] = spl

# #############################################################################
# MAIN: Get Through West -> East Spline
# #############################################################################
spl = fit_roadway_centerline_spline(
    centerl_Gessbrucke_WE + centerl_Usteristrasse_WE)
plot_spline_xy_2056(m, spl, label="Gessnerallee Centerline (N->S)",
                    linecolor=colors_dict['north'], linedashed=True, start_point=True)

splines_dict['W_2_E'] = spl

# #############################################################################
# MAIN: Get Through East -> West Spline
# #############################################################################
spl = fit_roadway_centerline_spline(
    centerl_Usteristrasse_EW + centerl_Gessbrucke_EW)
plot_spline_xy_2056(m, spl, label="Gessnerallee Centerline (N->S)",
                    linecolor=colors_dict['north'], linedashed=True, start_point=True)

splines_dict['E_2_W'] = spl

# #############################################################################
# MAIN: Get Through South -> West Spline
# #############################################################################

spl = fit_roadway_centerline_spline(
<<<<<<< Updated upstream
    centerl_Gessnerallee_S[::-1] + centerl_Gessbrucke_EW)
plot_spline_xy_2056(m, spl, label="Gessnerallee Centerline (N->S)",
                    linecolor=colors_dict['north'], linedashed=True, start_point=True)

splines_dict['S_2_W'] = spl

=======
    centerl_Gessnerallee_S[::-1] + centerl_Gessbrucke_EW, smoothing=0)
plot_spline_xy_2056(m, spl, label="Gessnerallee Centerline (S->W)",
                    linecolor=colors_dict['north'], linedashed=True, start_point=True)

plot_line_latlon(m, centerl_Gessnerallee_S, label="Gessnerallee (S)", linecolor='red', linedashed=False, start_point=True)
plot_line_latlon(m, centerl_Gessbrucke_EW, label="Gessnerbrucke (E->W)", linecolor='red', linedashed=False, start_point=True)

splines_dict['S_2_W'] = spl


# Clothoid
tmp_spl = fit_roadway_centerline_spline(centerl_Gessnerallee_S[::-1])
tck = tmp_spl[0]
x_spline, y_spline = splev(np.linspace(0, 1, 50), tck)
xy1 = np.column_stack((x_spline, y_spline))

tmp_spl = fit_roadway_centerline_spline(centerl_Gessbrucke_EW)
tck = tmp_spl[0]
x_spline, y_spline = splev(np.linspace(0, 1, 50), tck)
xy2 = np.column_stack((x_spline, y_spline))

south_west_merged_coords, _, _ = connect_lines_g2(xy1, xy2, n_connector=120, verbose=True)
south_west_spl_C = fit_roadway_centerline_spline(south_west_merged_coords, coordsys='2056')
plot_spline_xy_2056(m, south_west_spl_C, label="Clothoid (S->W)", 
                    linecolor="yellow", linedashed=True, start_point=True)


# compare curvatures
from tools_coordinate_transform import _compute_curvature, _compute_distance_traveled


def _compute_tangents(xy, eps=1e-06):
    dx = np.gradient(xy[:, 0])
    dy = np.gradient(xy[:, 1])
    return dy / (dx + eps), np.rad2deg(np.arctan2(dy, dx))


u = np.linspace(0, 1, 1000)
x_spline, y_spline = splev(u, south_west_spl_C[0])
xy_C = np.column_stack((x_spline, y_spline))
curv_C = _compute_curvature(xy_C)
s_C, _ = _compute_distance_traveled(xy_C)
_, tang_C = _compute_tangents(xy_C)
x_spline, y_spline = splev(u, spl[0])
xy_H = np.column_stack((x_spline, y_spline))
curv_H = _compute_curvature(xy_H)
_, tang_H = _compute_tangents(xy_H)
s_H, _ = _compute_distance_traveled(xy_H)

curv_1 = _compute_curvature(xy1)
_, tang_1 = _compute_tangents(xy1)
s_1, _ = _compute_distance_traveled(xy1)
curv_2 = _compute_curvature(xy2)
_, tang_2 = _compute_tangents(xy2)
s_2, _ = _compute_distance_traveled(xy2)
s_2 = s_2  + 93.4

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

>>>>>>> Stashed changes
# #############################################################################
# MAIN: Get Through South -> East Spline
# #############################################################################

spl = fit_roadway_centerline_spline(
    centerl_Gessnerallee_S[::-1] + centerl_Usteristrasse_WE)
plot_spline_xy_2056(m, spl, label="Gessnerallee Centerline (N->S)",
                    linecolor=colors_dict['north'], linedashed=True, start_point=True)

splines_dict['S_2_E'] = spl

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
<<<<<<< Updated upstream
m = create_swisstopo_map(center_lat=df["lat_ekf"].mean(
), center_lon=df["lon_ekf"].mean(), add_layer_control=False)
plot_all_centerlines_splines_xy_2056(m, splines_dict, add_layer_control=True)

m.save(f"../maps/road_centerlines_map_{date}_{intersection}.html")


m = create_swisstopo_map(center_lat=df["lat_ekf"].mean(
), center_lon=df["lon_ekf"].mean(), add_layer_control=False)
=======
m = create_swisstopo_map(center_lat=center_lat, center_lon=center_lon, add_layer_control=False)
plot_all_centerlines_splines_xy_2056(m, splines_dict, add_layer_control=True)

m.save(f"../maps/road_centerlines_map_{date}_{intersection}_{code}.html")


sys.exit(1)
m = create_swisstopo_map(center_lat=center_lat, center_lon=center_lon, add_layer_control=False)
>>>>>>> Stashed changes
plot_all_centerlines_splines_xy_2056(m, splines_dict, add_layer_control=False)
plot_bicycles_trajectories_xy_2056(
    m, df, linecolor='black', linealpha=0.25, add_layer_control=True)

m.save(
    f"../maps/trajectories_map_{date}_{intersection}_{timeslot}_{code}.html")
