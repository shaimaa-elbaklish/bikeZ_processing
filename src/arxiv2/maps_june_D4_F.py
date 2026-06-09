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

def get_clothoid_spl(centerl1, centerl2, nb_splev=50, nb_connectors=120):
    tmp_spl = fit_roadway_centerline_spline(centerl1)
    tck = tmp_spl[0]
    x_spline, y_spline = splev(np.linspace(0, 1, nb_splev), tck)
    xy1 = np.column_stack((x_spline, y_spline))

    tmp_spl = fit_roadway_centerline_spline(centerl2)
    tck = tmp_spl[0]
    x_spline, y_spline = splev(np.linspace(0, 1, nb_splev), tck)
    xy2 = np.column_stack((x_spline, y_spline))

    north_west_merged_coords, _, _ = connect_lines_g2(xy1, xy2, n_connector=nb_connectors,
                                                      verbose=True)
    spl = fit_roadway_centerline_spline(north_west_merged_coords, coordsys='2056')
    return spl
    


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
filename = f"trajectories_bikes_{date}_{intersection}_{timeslot}_{code}-1.csv"
df = pd.read_csv(data_root + f"{date}/{intersection}/{filename}")
# COLUMNS: ['veh_id', 'veh_type', 'speed(km/h)', 'a(m/s2)', 'time(s)', 'X_2056(m)', 'Y_2056(m)', 'longitude', 'latitude', 'datetime']
# add a column as a missing flag
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

# Fix time = -1 issues
# Find ref. datetime (i.e. datetime when time == 0)
ref_datetime = df['datetime'].min()
ref_time = df.loc[(df['datetime'] == ref_datetime) & (df['time'] >= 0), 'time'].unique()[0]
df['time'] = df['datetime'].apply(lambda x: np.round((x - ref_datetime).total_seconds() + ref_time, decimals=3))
df = df.sort_values(by=['veh_id', 'time'], ascending=True)    
# # Estimate heading angle (degrees)
# from tools_filtering import estimate_heading
# df = estimate_heading(df, speed_threshold=1.0, window_s=0.5)
# df['angle'] = df['heading'] * np.pi / 180 # convert to rad

center_lat, center_lon = df.loc[~df['missing'], "lat"].mean(), df.loc[~df['missing'], "lon"].mean()

# #############################################################################
# MAIN: Extract Remaining Centerlines from SwissTopo
# #############################################################################
# Link to edit drawing: https://map.geo.admin.ch/#/map?lang=en&center=2682815.13,1247903.71&z=13&topic=ech&layers=ch.swisstopo.zeitreihen@year=1864,f;ch.bfs.gebaeude_wohnungs_register,f;ch.bav.haltestellen-oev,f;ch.swisstopo.swisstlm3d-wanderwege,f;ch.vbs.schiessanzeigen,f;ch.astra.wanderland-sperrungen_umleitungen,f;KML%7Chttps://public.geo.admin.ch/api/kml/files/NUSEuRoGT_mOVN1edWFbEw,f;KML%7Chttps://public.geo.admin.ch/api/kml/files/AMdLJu9mRei9FqSDEVx51Q@adminId=vezMDMwQT0axzF76Z2JYjw&bgLayer=ch.swisstopo.swissimage&featureInfo=default
# Share Link: https://s.geo.admin.ch/me9562bvpx3u


kml_path = "../maps/from_swisstopo/gessnerbrucke_D4_F.kml"
gdf_swisstopo = gpd.read_file(kml_path, driver='KML')

# row = gdf_swisstopo[gdf_swisstopo['Description'] == 'Observed_Area'].copy()
# observed_area_polygon = row.geometry.item()

# Retrieving all relevant centerlines
centerl_Gessnerallee_NS = get_centerl_from_swisstopo(
    gdf_swisstopo, 'Gessnerallee_NS')
centerl_Gessnerallee_S_V = get_centerl_from_swisstopo(
    gdf_swisstopo, 'Gessnerallee_S_V')
centerl_Gessnerallee_S_B = get_centerl_from_swisstopo(
    gdf_swisstopo, 'Gessnerallee_S_B')
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
    center_lat=center_lat, center_lon=center_lon)

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
# spl = fit_roadway_centerline_spline(
#     centerl_Gessnerallee_NS + centerl_Gessnerallee_S)
spl = get_clothoid_spl(centerl_Gessnerallee_NS, centerl_Gessnerallee_S_V)
plot_spline_xy_2056(m, spl, label="Gessnerallee Centerline (N->S)",
                    linecolor=colors_dict['north'], linedashed=True, start_point=True)

splines_dict['N_2_S_V'] = spl

# #############################################################################
# MAIN: Get Through North -> West Spline
# #############################################################################
spl = get_clothoid_spl(
    centerl_Gessnerallee_NS, centerl_Gessbrucke_EW)
plot_spline_xy_2056(m, spl, label="Gessnerallee Centerline (N->S)",
                    linecolor=colors_dict['north'], linedashed=True, start_point=True)

splines_dict['N_2_W_A'] = spl

# #############################################################################
# MAIN: Get Through West -> South Spline
# #############################################################################
spl = get_clothoid_spl(
    centerl_Gessbrucke_WE, centerl_Gessnerallee_S_V)
plot_spline_xy_2056(m, spl, label="Gessnerallee Centerline (W->S)",
                    linecolor=colors_dict['north'], linedashed=True, start_point=True)

splines_dict['W_2_S_V'] = spl

# #############################################################################
# MAIN: Get Through West -> South Spline
# #############################################################################
spl = get_clothoid_spl(
    centerl_Gessbrucke_WE, centerl_Gessnerallee_S_B)
plot_spline_xy_2056(m, spl, label="Gessnerallee Centerline (W->S)",
                    linecolor=colors_dict['north'], linedashed=True, start_point=True)

splines_dict['W_2_S_B'] = spl

# #############################################################################
# MAIN: Get Through West -> East Spline
# #############################################################################
spl = get_clothoid_spl(
    centerl_Gessbrucke_WE, centerl_Usteristrasse_WE)
plot_spline_xy_2056(m, spl, label="Gessnerallee Centerline (N->S)",
                    linecolor=colors_dict['north'], linedashed=True, start_point=True)

splines_dict['W_2_E_A'] = spl


# #############################################################################
# MAIN: Get Through East -> West Spline
# #############################################################################
spl = get_clothoid_spl(
    centerl_Usteristrasse_EW, centerl_Gessbrucke_EW)
plot_spline_xy_2056(m, spl, label="Gessnerallee Centerline (N->S)",
                    linecolor=colors_dict['north'], linedashed=True, start_point=True)

splines_dict['E_2_W_A'] = spl

# #############################################################################
# MAIN: Get Through South -> West Spline
# #############################################################################

spl = get_clothoid_spl(
    centerl_Gessnerallee_S_B[::-1], centerl_Gessbrucke_EW)
plot_spline_xy_2056(m, spl, label="Gessnerallee Centerline (S->W)",
                    linecolor=colors_dict['north'], linedashed=True, start_point=True)

plot_line_latlon(m, centerl_Gessnerallee_S_V, label="Gessnerallee (S)", linecolor='red', linedashed=False, start_point=True)
plot_line_latlon(m, centerl_Gessbrucke_EW, label="Gessnerbrucke (E->W)", linecolor='red', linedashed=False, start_point=True)

splines_dict['S_2_W_A'] = spl

# #############################################################################
# MAIN: Get Through South -> East Spline
# #############################################################################
spl = get_clothoid_spl(
    centerl_Gessnerallee_S_B[::-1], centerl_Usteristrasse_WE)
plot_spline_xy_2056(m, spl, label="Gessnerallee Centerline (N->S)",
                    linecolor=colors_dict['north'], linedashed=True, start_point=True)

splines_dict['S_2_E_B'] = spl

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
m = create_swisstopo_map(center_lat=center_lat, center_lon=center_lon, add_layer_control=False)
plot_all_centerlines_splines_xy_2056(m, splines_dict, add_layer_control=True)

m.save(f"../maps/road_centerlines_map_{date}_{intersection}_{code}.html")


m = create_swisstopo_map(center_lat=center_lat, center_lon=center_lon, add_layer_control=False)
plot_all_centerlines_splines_xy_2056(m, splines_dict, add_layer_control=False)
plot_bicycles_trajectories_xy_2056(
    m, df, linecolor='black', linealpha=0.25, add_layer_control=True, ekf=False)

m.save(
    f"../maps/trajectories_map_{date}_{intersection}_{timeslot}_{code}.html")

# #############################################################################
# MAIN: Get Lane Boundary Splines
# #############################################################################
lane_boundaries_splines_dict = {}

bike_lb_spl = fit_roadway_centerline_spline(centerl_Gessbrucke_EW, smoothing=0.1) # tuple (tck, unew, cum_dist)
lane_boundaries_splines_dict['W_WB'] = bike_lb_spl

bike_lb_spl = fit_roadway_centerline_spline(centerl_Gessbrucke_WE, smoothing=0.1) # tuple (tck, unew, cum_dist)
lane_boundaries_splines_dict['W_EB'] = bike_lb_spl

bike_lb_spl = fit_roadway_centerline_spline(centerl_Gessnerallee_NS , smoothing=0.1) # tuple (tck, unew, cum_dist)
lane_boundaries_splines_dict['N_SB'] = bike_lb_spl

bike_lb_spl = fit_roadway_centerline_spline(centerl_Gessnerallee_S_B, smoothing=0.1) # tuple (tck, unew, cum_dist)
lane_boundaries_splines_dict['S_SB'] = bike_lb_spl

bike_lb_spl = fit_roadway_centerline_spline(centerl_Usteristrasse_EW, smoothing=0.1) # tuple (tck, unew, cum_dist)
lane_boundaries_splines_dict['E_WB'] = bike_lb_spl

bike_lb_spl = fit_roadway_centerline_spline(centerl_Usteristrasse_WE, smoothing=0.1) # tuple (tck, unew, cum_dist)
lane_boundaries_splines_dict['E_EB'] = bike_lb_spl

# Identify the rest (i.e. without boundaries) as "bicycles" or "cars" or "mixed"
lane_boundaries_splines_dict['E_WB'] = "mixed"

# Save
with open(f"../data/bike_lane_boundaries_splines_{date}_{intersection}.pkl", "wb") as f:
    pickle.dump(lane_boundaries_splines_dict, f)