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
date = BikeZ_Config.avail_dates[2]
campaign = f"Zurich_2025{date[5:7]}" # June or September
mode = BikeZ_Config.avail_modes[0] # Bike
data_root = BikeZ_Config.data_root[campaign][mode]

intersection, code = BikeZ_Config.avail_intersections[date][3]
timeslot = BikeZ_Config.avail_timeslots[date][(intersection, code)][0] # 'AM1' or 'PM1'

XY_2056_Bounds = BikeZ_Config.XY_2056_Bounds[date][(intersection, code)]
X_2056_offset = XY_2056_Bounds[0][0]
Y_2056_offset = XY_2056_Bounds[1][0]

sys.exit(1)

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
# Link to edit drawing: https://s.geo.admin.ch/klhfe96psuif
# Share Link: https://s.geo.admin.ch/3mllih039iu9


kml_path = "../maps/from_swisstopo/herdenstrasse_D2_E.kml"
gdf_swisstopo = gpd.read_file(kml_path, driver='KML')

#row = gdf_swisstopo[gdf_swisstopo['Description'] == 'Observed_Area'].copy()
#observed_area_polygon = row.geometry.item()

# Retrieving all relevant centerlines
centerl_Herdenstrasse_North_NS = get_centerl_from_swisstopo(gdf_swisstopo, 'Herdenstrasse_North_NS')
centerl_Herdenstrasse_North_SN = get_centerl_from_swisstopo(gdf_swisstopo, 'Herdenstrasse_North_SN')
centerl_Bullingerstrasse_East_EW = get_centerl_from_swisstopo(gdf_swisstopo, 'Bullingerstrasse_East_EW')
centerl_Bullingerstrasse_East_WE = get_centerl_from_swisstopo(gdf_swisstopo, 'Bullingerstrasse_East_WE')
centerl_Herdenstrasse_South_SN = get_centerl_from_swisstopo(gdf_swisstopo, 'Herdenstrasse_South_SN')
centerl_Herdenstrasse_South_NS = get_centerl_from_swisstopo(gdf_swisstopo, 'Herdenstrasse_South_NS')
centerl_Baslerstrasse_West_WE = get_centerl_from_swisstopo(gdf_swisstopo, 'Baslerstrasse_West_WE')
centerl_Baslerstrasse_West_EW = get_centerl_from_swisstopo(gdf_swisstopo, 'Baslerstrasse_West_EW')


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
tmp_spl = fit_roadway_centerline_spline(centerl_Herdenstrasse_North_NS)
tck = tmp_spl[0]
x_spline, y_spline = splev(np.linspace(0, 1, 50), tck)
xy1 = np.column_stack((x_spline, y_spline))

tmp_spl = fit_roadway_centerline_spline(centerl_Herdenstrasse_South_NS)
tck = tmp_spl[0]
x_spline, y_spline = splev(np.linspace(0, 1, 50), tck)
xy2 = np.column_stack((x_spline, y_spline))

#north_south_merged_coords, _, _ = connect_lines_g2(xy1, xy2, n_connector=120, verbose=True)
#spl = fit_roadway_centerline_spline(north_south_merged_coords, coordsys='2056')
spl = fit_roadway_centerline_spline(centerl_Herdenstrasse_North_NS + centerl_Herdenstrasse_South_NS)
plot_spline_xy_2056(m, spl, label="Herdenstrasse Centerline (N->S)", linecolor=colors_dict['north'], linedashed=True, start_point=True)

splines_dict['N_2_S_A'] = spl

# #############################################################################
# MAIN: Get Through North -> West Spline
# #############################################################################
tmp_spl = fit_roadway_centerline_spline(centerl_Herdenstrasse_North_NS)
tck = tmp_spl[0]
x_spline, y_spline = splev(np.linspace(0, 1, 50), tck)
xy1 = np.column_stack((x_spline, y_spline))

tmp_spl = fit_roadway_centerline_spline(centerl_Baslerstrasse_West_EW)
tck = tmp_spl[0]
x_spline, y_spline = splev(np.linspace(0, 1, 50), tck)
xy2 = np.column_stack((x_spline, y_spline))

north_west_merged_coords, _, _ = connect_lines_g2(xy1, xy2, n_connector=120, verbose=True)
spl = fit_roadway_centerline_spline(north_west_merged_coords, coordsys='2056')
plot_spline_xy_2056(m, spl, label="Herdenstrasse Centerline (N->W)", linecolor=colors_dict['north'], linedashed=True, start_point=True)

splines_dict['N_2_W_A'] = spl

# #############################################################################
# MAIN: Get Through North -> East Spline
# #############################################################################
tmp_spl = fit_roadway_centerline_spline(centerl_Herdenstrasse_North_NS)
tck = tmp_spl[0]
x_spline, y_spline = splev(np.linspace(0, 1, 50), tck)
xy1 = np.column_stack((x_spline, y_spline))

tmp_spl = fit_roadway_centerline_spline(centerl_Bullingerstrasse_East_WE)
tck = tmp_spl[0]
x_spline, y_spline = splev(np.linspace(0, 1, 50), tck)
xy2 = np.column_stack((x_spline, y_spline))

north_east_merged_coords, _, _ = connect_lines_g2(xy1, xy2, n_connector=120, verbose=True)
spl = fit_roadway_centerline_spline(north_east_merged_coords, coordsys='2056')
plot_spline_xy_2056(m, spl, label="Herdenstrasse Centerline (N->E)", linecolor=colors_dict['north'], linedashed=True, start_point=True)

splines_dict['N_2_E_A'] = spl

# #############################################################################
# MAIN: Get Through South -> North Spline
# #############################################################################
tmp_spl = fit_roadway_centerline_spline(centerl_Herdenstrasse_South_SN)
tck = tmp_spl[0]
x_spline, y_spline = splev(np.linspace(0, 1, 50), tck)
xy1 = np.column_stack((x_spline, y_spline))

tmp_spl = fit_roadway_centerline_spline(centerl_Herdenstrasse_North_SN)
tck = tmp_spl[0]
x_spline, y_spline = splev(np.linspace(0, 1, 50), tck)
xy2 = np.column_stack((x_spline, y_spline))

#south_north_merged_coords, _, _ = connect_lines_g2(xy1, xy2, n_connector=120, verbose=True)
#spl = fit_roadway_centerline_spline(south_north_merged_coords, coordsys='2056')
spl = fit_roadway_centerline_spline(centerl_Herdenstrasse_South_SN + centerl_Herdenstrasse_North_SN)
plot_spline_xy_2056(m, spl, label="Herdenstrasse Centerline (S->N)", linecolor=colors_dict['south'], linedashed=True, start_point=True)

splines_dict['S_2_N_A'] = spl

# #############################################################################
# MAIN: Get Through South -> East Spline
# #############################################################################
tmp_spl = fit_roadway_centerline_spline(centerl_Herdenstrasse_South_SN)
tck = tmp_spl[0]
x_spline, y_spline = splev(np.linspace(0, 1, 50), tck)
xy1 = np.column_stack((x_spline, y_spline))

tmp_spl = fit_roadway_centerline_spline(centerl_Bullingerstrasse_East_WE)
tck = tmp_spl[0]
x_spline, y_spline = splev(np.linspace(0, 1, 50), tck)
xy2 = np.column_stack((x_spline, y_spline))

south_east_merged_coords, _, _ = connect_lines_g2(xy1, xy2, n_connector=120, verbose=True)
spl = fit_roadway_centerline_spline(south_east_merged_coords, coordsys='2056')
plot_spline_xy_2056(m, spl, label="Herdenstrasse Centerline (S->E)", linecolor=colors_dict['south'], linedashed=True, start_point=True)

splines_dict['S_2_E_A'] = spl

# #############################################################################
# MAIN: Get Through South -> West Spline
# #############################################################################
tmp_spl = fit_roadway_centerline_spline(centerl_Herdenstrasse_South_SN)
tck = tmp_spl[0]
x_spline, y_spline = splev(np.linspace(0, 1, 50), tck)
xy1 = np.column_stack((x_spline, y_spline))

tmp_spl = fit_roadway_centerline_spline(centerl_Baslerstrasse_West_EW)
tck = tmp_spl[0]
x_spline, y_spline = splev(np.linspace(0, 1, 50), tck)
xy2 = np.column_stack((x_spline, y_spline))

south_west_merged_coords, _, _ = connect_lines_g2(xy1, xy2, n_connector=120, verbose=True)
spl = fit_roadway_centerline_spline(south_west_merged_coords, coordsys='2056')
plot_spline_xy_2056(m, spl, label="Herdenstrasse Centerline (S->W)", linecolor=colors_dict['south'], linedashed=True, start_point=True)

splines_dict['S_2_W_A'] = spl

# #############################################################################
# MAIN: Get Through East -> West Spline
# #############################################################################
tmp_spl = fit_roadway_centerline_spline(centerl_Bullingerstrasse_East_EW)
tck = tmp_spl[0]
x_spline, y_spline = splev(np.linspace(0, 1, 50), tck)
xy1 = np.column_stack((x_spline, y_spline))

tmp_spl = fit_roadway_centerline_spline(centerl_Baslerstrasse_West_EW)
tck = tmp_spl[0]
x_spline, y_spline = splev(np.linspace(0, 1, 50), tck)
xy2 = np.column_stack((x_spline, y_spline))

#east_west_merged_coords, _, _ = connect_lines_g2(xy1, xy2, n_connector=120, verbose=True)
#spl = fit_roadway_centerline_spline(east_west_merged_coords, coordsys='2056')
spl = fit_roadway_centerline_spline(centerl_Bullingerstrasse_East_EW + centerl_Baslerstrasse_West_EW)
plot_spline_xy_2056(m, spl, label="Herdenstrasse Centerline (E->W)", linecolor=colors_dict['east'], linedashed=True, start_point=True)

splines_dict['E_2_W_A'] = spl

# #############################################################################
# MAIN: Get Through East -> North Spline
# #############################################################################
tmp_spl = fit_roadway_centerline_spline(centerl_Bullingerstrasse_East_EW)
tck = tmp_spl[0]
x_spline, y_spline = splev(np.linspace(0, 1, 50), tck)
xy1 = np.column_stack((x_spline, y_spline))

tmp_spl = fit_roadway_centerline_spline(centerl_Herdenstrasse_North_SN)
tck = tmp_spl[0]
x_spline, y_spline = splev(np.linspace(0, 1, 50), tck)
xy2 = np.column_stack((x_spline, y_spline))

east_north_merged_coords, _, _ = connect_lines_g2(xy1, xy2, n_connector=120, verbose=True)
spl = fit_roadway_centerline_spline(east_north_merged_coords, coordsys='2056')
plot_spline_xy_2056(m, spl, label="Herdenstrasse Centerline (E->N)", linecolor=colors_dict['east'], linedashed=True, start_point=True)

splines_dict['E_2_N_A'] = spl

# #############################################################################
# MAIN: Get Through East -> South Spline
# #############################################################################
tmp_spl = fit_roadway_centerline_spline(centerl_Bullingerstrasse_East_EW)
tck = tmp_spl[0]
x_spline, y_spline = splev(np.linspace(0, 1, 50), tck)
xy1 = np.column_stack((x_spline, y_spline))

tmp_spl = fit_roadway_centerline_spline(centerl_Herdenstrasse_South_NS)
tck = tmp_spl[0]
x_spline, y_spline = splev(np.linspace(0, 1, 50), tck)
xy2 = np.column_stack((x_spline, y_spline))

east_south_merged_coords, _, _ = connect_lines_g2(xy1, xy2, n_connector=120, verbose=True)
spl = fit_roadway_centerline_spline(east_south_merged_coords, coordsys='2056')
plot_spline_xy_2056(m, spl, label="Herdenstrasse Centerline (E->S)", linecolor=colors_dict['east'], linedashed=True, start_point=True)

splines_dict['E_2_S_A'] = spl

# #############################################################################
# MAIN: Get Through West -> East Spline
# #############################################################################
tmp_spl = fit_roadway_centerline_spline(centerl_Baslerstrasse_West_WE)
tck = tmp_spl[0]
x_spline, y_spline = splev(np.linspace(0, 1, 50), tck)
xy1 = np.column_stack((x_spline, y_spline))

tmp_spl = fit_roadway_centerline_spline(centerl_Bullingerstrasse_East_WE)
tck = tmp_spl[0]
x_spline, y_spline = splev(np.linspace(0, 1, 50), tck)
xy2 = np.column_stack((x_spline, y_spline))

#west_east_merged_coords, _, _ = connect_lines_g2(xy1, xy2, n_connector=120, verbose=True)
#spl = fit_roadway_centerline_spline(west_east_merged_coords, coordsys='2056')
spl = fit_roadway_centerline_spline(centerl_Baslerstrasse_West_WE + centerl_Bullingerstrasse_East_WE)
plot_spline_xy_2056(m, spl, label="Herdenstrasse Centerline (W->E)", linecolor=colors_dict['west'], linedashed=True, start_point=True)

splines_dict['W_2_E_A'] = spl

# #############################################################################
# MAIN: Get Through West -> South Spline
# #############################################################################
tmp_spl = fit_roadway_centerline_spline(centerl_Baslerstrasse_West_WE)
tck = tmp_spl[0]
x_spline, y_spline = splev(np.linspace(0, 1, 50), tck)
xy1 = np.column_stack((x_spline, y_spline))

tmp_spl = fit_roadway_centerline_spline(centerl_Herdenstrasse_South_NS)
tck = tmp_spl[0]
x_spline, y_spline = splev(np.linspace(0, 1, 50), tck)
xy2 = np.column_stack((x_spline, y_spline))

west_south_merged_coords, _, _ = connect_lines_g2(xy1, xy2, n_connector=120, verbose=True)
spl = fit_roadway_centerline_spline(west_south_merged_coords, coordsys='2056')
plot_spline_xy_2056(m, spl, label="Herdenstrasse Centerline (W->S)", linecolor=colors_dict['west'], linedashed=True, start_point=True)

splines_dict['W_2_S_A'] = spl

# #############################################################################
# MAIN: Get Through West -> North Spline
# #############################################################################
tmp_spl = fit_roadway_centerline_spline(centerl_Baslerstrasse_West_WE)
tck = tmp_spl[0]
x_spline, y_spline = splev(np.linspace(0, 1, 50), tck)
xy1 = np.column_stack((x_spline, y_spline))

tmp_spl = fit_roadway_centerline_spline(centerl_Herdenstrasse_North_SN)
tck = tmp_spl[0]
x_spline, y_spline = splev(np.linspace(0, 1, 50), tck)
xy2 = np.column_stack((x_spline, y_spline))

west_north_merged_coords, _, _ = connect_lines_g2(xy1, xy2, n_connector=120, verbose=True)
spl = fit_roadway_centerline_spline(west_north_merged_coords, coordsys='2056')
plot_spline_xy_2056(m, spl, label="Herdenstrasse Centerline (W->N)", linecolor=colors_dict['west'], linedashed=True, start_point=True)

splines_dict['W_2_N_A'] = spl


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

bike_lb_spl = fit_roadway_centerline_spline(centerl_Herdenstrasse_North_NS, smoothing=0.1) # tuple (tck, unew, cum_dist)
lane_boundaries_splines_dict['N_SB'] = bike_lb_spl

bike_lb_spl = fit_roadway_centerline_spline(centerl_Herdenstrasse_North_SN, smoothing=0.1) # tuple (tck, unew, cum_dist)
lane_boundaries_splines_dict['N_NB'] = bike_lb_spl

bike_lb_spl = fit_roadway_centerline_spline(centerl_Herdenstrasse_South_SN, smoothing=0.1) # tuple (tck, unew, cum_dist)
lane_boundaries_splines_dict['S_NB'] = bike_lb_spl

bike_lb_spl = fit_roadway_centerline_spline(centerl_Herdenstrasse_South_NS, smoothing=0.1) # tuple (tck, unew, cum_dist)
lane_boundaries_splines_dict['S_SB'] = bike_lb_spl

bike_lb_spl = fit_roadway_centerline_spline(centerl_Baslerstrasse_West_WE, smoothing=0.1) # tuple (tck, unew, cum_dist)
lane_boundaries_splines_dict['W_EB'] = bike_lb_spl

bike_lb_spl = fit_roadway_centerline_spline(centerl_Baslerstrasse_West_EW, smoothing=0.1) # tuple (tck, unew, cum_dist)
lane_boundaries_splines_dict['W_WB'] = bike_lb_spl

# Identify the rest (i.e. without boundaries) as "bicycles" or "cars" or "mixed"
lane_boundaries_splines_dict['E_EB'] = "mixed"
lane_boundaries_splines_dict['E_WB'] = "mixed"

# Save
with open(f"../data/bike_lane_boundaries_splines_{date}_{intersection}.pkl", "wb") as f:
    pickle.dump(lane_boundaries_splines_dict, f)

