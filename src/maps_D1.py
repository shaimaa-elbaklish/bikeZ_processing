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
date = BikeZ_Config.avail_dates[0]
campaign = f"Zurich_2025{date[5:7]}" # June or September
mode = BikeZ_Config.avail_modes[0] # Bike
data_root = BikeZ_Config.data_root[campaign][mode]

intersection, code = BikeZ_Config.avail_intersections[date][0]
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
# Link to edit drawing: https://s.geo.admin.ch/cw2pvvqcgtkz
# Share Link: https://s.geo.admin.ch/wr83p7oil0jg


kml_path = "../maps/from_swisstopo/langstrasse_D1.kml"
gdf_swisstopo = gpd.read_file(kml_path, driver='KML')

#row = gdf_swisstopo[gdf_swisstopo['Description'] == 'Observed_Area'].copy()
#observed_area_polygon = row.geometry.item()

# Retrieving all relevant centerlines
centerl_Langstrasse_North_NS = get_centerl_from_swisstopo(gdf_swisstopo, 'Langstrasse_North_NS')
centerl_Langstrasse_North_SN = get_centerl_from_swisstopo(gdf_swisstopo, 'Langstrasse_North_SN')
centerl_Zollstrasse_EW = get_centerl_from_swisstopo(gdf_swisstopo, 'Zollstrasse_EW')
centerl_Zollstrasse_WE = get_centerl_from_swisstopo(gdf_swisstopo, 'Zollstrasse_WE')
centerl_Langstrasse_South_SN = get_centerl_from_swisstopo(gdf_swisstopo, 'Langstrasse_South_SN')
centerl_Langstrasse_South_NS = get_centerl_from_swisstopo(gdf_swisstopo, 'Langstrasse_South_NS')
centerl_Roentgenstrasse_WE = get_centerl_from_swisstopo(gdf_swisstopo, 'Roentgenstrasse_WE')
centerl_Roentgenstrasse_EW = get_centerl_from_swisstopo(gdf_swisstopo, 'Roentgenstrasse_EW')


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
spl = fit_roadway_centerline_spline(centerl_Langstrasse_North_NS + centerl_Langstrasse_South_NS)
plot_spline_xy_2056(m, spl, label="Langstrasse Centerline (N->S)", linecolor=colors_dict['north'], linedashed=True, start_point=True)

splines_dict['N_2_S'] = spl

# #############################################################################
# MAIN: Get Through North -> West Spline
# #############################################################################
spl = fit_roadway_centerline_spline(centerl_Langstrasse_North_NS + centerl_Roentgenstrasse_EW)
plot_spline_xy_2056(m, spl, label="Langstrasse Centerline (N->W)", linecolor=colors_dict['north'], linedashed=True, start_point=True)

splines_dict['N_2_W'] = spl

# #############################################################################
# MAIN: Get Through North -> East Spline
# #############################################################################
spl = fit_roadway_centerline_spline(centerl_Langstrasse_North_NS + centerl_Zollstrasse_WE)
plot_spline_xy_2056(m, spl, label="Langstrasse Centerline (N->E)", linecolor=colors_dict['north'], linedashed=True, start_point=True)

splines_dict['N_2_E'] = spl

# #############################################################################
# MAIN: Get Through South -> North Spline
# #############################################################################
spl = fit_roadway_centerline_spline(centerl_Langstrasse_South_SN + centerl_Langstrasse_North_SN, smoothing=0.25)
plot_spline_xy_2056(m, spl, label="Langstrasse Centerline (S->N)", linecolor=colors_dict['south'], linedashed=True, start_point=True)

splines_dict['S_2_N'] = spl

# #############################################################################
# MAIN: Get Through South -> East Spline
# #############################################################################
spl = fit_roadway_centerline_spline(centerl_Langstrasse_South_SN + centerl_Zollstrasse_WE, smoothing=0.25)
plot_spline_xy_2056(m, spl, label="Langstrasse Centerline (S->E)", linecolor=colors_dict['south'], linedashed=True, start_point=True)

splines_dict['S_2_E'] = spl

# #############################################################################
# MAIN: Get Through South -> West Spline
# #############################################################################
spl = fit_roadway_centerline_spline(centerl_Langstrasse_South_SN + centerl_Roentgenstrasse_EW, smoothing=0.25)
plot_spline_xy_2056(m, spl, label="Langstrasse Centerline (S->W)", linecolor=colors_dict['south'], linedashed=True, start_point=True)

splines_dict['S_2_W'] = spl

# #############################################################################
# MAIN: Get Through East -> West Spline
# #############################################################################
spl = fit_roadway_centerline_spline(centerl_Zollstrasse_EW + centerl_Roentgenstrasse_EW, smoothing=0.25)
plot_spline_xy_2056(m, spl, label="Langstrasse Centerline (E->W)", linecolor=colors_dict['east'], linedashed=True, start_point=True)

splines_dict['E_2_W'] = spl

# #############################################################################
# MAIN: Get Through East -> North Spline
# #############################################################################
spl = fit_roadway_centerline_spline(centerl_Zollstrasse_EW + centerl_Langstrasse_North_SN, smoothing=0.25)
plot_spline_xy_2056(m, spl, label="Langstrasse Centerline (E->N)", linecolor=colors_dict['east'], linedashed=True, start_point=True)

splines_dict['E_2_N'] = spl

# #############################################################################
# MAIN: Get Through East -> South Spline
# #############################################################################
spl = fit_roadway_centerline_spline(centerl_Zollstrasse_EW + centerl_Langstrasse_South_NS, smoothing=0.25)
plot_spline_xy_2056(m, spl, label="Langstrasse Centerline (E->S)", linecolor=colors_dict['east'], linedashed=True, start_point=True)

splines_dict['E_2_S'] = spl

# #############################################################################
# MAIN: Get Through West -> East Spline
# #############################################################################
spl = fit_roadway_centerline_spline(centerl_Roentgenstrasse_WE + centerl_Zollstrasse_WE, smoothing=0.25)
plot_spline_xy_2056(m, spl, label="Langstrasse Centerline (W->E)", linecolor=colors_dict['west'], linedashed=True, start_point=True)

splines_dict['W_2_E'] = spl

# #############################################################################
# MAIN: Get Through West -> South Spline
# #############################################################################
spl = fit_roadway_centerline_spline(centerl_Roentgenstrasse_WE + centerl_Langstrasse_South_NS, smoothing=0.25)
plot_spline_xy_2056(m, spl, label="Langstrasse Centerline (W->S)", linecolor=colors_dict['west'], linedashed=True, start_point=True)

splines_dict['W_2_S'] = spl

# #############################################################################
# MAIN: Get Through West -> North Spline
# #############################################################################
spl = fit_roadway_centerline_spline(centerl_Roentgenstrasse_WE + centerl_Langstrasse_North_SN, smoothing=0.25)
plot_spline_xy_2056(m, spl, label="Langstrasse Centerline (W->N)", linecolor=colors_dict['west'], linedashed=True, start_point=True)

splines_dict['W_2_N'] = spl


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



#m = create_swisstopo_map(center_lat=df["lat_ekf"].mean(), center_lon=df["lon_ekf"].mean(), add_layer_control=False)
#plot_all_centerlines_splines_xy_2056(m, splines_dict, add_layer_control=False)
#plot_bicycles_trajectories_xy_2056(m, df, linecolor='black', linealpha=0.25, add_layer_control=True)

#m.save(f"../maps/trajectories_map_{date}_{intersection}_{timeslot}_{code}.html")


# #############################################################################
# MAIN: Get Lane Boundary Splines
# #############################################################################
"""
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
"""
