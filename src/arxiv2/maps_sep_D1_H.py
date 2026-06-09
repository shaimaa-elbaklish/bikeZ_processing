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
import matplotlib.pyplot as plt

from pyproj import Transformer
from shapely.geometry import box

from _constants import BikeZ_Config
from tools_coordinate_transform import cut_line_at_stop
from tools_map_visualization import create_registry_map


from tools_osmnx import merge_osmnx_edges
from tools_osmnx import fit_spline_from_osmnx
from tools_osmnx import fit_spline_from_shapely
from tools_infrastructure_geometry import get_s_domain
from tools_infrastructure_geometry import build_segment_registry
from tools_infrastructure_geometry import add_bike_lane_boundaries
from tools_infrastructure_geometry import build_all_turns
from tools_infrastructure_geometry import build_movement_registry
from tools_infrastructure_geometry import restrict_segment_roles
from tools_infrastructure_geometry import serialize_registry
# from tools_infrastructure_geometry import build_turn_spline, sample_spline_near_boundary
from tools_plotting import plot_geometry_store
# from tools_plotting import plot_turn_debug
from tools_plotting import plot_turn_splines

# #############################################################################
# CONSTANTS
# #############################################################################
# Configuration
BikeZ_Config = BikeZ_Config()

# Specify Trajectory File
date = BikeZ_Config.avail_dates[-1]
campaign = f"Zurich_2025{date[5:7]}" # June or September
mode = BikeZ_Config.avail_modes[0] # Bike
data_root = BikeZ_Config.data_root[campaign][mode]

intersection, code = BikeZ_Config.avail_intersections[date][1]
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

center_lat, center_lon = df["lat_ekf"].mean(), df["lon_ekf"].mean()

# #############################################################################
# MAIN: Create Bounding Box for location
# #############################################################################
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

gdf = gdf_main[gdf_main['name'].isin(['Baslerstrasse', 'Freihofstrasse'])]
gdf = gdf[(gdf.geometry.type == "LineString") & (gdf['highway'].isin(main_road_types))]
gdf["geometry"] = gdf.geometry.intersection(bbox)
gdf = gdf[~gdf.is_empty] # Drop empty geometries
gdf.plot(column='name', legend=True)

