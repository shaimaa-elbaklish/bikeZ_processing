"""
TITLE OF PAPAER
-------------------------------------------
Authors:        Shaimaa El-Baklish
Organization:   ETH Zürich, Switzerland, IVT - Institute for Transportation Planning and Systems
Development:    2025
Submitted to:   JOURNAL
-------------------------------------------

maps_june_D4.py
-------------------------------------
Site definition — Gessnerbrucke / Gessnerallee / Usteristrasse
Zürich, Switzerland — June 2025 campaign (D3, E location)
 
Two intersections:
  MainInt  — 4-way: Gessnerbrucke × Gessnerallee × Usteristrasse
 
This file owns ALL geometry sourcing and calls the four builder phases.
Each phase takes plain dicts — customise here without touching the builder.
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
from shapely.geometry import Point
from shapely.plotting import plot_points
from shapely.plotting import plot_line

from _constants import BikeZ_Config
from tools_coordinate_transform import cut_line_at_stop
from tools_coordinate_transform import densify_linestring
from tools_site_builder import (
    fit_spline_from_osmnx,
    merge_osmnx_edges,
    fit_spline_from_shapely,
    register_geometries,
    build_segment_registry,
    add_bike_lane_boundaries,
    build_turns,
    build_intersection_polygon,
    build_movement_registry,
    serialize_registry
)

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

# Site constants
kml_path      = '../maps/from_swisstopo/June_D4.kml'
save_path     = f'../data/registry_{date}_{intersection}_{code}.pkl'
max_chain_len = 3

# OLD
# Share Link: https://s.geo.admin.ch/ge8picdgx322
# Edit Link: https://s.geo.admin.ch/tkhkli2704t4

# NEW
# Share Link: https://s.geo.admin.ch/4847z3l1p1dy
# Edit Link: https://s.geo.admin.ch/5dt9gh9beern

# #############################################################################
# MAIN
# #############################################################################

# =============================================================================
# STEP 0: load external data sources
print("Loading OSMnx features...")
transformer = Transformer.from_crs('EPSG:2056', 'EPSG:4326', always_xy=True)
lonlat      = transformer.transform(
    np.asarray(XY_2056_Bounds[0]) + np.asarray([-25, 25]),
    np.asarray(XY_2056_Bounds[1]) + np.asarray([-25, 25]),
)
bbox_geom = box(lonlat[0][0], lonlat[1][0], lonlat[0][1], lonlat[1][1])
 
gdf_main   = ox.features.features_from_place('Zürich, Switzerland',
                                              tags={'highway': True})
road_types = ['primary', 'secondary', 'tertiary',
              'residential', 'unclassified', 'cycleway']
gdf = gdf_main[
    gdf_main['name'].isin(
        ['Usteristrasse', 'Gessnerallee', 'Gessnerbrücke']
    )
]
gdf = gdf[
    (gdf.geometry.type == 'LineString') &
    (gdf['highway'].isin(road_types))
]
gdf['geometry'] = gdf.geometry.intersection(bbox_geom)
gdf = gdf[~gdf.is_empty]

# fig, ax = plt.subplots(1, 1)
# gdf.plot(ax=ax, column='name', legend=True)
# sys.exit(1)

# STEP 1: fit splines  (geometry sourcing, customise per road as needed)
print("\nFitting splines...")

# Baslerstrasse: split at Flurstrasse crossline into north and south branches
basler_full  = merge_osmnx_edges(gdf, 'Baslerstrasse')
flur_line = merge_osmnx_edges(gdf, 'Flurstrasse')
basler_west  = cut_line_at_stop(basler_full, flur_line, choose='last',  plotting=False)
basler_east  = cut_line_at_stop(basler_full, flur_line, choose='first', plotting=False)