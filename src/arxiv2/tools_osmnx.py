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
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import osmnx as ox
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from pyproj import Transformer
from scipy.interpolate import splev
from scipy.interpolate import interp1d
from shapely.ops import linemerge
from shapely.ops import unary_union
from shapely.geometry import LineString
from shapely.geometry import MultiPoint
from matplotlib.lines import Line2D

from tools_coordinate_transform import densify_linestring
from tools_coordinate_transform import fit_roadway_centerline_spline
from tools_coordinate_transform import convert_xy2056_to_roadway_coordinates
from tools_coordinate_transform import convert_roadway_to_xy2056_coordinates
from tools_coordinate_transform import connect_lines_g2

# #############################################################################
# METHODS
# #############################################################################
def merge_osmnx_edges(gdf, street_name):
    """
    Merge all OSMnx LineString fragments for a given street name
    into a single ordered LineString, clipped to the bounding box.
    """
    edges = gdf[gdf['name'] == street_name].copy()
    edges = edges[edges.geometry.type == 'LineString']
    if len(edges) == 0:
        raise ValueError(f"No edges found for street: {street_name}")
    merged = linemerge(unary_union(edges.geometry))
    if merged.geom_type == 'MultiLineString':
        # coords = []
        # for line in merged_centerline.geoms:
        #     coords.extend(list(line.coords))
            
        # Take the longest fragment if merge is incomplete
        merged = max(merged.geoms, key=lambda g: g.length)
    return merged   # Shapely LineString in WGS84


def osmnx_line_to_latlon(line):
    """Convert Shapely LineString (lon, lat) → list of (lat, lon) tuples."""
    return [(c[1], c[0]) for c in line.coords]


def fit_spline_from_osmnx(gdf_osmnx, street_name, smoothing=0, x_offset=0, y_offset=0):
    """
    Merge OSMnx edges for street_name, fit B-spline in EPSG:2056.
    Returns (tck, unew, cum_dist, total_length).
    """
    line    = merge_osmnx_edges(gdf_osmnx, street_name)
    latlon  = osmnx_line_to_latlon(line)
    tck, unew, cum_dist = fit_roadway_centerline_spline(
        latlon, smoothing=smoothing, coordsys='latlon', x_offset=x_offset, y_offset=y_offset
    )
    total_length = float(cum_dist[-1])
    return tck, unew, cum_dist, total_length


def fit_spline_from_shapely(line, smoothing=0, x_offset=0, y_offset=0):
    """Fit B-spline from a Shapely LineString in WGS84."""
    latlon = osmnx_line_to_latlon(line)
    try:
        tck, unew, cum_dist = fit_roadway_centerline_spline(
            latlon, smoothing=smoothing, coordsys='latlon', x_offset=x_offset, y_offset=y_offset
        )
    except:
        line = densify_linestring(line, num_segments=5)
        latlon = osmnx_line_to_latlon(line)
        tck, unew, cum_dist = fit_roadway_centerline_spline(
            latlon, smoothing=smoothing, coordsys='latlon', x_offset=x_offset, y_offset=y_offset
        )
    return tck, unew, cum_dist, float(cum_dist[-1])

