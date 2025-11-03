"""
TITLE
-------------------------------------------
Authors:        Shaimaa K. El-Baklish
Organization:   ETH Zürich, Switzerland, IVT - Institute for Transportation Planning and Systems
Development:    2025
Submitted to:   JOURNAL
-------------------------------------------
"""

###############################################################################
# IMPORTS
###############################################################################
import os
import sys
import pyproj

import osmnx as ox
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from functools import partial
from shapely.ops import linemerge, transform
from scipy.interpolate import splprep, splev, interp1d
from scipy.optimize import minimize_scalar

import _constants as cs

###############################################################################
# FUNCTIONS
###############################################################################
def extract_roadway_centerline(dataset: str):
    if dataset != "TUMDOT":
        raise NotImplementedError()
    
    tags = {"highway": True}
    gdf = ox.features.features_from_place(cs.TUMDOT_OSM_PLACE, tags=tags)

    # Filter for road name
    gdf = gdf[gdf['name'] == cs.TUMDOT_OSM_ROAD]

    # Optional: filter to LineStrings and main road types only
    main_road_types = ['primary', 'secondary', 'tertiary', 'residential', 'unclassified']
    gdf = gdf[(gdf.geometry.type == "LineString") & (gdf['highway'].isin(main_road_types))]
    
    # merge all linestrings for the centerline
    merged_centerline = linemerge(list(gdf.geometry))
  
    project = partial(
        pyproj.transform,
        pyproj.Proj("EPSG:4326"),   # WGS84 (lat/lon)
        pyproj.Proj("EPSG:2056")    # Swiss CH1903+ / LV95 (x/y)
    )

    utm_centerline = transform(project, merged_centerline)
    utm_centerline_coords = list(utm_centerline.coords)

    x, y = zip(*utm_centerline_coords)
    tck, u = splprep([x, y], s=0)
    unew = np.linspace(0, 1, num=500)
    spline_points = np.array(splev(unew, tck)).T  # shape (N, 2)

    # Compute cumulative distances along spline
    diffs = np.diff(spline_points, axis=0)
    dists = np.sqrt((diffs ** 2).sum(axis=1))
    cum_dist = np.insert(np.cumsum(dists), 0, 0)  # length N

    return tck, unew, cum_dist


def project_point_onto_spline(point, tck):
    distance_to_spline = lambda t, point, tck: np.sum((point - np.array(splev(t, tck))) ** 2)
    res = minimize_scalar(distance_to_spline, bounds=(0, 1), args=(point, tck), method='bounded')
    t_star = res.x
    closest_point = np.array(splev(t_star, tck))
    return t_star, closest_point


def convert_xy2056_to_roadway_coordinates(point, tck, unew, cum_dist):
    t_star, closest_point = project_point_onto_spline(point, tck)
    
    # Longitudinal s coordinate
    s = np.interp(t_star, unew, cum_dist)
    
    # Tangent vector at t_star
    dx_dt, dy_dt = splev(t_star, tck, der=1)
    tangent = np.array([dx_dt, dy_dt])
    tangent /= np.linalg.norm(tangent)
    
    # Normal vector
    normal = np.array([-tangent[1], tangent[0]])
    
    # Lateral offset d
    d = np.dot(point - closest_point, normal)
    
    return t_star, tangent, normal, s, d

def convert_roadway_to_xy2056_coordinates(s, d, tck, unew, cum_dist):
    # Step 1: find t such that spline arc length is s
    f_inv = interp1d(cum_dist, unew, bounds_error=False, fill_value=(unew[0], unew[-1]))
    t = f_inv(s)

    # Step 2: get centerline point at t
    x_c, y_c = splev(t, tck)
    center = np.array([x_c, y_c])

    # Step 3: get unit tangent vector at t
    dx_dt, dy_dt = splev(t, tck, der=1)
    tangent = np.array([dx_dt, dy_dt])
    tangent /= np.linalg.norm(tangent)

    # Step 4: compute normal vector
    normal = np.array([-tangent[1], tangent[0]])

    # Step 5: offset by d in the normal direction
    point = center + d * normal

    return point[0], point[1]  # returns (x, y)
