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

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

from pyproj import Transformer
from pyclothoids import Clothoid
from shapely.geometry import LineString, Point
from shapely.ops import transform, split, snap
from scipy.interpolate import splprep, splev, interp1d
from scipy.optimize import minimize_scalar

###############################################################################
# FUNCTIONS
###############################################################################
def fit_roadway_centerline_spline(centerline_coords: list, smoothing: float = 0, coordsys: str = 'latlon'):
    """
    Fits a B-spline for the roadway centerline in XY:2056 coordinate system.

    Parameters
    ----------
    centerline_coords : list
        List of tuples (latitude, longitude) for the centerline of a specific road.
    smoothing : float, optional
        B-spline smoothing factor, between 0 and 1. The default is 0.
    coordsys: str, optional
        String to denote the coordinate system used (latlon OR 2056)

    Returns
    -------
    tck : np.ndarray
        B-spline representation.
    unew : np.ndarray
        B-spline parameter array (between 0 and 1) of size N.
    cum_dist : np.ndarray
        Cumulative distance along the spline, array of size N.
    """
    if coordsys == 'latlon':
        # Convert list of (lat, lon) into LineString(lon, lat)
        merged_centerline = LineString([(lon, lat) for lat, lon in centerline_coords])
        
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:2056", always_xy=True)
        project = lambda x, y, z=None: transformer.transform(x, y)
    
        xy2056_centerline = transform(project, merged_centerline)
        xy2056_centerline_coords = list(xy2056_centerline.coords)
    elif coordsys == '2056':
        xy2056_centerline_coords = centerline_coords
    else:
        raise NotImplementedError

    x, y = zip(*xy2056_centerline_coords)
    tck, u = splprep([x, y], s=smoothing)
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


def cut_line_at_stop(line, stopline, choose='first', plotting=False):
    intersect_pt = line.intersection(stopline)

    if intersect_pt is None or intersect_pt.is_empty:
        return line
    
    # Handle precision issues in splitting
    # Make sure that the intersection point is on the line by projection and snapping
    distance_along_line = line.project(intersect_pt)
    interpolated_point = line.interpolate(distance_along_line)
    snapped_line = snap(line, interpolated_point, 1e-8)
    # Split the line into 2 pieces at the intersection
    pieces = split(snapped_line, interpolated_point)
    
    # Find which piece starts at the same point as the original line
    if choose == 'first':
        start_pt = Point(line.coords[0])
    elif choose == 'last':
        start_pt = Point(line.coords[-1])
    else:
        raise NotImplementedError
    first_piece = min(pieces.geoms, key=lambda g: start_pt.distance(Point(g.coords[0])))
    
    if plotting:
        g_line = gpd.GeoSeries([line], crs=None)
        g_point = gpd.GeoSeries([intersect_pt], crs=None)
        g_firstpiece = gpd.GeoSeries([first_piece], crs=None)
        
        piece_geoms = []
        try:
            for geom in pieces.geoms:
                piece_geoms.append(geom)
        except Exception:
            # pieces may be a single geometry
            piece_geoms = [pieces]
        
        g_pieces = gpd.GeoSeries(piece_geoms)
        print(len(g_pieces))
        
        fig, ax = plt.subplots(figsize=(8, 8))
        g_line.plot(ax=ax, color="gray", linewidth=2, label="original line")
        g_pieces.plot(ax=ax, color=["red", "blue", "orange", "green"][:len(g_pieces)], linewidth=3, alpha=0.8)
        g_point.plot(ax=ax, color="black", markersize=50, label="split point")
        g_firstpiece.plot(ax=ax, color="yellow", linewidth=2, label="CHOSEN line", linestyle='--')
        
        ax.set_aspect('equal')
        ax.legend()
        ax.set_title("Line split pieces")
        plt.show()
        
        sys.exit(1)

    return first_piece


def densify_linestring(line = None, latlon_pts = None, num_segments=5):
    """Add interpolated points every `spacing` meters along a LineString."""
    if line is None:
        line = LineString([(lon, lat) for lat, lon in latlon_pts])
    
    if line.length == 0:
        return line
    
    distances = np.linspace(0, line.length, num_segments + 1)
    new_points = [line.interpolate(distance) for distance in distances]
    return LineString(new_points)


def connect_lines(xy1, xy2, n_connector=100, scale=0.4,
                           return_full=True, angle_threshold_deg=5,
                           force_method=None, verbose=False):
    """
    Connect two centerline fragments smoothly using a clothoid when possible,
    falling back to a cubic Hermite for nearly straight cases.

    Parameters
    ----------
    xy1, xy2 : list-like of (x, y)
        Input coordinate sequences (e.g., EPSG:2056).
    n_connector : int
        Number of points in the connecting curve.
    scale : float
        Tangent magnitude scale for Hermite fallback.
    return_full : bool
        If True, returns merged coords (xy1 + connector + xy2).
    angle_threshold_deg : float
        If the heading difference is smaller than this, use Hermite/linear interpolation.
    force_method : {'clothoid', 'hermite', None}
        Force method regardless of angle. None = automatic.
    verbose : bool
        Print info if clothoid fitting fails or is skipped.

    Returns
    -------
    merged : list of (x, y)
    connector : ndarray (n_connector, 2)
    method : str ('clothoid' or 'hermite')
    """
    def _to_numpy(arr):
        arr = np.asarray(arr)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError("Input must be (N,2) array-like.")
        return arr

    def _unit(v):
        n = np.linalg.norm(v)
        return v / n if n > 1e-9 else v

    def hermite_connect(p0, p1, t0, t1, n_points):
        # Classic cubic Hermite interpolation
        s = np.linspace(0, 1, n_points)
        h00 = 2*s**3 - 3*s**2 + 1
        h10 = s**3 - 2*s**2 + s
        h01 = -2*s**3 + 3*s**2
        h11 = s**3 - s**2
        pts = np.outer(h00, p0) + np.outer(h10, t0) + np.outer(h01, p1) + np.outer(h11, t1)
        return pts

    a = _to_numpy(xy1)
    b = _to_numpy(xy2)
    if a.shape[0] < 2 or b.shape[0] < 2:
        raise ValueError("Each line must have at least two points to estimate headings.")

    p0, p1 = a[-1], b[0]
    v0 = _unit(a[-1] - a[-2])
    v1 = _unit(b[1] - b[0])
    theta0, theta1 = np.arctan2(v0[1], v0[0]), np.arctan2(v1[1], v1[0])
    angle_diff = np.rad2deg(np.arctan2(np.sin(theta1 - theta0), np.cos(theta1 - theta0)))
    angle_diff_abs = abs(angle_diff)

    # Degenerate case
    if np.allclose(p0, p1):
        merged = np.vstack([a, b])
        return (merged.tolist(), np.empty((0, 2)), "none")

    method = "clothoid"
    connector = None

    # Method override or automatic choice
    if force_method == "hermite" or (force_method is None and angle_diff_abs < angle_threshold_deg):
        method = "hermite"

    if method == "clothoid":
        try:
            clothoid = Clothoid.G1Hermite(p0[0], p0[1], theta0, p1[0], p1[1], theta1)
            x_vals, y_vals = clothoid.SampleXY(n_connector)
            connector = np.column_stack((x_vals, y_vals))
        except Exception as e:
            if verbose:
                print(f"[Warning] Clothoid fitting failed ({e}); using Hermite fallback.")
            method = "hermite"

    if method == "hermite":
        d = np.linalg.norm(p1 - p0)
        m0 = v0 * (scale * d)
        m1 = v1 * (scale * d)
        connector = hermite_connect(p0, p1, m0, m1, n_connector)

    # Clean up endpoints
    connector_inner = connector[1:-1] if connector.shape[0] > 2 else np.empty((0, 2))
    merged = np.vstack([a, connector_inner, b])
    return (merged.tolist(), connector, method)


def match_bicycle_to_centerline(bike_df, centerlines_start_end_pts_dict):
    distances = {}
    for key, pts in centerlines_start_end_pts_dict.items():
        start_pt, end_pt = pts
        start_dist = np.linalg.norm(np.asarray(start_pt) - bike_df.iloc[0][['x_act_ekf', 'y_act_ekf']].to_numpy())
        end_dist = np.linalg.norm(np.asarray(end_pt) - bike_df.iloc[-1][['x_act_ekf', 'y_act_ekf']].to_numpy())
        distances[key] = {'start': start_dist, 'end': end_dist}
    distances = pd.DataFrame(distances).T
    min_start_dist = distances['start'].min()
    distances = distances[abs(distances['start'] - min_start_dist) <= 1e-01]
    return distances['end'].idxmin()