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
from pyclothoids import Clothoid, SolveG2
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import transform, split, snap
from scipy.interpolate import splprep, splev, interp1d
from scipy.optimize import minimize_scalar
from numpy.lib.stride_tricks import as_strided

###############################################################################
# FUNCTIONS
###############################################################################
def _to_local_xy_polygon(polygon_wgs84, x_offset, y_offset):
    """Transform a WGS84 Polygon to the local EPSG:2056-offset frame
    used by geometry_store splines."""
    transformer_to_2056 = Transformer.from_crs(
        "EPSG:4326", "EPSG:2056", always_xy=True
    )
    coords_local = [
        (x - x_offset, y - y_offset)
        for x, y in (transformer_to_2056.transform(c[0], c[1])
                     for c in polygon_wgs84.exterior.coords)
    ]
    poly = Polygon(coords_local)
    if not poly.is_valid:
        poly = poly.buffer(0)
    return poly


def fit_roadway_centerline_spline(centerline_coords: list, smoothing: float = 0, 
                                  coordsys: str = 'latlon', x_offset: float = 0,
                                  y_offset: float = 0):
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
    x_offset: float, optional
        float to improve numerical stability, offset XY:2056 in x direction
    y_offset: float, optional
        float to improve numerical stability, offset XY:2056 in y direction

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
    x = np.asarray(x) - x_offset
    y = np.asarray(y) - y_offset
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


def convert_xy2056_to_roadway_coordinates(point, tck, unew, cum_dist, 
                                          x_offset: float = 0, y_offset: float = 0):
    point = point - np.asarray([x_offset, y_offset])
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


def convert_roadway_to_xy2056_coordinates(s, d, tck, unew, cum_dist, 
                                          x_offset: float = 0, y_offset: float = 0):
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
    point = point + np.asarray([x_offset, y_offset])

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


def project_line_onto_spline(line, stop_line, tck, unew, cum_dist, x_offset, y_offset):
    """
    Find the arc-length s where a stop/yield LineString crosses the spline.

    Parameters
    ----------
    line      : Shapely LineString — original centerline in WGS84
    stop_line : Shapely LineString — stop or yield line in WGS84
    tck, unew, cum_dist : spline representation (in EPSG:2056)

    Returns
    -------
    s : float — arc-length [m] where stop/yield line crosses the spline
    """
    transformer_to_2056 = Transformer.from_crs("EPSG:4326", "EPSG:2056", always_xy=True)

    # Step 1 — find intersection point in WGS84 using Shapely
    intersection = line.intersection(stop_line)

    if intersection.is_empty:
        raise ValueError("Centerline and stop/yield line do not intersect.")

    # Handle cases where intersection is a Point or MultiPoint
    if intersection.geom_type == 'Point':
        pt = intersection
    elif intersection.geom_type == 'MultiPoint':
        # Take the point closest to the stop_line midpoint
        mid = stop_line.interpolate(0.5, normalized=True)
        pt  = min(intersection.geoms, key=lambda p: p.distance(mid))
    else:
        # GeometryCollection — extract first Point
        pts = [g for g in intersection.geoms if g.geom_type == 'Point']
        if not pts:
            raise ValueError(f"Unexpected intersection geometry type: {intersection.geom_type}")
        pt = pts[0]

    # Step 2 — transform intersection point to EPSG:2056, and subtract local origin
    x_2056, y_2056 = transformer_to_2056.transform(pt.x, pt.y)
    point_2056 = np.array([x_2056, y_2056])

    # Step 3 — project onto spline to get s
    _, _, _, s, _ = convert_xy2056_to_roadway_coordinates(point_2056, tck, unew, cum_dist, x_offset, y_offset)
    return float(s)


def _compute_curvature(xy, eps=1e-6):
    """
    xy: array (N,2)
    returns curvature (N,)
    """
    dx = np.gradient(xy[:, 0])
    dy = np.gradient(xy[:, 1])
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    num = dx * ddy - dy * ddx
    den = (dx*dx + dy*dy)**1.5 + eps
    return num / den


def connect_lines_g2(xy1, xy2, n_connector=100, scale=0.4,
                     return_full=True, angle_threshold_deg=5,
                     force_method=None, verbose=False):
    """
    Connect two centerline fragments smoothly using a G2 clothoid when possible,
    falling back to a cubic Hermite for nearly straight cases.
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
        s = np.linspace(0, 1, n_points)
        h00 = 2*s**3 - 3*s**2 + 1
        h10 = s**3 - 2*s**2 + s
        h01 = -2*s**3 + 3*s**2
        h11 = s**3 - s**2
        pts = np.outer(h00, p0) + np.outer(h10, t0) + np.outer(h01, p1) + np.outer(h11, t1)
        return pts
    
    def remove_consecutive_duplicates(pts):
        pts = np.asarray(pts)
        diff = np.diff(pts, axis=0)
        mask = np.any(np.abs(diff) > 1e-8, axis=1)
        # Always keep first point
        return np.vstack([pts[0], pts[1:][mask]])


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

    # if np.allclose(p0, p1):
    #     merged = np.vstack([a, b])
    #     return (merged.tolist(), np.empty((0, 2)), "none")

    method = "clothoid"
    connector = None

    # Method override or automatic choice
    if force_method == "hermite" or (force_method is None and angle_diff_abs < angle_threshold_deg):
        method = "hermite"

    if method == "clothoid":
        try:
            # k0, k1 = 0.0, 0.0  # straight segment assumption
            k0 = _compute_curvature(xy1)[-1]
            k1 = _compute_curvature(xy2)[0]
            clothoid_triplet = SolveG2(p0[0], p0[1], theta0, k0,
                                       p1[0], p1[1], theta1, k1)
            # Sample points from each of the 3 segments
            connector_pts = []
            n_each = max(n_connector // 3, 2)
            for c in clothoid_triplet:
                x, y = c.SampleXY(n_each)
                connector_pts.append(np.column_stack((x, y)))
            connector = np.vstack(connector_pts)
        except Exception as e:
            if verbose:
                print(f"[Warning] SolveG2 fitting failed ({e}); using Hermite fallback.")
            method = "hermite"

    if method == "hermite":
        d = np.linalg.norm(p1 - p0)
        m0 = v0 * (scale * d)
        m1 = v1 * (scale * d)
        connector = hermite_connect(p0, p1, m0, m1, n_connector)

    connector_inner = connector[1:-1] if connector.shape[0] > 2 else np.empty((0, 2))
    merged = np.vstack([a, connector_inner, b])
    merged_clean = remove_consecutive_duplicates(merged)
    return (merged_clean.tolist(), connector, method)


def build_d_boundary_spline(boundary_line_wgs84, tck, unew, cum_dist, 
                            x_offset, y_offset):
    """
    Project a bike lane boundary polyline onto a road centerline spline,
    producing a 1D spline d_boundary(s) and deriving side from the
    sign of the projected d values.

    Parameters
    ----------
    boundary_line_wgs84 : Shapely LineString in WGS84
    tck, unew, cum_dist : centerline spline in EPSG:2056
    x_offset, y_offset  : XY:2056 offsets

    Returns
    -------
    d_boundary_spline : scipy interp1d — d_boundary(s)
    s_domain          : (s_min, s_max) in spline-native arc-length [m]
    side              : int (+1 or -1) — which side of centerline the
                        bike lane is on, derived from sign of d_boundary
    """
    transformer_to_2056 = Transformer.from_crs(
        "EPSG:4326", "EPSG:2056", always_xy=True
    )

    # Convert boundary vertices to EPSG:2056
    coords_2056 = [
        transformer_to_2056.transform(c[0], c[1])
        for c in boundary_line_wgs84.coords
    ]

    # Project each vertex onto centerline → (s_i, d_i)
    s_vals = []
    d_vals = []
    for x_b, y_b in coords_2056:
        _, _, _, s_i, d_i = convert_xy2056_to_roadway_coordinates(
            np.array([x_b, y_b]), tck, unew, cum_dist, x_offset, y_offset
        )
        s_vals.append(s_i)
        d_vals.append(d_i)

    s_vals = np.array(s_vals)
    d_vals = np.array(d_vals)

    # Derive side from sign of projected d values — consistent with
    # how convert_xy2056_to_roadway_coordinates defines lateral offset
    side = int(np.sign(np.mean(np.sign(d_vals))))
    if side == 0:
        side = 1   # fallback if d_vals are exactly zero (shouldn't happen)

    # Sort by s
    sort_idx = np.argsort(s_vals)
    s_vals   = s_vals[sort_idx]
    d_vals   = d_vals[sort_idx]

    # Remove duplicate s values
    _, unique_idx = np.unique(s_vals, return_index=True)
    s_vals = s_vals[unique_idx]
    d_vals = d_vals[unique_idx]

    # Fit 1D interpolant
    d_boundary_spline = interp1d(
        s_vals, d_vals,
        kind='linear',
        bounds_error=False,
        fill_value=(d_vals[0], d_vals[-1])
    )

    s_domain = (float(s_vals[0]), float(s_vals[-1]))
    return d_boundary_spline, s_domain, side

