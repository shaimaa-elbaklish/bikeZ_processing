"""
TITLE OF PAPAER
-------------------------------------------
Authors:        Shaimaa El-Baklish
Organization:   ETH Zürich, Switzerland, IVT - Institute for Transportation Planning and Systems
Development:    2025
Submitted to:   JOURNAL
-------------------------------------------

tools_site_builder.py
---------------------
Four-phase pipeline for building geometry_store, segment_registry,
and movement_registry from a site definition file.
 
Each phase takes plain dicts — no dataclasses, no fixed schema.
The site file owns all geometry sourcing (OSMnx, Shapely, KML, etc.)
and passes pre-fitted splines into Phase 1.
 
Pipeline
--------
Phase 1 — register_geometries()
    Projects s_stop, s_yield, computes s_change, assembles geometry_store.
    No algorithmic orientation check — positive_dir is user-verified from
    the Phase A validation plot.
 
Phase 2 — build_segment_registry()
    Determines is_forward from positive_dir, builds oriented-corridor
    validity polygons over the full spline domain (0, L).
    No approach/departure domain splitting — role is a label only.
 
Phase 3 — build_turns()
    Builds clothoid / Hermite turn splines, registers them in both
    geometry_store and segment_registry.
 
Phase 4 — build_movement_registry()
    Validates movement sequences (role is descriptive, no domain
    restriction applied).
 
Then call serialize_registry() to write the .pkl.
 
Design decisions
----------------
- approach / departure are labels in a movement sequence, not s-domains.
  Each lane segment's validity polygon covers the full spline (0, L).
  No restrict_segment_roles, no domain nulling, no polygon rebuilding.
 
- s_change = 0.5*(s_stop + s_yield) is the handoff boundary between
  lane segment and turn segment. Stored explicitly in geometry_store.
  s_stop and s_yield are also stored for downstream analysis.
 
- is_forward = (direction == positive_dir).
  For separated carriageways with flat geometry keys (e.g. 'LangstrS_NB'),
  positive_dir equals the direction, so is_forward is always True.
  For shared centerlines, one segment is forward and one is reverse —
  the scorer flips d-sign and heading reference for reverse segments.
"""

# #############################################################################
# IMPORTS
# #############################################################################
import os
import re
import sys
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import osmnx as ox
import geopandas as gpd
import xml.etree.ElementTree as ET

from datetime import date as _date
from scipy.interpolate import splev, splprep
from scipy.interpolate import interp1d
from shapely.ops import linemerge
from shapely.ops import unary_union
from shapely.geometry import LineString
from shapely.geometry import MultiPoint
from shapely.geometry import Polygon

from tools_coordinate_transform import (
    _to_local_xy_polygon,
    densify_linestring,
    fit_roadway_centerline_spline,
    convert_xy2056_to_roadway_coordinates,
    project_line_onto_spline,
    connect_lines_g2,
    build_d_boundary_spline,
)

# #############################################################################
# KML HELPERS
# #############################################################################
def _kml_color_to_hex(kml_color):
    # KML color format is aabbggrr -> convert to standard #rrggbb
    aa, bb, gg, rr = kml_color[0:2], kml_color[2:4], kml_color[4:6], kml_color[6:8]
    return f'#{rr}{gg}{bb}'.lower()


def _extract_colors(path):
    ns = {'kml': 'http://www.opengis.net/kml/2.2'}
    root = ET.parse(path).getroot()
    colors = []
    for pm in root.iter('{http://www.opengis.net/kml/2.2}Placemark'):
        color_el = pm.find('.//kml:LineStyle/kml:color', ns)
        colors.append(_kml_color_to_hex(color_el.text.strip()) if color_el is not None else None)
    return colors


def read_gis_kml(kml_path, color_to_name):
    gdf = gpd.read_file(kml_path, driver='KML')
    gdf['Color'] = _extract_colors(kml_path)
    gdf['Description'] = gdf['Color'].map(color_to_name)
    return gdf

# #############################################################################
# INTERNAL HELPERS
# #############################################################################
def _oriented_corridor_polygon(tck, unew, cum_dist,
                                s_start, s_end,
                                d_left, d_right,
                                is_forward=True,
                                n_sample=200):
    """
    Build a validity polygon as an oriented lateral corridor.
 
    Samples the centerline between s_start and s_end, offsets by
    d_left to the left and d_right to the right of the travel direction,
    then closes a ring: left-edge forward + right-edge reversed.
 
    Follows road curvature exactly — no convex-hull inflation.
 
    For reverse segments (is_forward=False) the spline tangent points
    opposite to the travel direction, so left/right are flipped.
 
    Parameters
    ----------
    tck, unew, cum_dist : spline
    s_start, s_end      : arc-length bounds [m] in native spline coords
    d_left              : width LEFT  of travel direction [m]
    d_right             : width RIGHT of travel direction [m]
    is_forward          : bool — True if travel direction == spline direction
    n_sample            : centerline sample count
 
    Returns
    -------
    shapely.geometry.Polygon
    """
    if s_end <= s_start + 0.1:
        return Polygon()
 
    s_vals = np.linspace(s_start, s_end, n_sample)
    t_vals = np.interp(s_vals, cum_dist, unew)
 
    x_c,  y_c  = splev(t_vals, tck, der=0)
    dx_c, dy_c = splev(t_vals, tck, der=1)
 
    tang = np.sqrt(dx_c**2 + dy_c**2)
    tang = np.where(tang > 1e-12, tang, 1.0)
    tx = dx_c / tang
    ty = dy_c / tang
 
    if not is_forward:
        tx = -tx
        ty = -ty
 
    nx = -ty
    ny =  tx
 
    left_x  = x_c + d_left  * nx
    left_y  = y_c + d_left  * ny
    right_x = x_c - d_right * nx
    right_y = y_c - d_right * ny
 
    left_pts  = list(zip(left_x,  left_y))
    right_pts = list(zip(right_x, right_y))
 
    ring = left_pts + right_pts[::-1] + [left_pts[0]]
    poly = Polygon(ring)
 
    if not poly.is_valid:
        poly = poly.buffer(0)
    return poly


def _turn_validity_polygon(tck, unew, cum_dist, s_start, s_end,
                            d_max, n_sample=60):
    """
    Build a validity polygon for a turn segment.
 
    Shape: convex hull of the laterally offset centerline points.
 
    Left and right edges expand by d_max from the centerline.
    Start cap (s=0) and end cap (s=L) are FLAT — perpendicular to the
    spline at each endpoint. No buffer() call, so no rounded corners.
 
    Avoids the bowtie self-intersection problem of the oriented corridor
    (which offsets each point independently and crosses on sharp bends)
    by taking the convex hull of all offset points instead.
 
    Parameters
    ----------
    tck, unew, cum_dist : turn spline
    s_start, s_end      : arc-length bounds [m]
    d_max               : lateral expansion [m] = max(d_left, d_right)
    n_sample            : centerline sample count
 
    Returns
    -------
    shapely.geometry.Polygon  — always valid, flat start/end caps
    """
    from shapely.geometry import MultiPoint
 
    if s_end <= s_start + 0.1:
        return Polygon()
 
    s_vals = np.linspace(s_start, s_end, n_sample)
    t_vals = np.interp(s_vals, cum_dist, unew)
    x_c, y_c   = splev(t_vals, tck, der=0)
    dx_c, dy_c = splev(t_vals, tck, der=1)
 
    # Unit left normals at each sample point
    tang = np.sqrt(dx_c**2 + dy_c**2)
    tang = np.where(tang > 1e-12, tang, 1.0)
    nx = -dy_c / tang
    ny =  dx_c / tang
 
    # Left and right offset edges
    left_x  = x_c + d_max * nx
    left_y  = y_c + d_max * ny
    right_x = x_c - d_max * nx
    right_y = y_c - d_max * ny
 
    # Convex hull of all offset points — naturally produces flat caps
    # at s=0 and s=L because the outermost points there are collinear
    all_pts = (list(zip(left_x, left_y)) +
               list(zip(right_x, right_y)))
    poly = MultiPoint(all_pts).convex_hull
 
    if not poly.is_valid:
        poly = poly.buffer(0)
    return poly


def _sample_near_boundary(seg_key, role, segment_registry, geometry_store,
                           s_change_key='s_change', n_pts=10, window_m=20.0):
    """
    Sample n_pts points near a junction boundary in local EPSG:2056,
    ordered in the vehicle's travel direction.
 
    The boundary is read from geometry_store[geom_key][s_change_key].
    Default key is 's_change' (primary intersection). Secondary junctions
    (e.g. Zollstr at Mattengasse) pass s_change_key='s_change_matt'.
 
    For approach: sample the 20 m window leading up to the boundary.
    For departure: sample the 20 m window leaving the boundary.
 
    Parameters
    ----------
    seg_key       : str
    role          : 'approach' | 'departure'
    segment_registry, geometry_store : dicts
    s_change_key  : str — key on the geometry entry for the boundary
    n_pts         : int
    window_m      : float
 
    Returns
    -------
    pts : (n_pts, 2) array, travel-direction order
    """
    entry               = segment_registry[seg_key]
    geom_key            = entry['geometry_key']
    geo                 = geometry_store[geom_key]
    tck, unew, cum_dist = geometry_store[geom_key]['spline']
    is_forward          = entry['is_forward']
    L                   = geometry_store[geom_key]['total_length']
 
    s_bnd = geometry_store[geom_key].get(s_change_key)
    if s_bnd is None:
        raise ValueError(
            f"_sample_near_boundary: geometry_store['{geom_key}'] has no "
            f"key '{s_change_key}'. Available keys: "
            f"{[k for k in geometry_store[geom_key] if k.startswith('s_')]}"
        )
    
    if geo.get('periodic'):
        # Closed curve: s wraps rather than clamping, and travel is always
        # +s because a ring is registered with direction == positive_dir.
        # Clamping here would silently shorten the window for any gate
        # within window_m of the seam.
        if role == 'approach':
            s_vals = s_bnd - np.linspace(window_m, 0.0, n_pts)
        else:
            s_vals = s_bnd + np.linspace(0.0, window_m, n_pts)
        s_vals = np.mod(s_vals, L)
    elif role == 'approach':
        if is_forward:
            s_vals = np.linspace(max(0.0, s_bnd - window_m), s_bnd, n_pts)
        else:
            s_vals = np.linspace(min(L, s_bnd + window_m), s_bnd, n_pts)
    else:   # departure
        if is_forward:
            s_vals = np.linspace(s_bnd, min(L, s_bnd + window_m), n_pts)
        else:
            s_vals = np.linspace(s_bnd, max(0.0, s_bnd - window_m), n_pts)
 
    t_vals         = np.interp(s_vals, cum_dist, unew)
    x_vals, y_vals = splev(t_vals, tck)
    return np.column_stack([x_vals, y_vals])
 
 
def _build_turn_spline(approach_seg, departure_seg,
                        segment_registry, geometry_store,
                        approach_s_change_key='s_change',
                        departure_s_change_key='s_change',
                        approach_window_m=20.0, departure_window_m=20.0,
                        n_pts=10, n_connector=100,
                        angle_threshold_deg=5, verbose=False):
    """
    Build a smooth clothoid/Hermite connector between the approach
    stop-line and departure yield-line, then fit a B-spline to it.
 
    approach_s_change_key / departure_s_change_key
        Which s_change value to use when sampling each segment near its
        boundary. Defaults to 's_change' (primary intersection).
        Pass 's_change_matt' etc. for secondary junctions.
 
    Returns
    -------
    tck, unew, cum_dist, total_length, connector, method
    """
    pts_app = _sample_near_boundary(
        approach_seg,  'approach',  segment_registry, geometry_store,
        s_change_key=approach_s_change_key,  n_pts=n_pts,
        window_m=approach_window_m,
    )
    pts_dep = _sample_near_boundary(
        departure_seg, 'departure', segment_registry, geometry_store,
        s_change_key=departure_s_change_key, n_pts=n_pts,
        window_m=departure_window_m,
    )
 
    _, connector, method = connect_lines_g2(
        pts_app, pts_dep,
        n_connector=n_connector,
        angle_threshold_deg=angle_threshold_deg,
        verbose=verbose,
    )
 
    if len(connector) < 2:
        raise RuntimeError("Connector is degenerate — too few points.")
 
    connector_clean = np.array(connector)
    keep = [0]
    for i in range(1, len(connector_clean)):
        if np.linalg.norm(connector_clean[i] - connector_clean[keep[-1]]) >= 1e-3:
            keep.append(i)
    connector_clean = connector_clean[keep]
 
    if len(connector_clean) < 4:
        raise RuntimeError(
            f"Only {len(connector_clean)} unique points after dedup "
            f"(need ≥ 4). Try increasing n_connector."
        )
 
    tck, unew, cum_dist = fit_roadway_centerline_spline(
        connector_clean.tolist(), coordsys='2056',
    )
    total_length = float(cum_dist[-1])
    return tck, unew, cum_dist, total_length, connector_clean, method


# #############################################################################
# PHASE 0: Spline fitting from osmnx / shapely
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
    # merged = linemerge(unary_union(edges.geometry))
    geom = unary_union(edges.geometry)
    merged = linemerge(geom) if geom.geom_type == "MultiLineString" else geom
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


# #############################################################################
# PHASE 1: register_geometries
# #############################################################################
def register_geometries(raw_axes, gdf_stop_yield, x_offset, y_offset):
    """
    Phase 1: project stop/yield lines, compute s_change values, assemble
    geometry_store.
 
    Orientation (positive_dir) is user-verified from the Phase A validation
    plot — no algorithmic check is performed here.
 
    Each entry in raw_axes is a dict with keys:
 
        name           str        — geometry_store key
        positive_dir   str | None — direction of increasing s
        spline         tuple      — (tck, unew, cum_dist)
        total_length   float      — arc-length [m]
        line_wgs84     LineString — WGS84, used for projection
        stop_line_id   str        — KML Description for stop line
        yield_line_id  str        — KML Description for yield line
 
        extra_changes  list       — optional, default []
            Additional boundary points on this through-road axis for
            secondary junctions (e.g. T-junctions).
 
            Each entry is a dict with ONE of two forms:
 
            Form A — single KML line, stored directly (no midpoint):
                key           str — name stored on geometry entry
                stop_line_id  str — KML Description of the line to project
                Use when the boundary is a single physical line
                (e.g. ZollstrE_Stop for a T-junction approach from east).
 
            Form B — stop+yield pair, stored as midpoint:
                key            str — name stored on geometry entry
                stop_line_id   str — KML Description of stop line
                yield_line_id  str — KML Description of yield line
                Use when the boundary is a box with two lines.
 
            Example (Zollstr at Mattengasse T-junction):
                extra_changes = [
                    {'key': 's_zollstr_east_stop',
                     'stop_line_id': 'ZollstrE_Stop'},       # Form A
                    {'key': 's_zollstr_west_yield',
                     'stop_line_id': 'ZollstrW_Yield'},      # Form A
                ]
 
    geometry_store schema per axis
    --------------------------------
        spline          (tck, unew, cum_dist)
        total_length    float [m]
        positive_dir    str | None
        line_wgs84      LineString
        s_stop          float — primary stop line [m]
        s_yield         float — primary yield line [m]
        s_change        float — 0.5*(s_stop+s_yield), primary handoff
        s_*             float — individual boundary from extra_changes Form A
        s_change_*      float — midpoint boundary from extra_changes Form B
 
    Parameters
    ----------
    raw_axes       : list of dicts
    gdf_stop_yield : GeoDataFrame — all KML stop/yield features
    x_offset, y_offset : float — EPSG:2056 local origin
 
    Returns
    -------
    geometry_store : dict
    """
    geometry_store = {
        'x_offset': x_offset,
        'y_offset': y_offset,
    }
 
    for ax in raw_axes:
        name             = ax['name']
        positive_dir     = ax.get('positive_dir')
        tck, unew, cum_dist = ax['spline']
        L                = float(ax['total_length'])
        line_wgs84       = ax['line_wgs84']
        stop_id          = ax['stop_line_id']
        yield_id         = ax['yield_line_id']
        extra_changes    = ax.get('extra_changes', [])
 
        print(f"  Registering {name}  (L={L:.1f} m, positive_dir={positive_dir})")
 
        # ── Primary stop / yield ──────────────────────────────────────────
        def _get_geom(feature_id):
            rows = gdf_stop_yield[gdf_stop_yield['Description'] == feature_id]
            if len(rows) == 0:
                raise ValueError(
                    f"register_geometries: KML feature '{feature_id}' "
                    f"not found for axis '{name}'."
                )
            return rows.geometry.iloc[0]
 
        def _project(geom):
            return float(project_line_onto_spline(
                line_wgs84, geom, tck, unew, cum_dist, x_offset, y_offset,
            ))
 
        s_stop  = _project(_get_geom(stop_id))
        s_yield = _project(_get_geom(yield_id))
 
        # s_change = 0.5 * (s_stop + s_yield)
        ratio    = ax.get('change_ratio', 0.6)
        s_change = (1-ratio) * s_stop +  ratio * s_yield
        frac     = s_change / L if L > 0 else 0.0
 
        print(f"    s_stop={s_stop:.2f}  s_yield={s_yield:.2f}  "
              f"s_change={s_change:.2f}  gap={s_yield-s_stop:.2f} m  "
              f"s_change/L={frac:.2f}"
              f"{'  ← CHECK positive_dir' if frac < 0.3 else ''}")
 
        entry = {
            'spline':       (tck, unew, cum_dist),
            'total_length': L,
            'positive_dir': positive_dir,
            'line_wgs84':   line_wgs84,
            's_stop':       s_stop,
            's_yield':      s_yield,
            's_change':     s_change,
        }
 
        # ── Extra change points (secondary junctions) ─────────────────────
        for ec in extra_changes:
            ec_key = ec['key']
 
            if 'yield_line_id' in ec:
                # Form B — stop+yield pair → store midpoint
                ec_stop  = _project(_get_geom(ec['stop_line_id']))
                ec_yield = _project(_get_geom(ec['yield_line_id']))
                if ec_stop > ec_yield:
                    ec_stop, ec_yield = ec_yield, ec_stop
                    print(f"    [INFO] {ec_key}: s values swapped")
                ec_val = 0.5 * (ec_stop + ec_yield)
                print(f"    {ec_key} [pair midpoint]: "
                      f"s_stop={ec_stop:.2f}  s_yield={ec_yield:.2f}  "
                      f"→ {ec_val:.2f} m")
            else:
                # Form A — single line → store projection directly
                ec_val = _project(_get_geom(ec['stop_line_id']))
                print(f"    {ec_key} [single line]: {ec_val:.2f} m")
 
            entry[ec_key] = ec_val
 
        geometry_store[name] = entry
 
    print(f"\nPhase 1 complete: {len(raw_axes)} axis/axes registered ✓\n")
    return geometry_store

# #############################################################################
# PHASE 2: build_segment_registry
# #############################################################################
def build_segment_registry(geometry_store, seg_defs):
    """
    Phase 2: build the segment registry.
 
    Each entry in seg_defs is a dict with keys:
 
        seg_key       str        — registry key, e.g. 'Roentgenstr_EB'
        geometry_key  str        — key into geometry_store
        direction     str        — travel direction, e.g. 'EB'
        mode          str        — 'shared' | 'bike' | 'car'
        bike_lane     dict|None  — {'w_bike': float} or None
        d_left        float      — validity polygon LEFT of travel [m]
        d_right       float      — validity polygon RIGHT of travel [m]
 
    Validity polygon covers the full spline (0, L).
    Role is a label only — no s-domain splitting.
 
    Segment registry schema
    -----------------------
        type            'lane'
        geometry_key    str
        direction       str
        is_forward      bool
        mode            str
        bike_lane       dict | None
        d_left, d_right, d_max  float
        validity_polygon        Polygon
 
    Returns
    -------
    segment_registry : dict
    """
    registry = {}
 
    for sd in seg_defs:
        seg_key   = sd['seg_key']
        geom_key  = sd['geometry_key']
        direction = sd['direction']
        mode      = sd.get('mode', 'shared')
        bike_lane = sd.get('bike_lane', None)
        d_left    = float(sd['d_left'])
        d_right   = float(sd['d_right'])
 
        geo              = geometry_store[geom_key]
        tck, unew, cum_dist = geo['spline']
        L                = geo['total_length']
        positive_dir     = geo['positive_dir']
        s_change         = geo.get('s_change')
 
        is_forward = (positive_dir is None) or (direction == positive_dir)
 
        # Clip validity polygon at s_change: beyond s_change belongs to
        # the intersection area and turn segments.
        # Choose the longer of (0, s_change) and (s_change, L) —
        # the approach arm is always the longer portion of the road.
        if s_change is not None:
            if s_change >= L - s_change:   # (0, s_change) is longer
                s_start, s_end = 0.0, s_change
            else:                           # (s_change, L) is longer
                s_start, s_end = s_change, L
        else:
            s_start, s_end = 0.0, L
 
        validity_poly = _oriented_corridor_polygon(
            tck, unew, cum_dist, s_start, s_end,
            d_left, d_right, is_forward=is_forward,
        )
 
        registry[seg_key] = {
            'type':             'lane',
            'geometry_key':     geom_key,
            'direction':        direction,
            'is_forward':       is_forward,
            'mode':             mode,
            'bike_lane':        dict(bike_lane) if bike_lane else None,
            'd_left':           d_left,
            'd_right':          d_right,
            'd_max':            max(d_left, d_right),
            'car_lane_d_bnd':     dict(sd.get('car_lane_d_bnd', {})),   # <-- new
            'validity_polygon': validity_poly,
        }
 
        fwd = 'fwd' if is_forward else 'rev'
        bl  = f"w={bike_lane['w_bike']}m" if bike_lane else 'None'
        print(f"  {seg_key}  ({fwd}, mode={mode}, bike_lane={bl}, "
              f"d_left={d_left:.1f} d_right={d_right:.1f})")
 
    print(f"\nPhase 2 complete: {len(seg_defs)} segment(s) registered ✓\n")
    return registry


def add_bike_lane_boundaries(segment_registry, geometry_store,
                              gdf_bike_boundaries):
    """
    Project bike lane boundary polylines onto centerline splines.
 
    KML features must have Description matching the seg_key exactly.
 
    Side convention:
        is_forward=True  → side = -1  (right of travel = right of spline)
        is_forward=False → side = +1  (right of travel = left of spline)
 
    Populates bike_lane['d_boundary_spline'], ['s_domain'], ['side'].
    """
    x_offset = geometry_store['x_offset']
    y_offset = geometry_store['y_offset']
    n_found  = 0
 
    for seg_key, entry in segment_registry.items():
        if entry.get('bike_lane') is None:
            continue
 
        geom_key            = entry['geometry_key']
        tck, unew, cum_dist = geometry_store[geom_key]['spline']
        is_forward          = entry['is_forward']
 
        rows = gdf_bike_boundaries[
            gdf_bike_boundaries['Description'] == seg_key
        ]
        if len(rows) == 0:
            print(f"  [WARN] no KML boundary for '{seg_key}' — stub kept")
            continue
 
        d_bnd_spl, s_domain, side_inferred = build_d_boundary_spline(
            rows.geometry.iloc[0], tck, unew, cum_dist, x_offset, y_offset,
        )
 
        side = -1 if is_forward else +1
        if side_inferred != side:
            print(f"  {seg_key}: side corrected {side_inferred:+d} → {side:+d}")
 
        entry['bike_lane']['d_boundary_spline'] = d_bnd_spl
        entry['bike_lane']['s_domain']          = s_domain
        entry['bike_lane']['side']              = side
 
        print(f"  {seg_key}: side={side:+d}  "
              f"s∈[{s_domain[0]:.1f}, {s_domain[1]:.1f}] m  "
              f"d_bnd∈[{d_bnd_spl(s_domain[0]):.2f}, "
              f"{d_bnd_spl(s_domain[1]):.2f}] m")
        n_found += 1
 
    print(f"\nStep 2b complete: {n_found} boundary/boundaries projected ✓\n")
    

def add_bike_lane_width_profile(segment_registry, geometry_store):
    """
    Build w_bike(s) for segments whose painted lane changes width.

    Each 'w_taper' names two boundary keys already projected onto the axis in
    Phase 1, plus the width on each side. The result is an interp1d that ramps
    linearly between them and holds constant outside:

        'bike_lane': {'w_bike': 2.5,                   # w_bike_at fallback
                      'w_taper': {'from_key': 'w_quaibr_end_2p5',   'w_from': 2.5,
                                  'to_key':   'w_quaibr_start_1p5', 'w_to':   1.5}}

    w_taper may also be a list of such dicts. Segments without one keep the
    scalar width and are left untouched.
    """
    n_found = 0

    for seg_key, entry in segment_registry.items():
        bike_lane = entry.get('bike_lane')
        if bike_lane is None or 'w_taper' not in bike_lane:
            continue

        geo    = geometry_store[entry['geometry_key']]
        tapers = bike_lane['w_taper']
        if isinstance(tapers, dict):
            tapers = [tapers]

        # (s, w) per transverse line, sorted along the axis
        knots = sorted((float(geo[t[s_key]]), float(t[w_key]))
                       for t in tapers
                       for s_key, w_key in (('from_key', 'w_from'),
                                            ('to_key',   'w_to')))
        s_k, w_k = (np.array(a) for a in zip(*knots))

        if np.any(np.diff(s_k) < 1e-6):
            raise ValueError(f"add_bike_lane_width_profile: '{seg_key}' has "
                             f"coincident width knots — a step change needs a "
                             f"non-zero taper length.")

        bike_lane['w_bike_spline'] = interp1d(
            s_k, w_k, kind='linear', bounds_error=False,
            fill_value=(w_k[0], w_k[-1]), assume_sorted=True,
        )
        bike_lane['w_range'] = (float(w_k.min()), float(w_k.max()))

        print(f"  {seg_key}: " + "  →  ".join(f"{w:.2f} m @ s={s:.2f}"
                                              for s, w in knots))
        n_found += 1

    print(f"\nVarying Bike Lane Widths: {n_found} width profile(s) built ✓\n")


def _d_bounds_from_polygon(polygon_local, tck, unew, cum_dist,
                            ds=0.5, d_search=15.0):
    """
    Slice a car-lane polygon (local XY) with the centerline's normal
    cross-sections to derive d_lb(s), d_ub(s).

    Returns
    -------
    d_lb_spline, d_ub_spline : scipy interp1d
    s_domain                 : (s_min, s_max)
    None, None, None if the polygon is not single-valued in s anywhere
    usable, or has no overlap with the centerline.
    """
    s_vtx = []
    for x, y in polygon_local.exterior.coords:
        _, _, _, s_i, _ = convert_xy2056_to_roadway_coordinates(
            np.array([x, y]), tck, unew, cum_dist, 0.0, 0.0
        )
        s_vtx.append(s_i)

    if not s_vtx:
        return None, None, None

    s_min = max(0.0,        min(s_vtx) - 2 * ds)
    s_max = min(cum_dist[-1], max(s_vtx) + 2 * ds)
    s_grid = np.arange(s_min, s_max, ds)

    s_valid, d_lb_vals, d_ub_vals = [], [], []
    n_multi = 0

    for s in s_grid:
        u = np.interp(s, cum_dist, unew)
        x0, y0 = splev(u, tck, der=0)
        dx, dy = splev(u, tck, der=1)
        norm   = np.hypot(dx, dy)
        if norm < 1e-12:
            continue
        nx, ny = -dy / norm, dx / norm

        cross = LineString([(x0 - nx * d_search, y0 - ny * d_search),
                             (x0 + nx * d_search, y0 + ny * d_search)])
        inter = polygon_local.intersection(cross)
        if inter.is_empty:
            continue

        if inter.geom_type == 'MultiLineString':
            n_multi += 1
            continue   # lane not single-valued in s at this cross-section
        if inter.geom_type == 'Point':
            continue   # tangent graze — no width here

        coords = list(inter.coords)
        d_vals = [(cx - x0) * nx + (cy - y0) * ny for cx, cy in coords]
        s_valid.append(s)
        d_lb_vals.append(min(d_vals))
        d_ub_vals.append(max(d_vals))

    if n_multi > 0:
        print(f"    [WARN] {n_multi}/{len(s_grid)} cross-section(s) "
              f"not single-valued in s — d_lb(s)/d_ub(s) skipped there")

    if len(s_valid) < 2:
        return None, None, None

    d_lb_spline = interp1d(s_valid, d_lb_vals, kind='linear',
                            bounds_error=False,
                            fill_value=(d_lb_vals[0], d_lb_vals[-1]))
    d_ub_spline = interp1d(s_valid, d_ub_vals, kind='linear',
                            bounds_error=False,
                            fill_value=(d_ub_vals[0], d_ub_vals[-1]))
    return d_lb_spline, d_ub_spline, (s_valid[0], s_valid[-1])


def add_car_lane_boundaries(segment_registry, geometry_store,
                             gdf_car_lane_polygons):
    """
    Attach car lane boundary polygons (and derived d_lb(s)/d_ub(s)
    bounds) to segment_registry entries.

    KML features must have Description == '<seg_key>_CarL<num>',
    e.g. 'Roentgenstr_EB_CarL1', 'Roentgenstr_EB_CarL2' — a segment
    may have multiple car lanes.

    Populates, per matched lane:
        entry['car_lane_d_bnd'][lane_id] = {
            'polygon':   Polygon              # local XY frame
            'd_bounds':  (d_lb, d_ub)         # each a scipy interp1d,
                                               # or None if unresolved
            's_domain':  (s_min, s_max) | None
        }
    """
    x_offset = geometry_store['x_offset']
    y_offset = geometry_store['y_offset']
    n_found  = 0
    n_total  = 0

    for seg_key, entry in segment_registry.items():
        # if not entry.get('car_lane_d_bnd'):
        #     continue

        geom_key             = entry['geometry_key']
        tck, unew, cum_dist  = geometry_store[geom_key]['spline']

        pattern = re.compile(rf'^{re.escape(seg_key)}_CarL(\d+)$')
        rows = gdf_car_lane_polygons[
            gdf_car_lane_polygons['Description'].str.match(pattern)
        ]

        if len(rows) == 0:
            print(f"  [WARN] no KML car lane polygon(s) for '{seg_key}' — stub kept")
            continue

        for _, row in rows.iterrows():
            lane_num = pattern.match(row['Description']).group(1)
            lane_id  = f"CarL{lane_num}"
            n_total += 1

            poly_local = _to_local_xy_polygon(row.geometry, x_offset, y_offset)
            d_lb, d_ub, s_domain = _d_bounds_from_polygon(
                poly_local, tck, unew, cum_dist
            )

            entry['car_lane_d_bnd'][int(lane_num)] = {
                'polygon':  poly_local,
                'd_bounds': (d_lb, d_ub),
                's_domain': s_domain,
            }

            if d_lb is not None:
                print(f"  {seg_key} [{lane_id}]: s∈[{s_domain[0]:.1f}, "
                      f"{s_domain[1]:.1f}] m  "
                      f"d_lb∈[{d_lb(s_domain[0]):.2f}, {d_lb(s_domain[1]):.2f}] m  "
                      f"d_ub∈[{d_ub(s_domain[0]):.2f}, {d_ub(s_domain[1]):.2f}] m")
                n_found += 1
            else:
                print(f"  [WARN] {seg_key} [{lane_id}]: polygon stored, "
                      f"but d_lb(s)/d_ub(s) could not be resolved "
                      f"(no single-valued overlap with centerline)")

    print(f"\ncar lane boundaries: {n_found}/{n_total} lane(s) resolved "
          f"with d(s) bounds ✓\n")

# #############################################################################
# PHASE 3: build_turns
# #############################################################################
def build_turns(geometry_store, segment_registry, turn_defs,
                n_pts=10, n_connector=100, angle_threshold_deg=5,
                verbose=False):
    """
    Phase 3: build clothoid / Hermite turn splines and register them.
 
    Each entry in turn_defs is a dict with keys:
 
        approach_seg          str   — seg_key of approach lane
        departure_seg         str   — seg_key of departure lane
        d_left                float — validity polygon LEFT  [m]
        d_right               float — validity polygon RIGHT [m]
 
        approach_s_change_key str   — optional, default 's_change'
            Which s_change key to use when sampling the approach segment
            near its boundary. Set to e.g. 's_change_matt' when the
            approach segment is a through-road at a secondary junction.
 
        departure_s_change_key str  — optional, default 's_change'
            Same for the departure segment.
 
    Example for a Mattengasse T-junction turn where Zollstr_EB is
    the departure road:
        {
            'approach_seg':           'Mattengasse_NE',
            'departure_seg':          'Zollstr_EB',
            'approach_s_change_key':  's_change',       # Mattengasse primary
            'departure_s_change_key': 's_change_matt',  # Zollstr at T-junction
            'd_left': 12.0, 'd_right': 12.0,
        }
 
    Modifies geometry_store and segment_registry in place.
 
    Returns
    -------
    turn_keys : list of str
    """
    turn_keys = []
 
    for td in turn_defs:
        app_seg   = td['approach_seg']
        dep_seg   = td['departure_seg']
        d_left    = float(td.get('d_left',  15.0))
        d_right   = float(td.get('d_right', 15.0))
        app_key   = td.get('approach_s_change_key',  's_change')
        dep_key   = td.get('departure_s_change_key', 's_change')
        app_win   = float(td.get('approach_window_m',  20.0))
        dep_win   = float(td.get('departure_window_m', 20.0))
        turn_key  = f'turn_{app_seg}_2_{dep_seg}'
 
        print(f"  Building {turn_key} ...", end=' ', flush=True)
 
        try:
            tck, unew, cum_dist, L, connector, method = _build_turn_spline(
                app_seg, dep_seg,
                segment_registry, geometry_store,
                approach_s_change_key=app_key,
                departure_s_change_key=dep_key,
                approach_window_m=app_win,
                departure_window_m=dep_win,
                n_pts=n_pts,
                n_connector=n_connector,
                angle_threshold_deg=angle_threshold_deg,
                verbose=verbose,
            )
        except Exception as exc:
            print(f"FAILED — {exc}")
            sys.exit(1)
 
        geometry_store[turn_key] = {
            'spline':       (tck, unew, cum_dist),
            'total_length': L,
            'positive_dir': None,
            's_stop':       None,
            's_yield':      None,
            's_change':     None,
            'line_wgs84':   None,
            'method':       method,
        }
 
        # Turn validity polygon: convex hull of sampled centerline points
        # expanded by d_max on all sides.
        # Oriented corridor is unreliable for short/sharp turns — large
        # lateral offsets fold back on themselves producing bowties or
        # MultiPolygons. Convex hull is always a valid Polygon and is
        # geometrically appropriate since cyclists can be anywhere in the
        # intersection box during a manoeuvre.
        validity_poly = _turn_validity_polygon(
            tck, unew, cum_dist, 0.0, L,
            d_max=max(d_left, d_right),
        )
 
        segment_registry[turn_key] = {
            'type':                  'turn',
            'geometry_key':          turn_key,
            'approach_seg':          app_seg,
            'departure_seg':         dep_seg,
            'approach_s_change_key': app_key,
            'departure_s_change_key':dep_key,
            'is_forward':            True,
            'mode':                  'shared',
            'bike_lane':             None,
            'd_left':                d_left,
            'd_right':               d_right,
            'd_max':                 max(d_left, d_right),
            'validity_polygon':      validity_poly,
        }
 
        print(f"✓  L={L:.1f} m  [{method}]")
        turn_keys.append(turn_key)
 
    print(f"\nPhase 3 complete: {len(turn_keys)} turn(s) built ✓\n")
    return turn_keys

# #############################################################################
# PHASE 4: build_movement_registry
# #############################################################################
def build_movement_registry(geometry_store, segment_registry, movement_defs):
    """
    Phase 4: validate movement sequences and build movement_registry.
 
    Each entry in movement_defs is a dict:
        key       str  — movement key
        sequence  list — [(seg_key, role), …]
                         first role must be 'approach'
                         last role must be 'departure'
                         middle roles must be 'turn'
 
    Role is a descriptive label only — no s-domain restriction applied.
 
    Raises ValueError listing all errors if validation fails.
 
    Returns
    -------
    movement_registry : dict  {key: [(seg_key, role), …]}
    """
    VALID_ROLES = {'approach', 'turn', 'ring', 'departure'}
    movement_registry = {}
    errors = []
 
    for md in movement_defs:
        key      = md['key']
        sequence = md['sequence']
        roles    = [r for _, r in sequence]
 
        if not roles or roles[0] != 'approach':
            errors.append(f"  {key}: must start with 'approach'")
            continue
        if roles[-1] != 'departure':
            errors.append(f"  {key}: must end with 'departure'")
            continue
        if any(r not in VALID_ROLES for r in roles):
            errors.append(f"  {key}: unknown role(s) {[r for r in roles if r not in VALID_ROLES]}")
            continue
 
        missing = [sk for sk, _ in sequence if sk not in segment_registry]
        if missing:
            errors.append(f"  {key}: missing segment keys {missing}")
            continue
 
        ok = True
        for sk, role in sequence:
            seg_type = segment_registry[sk]['type']
            if role in ('approach', 'departure') and seg_type != 'lane':
                errors.append(f"  {key}: '{sk}' type='{seg_type}' but role='{role}'")
                ok = False
            if role == 'turn' and seg_type != 'turn':
                errors.append(f"  {key}: '{sk}' type='{seg_type}' but role='turn'")
                ok = False
            if role == 'ring' and seg_type != 'ring':
                errors.append(f"  {key}: '{sk}' type='{seg_type}' but role='ring'")
                ok = False
        if not ok:
            continue
 
        movement_registry[key] = sequence
 
    if errors:
        print(f"\nPhase 4 ERRORS ({len(errors)}):")
        for e in errors:
            print(e)
        raise ValueError("build_movement_registry: fix errors above.")
 
    print(f"\nPhase 4: {len(movement_registry)} movement(s) validated ✓")
    for key, seq in movement_registry.items():
        print(f"  {key}: {' → '.join(f'{sk}[{r}]' for sk, r in seq)}")
 
    print(f"\nPhase 4 complete ✓\n")
    return movement_registry



def serialize_registry(geometry_store, segment_registry,
                        movement_registry, max_chain_length, 
                        intersection, date, save_path):
    """
    Serialize all three registry layers to a single .pkl file.

    Note: d_boundary_spline objects (scipy interp1d) pickle cleanly.
          pyclothoids objects do NOT pickle — but we use B-splines for
          turns so this is not an issue.

    Parameters
    ----------
    geometry_store    : dict
    segment_registry  : dict
    movement_registry : dict
    intersection      : str — e.g. 'Langstr_Roentgenstr'
    date              : str — e.g. '2025-06'
    save_path         : str — full path to output .pkl
    """
    registry = {
        'metadata': {
            'max_chain_length': max_chain_length, 
            'intersection': intersection,
            'date':         date,
            'crs':          'EPSG:2056',
            'x_offset':     geometry_store['x_offset'],
            'y_offset':     geometry_store['y_offset'],
            'created':      str(_date.today()),
            'n_segments':   len(segment_registry),
            'n_movements':  len(movement_registry),
            'n_turns':      sum(
                1 for e in segment_registry.values()
                if e['type'] == 'turn'
            ),
            'n_validity_polygons': sum(
                1 for e in segment_registry.values()
                if e.get('validity_polygon') is not None
                and not e['validity_polygon'].is_empty
            ),
        },
        'geometry_store':    geometry_store,
        'segment_registry':  segment_registry,
        'movement_registry': movement_registry,
    }

    with open(save_path, 'wb') as f:
        pickle.dump(registry, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Verify by reloading
    with open(save_path, 'rb') as f:
        verify = pickle.load(f)

    print(f"\nA4: Registry serialized to {save_path}")
    print(f"  intersection     : {verify['metadata']['intersection']}")
    print(f"  date             : {verify['metadata']['date']}")
    print(f"  created          : {verify['metadata']['created']}")
    print(f"  x_offset         : {verify['metadata']['x_offset']:.2f}")
    print(f"  y_offset         : {verify['metadata']['y_offset']:.2f}")
    print(f"  segments         : {verify['metadata']['n_segments']}")
    print(f"  movements        : {verify['metadata']['n_movements']}")
    print(f"  turns            : {verify['metadata']['n_turns']}")
    print(f"  validity polygons: {verify['metadata']['n_validity_polygons']}")
    print(f"  file size        : {os.path.getsize(save_path) / 1024:.1f} KB")
    print("  Reload verified ✓")
    

# =============================================================================
# INTERSECTION AREA POLYGON BUILDER
# =============================================================================

def build_intersection_polygon(arm_defs, geometry_store, segment_registry):
    """
    Build an intersection area polygon from s_change points.
 
    For each road arm, evaluates the spline normal at the s_change point
    and draws a line spanning the full carriageway width:
        left  side: d_right of the positive-direction segment
        right side: d_right of the opposite-direction segment
 
    The endpoints of all arm normal lines are joined into a convex polygon
    via convex_hull.
 
    Parameters
    ----------
    arm_defs : list of dicts, one per road arm entering the intersection.
        Each dict has keys:
            geom_key      str   — geometry_store key, e.g. 'Roentgenstr'
            s_change_key  str   — which s_change to use, e.g. 's_change'
                                  or 's_change_matt' for secondary junctions
            pos_seg_key   str   — seg_key of the positive-direction segment
                                  e.g. 'Roentgenstr_WB'
            opp_seg_key   str   — seg_key of the opposite-direction segment
                                  e.g. 'Roentgenstr_EB'
 
        Width rule:
            d_right(pos_seg) → extends LEFT  of spline (positive-dir side)
            d_right(opp_seg) → extends RIGHT of spline (opposite-dir side)
 
        For a road with only one direction (e.g. one-way), set opp_seg_key
        to None and provide opp_d_right explicitly:
            opp_d_right   float — fallback width when opp_seg_key is None
 
    geometry_store   : dict
    segment_registry : dict
 
    Returns
    -------
    shapely.geometry.Polygon — convex hull of all normal-line endpoints
                               in local EPSG:2056 coordinates
 
    Example (MainInt, 4-arm intersection)
    ----------------------------------------
    arm_defs = [
        {'geom_key': 'Roentgenstr', 's_change_key': 's_change',
         'pos_seg_key': 'Roentgenstr_WB', 'opp_seg_key': 'Roentgenstr_EB'},
        {'geom_key': 'LangstrN',    's_change_key': 's_change',
         'pos_seg_key': 'LangstrN_NB',    'opp_seg_key': 'LangstrN_SB'},
        {'geom_key': 'Zollstr',     's_change_key': 's_change',
         'pos_seg_key': 'Zollstr_EB',     'opp_seg_key': 'Zollstr_WB'},
        {'geom_key': 'LangstrS',    's_change_key': 's_change',
         'pos_seg_key': 'LangstrS_NB',    'opp_seg_key': 'LangstrS_SB'},
    ]
 
    Example (MattInt, T-junction — Zollstr arm uses s_change_matt)
    ----------------------------------------------------------------
    arm_defs = [
        {'geom_key': 'Zollstr',  's_change_key': 's_change_matt',
         'pos_seg_key': 'Zollstr_EB',  'opp_seg_key': 'Zollstr_WB'},
        {'geom_key': 'Matteng',  's_change_key': 's_change',
         'pos_seg_key': 'Matteng_SB',  'opp_seg_key': 'Matteng_NB'},
    ]
    """
    from shapely.geometry import MultiPoint

    all_pts = []

    for arm in arm_defs:
        geom_key     = arm['geom_key']
        s_change_key = arm['s_change_key']
        pos_seg_key  = arm['pos_seg_key']
        opp_seg_key  = arm.get('opp_seg_key')
        opp_d_right  = arm.get('opp_d_right', 0.0)
 
        geo              = geometry_store[geom_key]
        tck, unew, cum_dist = geo['spline']
        s_val            = geo[s_change_key]
 
        # Evaluate spline at s_change
        t_val        = float(np.interp(s_val, cum_dist, unew))
        x_c,  y_c   = splev(t_val, tck, der=0)
        dx_c, dy_c  = splev(t_val, tck, der=1)
 
        # Unit normal (left of spline = positive-direction side)
        tang = float(np.sqrt(dx_c**2 + dy_c**2))
        if tang < 1e-12:
            tang = 1.0
        nx = float(-dy_c) / tang
        ny = float( dx_c) / tang
 
        # Road widths from segment registry
        d_pos = float(segment_registry[pos_seg_key]['d_right'])
        d_opp = float(segment_registry[opp_seg_key]['d_right']) \
                if opp_seg_key and opp_seg_key in segment_registry \
                else float(opp_d_right)
 
        # Two endpoints spanning the full carriageway
        # Left  = positive-direction side  → offset +d_pos along normal
        # Right = opposite-direction side  → offset -d_opp along normal
        p_left  = (float(x_c) + d_pos * nx,
                   float(y_c) + d_pos * ny)
        p_right = (float(x_c) - d_opp * nx,
                   float(y_c) - d_opp * ny)
 
        all_pts.extend([p_left, p_right])
 
        print(f"  {geom_key} ({s_change_key}={s_val:.2f}m): "
              f"d_pos={d_pos:.1f}m  d_opp={d_opp:.1f}m  "
              f"L=({p_left[0]:.1f},{p_left[1]:.1f})  "
              f"R=({p_right[0]:.1f},{p_right[1]:.1f})")
 
    poly = MultiPoint(all_pts).convex_hull

    if not poly.is_valid:
        poly = poly.buffer(0)

    # ── Clip against each arm's stop line ────────────────────────────────
    # The intersection area must not extend past any arm's stop line.
    # For each arm, build a half-plane on the intersection-inward side of
    # s_stop and intersect the polygon with it.
    #
    # Inward side determined by approach_seg_key:
    #   approach == pos_seg  → cyclist approaches in positive direction (0→L)
    #                        → intersection is at the s=L end
    #                        → keep the HIGH-s side of the stop line
    #   approach != pos_seg  → cyclist approaches in opposite direction (L→0)
    #                        → intersection is at the s=0 end
    #                        → keep the LOW-s side of the stop line
    CLIP_WIDTH = 2000.0   # half-plane extent [m] — large enough to cover site

    for arm in arm_defs:
        geom_key        = arm['geom_key']
        s_change_key    = arm['s_change_key']
        pos_seg_key     = arm['pos_seg_key']
        approach_seg_key= arm.get('approach_seg_key')

        if approach_seg_key is None:
            continue   # no approach defined — skip clipping for this arm

        geo                  = geometry_store[geom_key]
        tck, unew, cum_dist  = geo['spline']
        s_stop_val           = geo.get(s_change_key)
        # if s_change_key == 's_change':
        #     s_stop_val           = geo.get('s_stop')
        # else:
        #     s_stop_val           = geo.get(s_change_key)

        if s_stop_val is None:
            continue

        # Evaluate spline at s_stop — stop line position and orientation
        t_stop      = float(np.interp(s_stop_val, cum_dist, unew))
        x_s,  y_s  = splev(t_stop, tck, der=0)
        dx_s, dy_s = splev(t_stop, tck, der=1)
        x_s, y_s   = float(x_s), float(y_s)

        tang = float(np.sqrt(dx_s**2 + dy_s**2))
        if tang < 1e-12:
            tang = 1.0
        tx = float(dx_s) / tang   # unit tangent (positive spline direction)
        ty = float(dy_s) / tang
        nx = -ty;  ny = tx        # unit normal (left of spline)

        # Stop line endpoints (wide enough to clip any polygon)
        stop_l = (x_s + CLIP_WIDTH * nx, y_s + CLIP_WIDTH * ny)
        stop_r = (x_s - CLIP_WIDTH * nx, y_s - CLIP_WIDTH * ny)

        # Inward side: approach == pos_seg → intersection at s=L → keep +tx side
        #              approach != pos_seg → intersection at s=0 → keep -tx side
        intersection_at_high_s = (approach_seg_key == pos_seg_key)

        if intersection_at_high_s:
            # Keep the HIGH-s side (extend rectangle in +tangent direction)
            half_plane = Polygon([
                stop_l,
                stop_r,
                (stop_r[0] + CLIP_WIDTH * tx, stop_r[1] + CLIP_WIDTH * ty),
                (stop_l[0] + CLIP_WIDTH * tx, stop_l[1] + CLIP_WIDTH * ty),
            ])
        else:
            # Keep the LOW-s side (extend rectangle in -tangent direction)
            half_plane = Polygon([
                stop_l,
                stop_r,
                (stop_r[0] - CLIP_WIDTH * tx, stop_r[1] - CLIP_WIDTH * ty),
                (stop_l[0] - CLIP_WIDTH * tx, stop_l[1] - CLIP_WIDTH * ty),
            ])

        poly = poly.intersection(half_plane)
        if poly.is_empty:
            break

    if not poly.is_valid:
        poly = poly.buffer(0)

    print(f"  → polygon area={poly.area:.1f} m²  "
          f"({len(all_pts)//2} arm(s), {len(all_pts)} boundary points)")

    return poly


# #############################################################################
# Location 7: Handling Ring Geometries for Roundabouts
# #############################################################################
from shapely.ops import transform
from tools_utils import _PROJ_LONLAT_TO_2056, _PROJ_2056_TO_LONLAT

# A gate ray is drawn outward from the center point; its inner endpoint is
# only as accurate as the drawing. Anything further than this from the
# center means the feature is not a radial ray and the file needs checking.
RAY_ORIGIN_TOL_M = 1.5
 
# Disagreement between the projected gate and the ray's own tip bearing
# above which the ray is probably drawn carelessly.
GATE_XCHECK_M = 0.25
 
# A gate closer than this to s = 0 could be reported as either ~0 or ~L.
SEAM_CLEARANCE_M = 0.5


def read_ring_kml(gdf_kml, prefix='Bullingerpl_RA', verbose=True):
    """
    Read a swisstopo roundabout drawing.
 
    Takes the GeoDataFrame the site file has already loaded.
    Expected features, labelled in the KML description:
        <prefix>_C                Point       — center
        <prefix>_Rinner           LineString  — radial ray, center to inner kerb
        <prefix>_Router           LineString  — radial ray, center to outer kerb
        <prefix>_Entry_<leg>      LineString  — radial ray, see below
        <prefix>_Exit_<leg>       LineString  — radial ray, see below
        <leg>_Stop, <leg>_Yield   LineString  — optional, used only to check
                                                circulation sense
 
    Entry and Exit are named for the RING, not for the leg:
        Entry_<leg>   where traffic coming from <leg> JOINS the ring
        Exit_<leg>    where traffic LEAVES the ring onto <leg>
 
    This function returns GEOMETRY, not arc-length.
 
    Returns
    -------
    ring : dict
        prefix      str
        center      (2,) array — EPSG:2056, NOT offset
        r_inner     float [m]
        r_outer     float [m]
        R           float [m] — circulating centerline radius
        L_ring      float [m] — 2 pi R
        legs        list of str, sorted
        gates       {(leg, 'Entry'|'Exit'): LineString} — WGS84, as drawn
        theta_tip   {(leg, 'Entry'|'Exit'): float} — deg CCW from east
        checks      dict — see _validate_ring()
    """
    # fiona calls it 'Description', pyogrio 'description'; geopandas 1.0
    # switched engines, so match case-insensitively.
    if 'Description' in gdf_kml.columns:
        col = 'Description'
    elif 'description' in gdf_kml.columns:
        col = 'description'
    else:
        raise ValueError(f"read_ring_kml: no description column. Columns are "
                         f"{list(gdf_kml.columns)}.")
 
    # --- index the frame, and keep a metric copy for the radii / bearings ----
    # Offsets are NOT applied: radii and angles are offset-invariant, and the
    # local origin only enters at Phase 1 in register_ring_geometry.
    to_2056 = lambda x, y, z=None: _PROJ_LONLAT_TO_2056.transform(x, y)
    geoms, feats = {}, {}
    for desc, geom in zip(gdf_kml[col], gdf_kml.geometry):
        if geom is None or geom.is_empty:
            continue
        desc = str(desc).strip()
        if not desc:
            continue
        if desc in geoms:
            raise ValueError(f"read_ring_kml: duplicate description '{desc}'.")
        g = transform(to_2056, geom)
        coords = (g.exterior.coords if g.geom_type == 'Polygon'
                  else g.coords if hasattr(g, 'coords') else None)
        if coords is None:
            continue
        geoms[desc] = geom
        feats[desc] = np.asarray(coords, dtype=float)[:, :2]
 
    if f'{prefix}_C' not in feats:
        raise ValueError(f"read_ring_kml: center '{prefix}_C' not found among "
                         f"{len(feats)} features; check the prefix.")
    center = feats[f'{prefix}_C'][0]
 
    def _ray(desc):
        """
        (tip bearing [deg CCW from east], tip radius [m]) of a radial ray.
 
        The bearing is taken at the FAR endpoint. The near endpoint sits
        within centimetres of the center, where a small drawing error is tens
        of degrees, so it carries almost no angular information.
        """
        if desc not in feats:
            raise ValueError(f"read_ring_kml: feature '{desc}' not found.")
        r = np.hypot(*(feats[desc] - center).T)
        if r.min() > RAY_ORIGIN_TOL_M:
            raise ValueError(f"read_ring_kml: '{desc}' does not start at the "
                             f"center (nearest endpoint {r.min():.2f} m away). "
                             f"Expected a radial ray drawn out from {prefix}_C.")
        v = feats[desc][int(np.argmax(r))] - center
        return float(np.degrees(np.arctan2(v[1], v[0])) % 360.0), float(r.max())
 
    # --- radii ---------------------------------------------------------------
    _, r_inner = _ray(f'{prefix}_Rinner')
    _, r_outer = _ray(f'{prefix}_Router')
    if not 0.0 < r_inner < r_outer:
        raise ValueError(f"read_ring_kml: implausible radii "
                         f"r_inner={r_inner:.2f} r_outer={r_outer:.2f}")
    R = 0.5 * (r_inner + r_outer)
 
    # --- gates ---------------------------------------------------------------
    gates, theta_tip, legs = {}, {}, set()
    for desc in feats:
        for kind in ('Entry', 'Exit'):
            token = f'{prefix}_{kind}_'
            if not desc.startswith(token):
                continue
            leg = desc[len(token):]
            th, r_far = _ray(desc)
            if r_far < r_outer:
                raise ValueError(f"read_ring_kml: '{desc}' ends inside the "
                                 f"outer kerb ({r_far:.2f} m < {r_outer:.2f} m), "
                                 f"so it never crosses the circulating "
                                 f"centerline and cannot be projected.")
            gates[(leg, kind)] = geoms[desc]
            theta_tip[(leg, kind)] = th
            legs.add(leg)
 
    legs = sorted(legs)
    if not legs:
        raise ValueError(f"read_ring_kml: no '{prefix}_Entry_*' features found.")
    missing = [(l, k) for l in legs for k in ('Entry', 'Exit')
               if (l, k) not in gates]
    if missing:
        raise ValueError(f"read_ring_kml: incomplete Entry/Exit pairs: {missing}")
 
    ring = {'prefix': prefix, 'center': np.asarray(center, dtype=float),
            'r_inner': r_inner, 'r_outer': r_outer, 'R': R,
            'L_ring': 2.0 * np.pi * R, 'legs': legs,
            'gates': gates, 'theta_tip': theta_tip}
    ring['checks'] = _validate_ring(ring, feats, verbose=verbose)
 
    if verbose:
        lon, lat = _PROJ_2056_TO_LONLAT.transform(center[0], center[1])
        print(f"  Ring '{prefix}': center=({center[0]:.3f}, {center[1]:.3f}) "
              f"LV95  ({lat:.6f}, {lon:.6f})")
        print(f"    r_inner={r_inner:.2f}  r_outer={r_outer:.2f}  "
              f"R={R:.2f}  L_ring={ring['L_ring']:.2f} m")
        print(f"    {len(legs)} legs: {', '.join(legs)}")
 
    return ring
 

def _validate_ring(ring, feats, verbose=True):
    """
    Geometric checks that need nothing but the drawing.
 
    Run on the tip bearings rather than the projected gates, because these
    are coarse tests — a leg mouth is 45-60 deg wide and the two estimators
    differ by under 0.3 deg — and because they must run before the spline
    exists.
 
    1. circulation sense — a leg's mouth is the CCW arc from where traffic
       LEAVES the ring onto that leg (Exit) to where traffic from that leg
       JOINS the ring (Entry). That arc must contain the leg's own Yield
       line. A leg failing this has its Entry/Exit labels swapped, or the
       roundabout circulates clockwise.
    2. mouth nesting — mouths of adjacent legs must not overlap. Overlap
       means one leg's rays are drawn too wide, and every movement arc
       through that pair is then wrong.
    3. annulus width — the circulating carriageway must be wide enough to be
       worth modelling as a corridor.
    """
    legs, theta = ring['legs'], ring['theta_tip']
    out = {'sense_ok': [], 'sense_bad': [], 'overlaps': [],
           'annulus_width': ring['r_outer'] - ring['r_inner']}
 
    for leg in legs:
        th_ex, th_en = theta[(leg, 'Exit')], theta[(leg, 'Entry')]
        mouth = (th_en - th_ex) % 360.0
        yld = feats.get(f'{leg}_Yield')
        if yld is None:
            continue
        v = yld.mean(axis=0) - ring['center']
        th_y = np.degrees(np.arctan2(v[1], v[0])) % 360.0
        (out['sense_ok'] if ((th_y - th_ex) % 360.0) <= mouth
         else out['sense_bad']).append(leg)
 
    if out['sense_bad'] and verbose:
        print(f"    [WARN] circulation sense fails for {out['sense_bad']} — "
              f"Entry/Exit may be swapped, or this roundabout is not "
              f"counter-clockwise.")
 
    for i, a in enumerate(legs):
        for b in legs[i + 1:]:
            a0 = theta[(a, 'Exit')]
            aw = (theta[(a, 'Entry')] - a0) % 360.0
            b0 = theta[(b, 'Exit')]
            bw = (theta[(b, 'Entry')] - b0) % 360.0
            if ((b0 - a0) % 360.0) < aw or ((a0 - b0) % 360.0) < bw:
                out['overlaps'].append((a, b))
 
    if out['overlaps'] and verbose:
        for a, b in out['overlaps']:
            print(f"    [WARN] mouths of '{a}' and '{b}' overlap on the ring — "
                  f"movement arcs through this pair are unreliable. Redraw "
                  f"the Entry/Exit rays.")
 
    if out['annulus_width'] < 2.0 and verbose:
        print(f"    [WARN] annulus is only {out['annulus_width']:.2f} m wide.")
 
    return out


def _ring_spline(cx, cy, R, theta0_rad, x_offset, y_offset, n_ctrl=180):
    """
    Periodic B-spline of the circulating centerline, parameterised
    counter-clockwise from theta0, plus the matching closed WGS84 ring.
 
    (cx, cy) are LOCAL; the WGS84 ring adds the offsets back.
 
    Delegates to fit_roadway_centerline_spline(periodic=True) so the cum_dist
    convention is defined in exactly one place. splprep(per=1) requires the
    sample set to close, i.e. the last point repeats the first.
    """
    th = theta0_rad + np.linspace(0.0, 2.0 * np.pi, n_ctrl + 1)
    x, y = cx + R * np.cos(th), cy + R * np.sin(th)
 
    tck, unew, cum_dist = fit_roadway_centerline_spline(
        list(zip(x, y)), smoothing=0, coordsys='2056', periodic=True,
    )
    lon, lat = _PROJ_2056_TO_LONLAT.transform(x + x_offset, y + y_offset)
    return tck, unew, cum_dist, LineString(np.column_stack([lon, lat]))
 
 
def register_ring_geometry(ring, x_offset, y_offset, name=None, verbose=True):
    """
    Phase 1 for a roundabout: build the geometry_store entry for the
    circulating carriageway.
 
    Returns ONE entry, to be merged into the geometry_store produced by
    register_geometries():
        geometry_store[name] = register_ring_geometry(ring, X_off, Y_off)
 
    Gate arc-lengths: Each gate line is projected onto the circulating centerline with
    project_line_onto_spline
 
    Anchor: s = 0 is due EAST of the center, always, with no per-site choice.
 
    Orientation: positive_dir is 'CCW' and the spline is parameterised counter-clockwise,
    i.e. s increases in the legal direction of travel.
 
    Gates: Each leg contributes 's_entry_<leg>' and 's_exit_<leg>' as plain scalars.
 
    Parameters
    ----------
    ring     : dict from read_ring_kml()
    x_offset, y_offset : float — EPSG:2056 local origin
    name     : str — geometry_store key. Defaults to '<prefix>_Ring'.
 
    Returns
    -------
    entry : dict
    """
    name = name or f"{ring['prefix']}_Ring"
    R, L = ring['R'], ring['L_ring']
    cx = float(ring['center'][0]) - x_offset
    cy = float(ring['center'][1]) - y_offset
 
    tck, unew, cum, line_wgs84 = _ring_spline(cx, cy, R, 0.0,
                                              x_offset, y_offset)
    if abs(float(cum[-1]) - L) > 0.05:
        raise ValueError(f"register_ring_geometry: fitted ring length "
                         f"{cum[-1]:.3f} m disagrees with 2*pi*R = {L:.3f} m")
 
    entry = {
        # --- standard geometry_store interface -------------------------------
        'spline':       (tck, unew, cum),
        'total_length': L,
        'positive_dir': 'CCW',
        'line_wgs84':   line_wgs84,
 
        # --- ring-specific: consumed by the periodic branch in V5 ------------
        'periodic':     True,
        'center':       (cx, cy),          # LOCAL, offsets already applied
        'radius':       R,
        'r_inner':      ring['r_inner'],
        'r_outer':      ring['r_outer'],
        'theta0_deg':   0.0,               # s = 0 due east, by definition
        'ring_legs':    list(ring['legs']),
    }
 
    for (leg, kind), gate_line in ring['gates'].items():
        s = float(project_line_onto_spline(
            line_wgs84, gate_line, tck, unew, cum, x_offset, y_offset,
        ))
        entry[f's_{kind.lower()}_{leg}'] = s
 
        # Cross-check against the ray's own tip bearing. The two answer
        # slightly different questions — where the LINE crosses the ring
        # versus where the TIP points — so they agree only if the ray really
        # is radial about the center.
        s_tip = ring['theta_tip'][(leg, kind)] * np.pi / 180.0 * R
        gap = abs((s - s_tip + L / 2.0) % L - L / 2.0)
        if gap > GATE_XCHECK_M and verbose:
            print(f"    [WARN] {kind}_{leg}: projected gate and ray tip "
                  f"bearing disagree by {gap:.2f} m — check that ray is "
                  f"drawn through the gate.")
 
        if min(s, L - s) < SEAM_CLEARANCE_M and verbose:
            print(f"    [WARN] {kind}_{leg} sits {min(s, L - s):.2f} m from "
                  f"the s = 0 seam; wrap comparisons against it are fragile.")
 
    if verbose:
        print(f"  Registering {name}  (L={L:.1f} m, positive_dir=CCW, "
              f"s=0 due east)")
        gates = sorted(((k, v) for k, v in entry.items()
                        if k.startswith('s_')), key=lambda kv: kv[1])
        for k, v in gates:
            print(f"    {k:34s} s={v:7.2f} m")
        order = [k for k, _ in gates]
        if any(order[i].startswith('s_entry') and order[i + 1].startswith('s_entry')
               for i in range(len(order) - 1)):
            print("    [INFO] two consecutive entry gates — adjacent legs "
                  "share a mouth; check the movement arcs for that pair.")
 
    return entry


def _ring_corridor_polygon(center, d_left, d_right, radius, n_sample=256):
    """
    Validity polygon for a circulating carriageway: an annulus.
 
    Built from the corridor half-widths rather than the kerb radii, so the
    polygon and the d bounds in the segment registry can never disagree.
        d_left  = radius - r_inner     (inward  half-width)
        d_right = r_outer - radius     (outward half-width)
 
    Parameters
    ----------
    center    : (cx, cy) — LOCAL frame, offsets already applied
    d_left    : float [m] — corridor extent toward the center
    d_right   : float [m] — corridor extent away from the center
    radius    : float [m] — circulating centerline radius
    n_sample  : int — vertices per ring
 
    Returns
    -------
    shapely.geometry.Polygon — with one interior ring
    """
    cx, cy = float(center[0]), float(center[1])
    r_in = radius - d_left
    r_out = radius + d_right
 
    if r_in <= 0.0:
        raise ValueError(f"_ring_corridor_polygon: d_left={d_left:.2f} m "
                         f"reaches past the center (radius={radius:.2f} m); "
                         f"the corridor would not be an annulus.")
    if r_out <= r_in:
        raise ValueError(f"_ring_corridor_polygon: inner radius {r_in:.2f} m "
                         f"is not smaller than outer {r_out:.2f} m.")
 
    th = np.linspace(0.0, 2.0 * np.pi, n_sample + 1)[:-1]
    outer = np.column_stack([cx + r_out * np.cos(th), cy + r_out * np.sin(th)])
    inner = np.column_stack([cx + r_in * np.cos(th), cy + r_in * np.sin(th)])
 
    # Shapely convention: shell CCW, holes CW.
    poly = Polygon(outer, [inner[::-1]])
    if not poly.is_valid:
        poly = poly.buffer(0)
    return poly
 
 
def build_ring_segment(geometry_store, segment_registry, ring_seg_def,
                       verbose=True):
    """
    Register the circulating carriageway as one segment, and add it to 
    segment_registry in place.
    Registered with type='ring' rather than 'turn'.
 
    ring_seg_def keys
    -----------------
        seg_key       str        — registry key, e.g. 'Bullingerpl_RA_CCW'
        geometry_key  str        — key into geometry_store
        mode          str        — 'shared' | 'bike' | 'car'. Default 'shared'
        d_left        float      — optional [m], inward.  Default radius-r_inner
        d_right       float      — optional [m], outward. Default r_outer-radius
 
    Returns
    -------
    segment_registry : dict — same object, mutated
    """
    seg_key = ring_seg_def['seg_key']
    geom_key = ring_seg_def['geometry_key']
    mode = ring_seg_def.get('mode', 'shared')
 
    geo = geometry_store[geom_key]
    if not geo.get('periodic'):
        raise ValueError(f"build_ring_segment: geometry '{geom_key}' is not a "
                         f"ring (no 'periodic' flag). Register it with "
                         f"register_ring_geometry() first.")
    if geo.get('positive_dir') != 'CCW':
        raise ValueError(f"build_ring_segment: geometry '{geom_key}' has "
                         f"positive_dir={geo.get('positive_dir')!r}, expected "
                         f"'CCW'.")
 
    R = geo['radius']
    d_left = float(ring_seg_def.get('d_left', R - geo['r_inner']))
    d_right = float(ring_seg_def.get('d_right', geo['r_outer'] - R))
 
    validity_poly = _ring_corridor_polygon(geo['center'], d_left, d_right, R)
 
    segment_registry[seg_key] = {
        'type':             'ring',
        'geometry_key':     geom_key,
        'direction':        'CCW',
        # positive_dir == direction, so travel and spline agree by
        # construction; the mirrored state carries clockwise riders.
        'is_forward':       True,
        'mode':             mode,
        'bike_lane':        None,
        'd_left':           d_left,
        'd_right':          d_right,
        'd_max':            max(d_left, d_right),
        'car_lane_d_bnd':   {},
        'validity_polygon': validity_poly,
    }
 
    if verbose:
        r_in, r_out = R - d_left, R + d_right
        exp = np.pi * (r_out ** 2 - r_in ** 2)
        print(f"  {seg_key}  (ring, mode={mode}, "
              f"d_left={d_left:.2f} in / d_right={d_right:.2f} out, "
              f"r={r_in:.2f}..{r_out:.2f} m)")
        print(f"    annulus area {validity_poly.area:.1f} m^2 "
              f"(analytic {exp:.1f} m^2), "
              f"{len(geo.get('ring_legs', []))} legs")
 
    return segment_registry

