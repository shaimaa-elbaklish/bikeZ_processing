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
import os
import sys
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import geopandas as gpd

from pyproj import Transformer
from scipy.interpolate import splev
from scipy.interpolate import interp1d
from shapely.geometry import LineString
from shapely.geometry import MultiPoint
from shapely.geometry import Polygon
from datetime import date as _date

from tools_coordinate_transform import fit_roadway_centerline_spline
from tools_coordinate_transform import convert_xy2056_to_roadway_coordinates
from tools_coordinate_transform import connect_lines_g2

# #############################################################################
# METHODS
# #############################################################################
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


def resolve_geometry(geometry_store, geom_key, direction=None):
    """
    Return the correct spline entry for a (geom_key, direction) pair.
    
    If geometry_store[geom_key] has an 'opposite' sub-dict AND the
    requested direction is opposite to positive_dir, return the
    'opposite' entry merged with top-level metadata.
    Otherwise return the top-level entry as-is (D1 behaviour).
    """
    entry    = geometry_store[geom_key]
    pos_dir  = entry['positive_dir']
    is_forward = (direction is None) or (direction == pos_dir)

    if not is_forward and 'opposite' in entry:
        # Return opposite spline but keep shared metadata
        return {
            **entry,
            'spline':       entry['opposite']['spline'],
            'total_length': entry['opposite']['total_length'],
            'line_wgs84':   entry['opposite']['line_wgs84'],
            's_stop':       entry['opposite'].get('s_stop',  entry.get('s_stop')),
            's_yield':      entry['opposite'].get('s_yield', entry.get('s_yield')),
            's_change':  entry['opposite'].get('s_change', entry.get('s_change')),  # ADD
        }
    return entry


def get_s_domain(geometry_key, geometry_store, gdf_stop_yield_lines):
    """
    Compute s_stop and s_yield for a geometry in spline-native arc-length,
    and store them directly into geometry_store[geometry_key].

    s_stop  : arc-length where the stop-line crosses the spline [m]
    s_yield : arc-length where the yield-line crosses the spline [m]

    Both are in spline-native coords (increasing in positive_dir).
    s_stop < s_yield is guaranteed — if the projection returns them
    in the wrong order (e.g. because the spline runs opposite to the
    stop/yield line ordering), they are swapped.
    """
    x_offset = geometry_store['x_offset']
    y_offset = geometry_store['y_offset']
    entry    = geometry_store[geometry_key]

    # ── Forward (positive) direction ─────────────────────────────────────────
    line                = entry['line_wgs84']
    tck, unew, cum_dist = entry['spline']

    stop_row  = gdf_stop_yield_lines[
        gdf_stop_yield_lines['Description'] == f'{geometry_key}_Stop'
    ].copy()
    yield_row = gdf_stop_yield_lines[
        gdf_stop_yield_lines['Description'] == f'{geometry_key}_Yield'
    ].copy()

    if len(stop_row) == 0:
        raise ValueError(f"No stop-line found for: {geometry_key}")
    if len(yield_row) == 0:
        raise ValueError(f"No yield-line found for: {geometry_key}")

    s_stop  = project_line_onto_spline(
        line, stop_row.geometry.iloc[0], tck, unew, cum_dist, x_offset, y_offset
    )
    s_yield = project_line_onto_spline(
        line, yield_row.geometry.iloc[0], tck, unew, cum_dist, x_offset, y_offset
    )
    entry['s_stop']   = float(s_stop)
    entry['s_yield']  = float(s_yield)
    entry['s_change'] = 0.5 * (float(s_stop) + float(s_yield))  # ADD

    # ── Opposite direction (if defined) ──────────────────────────────────────
    if 'opposite' in entry:
        opp             = entry['opposite']
        line_opp        = opp['line_wgs84']
        tck_o, unew_o, cum_o = opp['spline']

        s_stop_opp  = project_line_onto_spline(
            line_opp, stop_row.geometry.iloc[0], tck_o, unew_o, cum_o, x_offset, y_offset
        )
        s_yield_opp = project_line_onto_spline(
            line_opp, yield_row.geometry.iloc[0], tck_o, unew_o, cum_o, x_offset, y_offset
        )
        opp['s_stop']   = float(s_stop_opp)
        opp['s_yield']  = float(s_yield_opp)
        opp['s_change'] = 0.5 * (float(s_stop_opp) + float(s_yield_opp))  # ADD
    return


def _resolve_d_lateral(d_spec, default=10.0):
    """
    Resolve a d_max specification into (d_left, d_right).

    Accepts three forms:
      - float / int  → symmetric: d_left = d_right = value
      - {'d_left': x, 'd_right': y} → asymmetric
      - None  → symmetric default

    d_left  : lateral offset toward the LEFT of travel direction  [m]
    d_right : lateral offset toward the RIGHT of travel direction [m]

    For lane segments the normal convention is:
      - The correct carriageway is to the RIGHT of travel (cyclists keep right)
      - d_right should cover the full carriageway + any bike lane
      - d_left is a small GPS-noise tolerance (e.g. 1–3 m)

    Returns
    -------
    d_left  : float
    d_right : float
    d_max   : float — max(d_left, d_right), used for hard lateral veto
    """
    if d_spec is None:
        return float(default), float(default), float(default)
    if isinstance(d_spec, (int, float)):
        v = float(d_spec)
        return v, v, v
    if isinstance(d_spec, dict):
        d_left  = float(d_spec.get('d_left',  default))
        d_right = float(d_spec.get('d_right', default))
        return d_left, d_right, max(d_left, d_right)
    raise TypeError(f"d_max spec must be float or dict, got {type(d_spec)}")


def build_validity_polygon(seg_key, segment_registry, geometry_store,
                           d_left, d_right, n_sample=200):
    """
    Build a lateral corridor polygon representing the valid spatial region
    for a directed segment.

    The polygon is constructed by offsetting the centerline within the
    segment's valid s-domain:
      - d_right metres to the RIGHT of the travel direction
      - d_left  metres to the LEFT  of the travel direction

    For lane segments, d_right should cover the carriageway the cyclist
    is expected to use (right side of travel), while d_left is a small
    GPS-noise tolerance. This separates opposing-direction lanes that
    share a centerline spline (e.g. Roentgenstr_EB vs Roentgenstr_WB).

    For turn segments, symmetric values are typical (d_left == d_right).

    Expansion is purely lateral — perpendicular to the centerline tangent —
    so the longitudinal extent of the segment is respected exactly.

    Parameters
    ----------
    seg_key          : str — key into segment_registry
    segment_registry : dict
    geometry_store   : dict
    d_left           : float — offset to the LEFT  of travel direction [m]
    d_right          : float — offset to the RIGHT of travel direction [m]
    n_sample         : int  — number of centerline points sampled

    Returns
    -------
    polygon : shapely.geometry.Polygon — convex hull of the lateral corridor
              in local EPSG:2056 coordinates (offset already subtracted)
    """
    entry      = segment_registry[seg_key]
    geom_key   = entry['geometry_key']
    seg_type   = entry['type']
    is_forward = entry['is_forward']
    direction           = segment_registry[seg_key].get('direction')
    resolved            = resolve_geometry(geometry_store, geom_key, direction)
    tck, unew, cum_dist = resolved['spline']
    L                   = resolved['total_length']

    # ── Determine valid s-domain in native spline coords ─────────────────────
    s_start, s_end = 0.0, L

    # ── Sample centerline densely within s-domain ─────────────────────────────
    s_vals = np.linspace(s_start, s_end, n_sample)
    t_vals = np.interp(s_vals, cum_dist, unew)

    x_vals,  y_vals  = splev(t_vals, tck, der=0)
    dx_vals, dy_vals = splev(t_vals, tck, der=1)

    # Normalise tangents → unit vectors in spline-native direction
    tang_len = np.sqrt(dx_vals**2 + dy_vals**2)
    tang_len = np.where(tang_len > 1e-12, tang_len, 1.0)
    tx_nat = dx_vals / tang_len
    ty_nat = dy_vals / tang_len

    # For reverse segments the travel direction is opposite to the spline
    # tangent — flip so tx/ty always point in the vehicle's travel direction
    if not is_forward:
        tx_nat = -tx_nat
        ty_nat = -ty_nat

    # Normal pointing LEFT of travel direction (rotate tangent 90° CCW)
    nx_left = -ty_nat
    ny_left =  tx_nat

    # ── Offset by d_left (left) and d_right (right) ───────────────────────────
    # Right of travel = opposite of left normal
    left_x  = x_vals + d_left  * nx_left
    left_y  = y_vals + d_left  * ny_left
    right_x = x_vals - d_right * nx_left
    right_y = y_vals - d_right * ny_left

    all_pts = np.column_stack([
        np.concatenate([left_x,  right_x]),
        np.concatenate([left_y,  right_y]),
    ])
    return MultiPoint(all_pts).convex_hull


def rebuild_validity_polygons(segment_registry, geometry_store):
    """
    Rebuild validity_polygon for all segments in segment_registry.

    Call this after restrict_segment_roles() has nulled some
    approach_native / departure_native domains — the polygons must
    reflect the final restricted s-domains.

    Modifies segment_registry in place.

    Parameters
    ----------
    segment_registry : dict — modified in place
    geometry_store   : dict

    Returns
    -------
    segment_registry : dict (same object, modified in place)
    """
    for seg_key, entry in segment_registry.items():
        d_left  = entry.get('d_left',  10.0)
        d_right = entry.get('d_right', 10.0)
        entry['validity_polygon'] = build_validity_polygon(
            seg_key, segment_registry, geometry_store,
            d_left=d_left, d_right=d_right
        )
    n = len(segment_registry)
    print(f"\nrebuild_validity_polygons: {n} validity polygon(s) built ✓")
    return segment_registry


def build_segment_registry(geometry_store, directed_segments,
                            bike_lane_info, mode, d_max_map=None):
    """
    Build segment_registry from geometry_store and directed segment metadata.

    For each directed segment, computes approach and departure domains
    in spline-native arc-length using s_stop, s_yield, and positive_dir
    from geometry_store. No coordinate flipping — native coords throughout.

    Also builds a validity_polygon per segment — an asymmetric lateral
    corridor polygon in local EPSG:2056 representing the valid spatial
    region for matching. See build_validity_polygon() for details.

    Parameters
    ----------
    geometry_store     : dict — as assembled in Phase A
    directed_segments  : list of (geom_key, direction) tuples
    bike_lane_info     : dict — {seg_key: {'w_bike': float, 'side': int} | None}
    mode               : dict — {seg_key: 'car' | 'bike' | 'shared'}
    d_max_map          : dict or None
                         Values can be:
                           float  → symmetric: d_left = d_right = value
                           {'d_left': x, 'd_right': y} → asymmetric
                         Missing keys default to 10.0 m symmetric.
                         d_right should cover the expected carriageway
                         (right side of travel); d_left is a small
                         GPS-noise tolerance.

    Returns
    -------
    segment_registry : dict
    """
    registry = {}

    for geom_key, direction in directed_segments:
        seg_key    = f'{geom_key}_{direction}'
        is_forward = (direction == geometry_store[geom_key]['positive_dir'])
        entry      = resolve_geometry(geometry_store, geom_key, direction)
        # pos_dir    = entry['positive_dir']
        # L          = entry['total_length']
        # s_stop     = entry['s_stop']
        # s_yield    = entry['s_yield']

        # Approach/departure in spline-native arc-length
        s_change         = entry['s_change']
        # approach_native  = None   # determined by TURNING_MOVEMENTS role
        # departure_native = None   # determined by TURNING_MOVEMENTS role

        d_spec  = (d_max_map or {}).get(seg_key, 10.0)
        d_left, d_right, d_max = _resolve_d_lateral(d_spec)

        registry[seg_key] = {
            'type':             'lane',
            'geometry_key':     geom_key,
            'direction':        direction,
            'is_forward':       is_forward,
            'mode':             mode.get(seg_key, 'shared'),
            's_stop':           entry['s_stop'],
            's_yield':          entry['s_yield'],
            's_change':         s_change,           # single boundary point
            'approach_native':  None,               # set by restrict_segment_roles
            'departure_native': None,               # set by restrict_segment_roles
            'bike_lane':        bike_lane_info.get(seg_key, None),
            'd_left':           d_left,    # offset to LEFT  of travel [m]
            'd_right':          d_right,   # offset to RIGHT of travel [m]
            'd_max':            d_max,     # max(d_left, d_right) — hard veto
            'validity_polygon': None,      # populated below
        }

    # ── Build validity polygons ───────────────────────────────────────────────
    # Two-step: all entries must exist first so build_validity_polygon can
    # look them up. Rebuilt after restrict_segment_roles() nulls some domains
    # via rebuild_validity_polygons().
    for seg_key in registry:
        e = registry[seg_key]
        registry[seg_key]['validity_polygon'] = build_validity_polygon(
            seg_key, registry, geometry_store,
            d_left=e['d_left'], d_right=e['d_right']
        )

    return registry


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


def add_bike_lane_boundaries(segment_registry, geometry_store,
                              gdf_bike_boundaries):
    x_offset = geometry_store['x_offset']
    y_offset = geometry_store['y_offset']
    for seg_key, entry in segment_registry.items():
        if entry['bike_lane'] is None:
            continue

        geom_key        = entry['geometry_key']
        direction       = entry['direction']
        resolved        = resolve_geometry(geometry_store, geom_key, direction)
        tck, unew, cum_dist = resolved['spline']

        boundary_row = gdf_bike_boundaries[
            gdf_bike_boundaries['Description'] == seg_key
        ]
        if len(boundary_row) == 0:
            print(f"  [WARNING] No boundary found for {seg_key} — skipping")
            continue

        d_boundary_spline, s_domain, side_inferred = build_d_boundary_spline(
            boundary_row.geometry.iloc[0], tck, unew, cum_dist,
            x_offset, y_offset
        )
        
        # ── Correct side using direction rule ─────────────────────────────
        is_forward      = entry['is_forward']
        direction       = entry['direction']
        positive_dir    = geometry_store[geom_key]['positive_dir']
        side_expected = -1 if is_forward else +1
        if side_inferred != side_expected:
            print(f"  {seg_key}: side corrected "
                  f"{side_inferred:+d} → {side_expected:+d} "
                  f"(direction={direction}, positive_dir={positive_dir})")
        side = side_expected   # always use direction-derived side
        
        entry['bike_lane']['d_boundary_spline'] = d_boundary_spline
        entry['bike_lane']['s_domain']          = s_domain
        entry['bike_lane']['side']              = side   # derived, not hardcoded

        print(f"  {seg_key}: side={side:+d}  "
              f"s∈[{s_domain[0]:.1f}, {s_domain[1]:.1f}] m  "
              f"d_boundary∈[{d_boundary_spline(s_domain[0]):.2f}, "
              f"{d_boundary_spline(s_domain[1]):.2f}] m")
    return


def sample_spline_near_boundary(seg_key, role, segment_registry,
                                 geometry_store, n_pts=10):
    """
    Sample n_pts points near the intersection boundary in local EPSG:2056
    coords, ordered in the vehicle's travel direction.

    Parameters
    ----------
    seg_key          : str — e.g. 'Roentgenstr_EB'
    role             : 'approach' or 'departure'
    segment_registry : dict
    geometry_store   : dict
    n_pts            : int — number of points to sample

    Returns
    -------
    pts : (n_pts, 2) array in local EPSG:2056 coords (offset subtracted)
          ordered in travel direction
    """
    entry           = segment_registry[seg_key]
    geom_key        = entry['geometry_key']
    direction       = entry['direction']
    resolved        = resolve_geometry(geometry_store, geom_key, direction)
    tck, unew, cum_dist = resolved['spline']
    is_forward      = entry['is_forward']
    s_change        = entry['s_change']
    L               = resolved['total_length']
    
    if role == 'approach':
        if is_forward:
            s_vals = np.linspace(max(0.0, s_change - 20.0), s_change, n_pts)
        else:
            s_vals = np.linspace(min(L, s_change + 20.0), s_change, n_pts)
    else:  # departure
        if is_forward:
            s_vals = np.linspace(s_change, min(L, s_change + 20.0), n_pts)
        else:
            s_vals = np.linspace(s_change, max(0.0, s_change - 20.0), n_pts)

    # Evaluate spline — no flip needed since s_vals already in travel order
    # returns local coords directly (offset already in spline)
    t_vals         = np.interp(s_vals, cum_dist, unew)
    x_vals, y_vals = splev(t_vals, tck)
    pts            = np.column_stack([x_vals, y_vals])

    return pts


def build_turn_spline(approach_seg, departure_seg,
                       segment_registry, geometry_store,
                       n_pts=10, n_connector=50,
                       angle_threshold_deg=5,
                       verbose=False):
    """
    Build a smooth connector between approach stop-line and departure
    yield-line using connect_lines_g2, then fit a B-spline to the result.
    
    All coordinates are in local EPSG:2056 (offset subtracted).
    No offset handling needed here — splines already live in local coords.

    Parameters
    ----------
    approach_seg, departure_seg : str — segment keys
    segment_registry, geometry_store : dicts
    n_pts           : points sampled from each spline near boundary
    n_connector     : points in the connector curve
    angle_threshold_deg : below this heading difference, use Hermite
    verbose         : print diagnostics

    Returns
    -------
    tck, unew, cum_dist, total_length : spline representation
    method : 'clothoid' or 'hermite'
    """
    # Sample points near boundary from each segment
    pts_approach  = sample_spline_near_boundary(
        approach_seg,  'approach',  segment_registry, geometry_store, n_pts
    )
    pts_departure = sample_spline_near_boundary(
        departure_seg, 'departure', segment_registry, geometry_store, n_pts
    )
    
    # # ── DEBUG ────────────────────────────────────────────────────────────────
    # print(f"\n  DEBUG {approach_seg} → {departure_seg}")
    # print(f"    pts_approach  start={pts_approach[0]}  end={pts_approach[-1]}")
    # print(f"    pts_departure start={pts_departure[0]}  end={pts_departure[-1]}")

    # # Check heading difference
    # v0 = pts_approach[-1]  - pts_approach[-2]
    # v1 = pts_departure[1]  - pts_departure[0]
    # v0 = v0 / (np.linalg.norm(v0) + 1e-9)
    # v1 = v1 / (np.linalg.norm(v1) + 1e-9)
    # theta0 = np.degrees(np.arctan2(v0[1], v0[0]))
    # theta1 = np.degrees(np.arctan2(v1[1], v1[0]))
    # angle_diff = abs(np.degrees(
    #     np.arctan2(np.sin(np.radians(theta1 - theta0)),
    #                np.cos(np.radians(theta1 - theta0)))
    # ))
    # print(f"    theta0={theta0:.1f}°  theta1={theta1:.1f}°  "
    #       f"angle_diff={angle_diff:.1f}°  "
    #       f"threshold={angle_threshold_deg}°")
    # print(f"    p0={pts_approach[-1]}  p1={pts_departure[0]}")
    # print(f"    distance p0→p1 = "
    #       f"{np.linalg.norm(pts_departure[0] - pts_approach[-1]):.2f} m")
    # # ── END DEBUG ────────────────────────────────────────────────────────────
    
    # Build connector
    _, connector, method = connect_lines_g2(
        pts_approach, pts_departure,
        n_connector=n_connector,
        angle_threshold_deg=angle_threshold_deg,
        verbose=verbose
    )

    if verbose:
        print(f"    method={method}  connector pts={len(connector)}")

    # Fit B-spline to connector only (the turn geometry)
    # connector already in EPSG:2056
    if len(connector) < 2:
        raise RuntimeError("Connector is degenerate — too few points.")
    
    # Use merged (full path: approach anchor + connector + departure anchor)
    connector_clean = np.array(connector)

    # Remove near-duplicate points (splprep fails on zero-length segments)
    keep = [0]
    for i in range(1, len(connector_clean)):
        if np.linalg.norm(connector_clean[i] - connector_clean[keep[-1]]) >= 1e-03:
            keep.append(i)
    connector_clean = connector_clean[keep]

    if verbose:
        print(f"    merged: {len(connector)} pts → {len(connector_clean)} "
              f"after dedup")

    if len(connector_clean) < 4:
        raise RuntimeError(
            f"Only {len(connector_clean)} unique points after deduplication "
            f"(need ≥ 4). Try increasing n_connector."
        )

    # Fit spline — coordsys='2056', no offset needed (already local)
    tck, unew, cum_dist = fit_roadway_centerline_spline(
        connector_clean.tolist(), coordsys='2056'
    )
    total_length = float(cum_dist[-1])

    return tck, unew, cum_dist, total_length, connector_clean, method


def build_all_turns(turning_movements, segment_registry,
                    geometry_store, n_pts=10, n_connector=50,
                    angle_threshold_deg=5, d_max_map=None,
                    verbose=False):
    """
    Build all turning movement splines and register them in
    segment_registry and geometry_store.

    Turn keys: 'turn_{approach}_2_{departure}'

    Parameters
    ----------
    turning_movements : list of (approach_seg, departure_seg)
    segment_registry  : dict — modified in place
    geometry_store    : dict — modified in place
    d_max_map         : dict or None — {turn_key: float} lateral half-width [m]
                        for validity polygon. Missing keys default to 15.0 m.

    Returns
    -------
    turn_keys : list of registered turn keys
    """
    turn_keys = []

    for approach_seg, departure_seg in turning_movements:
        turn_key = f'turn_{approach_seg}_2_{departure_seg}'
        print(f"  Building: {turn_key} ...", end=' ')

        try:
            tck, unew, cum_dist, total_length, connector, method = build_turn_spline(
                approach_seg, departure_seg,
                segment_registry, geometry_store,
                n_pts=n_pts, n_connector=n_connector,
                angle_threshold_deg=angle_threshold_deg,
                verbose=verbose
            )
        except Exception as e:
            print(f"FAILED — {e}")
            sys.exit(1)

        # Add to geometry_store
        geometry_store[turn_key] = {
            'spline':       (tck, unew, cum_dist),
            'total_length': total_length,
            'positive_dir': None,   # turns have no cardinal direction
            's_stop':       None,
            's_yield':      None,
            'line_wgs84':   None,
            'method':       method,
        }

        # d_left/d_right for this turn — turns accept scalar or dict,
        # but are typically symmetric. Default 15.0 m each side.
        d_spec  = (d_max_map or {}).get(turn_key, 15.0)
        d_left, d_right, d_max = _resolve_d_lateral(d_spec, default=15.0)

        # Insert entry first (validity_polygon=None), then build polygon below.
        # Two-step required: build_validity_polygon looks up segment_registry[seg_key],
        # so the entry must exist before the polygon can be built.
        segment_registry[turn_key] = {
            'type':             'turn',
            'geometry_key':     turn_key,
            'approach_seg':     approach_seg,
            'departure_seg':    departure_seg,
            'is_forward':       True,   # turns always traversed forward
            'mode':             'shared',
            'approach_native':  (0.0, total_length),
            'departure_native': (0.0, total_length),
            'bike_lane':        None,
            'd_left':           d_left,
            'd_right':          d_right,
            'd_max':            d_max,
            'validity_polygon': None,
        }
        # Now the entry exists — build and assign the polygon
        segment_registry[turn_key]['validity_polygon'] = build_validity_polygon(
            turn_key, segment_registry, geometry_store,
            d_left=d_left, d_right=d_right
        )

        print(f"✓  L={total_length:.1f} m  [{method}]")
        turn_keys.append(turn_key)

    return turn_keys


def build_movement_registry(movements, segment_registry):
    """
    Build movement_registry from MOVEMENTS list.

    Validates that all segment keys exist in segment_registry.
    Validates that role sequence is always approach → turn → departure.

    Parameters
    ----------
    movements        : list of (movement_key, [(seg_key, role), ...])
    segment_registry : dict

    Returns
    -------
    movement_registry : dict
        {movement_key: [(seg_key, role), ...]}
    """
    VALID_ROLES    = ('approach', 'turn', 'departure')
    VALID_SEQUENCE = ('approach', 'turn', 'departure')

    movement_registry = {}
    errors            = []

    for mov_key, sequence in movements:
        # Validate role sequence
        roles = tuple(role for _, role in sequence)
        if roles != VALID_SEQUENCE:
            errors.append(
                f"  {mov_key}: invalid role sequence {roles} "
                f"(expected {VALID_SEQUENCE})"
            )
            continue

        # Validate all segment keys exist
        missing = [
            seg_key for seg_key, _ in sequence
            if seg_key not in segment_registry
        ]
        if missing:
            errors.append(
                f"  {mov_key}: missing segment keys {missing}"
            )
            continue

        # Validate approach/departure roles match segment types
        seg_types = {
            seg_key: segment_registry[seg_key]['type']
            for seg_key, _ in sequence
        }
        for seg_key, role in sequence:
            seg_type = seg_types[seg_key]
            if role in ('approach', 'departure') and seg_type != 'lane':
                errors.append(
                    f"  {mov_key}: {seg_key} has role='{role}' "
                    f"but type='{seg_type}' (expected 'lane')"
                )
            if role == 'turn' and seg_type != 'turn':
                errors.append(
                    f"  {mov_key}: {seg_key} has role='turn' "
                    f"but type='{seg_type}' (expected 'turn')"
                )

        movement_registry[mov_key] = sequence

    # Report
    print(f"\nmovement_registry: {len(movement_registry)} movements registered")
    if errors:
        print(f"  {len(errors)} error(s):")
        for e in errors:
            print(e)
    else:
        print("  All movements validated ✓")

    for mov_key, sequence in movement_registry.items():
        segs = ' → '.join(
            f"{seg_key}[{role}]" for seg_key, role in sequence
        )
        print(f"  {mov_key}: {segs}")

    return movement_registry


def restrict_segment_roles(segment_registry, movement_registry, geometry_store):
    """
    Post-build pass: set each lane segment's approach_native and
    departure_native domains based on the roles it actually plays
    in movement_registry, using s_change as the boundary point.

    For each lane segment:
      - If it plays 'approach':  set approach_native  from s_change
      - If it plays 'departure': set departure_native from s_change
      - If it plays both:        set both domains
      - If it plays neither:     null both domains

    Parameters
    ----------
    segment_registry : dict — modified in place
    movement_registry: dict
    geometry_store   : dict

    Returns
    -------
    segment_registry : dict (same object, modified in place)
    """
    # Collect roles per segment from movement_registry
    seg_roles = {}
    for mov_key, sequence in movement_registry.items():
        for seg_key, role in sequence:
            seg_roles.setdefault(seg_key, set()).add(role)

    restricted = []
    for seg_key, entry in segment_registry.items():
        if entry['type'] != 'lane':
            continue

        roles      = seg_roles.get(seg_key, set())
        s_change   = entry['s_change']
        is_forward = entry['is_forward']
        resolved   = resolve_geometry(
            geometry_store, entry['geometry_key'], entry['direction']
        )
        L = resolved['total_length']

        if not roles:
            entry['approach_native']  = None
            entry['departure_native'] = None
            restricted.append(f'  {seg_key}: not in any movement → both nulled')

        else:
            # Approach domain: from segment start (in travel direction) to s_change
            if 'approach' in roles:
                entry['approach_native'] = (0.0, s_change) if is_forward \
                                           else (s_change, L)
            else:
                entry['approach_native'] = None
                restricted.append(f'  {seg_key}: no approach role → approach_native nulled')

            # Departure domain: from s_change to segment end (in travel direction)
            if 'departure' in roles:
                entry['departure_native'] = (s_change, L) if is_forward \
                                            else (0.0, s_change)
            else:
                entry['departure_native'] = None
                restricted.append(f'  {seg_key}: no departure role → departure_native nulled')

    print(f"\nrestrict_segment_roles: {len(restricted)} segment(s) restricted")
    for msg in restricted:
        print(msg)

    return segment_registry


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