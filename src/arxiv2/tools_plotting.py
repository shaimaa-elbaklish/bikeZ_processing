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
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from pyproj import Transformer
from scipy.interpolate import splev
from matplotlib.lines import Line2D

from tools_coordinate_transform import convert_roadway_to_xy2056_coordinates
from tools_infrastructure_geometry import resolve_geometry

# #############################################################################
# METHODS
# #############################################################################
LANE_COLOR_PALETTE = [
    'steelblue', 'tomato', 'mediumpurple', 'darkorange',
    'seagreen',  'crimson', 'goldenrod',   'teal',
    'slategray', 'orchid',  'sienna',      'cornflowerblue',
]


def build_lane_color_map(geometry_store):
    """
    Build a {geom_key: color} map for all lane geometry entries.

    Colors are assigned from LANE_COLOR_PALETTE in insertion order.
    Call once before a plotting loop and pass the result into
    plot_geometry_store, plot_turn_splines, and plot_lane_coord_debug
    so colors are consistent across all plots for the same registry.

    Parameters
    ----------
    geometry_store : dict

    Returns
    -------
    lane_color_map : dict  geom_key -> color str
    """
    skip = {'x_offset', 'y_offset'}
    lane_keys = [
        k for k, v in geometry_store.items()
        if k not in skip and v.get('positive_dir') is not None
    ]
    return {
        k: LANE_COLOR_PALETTE[i % len(LANE_COLOR_PALETTE)]
        for i, k in enumerate(lane_keys)
    }


def plot_geometry_store(geometry_store, gdf_swisstopo,
                        offset_m=3.0, save_path=None):
    """
    Plot geometry_store for visual inspection.

    For each geometry_key, plots two offset lines (forward and reverse
    directions) to separate EB/WB or NB/SB visually. Approach domains
    are solid, departure domains are dashed, inside-intersection is dotted.

    Parameters
    ----------
    geometry_store : dict
        As assembled in Phase A — must contain s_stop and s_yield.
    gdf_swisstopo : GeoDataFrame
        Full KML GeoDataFrame with Description column.
    offset_m : float
        Lateral offset in metres to separate the two directions.
    save_path : str or None
        If given, saves the figure to this path.
    """
    transformer_to_2056 = Transformer.from_crs(
        "EPSG:4326", "EPSG:2056", always_xy=True
    )
    x_offset            = geometry_store['x_offset']
    y_offset            = geometry_store['y_offset']

    DIRECTION_SUFFIXES = ('_NB', '_SB', '_EB', '_WB')
    COLORS = {
        'Roentgenstr': 'steelblue',
        'Zollstr':     'tomato',
        'LangstrN':    'mediumpurple',
        'LangstrS':    'darkorange',
    }

    fig, ax = plt.subplots(figsize=(14, 14))
    ax.set_title(
        'Geometry Store — Visual Inspection\n'
        'Solid = approach  |  Dashed = departure  |'
        '  Dotted = inside intersection\n'
        'Offset +/- = forward / reverse direction',
        fontsize=10
    )

    # ── Helper: plot one arc-length interval with lateral offset ─────────────
    def plot_offset_interval(s_min, s_max, d_offset,
                             tck, unew, cum_dist,
                             color, lw, ls, zorder, label=None):
        """Plot [s_min, s_max] offset by d_offset metres from centerline."""
        if s_max <= s_min + 0.1:
            return
        all_s = np.linspace(s_min, s_max, num=150)
        x_seg = np.zeros_like(all_s)
        y_seg = np.zeros_like(all_s)
        for i in range(len(all_s)):
            x_seg[i], y_seg[i] = convert_roadway_to_xy2056_coordinates(
                all_s[i], d_offset, tck, unew, cum_dist, x_offset, y_offset
            )
        ax.plot(x_seg, y_seg, color=color, linewidth=lw,
                linestyle=ls, zorder=zorder,
                solid_capstyle='round', label=label)
        return x_seg, y_seg

    # ── Intersection polygon ──────────────────────────────────────────────────
    poly_row = gdf_swisstopo[
        gdf_swisstopo['Description'] == 'Intersection_Area'
    ]
    if len(poly_row) > 0:
        poly      = poly_row.geometry.iloc[0]
        xs, ys    = poly.exterior.xy
        xs_m, ys_m = transformer_to_2056.transform(xs, ys)
        ax.fill(xs_m, ys_m, alpha=0.12, color='yellow', zorder=1)
        ax.plot(xs_m, ys_m, color='gold', linewidth=2,
                zorder=2, label='Intersection box')

    # ── Per-geometry spline domains ───────────────────────────────────────────
    first = True
    for geom_key, entry in geometry_store.items():
        if geom_key == 'x_offset' or geom_key == 'y_offset':
            continue
        pos_dir = entry['positive_dir']
        col     = COLORS.get(geom_key, 'black')

        for is_forward in [True, False]:
            d_offset  = +offset_m if is_forward else -offset_m
            direction = pos_dir if is_forward else {
                'EB': 'WB', 'WB': 'EB',
                'NB': 'SB', 'SB': 'NB'
            }[pos_dir]
        
            resolved        = resolve_geometry(geometry_store, geom_key, direction)
            tck, unew, cum_dist = resolved['spline']
            L               = resolved['total_length']
            s_stop          = resolved['s_stop']
            s_yield         = resolved['s_yield']
        
            if is_forward:
                s_app_min,  s_app_max  = 0.0,     s_stop
                s_dep_min,  s_dep_max  = s_yield,  L
            else:
                s_app_min,  s_app_max  = s_yield,  L
                s_dep_min,  s_dep_max  = 0.0,      s_stop

            # Labels — only on very first segment
            lbl_app = 'Approach'            if first else '_nolegend_'
            lbl_int = 'Inside intersection' if first else '_nolegend_'
            lbl_dep = 'Departure'           if first else '_nolegend_'

            # Approach — solid
            plot_offset_interval(
                s_app_min, s_app_max, d_offset,
                tck, unew, cum_dist,
                col, lw=4, ls='-', zorder=5, label=lbl_app
            )

            # Inside intersection — thin dotted gray
            plot_offset_interval(
                min(s_stop, s_yield), max(s_stop, s_yield), d_offset,
                tck, unew, cum_dist,
                'gray', lw=1.5, ls=':', zorder=3, label=lbl_int
            )

            # Departure — dashed
            plot_offset_interval(
                s_dep_min, s_dep_max, d_offset,
                tck, unew, cum_dist,
                col, lw=4, ls='--', zorder=5, label=lbl_dep
            )

            # Direction label at approach midpoint
            s_label = (s_app_min + s_app_max) / 2.0
            if s_app_max > s_app_min + 1.0:
                x_lbl, y_lbl = convert_roadway_to_xy2056_coordinates(
                    s_label, d_offset, tck, unew, cum_dist, x_offset, y_offset
                )
                ax.annotate(
                    f'{geom_key}_{direction}',
                    (x_lbl, y_lbl),
                    fontsize=7, color=col, fontweight='bold',
                    xytext=(5, 5), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                              alpha=0.8, edgecolor=col)
                )
            first = False

        # Stop and yield crossing points on centerline (no offset)
        for s_val, marker, color, lbl_suffix in [
            (s_stop,  'o', 'red',       'Stop crossing'),
            (s_yield, 'D', 'darkorange','Yield crossing'),
        ]:
            t_val      = float(np.interp(s_val, cum_dist, unew))
            x_val, y_val = splev(t_val, tck)
            x_val = x_val + x_offset
            y_val = y_val + y_offset
            ax.scatter(x_val, y_val, color=color, s=120, zorder=8, marker=marker,
                       label=lbl_suffix if geom_key == list(geometry_store.keys())[0]
                       else '_nolegend_')

    # ── Stop-lines ────────────────────────────────────────────────────────────
    first_stop = True
    for _, row in gdf_swisstopo.iterrows():
        if not row['Description'].endswith('_Stop'):
            continue
        xs, ys     = [c[0] for c in row.geometry.coords], [c[1] for c in row.geometry.coords]
        xs_m, ys_m = transformer_to_2056.transform(xs, ys)
        lbl = 'Stop-line' if first_stop else '_nolegend_'
        ax.plot(xs_m, ys_m, color='red', linewidth=2.5,
                zorder=6, solid_capstyle='round', label=lbl)
        mid = len(xs_m) // 2
        ax.annotate(row['Description'], (xs_m[mid], ys_m[mid]),
                    fontsize=7, color='darkred',
                    xytext=(4, -12), textcoords='offset points')
        first_stop = False

    # ── Yield-lines ───────────────────────────────────────────────────────────
    first_yield = True
    for _, row in gdf_swisstopo.iterrows():
        if not row['Description'].endswith('_Yield'):
            continue
        xs, ys     = [c[0] for c in row.geometry.coords], [c[1] for c in row.geometry.coords]
        xs_m, ys_m = transformer_to_2056.transform(xs, ys)
        lbl = 'Yield-line' if first_yield else '_nolegend_'
        ax.plot(xs_m, ys_m, color='darkorange', linewidth=2.5,
                linestyle='--', zorder=6, solid_capstyle='round', label=lbl)
        mid = len(xs_m) // 2
        ax.annotate(row['Description'], (xs_m[mid], ys_m[mid]),
                    fontsize=7, color='darkorange',
                    xytext=(4, 4), textcoords='offset points')
        first_yield = False

    # ── Bike lane boundaries ──────────────────────────────────────────────────
    first_bl = True
    for _, row in gdf_swisstopo.iterrows():
        if not row['Description'].endswith(DIRECTION_SUFFIXES):
            continue
        xs, ys     = [c[0] for c in row.geometry.coords], [c[1] for c in row.geometry.coords]
        xs_m, ys_m = transformer_to_2056.transform(xs, ys)
        lbl = 'Bike lane boundary' if first_bl else '_nolegend_'
        ax.plot(xs_m, ys_m, color='cyan', linewidth=2.5,
                zorder=5, solid_capstyle='round', label=lbl)
        mid = len(xs_m) // 2
        ax.annotate(row['Description'], (xs_m[mid], ys_m[mid]),
                    fontsize=7, color='teal',
                    xytext=(4, 4), textcoords='offset points')
        first_bl = False

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_elements = [
        # Domains
        Line2D([0], [0], color='dimgray', lw=4, ls='-',
               label='Approach (solid)'),
        Line2D([0], [0], color='dimgray', lw=4, ls='--',
               label='Departure (dashed)'),
        Line2D([0], [0], color='gray',    lw=1.5, ls=':',
               label='Inside intersection (dotted)'),
        # Crossing points
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor='red',       markersize=10,
               label='Stop-line crossing'),
        Line2D([0], [0], marker='D', color='w',
               markerfacecolor='darkorange', markersize=10,
               label='Yield-line crossing'),
        # Drawn geometry
        Line2D([0], [0], color='red',       lw=2.5,
               label='Stop-line (swisstopo)'),
        Line2D([0], [0], color='darkorange', lw=2.5, ls='--',
               label='Yield-line (swisstopo)'),
        Line2D([0], [0], color='cyan',       lw=2.5,
               label='Bike lane boundary'),
        mpatches.Patch(facecolor='yellow', alpha=0.4, edgecolor='gold',
                       label='Intersection box'),
        # Offset convention
        Line2D([0], [0], color='gray', lw=2,
               label='+ offset = forward dir  |  − offset = reverse dir'),
        # Per-street colors
        Line2D([0], [0], color='steelblue',    lw=4, label='Roentgenstr'),
        Line2D([0], [0], color='tomato',       lw=4, label='Zollstr'),
        Line2D([0], [0], color='mediumpurple', lw=4, label='LangstrN'),
        Line2D([0], [0], color='darkorange',   lw=4, label='LangstrS'),
    ]
    ax.legend(handles=legend_elements, fontsize=8,
              loc='upper right', framealpha=0.9, edgecolor='gray')

    ax.set_xlabel('X [EPSG:2056]', fontsize=10)
    ax.set_ylabel('Y [EPSG:2056]', fontsize=10)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    plt.show()
    return


def plot_turn_debug(approach_seg, departure_seg,
                    pts_approach, pts_departure,
                    connector, method,
                    geometry_store, segment_registry,
                    gdf_swisstopo, save_path):
    """
    Debug plot for a single turning movement showing:
    - Approach/departure splines (gray)
    - Sampled boundary points (green/red)
    - Travel direction arrows at p0 and p1
    - Connector curve
    - Intersection polygon
    """
    x_offset = geometry_store['x_offset']
    y_offset = geometry_store['y_offset']
    transformer_to_2056 = Transformer.from_crs(
        "EPSG:4326", "EPSG:2056", always_xy=True
    )
    def to_full(pts_local):
        """Convert local coords → full LV95 for plotting."""
        pts = np.asarray(pts_local)
        return pts + np.array([x_offset, y_offset])

    def kml_to_full(geometry):
        """WGS84 geometry → full LV95 coords."""
        xs, ys = zip(*[(c[0], c[1]) for c in geometry.coords])
        xs_m, ys_m = transformer_to_2056.transform(xs, ys)
        return np.array(xs_m), np.array(ys_m)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title(
        f'Turn Debug: {approach_seg} → {departure_seg}\n'
        f'method={method}  (full LV95 coords)',
        fontsize=10
    )

    # Intersection polygon
    poly_row = gdf_swisstopo[
        gdf_swisstopo['Description'] == 'Intersection_Area'
    ]
    if len(poly_row) > 0:
        xs_m, ys_m = kml_to_full(poly_row.geometry.iloc[0].exterior)
        ax.fill(xs_m, ys_m, alpha=0.12, color='yellow', zorder=1)
        ax.plot(xs_m, ys_m, color='gold', linewidth=2, zorder=2)

    # Full approach and departure splines — gray
    for seg_key in [approach_seg, departure_seg]:
        geom_key        = segment_registry[seg_key]['geometry_key']
        tck, unew, _    = geometry_store[geom_key]['spline']
        x_cl, y_cl      = splev(unew, tck)
        # Add offset back for full LV95 plotting
        ax.plot(x_cl + x_offset, y_cl + y_offset,
                color='lightgray', linewidth=4,
                zorder=2, solid_capstyle='round')

    # Approach points — green, in full LV95
    pts_app_full = to_full(pts_approach)
    ax.plot(pts_app_full[:, 0], pts_app_full[:, 1],
            color='green', linewidth=2, zorder=5,
            label='pts_approach (travel order)')
    ax.scatter(pts_app_full[:, 0], pts_app_full[:, 1],
               color='green', s=30, zorder=6)
    ax.scatter(*pts_app_full[0],  color='green',     s=120,
               marker='o', zorder=7, label='approach[0] (entry)')
    ax.scatter(*pts_app_full[-1], color='darkgreen', s=120,
               marker='s', zorder=7, label='approach[-1] = p0')

    # Travel direction arrow at p0
    p0 = pts_app_full[-1]
    v0 = pts_approach[-1] - pts_approach[-2]   # use local for direction
    v0 = v0 / (np.linalg.norm(v0) + 1e-9)
    theta0 = np.degrees(np.arctan2(v0[1], v0[0]))
    ax.annotate('', xy=p0 + v0 * 8, xytext=p0,
                arrowprops=dict(arrowstyle='->', color='green', lw=2.5))
    ax.text(p0[0], p0[1] + 3, f'p0\nθ={theta0:.1f}°',
            fontsize=8, color='green', ha='center')

    # Departure points — red, in full LV95
    pts_dep_full = to_full(pts_departure)
    ax.plot(pts_dep_full[:, 0], pts_dep_full[:, 1],
            color='red', linewidth=2, zorder=5,
            label='pts_departure (travel order)')
    ax.scatter(pts_dep_full[:, 0], pts_dep_full[:, 1],
               color='red', s=30, zorder=6)
    ax.scatter(*pts_dep_full[0],  color='red',     s=120,
               marker='o', zorder=7, label='departure[0] = p1')
    ax.scatter(*pts_dep_full[-1], color='darkred', s=120,
               marker='s', zorder=7, label='departure[-1] (exit)')

    # Travel direction arrow at p1
    p1 = pts_dep_full[0]
    v1 = pts_departure[1] - pts_departure[0]   # use local for direction
    v1 = v1 / (np.linalg.norm(v1) + 1e-9)
    theta1 = np.degrees(np.arctan2(v1[1], v1[0]))
    ax.annotate('', xy=p1 + v1 * 8, xytext=p1,
                arrowprops=dict(arrowstyle='->', color='red', lw=2.5))
    ax.text(p1[0], p1[1] - 5, f'p1\nθ={theta1:.1f}°',
            fontsize=8, color='red', ha='center')

    # Connector — in full LV95
    if connector is not None and len(connector) > 1:
        conn_full = to_full(connector)
        ax.plot(conn_full[:, 0], conn_full[:, 1],
                color='blue', linewidth=2.5, zorder=6,
                label=f'Connector [{method}]')
        ax.scatter(*conn_full[0],  color='blue', s=80,
                   marker='o', zorder=8)
        ax.scatter(*conn_full[-1], color='blue', s=80,
                   marker='D', zorder=8)

    # Segment info
    for seg_key, color, role in [
        (approach_seg,  'green', 'approach'),
        (departure_seg, 'red',   'departure'),
    ]:
        entry      = segment_registry[seg_key]
        is_forward = entry['is_forward']
        pos_dir    = geometry_store[entry['geometry_key']]['positive_dir']
        ax.text(
            0.02, 0.98 if role == 'approach' else 0.94,
            f'{seg_key}: is_forward={is_forward}, positive_dir={pos_dir}',
            transform=ax.transAxes,
            fontsize=8, color=color, va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )

    ax.set_xlabel('X [EPSG:2056]', fontsize=10)
    ax.set_ylabel('Y [EPSG:2056]', fontsize=10)
    ax.legend(fontsize=8, loc='lower right', framealpha=0.9)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    # plt.savefig(
    #     f'../debugging/turn_debug_{approach_seg}_2_{departure_seg}.png',
    #     dpi=150, bbox_inches='tight'
    # )
    plt.show()
    return 
    

def plot_turn_splines(turn_keys, segment_registry, geometry_store,
                      gdf_swisstopo, offset_m=3.0, turn_offset_m=2.0,
                      save_path=None):
    """
    Plot all turning movement splines with lateral offsets to separate
    overlapping turns visually.

    turn_offset_m : float — lateral offset applied to each turn spline
                    to separate overlapping movements. Each turn gets a
                    unique offset index so they fan out.
    """
    transformer_to_2056 = Transformer.from_crs(
        "EPSG:4326", "EPSG:2056", always_xy=True
    )
    x_offset  = geometry_store['x_offset']
    y_offset  = geometry_store['y_offset']
    skip_keys = {'x_offset', 'y_offset'}

    COLORS = {
        'Roentgenstr': 'steelblue',
        'Zollstr':     'tomato',
        'LangstrN':    'mediumpurple',
        'LangstrS':    'darkorange',
    }
    DIRECTION_SUFFIXES = ('_NB', '_SB', '_EB', '_WB')

    def kml_to_local(geometry):
        xs, ys     = zip(*[(c[0], c[1]) for c in geometry.coords])
        xs_m, ys_m = transformer_to_2056.transform(xs, ys)
        return (np.array(xs_m) - x_offset,
                np.array(ys_m) - y_offset)

    fig, ax = plt.subplots(figsize=(16, 16))
    ax.set_title(
        'A2 Validation — Turning Movement Splines\n'
        'Gray = lane splines  |  Colored = turns (laterally offset for clarity)\n'
        'Solid = clothoid  |  Dashed = Hermite fallback  |  '
        'o = start (stop-line)  D = end (yield-line)',
        fontsize=10
    )

    # ── Intersection polygon ──────────────────────────────────────────────────
    poly_row = gdf_swisstopo[
        gdf_swisstopo['Description'] == 'Intersection_Area'
    ]
    if len(poly_row) > 0:
        xs_loc, ys_loc = kml_to_local(poly_row.geometry.iloc[0].exterior)
        ax.fill(xs_loc, ys_loc, alpha=0.08, color='yellow', zorder=1)
        ax.plot(xs_loc, ys_loc, color='gold', linewidth=2,
                zorder=2, label='Intersection box')

    # ── Lane splines — gray, faint background ────────────────────────────────
    opp_dir = {'EB':'WB','WB':'EB','NB':'SB','SB':'NB'}
    for geom_key, entry in geometry_store.items():
        if geom_key in skip_keys:
            continue
        if entry.get('positive_dir') is None:
            continue
        tck, unew, cum_dist = entry['spline']
        L       = entry['total_length']
        s_stop  = entry['s_stop']
        s_yield = entry['s_yield']
        col     = COLORS.get(geom_key, 'dimgray')

        for is_forward in [True, False]:
            d_off = +offset_m if is_forward else -offset_m
            if is_forward:
                segs = [(0.0, s_stop, '-'), (s_yield, L, '--'),
                        (min(s_stop,s_yield), max(s_stop,s_yield), ':')]
            else:
                segs = [(s_yield, L, '-'), (0.0, s_stop, '--'),
                        (min(s_stop,s_yield), max(s_stop,s_yield), ':')]
            for s_min, s_max, ls in segs:
                if s_max <= s_min + 0.1:
                    continue
                all_s  = np.linspace(s_min, s_max, 80)
                x_seg  = np.zeros_like(all_s)
                y_seg  = np.zeros_like(all_s)
                for i in range(len(all_s)):
                    x_seg[i], y_seg[i] = convert_roadway_to_xy2056_coordinates(
                        all_s[i], d_off, tck, unew, cum_dist
                    )
                ax.plot(x_seg, y_seg, color=col, linewidth=2.5,
                        linestyle=ls, zorder=3, alpha=0.35,
                        solid_capstyle='round')

    # ── Turn splines — laterally offset ──────────────────────────────────────
    # Group turns by approach segment so overlapping turns fan out
    # Each turn gets a unique signed lateral offset index
    from collections import defaultdict
    approach_counts = defaultdict(list)
    for turn_key in turn_keys:
        app_seg = segment_registry[turn_key]['approach_seg']
        approach_counts[app_seg].append(turn_key)

    # Assign offset index per turn within each approach group
    turn_d_offsets = {}
    for app_seg, keys in approach_counts.items():
        n = len(keys)
        # Centre the offsets around 0
        offsets = np.linspace(-(n-1)/2, (n-1)/2, n) * turn_offset_m
        for key, d_off in zip(keys, offsets):
            turn_d_offsets[key] = d_off

    cmap   = plt.cm.tab20
    colors = [cmap(i / max(len(turn_keys), 1))
              for i in range(len(turn_keys))]

    for i, turn_key in enumerate(turn_keys):
        entry            = geometry_store[turn_key]
        tck, unew, cum_dist = entry['spline']
        col              = colors[i]
        method           = entry.get('method', '?')
        ls               = '-' if method == 'clothoid' else '--'
        d_off            = turn_d_offsets[turn_key]
        L                = entry['total_length']

        # Apply lateral offset to turn spline for visual separation
        all_s  = np.linspace(0, L, 200)
        x_seg  = np.zeros_like(all_s)
        y_seg  = np.zeros_like(all_s)
        for j in range(len(all_s)):
            x_seg[j], y_seg[j] = convert_roadway_to_xy2056_coordinates(
                all_s[j], d_off, tck, unew, cum_dist
            )
        ax.plot(x_seg, y_seg, color=col, linewidth=2.5,
                linestyle=ls, zorder=5)

        # Start / end markers at true spline endpoints (no offset)
        x_t, y_t = splev(unew, tck)
        ax.scatter(x_t[0],  y_t[0],  color=col, s=120,
                   marker='o', zorder=7,
                   label='Turn start (stop-line)' if i == 0
                   else '_nolegend_')
        ax.scatter(x_t[-1], y_t[-1], color=col, s=120,
                   marker='D', zorder=7,
                   label='Turn end (yield-line)' if i == 0
                   else '_nolegend_')

        # Label at offset midpoint
        mid   = len(x_seg) // 2
        short = turn_key.replace('turn_', '').replace('_2_', '→')
        ax.annotate(
            f'{short}\n[{method}]',
            (x_seg[mid], y_seg[mid]),
            fontsize=6, color=col, fontweight='bold',
            xytext=(4, 4), textcoords='offset points',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                      alpha=0.85, edgecolor=col)
        )

        # Arrow at midpoint showing travel direction
        if mid > 0:
            dx = x_seg[mid] - x_seg[mid-1]
            dy = y_seg[mid] - y_seg[mid-1]
            ax.annotate('',
                xy=(x_seg[mid] + dx*3, y_seg[mid] + dy*3),
                xytext=(x_seg[mid], y_seg[mid]),
                arrowprops=dict(arrowstyle='->', color=col, lw=1.5)
            )

    # ── Stop-lines ────────────────────────────────────────────────────────────
    first_stop = True
    for _, row in gdf_swisstopo.iterrows():
        if not row['Description'].endswith('_Stop'):
            continue
        xs_loc, ys_loc = kml_to_local(row.geometry)
        lbl = 'Stop-line' if first_stop else '_nolegend_'
        ax.plot(xs_loc, ys_loc, color='red', linewidth=3,
                zorder=6, solid_capstyle='round', label=lbl)
        mid = len(xs_loc) // 2
        ax.annotate(row['Description'], (xs_loc[mid], ys_loc[mid]),
                    fontsize=8, color='darkred', fontweight='bold',
                    xytext=(4, -12), textcoords='offset points')
        first_stop = False

    # ── Yield-lines ───────────────────────────────────────────────────────────
    first_yield = True
    for _, row in gdf_swisstopo.iterrows():
        if not row['Description'].endswith('_Yield'):
            continue
        xs_loc, ys_loc = kml_to_local(row.geometry)
        lbl = 'Yield-line' if first_yield else '_nolegend_'
        ax.plot(xs_loc, ys_loc, color='darkorange', linewidth=3,
                linestyle='--', zorder=6, solid_capstyle='round', label=lbl)
        mid = len(xs_loc) // 2
        ax.annotate(row['Description'], (xs_loc[mid], ys_loc[mid]),
                    fontsize=8, color='darkorange', fontweight='bold',
                    xytext=(4, 6), textcoords='offset points')
        first_yield = False

    # ── Bike lane boundaries ──────────────────────────────────────────────────
    first_bl = True
    for _, row in gdf_swisstopo.iterrows():
        if not row['Description'].endswith(DIRECTION_SUFFIXES):
            continue
        xs_loc, ys_loc = kml_to_local(row.geometry)
        lbl = 'Bike lane boundary' if first_bl else '_nolegend_'
        ax.plot(xs_loc, ys_loc, color='cyan', linewidth=2,
                zorder=4, alpha=0.6, solid_capstyle='round', label=lbl)
        first_bl = False

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_elements = [
        Line2D([0], [0], color='dimgray', lw=2.5, ls='-',  alpha=0.4,
               label='Lane approach'),
        Line2D([0], [0], color='dimgray', lw=2.5, ls='--', alpha=0.4,
               label='Lane departure'),
        Line2D([0], [0], color='gray',    lw=1.5, ls=':', alpha=0.4,
               label='Inside intersection'),
        Line2D([0], [0], color='dimgray', lw=2.5, ls='-',
               label='Turn spline (clothoid)'),
        Line2D([0], [0], color='dimgray', lw=2.5, ls='--',
               label='Turn spline (Hermite)'),
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor='dimgray', markersize=9,
               label='Turn start (stop-line)'),
        Line2D([0], [0], marker='D', color='w',
               markerfacecolor='dimgray', markersize=9,
               label='Turn end (yield-line)'),
        Line2D([0], [0], color='red',        lw=3,
               label='Stop-line (swisstopo)'),
        Line2D([0], [0], color='darkorange', lw=3, ls='--',
               label='Yield-line (swisstopo)'),
        Line2D([0], [0], color='cyan',       lw=2, alpha=0.6,
               label='Bike lane boundary'),
        mpatches.Patch(facecolor='yellow', alpha=0.3, edgecolor='gold',
                       label='Intersection box'),
        Line2D([0], [0], color='steelblue',    lw=3, alpha=0.5,
               label='Roentgenstr'),
        Line2D([0], [0], color='tomato',       lw=3, alpha=0.5,
               label='Zollstr'),
        Line2D([0], [0], color='mediumpurple', lw=3, alpha=0.5,
               label='LangstrN'),
        Line2D([0], [0], color='darkorange',   lw=3, alpha=0.5,
               label='LangstrS'),
    ]
    ax.legend(handles=legend_elements, fontsize=7,
              loc='upper right', framealpha=0.9, edgecolor='gray')

    ax.set_xlabel('X [local, m]', fontsize=10)
    ax.set_ylabel('Y [local, m]', fontsize=10)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    plt.show()
    return


def _chain_from_df(bike_df):
    """
    Reconstruct (chain, movement_key) from a bike_df already processed
    by to_lane_coordinates().

    Returns
    -------
    chain        : list of dicts with seg_key, role, df_indices, score
    movement_key : str
    """
    matched = bike_df['segment_id'].notna()
    if not matched.any():
        return [], 'unmatched'

    movement_key = str(
        bike_df.loc[matched, 'movement_key'].dropna().iloc[0]
    )

    seg_arr  = bike_df['segment_id'].to_numpy()
    role_arr = bike_df['segment_role'].to_numpy()
    n        = len(seg_arr)

    chain = []
    i     = 0
    while i < n:
        val = seg_arr[i]
        if val is None or (isinstance(val, float)):
            i += 1
            continue
        seg_key = val
        role    = role_arr[i]
        j       = i
        while j < n and seg_arr[j] == seg_key:
            j += 1
        chain.append({
            'seg_key':    seg_key,
            'role':       role,
            'df_indices': list(range(i, j)),
            'score':      float('nan'),
        })
        i = j

    return chain, movement_key


def plot_lane_coord_debug(bike_df,
                           segment_registry, geometry_store,
                           XY_2056_Bounds, veh_id,
                           chain=None, movement_key=None,
                           lane_color_map=None,
                           save_path=None):
    """
    Debug plot for lane coordinate transform output.

    Subplot [0,0]: XY map — all lane splines labeled + trajectory
                   colored by matched segment
    Subplot [0,1]: s vs d — lane coordinates
    Subplot [1,0]: Speed profiles — speed_ekf, s_dot, d_dot
    Subplot [1,1]: Acceleration profiles — a, s_ddot, d_ddot

    Parameters
    ----------
    bike_df          : DataFrame — output of to_lane_coordinates()
    chain            : list of dicts from assign_segments(), or None.
                       If None, reconstructed from bike_df columns.
    movement_key     : str or None. If None, taken from bike_df.
    segment_registry : dict
    geometry_store   : dict
    XY_2056_Bounds   : [(x_min, x_max), (y_min, y_max)] in full LV95
    veh_id           : vehicle/bike ID for title
    save_path        : str or None
    """
    # Reconstruct chain/movement_key from bike_df if not provided
    if chain is None or movement_key is None:
        chain_rc, movement_key_rc = _chain_from_df(bike_df)
        if chain is None:
            chain = chain_rc
        if movement_key is None:
            movement_key = movement_key_rc

    x_offset  = geometry_store['x_offset']
    y_offset  = geometry_store['y_offset']
    skip_keys = {'x_offset', 'y_offset'}

    CHAIN_COLORS = [
        'tab:blue', 'tab:orange', 'tab:green',
        'tab:red',  'tab:purple', 'tab:brown',
    ]
    LANE_COLORS = lane_color_map if lane_color_map is not None \
                  else build_lane_color_map(geometry_store)

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    chain_str = (
        ' → '.join(e['seg_key'] for e in chain)
        if chain else 'EMPTY — check matching'
    )
    fig.suptitle(
        f'Lane Coordinate Transformation — veh_id={veh_id}\n'
        f'movement={movement_key}   chain: {chain_str}',
        fontsize=12
    )

    # =========================================================================
    # [0,0] XY MAP
    # =========================================================================
    ax = axs[0, 0]
    ax.set_title('XY Map (full LV95)', fontsize=11)

    # Axis limits from XY_2056_Bounds
    ax.set_xlim(XY_2056_Bounds[0])
    ax.set_ylim(XY_2056_Bounds[1])
    ax.set_aspect('equal')
    ax.set_xlabel('X [LV95, m]', fontsize=10)
    ax.set_ylabel('Y [LV95, m]', fontsize=10)

    # ── All lane splines — gray background, labeled ───────────────────────────
    for geom_key, entry in geometry_store.items():
        if geom_key in skip_keys:
            continue
        if entry.get('positive_dir') is None:
            continue   # skip turn entries
        tck, unew, cum_dist = entry['spline']
        col = LANE_COLORS.get(geom_key, 'dimgray')
        L   = entry['total_length']
        s_stop  = entry['s_stop']
        s_yield = entry['s_yield']

        # Full spline in full LV95 coords
        x_loc, y_loc = splev(unew, tck)
        x_full = x_loc + x_offset
        y_full = y_loc + y_offset
        ax.plot(x_full, y_full, color=col, linewidth=2,
                alpha=0.4, zorder=2)

        # Stop/yield crossing markers
        for s_val, marker, mc in [
            (s_stop,  'o', 'red'),
            (s_yield, 'D', 'darkorange'),
        ]:
            t_val    = float(np.interp(s_val, cum_dist, unew))
            xv, yv   = splev(t_val, tck)
            ax.scatter(xv + x_offset, yv + y_offset,
                       color=mc, s=60, marker=marker,
                       zorder=5)

        # Label at spline midpoint
        t_mid    = float(np.interp(L / 2, cum_dist, unew))
        xm, ym   = splev(t_mid, tck)
        ax.annotate(
            geom_key,
            (xm + x_offset, ym + y_offset),
            fontsize=7, color=col,
            xytext=(4, 4), textcoords='offset points',
            bbox=dict(boxstyle='round,pad=0.15',
                      facecolor='white', alpha=0.7,
                      edgecolor=col)
        )
    
    # ── Bike lane boundaries on XY map ───────────────────────────────────────
    first_bl = True
    for seg_key, entry in segment_registry.items():
        if entry['type'] != 'lane':
            continue
        bike_lane = entry.get('bike_lane')
        if bike_lane is None or 'd_boundary_spline' not in bike_lane:
            continue
    
        geom_key             = entry['geometry_key']
        tck, unew, cum_dist  = geometry_store[geom_key]['spline']
        d_bnd_spl            = bike_lane['d_boundary_spline']
        w_bike               = bike_lane['w_bike']
        side                 = bike_lane['side']
        s_bl_min, s_bl_max   = bike_lane['s_domain']
    
        s_query = np.linspace(s_bl_min, s_bl_max, 150)
        x_bnd, y_bnd = np.zeros_like(s_query), np.zeros_like(s_query)
        x_far, y_far = np.zeros_like(s_query), np.zeros_like(s_query)
    
        for j, s_j in enumerate(s_query):
            d_bnd_j = float(d_bnd_spl(s_j))
            d_far_j = d_bnd_j + side * w_bike
            x_bnd[j], y_bnd[j] = convert_roadway_to_xy2056_coordinates(
                s_j, d_bnd_j, tck, unew, cum_dist
            )
            x_far[j], y_far[j] = convert_roadway_to_xy2056_coordinates(
                s_j, d_far_j, tck, unew, cum_dist
            )
    
        # Convert local → full LV95
        x_bnd += x_offset;  y_bnd += y_offset
        x_far += x_offset;  y_far += y_offset
    
        lbl = 'bike lane' if first_bl else '_nolegend_'
    
        # Boundary line (car/bike boundary)
        axs[0, 0].plot(x_bnd, y_bnd, color='cyan', linewidth=1.5,
                       linestyle='--', zorder=4, alpha=0.9, label=lbl)
    
        # Bike lane band (filled)
        axs[0, 0].fill(
            np.concatenate([x_bnd, x_far[::-1]]),
            np.concatenate([y_bnd, y_far[::-1]]),
            alpha=0.15, color='cyan', zorder=3
        )
    
        first_bl = False

    # ── All turn splines — thin gray ──────────────────────────────────────────
    for seg_key, entry in segment_registry.items():
        if entry['type'] != 'turn':
            continue
        geom_key        = entry['geometry_key']
        tck, unew, _    = geometry_store[geom_key]['spline']
        x_loc, y_loc    = splev(unew, tck)
        ax.plot(x_loc + x_offset, y_loc + y_offset,
                color='lightgray', linewidth=1,
                linestyle=':', zorder=1, alpha=0.6)

    # ── Trajectory — colored by matched segment ───────────────────────────────
    # First plot unmatched points in gray
    all_matched_idx = set()
    for entry in chain:
        all_matched_idx.update(entry['df_indices'])
    unmatched_idx = [i for i in bike_df.index
                     if i not in all_matched_idx]

    if unmatched_idx:
        ax.scatter(
            bike_df.loc[unmatched_idx, 'x_act_ekf'],
            bike_df.loc[unmatched_idx, 'y_act_ekf'],
            color='lightgray', s=8, zorder=3, alpha=0.5,
            label='unmatched'
        )

    # Plot each chain segment in its own color
    for i, seg_entry in enumerate(chain):
        seg_key    = seg_entry['seg_key']
        role       = seg_entry['role']
        df_indices = seg_entry['df_indices']
        col        = CHAIN_COLORS[i % len(CHAIN_COLORS)]

        ax.scatter(
            bike_df.iloc[df_indices]['x_act_ekf'],
            bike_df.iloc[df_indices]['y_act_ekf'],
            color=col, s=12, zorder=4, alpha=0.8,
            label=f'{seg_key} [{role}]'
        )
        # Start marker
        ax.scatter(
            bike_df.iloc[df_indices[0]]['x_act_ekf'],
            bike_df.iloc[df_indices[0]]['y_act_ekf'],
            color=col, s=80, marker='o', zorder=6
        )
        # End marker
        ax.scatter(
            bike_df.iloc[df_indices[-1]]['x_act_ekf'],
            bike_df.iloc[df_indices[-1]]['y_act_ekf'],
            color=col, s=80, marker='s', zorder=6
        )

    # Overall trajectory start/end
    ax.scatter(
        bike_df['x_act_ekf'].iloc[0],
        bike_df['y_act_ekf'].iloc[0],
        color='black', s=100, marker='^',
        zorder=7, label='Traj start'
    )
    ax.scatter(
        bike_df['x_act_ekf'].iloc[-1],
        bike_df['y_act_ekf'].iloc[-1],
        color='black', s=100, marker='v',
        zorder=7, label='Traj end'
    )
    ax.legend(fontsize=8, loc='best', framealpha=0.8)

    # ── Matching diagnostics text ─────────────────────────────────────────────
    diag_lines = [f'chain length: {len(chain)}']
    for e in chain:
        diag_lines.append(
            f"  {e['seg_key']} [{e['role']}]: "
            f"{len(e['df_indices'])} pts  "
            # f"score={e.get('score', float('nan')):.3f}"
        )
    if not chain:
        diag_lines.append('  *** No segments matched ***')
        diag_lines.append(
            f"  traj pts: {len(bike_df)}  "
            f"x∈[{bike_df['x_act_ekf'].min():.1f}, "
            f"{bike_df['x_act_ekf'].max():.1f}]"
        )
        diag_lines.append(
            f"  y∈[{bike_df['y_act_ekf'].min():.1f}, "
            f"{bike_df['y_act_ekf'].max():.1f}]"
        )
    ax.text(
        0.02, 0.02, '\n'.join(diag_lines),
        transform=ax.transAxes,
        fontsize=7, va='bottom', family='monospace',
        bbox=dict(boxstyle='round', facecolor='white',
                  alpha=0.85, edgecolor='gray')
    )

    # =========================================================================
    # [0,1] s vs d  — using cumulative distance traveled (always from 0)
    # =========================================================================
    ax = axs[0, 1]
    ax.set_title('Lane Coordinates: s vs d', fontsize=11)
    ax.set_xlabel('Cumulative distance travelled [m]', fontsize=10)
    ax.set_ylabel('Lateral offset d [m]', fontsize=10)
    ax.axhline(0, color='gray', linewidth=0.8, linestyle='--')
    
    cumulative_s = 0.0
    
    for i, seg_entry in enumerate(chain):
        seg_key    = seg_entry['seg_key']
        role       = seg_entry['role']
        df_indices = seg_entry['df_indices']
        col        = CHAIN_COLORS[i % len(CHAIN_COLORS)]
    
        s_vals    = bike_df.iloc[df_indices]['s'].to_numpy()
        d_vals    = bike_df.iloc[df_indices]['d'].to_numpy()
        in_bl_vals = bike_df.iloc[df_indices]['in_bike_lane'].to_numpy() \
                     if 'in_bike_lane' in bike_df.columns \
                     else np.full(len(s_vals), np.nan)
        valid     = ~(np.isnan(s_vals) | np.isnan(d_vals))
    
        if not valid.any():
            continue
    
        s_valid    = s_vals[valid]
        d_valid    = d_vals[valid]
        in_bl_valid = in_bl_vals[valid]
    
        # Cumulative distance — monotonically increasing
        ds          = np.abs(np.diff(s_valid, prepend=s_valid[0]))
        s_cum_local = np.cumsum(ds)
        s_plot      = s_cum_local + cumulative_s
    
        # Segment separator
        if i > 0:
            ax.axvline(cumulative_s, color='gray', linewidth=0.8,
                       linestyle=':', alpha=0.5)
    
        # ── in_bike_lane highlight — vertical spans ───────────────────────────
        in_bl_bool = (in_bl_valid == True)
        if in_bl_bool.any():
            padded = np.concatenate([[False], in_bl_bool, [False]])
            starts = np.where(~padded[:-1] &  padded[1:])[0]
            ends   = np.where( padded[:-1] & ~padded[1:])[0]
            for k, (s_s, s_e) in enumerate(zip(starts, ends)):
                ax.axvspan(
                    s_plot[s_s],
                    s_plot[min(s_e, len(s_plot)-1)],
                    alpha=0.18, color='lime', zorder=2,
                    label='in bike lane' if (k == 0 and i == 0)
                    else '_nolegend_'
                )
    
        # ── Trajectory line ───────────────────────────────────────────────────
        ax.plot(s_plot, d_valid, color=col, linewidth=1.5,
                label=f'{seg_key} [{role}]', zorder=5)
        ax.scatter(s_plot[0],  d_valid[0],
                   color=col, s=60, marker='o', zorder=6)
        ax.scatter(s_plot[-1], d_valid[-1],
                   color=col, s=60, marker='s', zorder=6)
    
        # Segment label
        mid = len(s_plot) // 2
        ax.annotate(seg_key, (s_plot[mid], d_valid[mid]),
                    fontsize=8, color=col, alpha=0.7,
                    xytext=(0, 8), textcoords='offset points', ha='center')
    
        cumulative_s += float(s_cum_local[-1])
    
    if not chain:
        ax.text(0.5, 0.5, 'No matched segments\n(s, d all NaN)',
                transform=ax.transAxes, ha='center', va='center',
                fontsize=10, color='red')
    
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)
    # =========================================================================
    # [1,0] SPEED PROFILES
    # =========================================================================
    ax = axs[1, 0]
    ax.set_title('Speed Profiles', fontsize=11)
    ax.set_xlabel('Time [s]', fontsize=10)
    ax.set_ylabel('Speed [km/h]', fontsize=10)
    ax.set_ylim([-10, 38])

    time = bike_df['time'].to_numpy() if 'time' in bike_df.columns \
           else np.arange(len(bike_df)) / 20.0

    # Original speed — always available
    ax.plot(time, bike_df['speed_ekf'].to_numpy(),
            color='black', linewidth=1.5, linestyle='--',
            alpha=0.6, label='speed_ekf (total)')

    # Lane coordinate speeds — per segment
    for i, seg_entry in enumerate(chain):
        df_indices = seg_entry['df_indices']
        seg_key    = seg_entry['seg_key']
        col        = CHAIN_COLORS[i % len(CHAIN_COLORS)]
        t_seg      = time[df_indices]

        s_dot = bike_df.iloc[df_indices]['s_dot'].to_numpy()
        d_dot = bike_df.iloc[df_indices]['d_dot'].to_numpy()
        valid = ~(np.isnan(s_dot) | np.isnan(d_dot))

        if valid.any():
            ax.plot(t_seg[valid], s_dot[valid],
                    color=col, linewidth=1.5,
                    label=f's_dot [{seg_key}]')
            ax.plot(t_seg[valid], d_dot[valid],
                    color=col, linewidth=1.5,
                    linestyle=':', alpha=0.7,
                    label=f'd_dot [{seg_key}]')

    ax.axhline(0, color='gray', linewidth=0.5)
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)

    # =========================================================================
    # [1,1] ACCELERATION PROFILES
    # =========================================================================
    ax = axs[1, 1]
    ax.set_title('Acceleration Profiles', fontsize=11)
    ax.set_xlabel('Time [s]', fontsize=10)
    ax.set_ylabel('Acceleration [m/s²]', fontsize=10)

    # Original acceleration — always available
    accel_col = 'a_ekf' if 'a_ekf' in bike_df.columns else 'a'
    if accel_col in bike_df.columns:
        ax.plot(time, bike_df[accel_col].to_numpy(),
                color='black', linewidth=1.5, linestyle='--',
                alpha=0.6, label=f'{accel_col} (total)')

    # Lane coordinate accelerations — per segment
    for i, seg_entry in enumerate(chain):
        df_indices = seg_entry['df_indices']
        seg_key    = seg_entry['seg_key']
        col        = CHAIN_COLORS[i % len(CHAIN_COLORS)]
        t_seg      = time[df_indices]

        s_ddot = bike_df.iloc[df_indices]['s_ddot'].to_numpy()
        d_ddot = bike_df.iloc[df_indices]['d_ddot'].to_numpy()
        valid  = ~(np.isnan(s_ddot) | np.isnan(d_ddot))

        if valid.any():
            ax.plot(t_seg[valid], s_ddot[valid],
                    color=col, linewidth=1.5,
                    label=f's_ddot [{seg_key}]')
            ax.plot(t_seg[valid], d_ddot[valid],
                    color=col, linewidth=1.5,
                    linestyle=':', alpha=0.7,
                    label=f'd_ddot [{seg_key}]')

    ax.axhline(0, color='gray', linewidth=0.5)
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)

    # =========================================================================
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        # print(f"Saved: {save_path}")
        plt.close()
        return fig
    # plt.show()
    return fig

