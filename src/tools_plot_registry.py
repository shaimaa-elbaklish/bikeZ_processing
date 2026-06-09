"""
tools_plot_registry.py
-----------------------
Plotting functions for geometry_store and segment_registry validation.

Two functions:

    plot_geometry_store(geometry_store, gdf_swisstopo, ...)
        Phase 1 validation plot. Shows each road axis as two offset
        lines (forward / reverse direction), with s_change markers,
        stop/yield lines from KML, and bike lane boundaries.

    plot_segment_registry(geometry_store, segment_registry, gdf_swisstopo, ...)
        Phase 2 + 3 validation plot. Shows validity polygons per segment
        and turn splines, colored by road axis.

Both work directly with the new geometry_store schema:
    - No resolve_geometry() calls — all lookups are direct
    - Uses s_change (not approach_native / departure_native)
    - Handles extra_changes (s_change_matt, etc.)
    - Color map is derived automatically from geometry_store keys

Authors : ETH Zürich IVT
"""

# =============================================================================
# IMPORTS
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from collections import defaultdict

from pyproj import Transformer
from scipy.interpolate import splev

# =============================================================================
# HELPERS
# =============================================================================

def _is_axis_entry(key, val):
    """True if geometry_store entry is a spline-based axis dict."""
    if key in ('x_offset', 'y_offset'):
        return False
    if key.startswith('intersection_area') or key.startswith('__'):
        return False
    return isinstance(val, dict)


# =============================================================================
# COLOR PALETTE
# =============================================================================
# Assigned in geometry_store insertion order, skipping x_offset / y_offset.
# Extra entries (turns) get a fallback gray.
_PALETTE = [
    'steelblue', 'tomato', 'mediumpurple', 'darkorange',
    'seagreen',  'crimson', 'goldenrod',   'teal',
    'slategray', 'orchid',  'sienna',      'cornflowerblue',
    'deeppink',  'olive',   'peru',        'dodgerblue',
]

def _build_color_map(geometry_store):
    """
    Return {geom_key: color} for all lane axes (non-turn entries).
    Turn entries (s_stop=None) get 'dimgray'.
    """
    lane_keys = [
        k for k, v in geometry_store.items()
        if _is_axis_entry(k, v) and v.get('s_stop') is not None
    ]
    cmap = {k: _PALETTE[i % len(_PALETTE)] for i, k in enumerate(lane_keys)}
    # Turns
    for k, v in geometry_store.items():
        if _is_axis_entry(k, v) and k not in cmap:
            cmap[k] = 'dimgray'
    return cmap


def _spline_xy(tck, unew, cum_dist, s_start, s_end,
               d_offset=0.0, n=150):
    """
    Evaluate spline at n points between s_start and s_end,
    laterally offset by d_offset metres (left = positive).

    Returns (x, y) arrays in local EPSG:2056 coords.
    d_offset=0 gives the centerline.
    """
    s_vals = np.linspace(s_start, s_end, n)
    t_vals = np.interp(s_vals, cum_dist, unew)

    x_c,  y_c  = splev(t_vals, tck, der=0)
    dx_c, dy_c = splev(t_vals, tck, der=1)

    if d_offset != 0.0:
        tang = np.sqrt(dx_c**2 + dy_c**2)
        tang = np.where(tang > 1e-12, tang, 1.0)
        nx = -dy_c / tang   # left normal
        ny =  dx_c / tang
        x_c = x_c + d_offset * nx
        y_c = y_c + d_offset * ny

    return x_c, y_c


def _kml_to_local(geometry, transformer, x_offset, y_offset):
    """Convert a KML WGS84 geometry to local EPSG:2056 coords."""
    xs = [c[0] for c in geometry.coords]
    ys = [c[1] for c in geometry.coords]
    xs_m, ys_m = transformer.transform(xs, ys)
    return np.array(xs_m) - x_offset, np.array(ys_m) - y_offset


# =============================================================================
# PLOT 1 — geometry_store
# =============================================================================

def plot_geometry_store(geometry_store, gdf_swisstopo,
                        offset_m=3.0, save_path=None):
    """
    Phase 1 validation plot.

    For each road axis in geometry_store:
      - Forward direction: solid line, offset +offset_m to the left
      - Reverse direction: dashed line, offset -offset_m to the right
      - s_change marker: vertical tick on the centerline
      - extra s_change_* markers: labeled differently

    KML overlays:
      - Stop lines   (red, solid)
      - Yield lines  (orange, dashed)
      - Bike lane boundaries (cyan, solid)
      - Intersection area polygon (yellow fill)

    Parameters
    ----------
    geometry_store : dict — from Phase 1
    gdf_swisstopo  : GeoDataFrame — full KML features
    offset_m       : float — lateral separation between forward/reverse [m]
    save_path      : str | None — if given, saves figure and closes
    """
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:2056", always_xy=True)
    x_offset    = geometry_store['x_offset']
    y_offset    = geometry_store['y_offset']
    color_map   = _build_color_map(geometry_store)

    fig, ax = plt.subplots(figsize=(14, 14))
    ax.set_title(
        'Geometry store — Phase 1 validation\n'
        'Solid = forward (positive_dir)  |  Dashed = reverse direction\n'
        '▼ = s_change  ▽ = extra s_change_*',
        fontsize=10
    )

    # ── Intersection area ─────────────────────────────────────────────────────
    for _, row in gdf_swisstopo.iterrows():
        if row['Description'] == 'Intersection_Area':
            xs_loc, ys_loc = _kml_to_local(
                row.geometry.exterior, transformer, x_offset, y_offset
            )
            ax.fill(xs_loc, ys_loc, alpha=0.10, color='yellow', zorder=1)
            ax.plot(xs_loc, ys_loc, color='gold', linewidth=1.5,
                    zorder=2, label='Intersection area')

    # ── Road axis splines ─────────────────────────────────────────────────────
    first_axis = True
    for geom_key, geo in geometry_store.items():
        if not _is_axis_entry(geom_key, geo):
            continue
        if geo.get('s_stop') is None:
            continue   # skip turn entries

        tck, unew, cum_dist = geo['spline']
        L            = geo['total_length']
        s_change     = geo['s_change']
        positive_dir = geo.get('positive_dir')
        col          = color_map[geom_key]

        # Forward direction — solid, offset left
        x_fwd, y_fwd = _spline_xy(tck, unew, cum_dist, 0, L,
                                   d_offset=+offset_m)
        ax.plot(x_fwd, y_fwd, color=col, linewidth=3, linestyle='-',
                zorder=4, solid_capstyle='round',
                label=f'{geom_key} ({positive_dir}↑)')

        # Reverse direction — dashed, offset right
        x_rev, y_rev = _spline_xy(tck, unew, cum_dist, 0, L,
                                   d_offset=-offset_m)
        ax.plot(x_rev, y_rev, color=col, linewidth=2, linestyle='--',
                zorder=4, alpha=0.6, solid_capstyle='round',
                label='_nolegend_')

        # Annotate direction labels at midpoints
        mid = L / 2.0
        t_mid = float(np.interp(mid, cum_dist, unew))
        xm, ym = splev(t_mid, tck)
        ax.annotate(
            geom_key,
            (xm + offset_m * 1.5, ym + offset_m * 1.5),
            fontsize=7, color=col, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                      alpha=0.8, edgecolor=col)
        )

        # s_change marker — primary intersection boundary
        t_sc = float(np.interp(s_change, cum_dist, unew))
        xsc, ysc = splev(t_sc, tck)
        ax.scatter(xsc, ysc, color=col, s=120, marker='v',
                   zorder=7,
                   label='s_change' if first_axis else '_nolegend_')
        ax.annotate(
            f's_change={s_change:.1f}m',
            (xsc, ysc),
            fontsize=6, color=col, alpha=0.8,
            xytext=(6, 6), textcoords='offset points'
        )

        # Extra s_change_* markers — secondary junctions
        for key in [k for k in geo if k.startswith('s_change_')]:
            sc_extra = geo[key]
            t_sc2    = float(np.interp(sc_extra, cum_dist, unew))
            xsc2, ysc2 = splev(t_sc2, tck)
            ax.scatter(xsc2, ysc2, color=col, s=100, marker='v',
                       zorder=7, alpha=0.6,
                       label=key if first_axis else '_nolegend_')
            ax.annotate(
                f'{key}={sc_extra:.1f}m',
                (xsc2, ysc2),
                fontsize=6, color=col, alpha=0.7,
                xytext=(6, -12), textcoords='offset points'
            )

        first_axis = False

    # ── Stop lines ────────────────────────────────────────────────────────────
    first_stop = True
    for _, row in gdf_swisstopo.iterrows():
        if not row['Description'].endswith('_Stop'):
            continue
        xs_loc, ys_loc = _kml_to_local(
            row.geometry, transformer, x_offset, y_offset
        )
        ax.plot(xs_loc, ys_loc, color='red', linewidth=2.5,
                zorder=6, solid_capstyle='round',
                label='Stop line' if first_stop else '_nolegend_')
        mid = len(xs_loc) // 2
        ax.annotate(row['Description'], (xs_loc[mid], ys_loc[mid]),
                    fontsize=6, color='darkred',
                    xytext=(4, -10), textcoords='offset points')
        first_stop = False

    # ── Yield lines ───────────────────────────────────────────────────────────
    first_yield = True
    for _, row in gdf_swisstopo.iterrows():
        if not row['Description'].endswith('_Yield'):
            continue
        xs_loc, ys_loc = _kml_to_local(
            row.geometry, transformer, x_offset, y_offset
        )
        ax.plot(xs_loc, ys_loc, color='darkorange', linewidth=2.5,
                linestyle='--', zorder=6, solid_capstyle='round',
                label='Yield line' if first_yield else '_nolegend_')
        mid = len(xs_loc) // 2
        ax.annotate(row['Description'], (xs_loc[mid], ys_loc[mid]),
                    fontsize=6, color='darkorange',
                    xytext=(4, 4), textcoords='offset points')
        first_yield = False

    # ── Bike lane boundaries ──────────────────────────────────────────────────
    DIRECTION_SUFFIXES = ('_NB', '_SB', '_EB', '_WB', '_NE', '_SW')
    first_bl = True
    for _, row in gdf_swisstopo.iterrows():
        desc = row['Description']
        if not any(desc.endswith(s) for s in DIRECTION_SUFFIXES):
            continue
        if desc.endswith('_Stop') or desc.endswith('_Yield'):
            continue
        xs_loc, ys_loc = _kml_to_local(
            row.geometry, transformer, x_offset, y_offset
        )
        ax.plot(xs_loc, ys_loc, color='cyan', linewidth=2,
                zorder=5, solid_capstyle='round',
                label='Bike lane boundary' if first_bl else '_nolegend_')
        mid = len(xs_loc) // 2
        ax.annotate(desc, (xs_loc[mid], ys_loc[mid]),
                    fontsize=6, color='teal',
                    xytext=(4, 4), textcoords='offset points')
        first_bl = False

    ax.set_xlabel('X local [m]', fontsize=10)
    ax.set_ylabel('Y local [m]', fontsize=10)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc='upper right', framealpha=0.9)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
        plt.close()
    else:
        plt.show()


# =============================================================================
# PLOT 2 — segment_registry
# =============================================================================

def plot_segment_registry(geometry_store, segment_registry, gdf_swisstopo,
                           offset_m=3.0, turn_offset_m=1.5,
                           show_validity_polygons=True,
                           save_path=None):
    """
    Phase 2 + 3 validation plot.

    For each lane segment:
      - Centerline spline (with direction offset) — colored by road axis
      - Validity polygon (semi-transparent fill) — same color
      - s_change tick on the centerline — where handoff occurs

    For each turn segment:
      - Turn spline (colored, laterally fanned out per approach group)
      - Start marker (o) and end marker (D)

    KML overlays same as plot_geometry_store (stop/yield/bike lane/polygon).

    Parameters
    ----------
    geometry_store          : dict
    segment_registry        : dict
    gdf_swisstopo           : GeoDataFrame
    offset_m                : float — lane direction separation [m]
    turn_offset_m           : float — fan separation between turns [m]
    show_validity_polygons  : bool — draw validity polygon fills
    save_path               : str | None
    """
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:2056", always_xy=True)
    x_offset    = geometry_store['x_offset']
    y_offset    = geometry_store['y_offset']
    color_map   = _build_color_map(geometry_store)

    fig, ax = plt.subplots(figsize=(16, 16))
    ax.set_title(
        'Segment registry — Phase 2/3 validation\n'
        'Filled = validity polygon  |  Solid = forward  |  Dashed = reverse\n'
        '▼ = s_change  o = turn start  D = turn end',
        fontsize=10
    )

    # ── Intersection area ─────────────────────────────────────────────────────
    for _, row in gdf_swisstopo.iterrows():
        if row['Description'] == 'Intersection_Area':
            xs_loc, ys_loc = _kml_to_local(
                row.geometry.exterior, transformer, x_offset, y_offset
            )
            ax.fill(xs_loc, ys_loc, alpha=0.08, color='yellow', zorder=1)
            ax.plot(xs_loc, ys_loc, color='gold', linewidth=1.5,
                    zorder=2, label='Intersection area')

    # ── Lane segments ─────────────────────────────────────────────────────────
    plotted_geom_keys = set()

    for seg_key, entry in segment_registry.items():
        if entry['type'] != 'lane':
            continue

        geom_key   = entry['geometry_key']
        is_forward = entry['is_forward']
        d_left     = entry['d_left']
        d_right    = entry['d_right']
        col        = color_map.get(geom_key, 'dimgray')

        geo              = geometry_store[geom_key]
        tck, unew, cum_dist = geo['spline']
        L                = geo['total_length']
        s_change         = geo['s_change']

        d_offset = +offset_m if is_forward else -offset_m
        ls       = '-'       if is_forward else '--'
        lw       = 3.0       if is_forward else 2.0

        # Centerline with direction offset
        x_cl, y_cl = _spline_xy(tck, unew, cum_dist, 0, L,
                                 d_offset=d_offset)
        label = f'{seg_key}' if seg_key not in plotted_geom_keys else '_nolegend_'
        ax.plot(x_cl, y_cl, color=col, linewidth=lw, linestyle=ls,
                zorder=5, solid_capstyle='round', label=label)

        # Segment key label at midpoint
        mid_t = float(np.interp(L / 2, cum_dist, unew))
        xm, ym = splev(mid_t, tck)
        ax.annotate(
            seg_key,
            (xm + d_offset * 0.8, ym + d_offset * 0.8),
            fontsize=6, color=col, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                      alpha=0.75, edgecolor=col)
        )

        # s_change tick — primary handoff boundary
        t_sc = float(np.interp(s_change, cum_dist, unew))
        xsc, ysc = splev(t_sc, tck)
        ax.scatter(xsc, ysc, color=col, s=80, marker='v', zorder=8,
                   label='s_change' if len(plotted_geom_keys) == 0
                   else '_nolegend_')

        # Extra s_change_* ticks
        for key in [k for k in geo if k.startswith('s_change_')]:
            sc_extra = geo[key]
            t_sc2    = float(np.interp(sc_extra, cum_dist, unew))
            xsc2, ysc2 = splev(t_sc2, tck)
            ax.scatter(xsc2, ysc2, color=col, s=60, marker='v',
                       zorder=8, alpha=0.6,
                       label=key if len(plotted_geom_keys) == 0
                       else '_nolegend_')

        # Validity polygon
        if show_validity_polygons:
            poly = entry.get('validity_polygon')
            if poly is not None and not poly.is_empty:
                try:
                    px, py = poly.exterior.xy
                    ax.fill(px, py, alpha=0.10, color=col, zorder=3)
                    ax.plot(px, py, color=col, linewidth=0.8,
                            linestyle=':', zorder=3, alpha=0.5)
                except Exception:
                    pass   # MultiPolygon edge case — skip

        plotted_geom_keys.add(seg_key)

    # ── Turn segments ─────────────────────────────────────────────────────────
    turn_entries = {k: v for k, v in segment_registry.items()
                    if v['type'] == 'turn'}

    # Fan turns laterally, grouped by approach segment
    approach_groups = defaultdict(list)
    for tk, te in turn_entries.items():
        approach_groups[te['approach_seg']].append(tk)

    turn_d_offsets = {}
    for app_seg, keys in approach_groups.items():
        n = len(keys)
        offsets = np.linspace(-(n - 1) / 2, (n - 1) / 2, n) * turn_offset_m
        for tk, d_off in zip(keys, offsets):
            turn_d_offsets[tk] = d_off

    cmap_turn  = plt.cm.tab20
    turn_list  = list(turn_entries.keys())
    turn_colors = {tk: cmap_turn(i / max(len(turn_list), 1))
                   for i, tk in enumerate(turn_list)}

    for turn_key, te in turn_entries.items():
        geom     = geometry_store[turn_key]
        tck_t, unew_t, cum_t = geom['spline']
        L_t      = geom['total_length']
        col_t    = turn_colors[turn_key]
        method   = geom.get('method', '?')
        ls_t     = '-' if method == 'clothoid' else '--'
        d_off    = turn_d_offsets.get(turn_key, 0.0)

        x_t, y_t = _spline_xy(tck_t, unew_t, cum_t, 0, L_t,
                               d_offset=d_off)
        ax.plot(x_t, y_t, color=col_t, linewidth=2.5, linestyle=ls_t,
                zorder=6, label=f'{turn_key} [{method}]')

        # Start / end markers at true spline endpoints
        xs_t, ys_t = splev(unew_t[[0, -1]], tck_t)
        ax.scatter(xs_t[0], ys_t[0], color=col_t, s=100,
                   marker='o', zorder=9)
        ax.scatter(xs_t[1], ys_t[1], color=col_t, s=100,
                   marker='D', zorder=9)

        # Validity polygon
        if show_validity_polygons:
            poly = te.get('validity_polygon')
            if poly is not None and not poly.is_empty:
                try:
                    px, py = poly.exterior.xy
                    ax.fill(px, py, alpha=0.06, color=col_t, zorder=2)
                    ax.plot(px, py, color=col_t, linewidth=0.6,
                            linestyle=':', zorder=2, alpha=0.4)
                except Exception:
                    pass

    # ── Stop / yield / bike lane overlays ─────────────────────────────────────
    DIRECTION_SUFFIXES = ('_NB', '_SB', '_EB', '_WB', '_NE', '_SW')
    first_stop = first_yield = first_bl = True

    for _, row in gdf_swisstopo.iterrows():
        desc = row['Description']
        if desc == 'Intersection_Area':
            continue

        if desc.endswith('_Stop'):
            xs_loc, ys_loc = _kml_to_local(
                row.geometry, transformer, x_offset, y_offset
            )
            ax.plot(xs_loc, ys_loc, color='red', linewidth=2,
                    zorder=7, solid_capstyle='round',
                    label='Stop line' if first_stop else '_nolegend_')
            mid = len(xs_loc) // 2
            ax.annotate(desc, (xs_loc[mid], ys_loc[mid]),
                        fontsize=6, color='darkred',
                        xytext=(3, -9), textcoords='offset points')
            first_stop = False

        elif desc.endswith('_Yield'):
            xs_loc, ys_loc = _kml_to_local(
                row.geometry, transformer, x_offset, y_offset
            )
            ax.plot(xs_loc, ys_loc, color='darkorange', linewidth=2,
                    linestyle='--', zorder=7, solid_capstyle='round',
                    label='Yield line' if first_yield else '_nolegend_')
            mid = len(xs_loc) // 2
            ax.annotate(desc, (xs_loc[mid], ys_loc[mid]),
                        fontsize=6, color='darkorange',
                        xytext=(3, 4), textcoords='offset points')
            first_yield = False

        elif any(desc.endswith(s) for s in DIRECTION_SUFFIXES):
            xs_loc, ys_loc = _kml_to_local(
                row.geometry, transformer, x_offset, y_offset
            )
            ax.plot(xs_loc, ys_loc, color='cyan', linewidth=1.8,
                    zorder=5, solid_capstyle='round',
                    label='Bike lane boundary' if first_bl else '_nolegend_')
            mid = len(xs_loc) // 2
            ax.annotate(desc, (xs_loc[mid], ys_loc[mid]),
                        fontsize=6, color='teal',
                        xytext=(3, 3), textcoords='offset points')
            first_bl = False

    # ── Legend summary ────────────────────────────────────────────────────────
    legend_extra = [
        Line2D([0], [0], color='dimgray', lw=3, ls='-',
               label='Lane — forward direction'),
        Line2D([0], [0], color='dimgray', lw=2, ls='--',
               label='Lane — reverse direction'),
        mpatches.Patch(facecolor='dimgray', alpha=0.15,
                       label='Validity polygon'),
        Line2D([0], [0], color='dimgray', lw=2, ls='-',
               marker='v', markersize=8, label='s_change boundary'),
        Line2D([0], [0], color='dimgray', lw=2.5, ls='-',
               label='Turn spline (clothoid)'),
        Line2D([0], [0], color='dimgray', lw=2.5, ls='--',
               label='Turn spline (Hermite)'),
        Line2D([0], [0], color='w', marker='o',
               markerfacecolor='dimgray', markersize=9,
               label='Turn start'),
        Line2D([0], [0], color='w', marker='D',
               markerfacecolor='dimgray', markersize=9,
               label='Turn end'),
    ]

    handles, labels = ax.get_legend_handles_labels()
    # Deduplicate labels
    seen   = set()
    unique = [(h, l) for h, l in zip(handles, labels)
              if l not in seen and not seen.add(l)
              and not l.startswith('_')]
    u_h, u_l = zip(*unique) if unique else ([], [])

    ax.legend(
        list(u_h) + legend_extra,
        list(u_l) + [e.get_label() for e in legend_extra],
        fontsize=7, loc='upper right',
        framealpha=0.9, ncol=2
    )

    ax.set_xlabel('X local [m]', fontsize=10)
    ax.set_ylabel('Y local [m]', fontsize=10)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
        plt.close()
    else:
        plt.show()