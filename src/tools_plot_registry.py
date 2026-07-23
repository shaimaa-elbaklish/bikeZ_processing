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

from tools_utils import _is_axis_entry, _spline_xy, _build_color_map

# =============================================================================
# HELPERS
# =============================================================================
def _kml_to_local(geometry, transformer, x_offset, y_offset):
    """Convert a KML WGS84 geometry to local EPSG:2056 coords."""
    xs = [c[0] for c in geometry.coords]
    ys = [c[1] for c in geometry.coords]
    xs_m, ys_m = transformer.transform(xs, ys)
    return np.array(xs_m) - x_offset, np.array(ys_m) - y_offset


# =============================================================================
# PLOT 1 — geometry_store
# =============================================================================

def plot_geometry_store(geometry_store, gdf_swisstopo=None,
                        offset_m=3.0, figuresize=(14, 14), save_path=None):
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

    fig, ax = plt.subplots(figsize=figuresize)
    ax.set_title(
        'Geometry store — Phase 1\n'
        'Solid = forward (positive_dir)  |  Dashed = reverse direction\n'
        '▼ = s_change  ▽ = extra s_change_*',
        fontsize=10
    )

    # ── Intersection area ─────────────────────────────────────────────────────
    if gdf_swisstopo is not None:
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
    if gdf_swisstopo is not None:
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
    if gdf_swisstopo is not None:
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
    if gdf_swisstopo is not None:
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

def plot_segment_registry(geometry_store, segment_registry, gdf_swisstopo=None,
                           offset_m=3.0, turn_offset_m=1.5,
                           show_validity_polygons=True, figuresize=(16, 16),
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

    fig, ax = plt.subplots(figsize=figuresize)
    ax.set_title(
        'Segment registry — Phase 2/3\n'
        'Filled = validity polygon  |  Solid = forward  |  Dashed = reverse\n'
        '▼ = s_change  o = turn start  D = turn end',
        fontsize=10
    )

    # ── Intersection area ─────────────────────────────────────────────────────
    if gdf_swisstopo is not None:
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
    if gdf_swisstopo is not None:
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
        
        
# =============================================================================
# PLOT 2 — segment_registry USING PLOTLY
# =============================================================================
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import plotly.graph_objects as go


def _to_rgba_str(color, alpha=1.0):
    """Convert any matplotlib-compatible color (name, hex, or RGBA tuple)
    into a Plotly-compatible 'rgba(r,g,b,a)' string."""
    r, g, b, a = mcolors.to_rgba(color, alpha=alpha)
    return f'rgba({int(r * 255)},{int(g * 255)},{int(b * 255)},{a:.3f})'
 
 
def plot_segment_registry_plotly(geometry_store, segment_registry,
                                  offset_m=3.0, turn_offset_m=1.5,
                                  show_validity_polygons=True,
                                  width=1000, height=1000):
    """
    Phase 2 + 3 validation plot — Plotly version.
 
    Each lane segment and each turn segment gets its own `legendgroup`.
    Clicking that segment's legend entry toggles its centerline (or turn
    spline), validity polygon, s_change ticks, label, and start/end
    markers together — thanks to `legend.groupclick = 'togglegroup'`.
 
    Stop lines, yield lines, and bike-lane boundaries are each their own
    toggleable group (one legend entry per category, not per line).
 
    A block of inert "style guide" legend entries at the end reproduces
    the reference key from the matplotlib version (line styles, marker
    meanings) — these carry no data and toggling them does nothing.
 
    Parameters
    ----------
    geometry_store, segment_registry : same as before
    offset_m, turn_offset_m, show_validity_polygons  : same as before
    width, height    : figure size in pixels
 
    Returns
    -------
    fig : plotly.graph_objects.Figure
    """
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:2056", always_xy=True)
    x_offset    = geometry_store['x_offset']
    y_offset    = geometry_store['y_offset']
    color_map   = _build_color_map(geometry_store)
 
    fig = go.Figure()
 
    # ── Lane segments ────────────────────────────────────────────────────
    for seg_key, entry in segment_registry.items():
        if entry['type'] != 'lane':
            continue
 
        geom_key   = entry['geometry_key']
        is_forward = entry['is_forward']
        col        = color_map.get(geom_key, 'dimgray')
 
        geo                  = geometry_store[geom_key]
        tck, unew, cum_dist  = geo['spline']
        L                    = geo['total_length']
        s_change             = geo['s_change']
 
        d_offset = +offset_m if is_forward else -offset_m
        dash     = 'solid'   if is_forward else 'dash'
        lw       = 3.0       if is_forward else 2.0
 
        legend_group = seg_key
 
        # Centerline with direction offset — this is the toggle handle
        x_cl, y_cl = _spline_xy(tck, unew, cum_dist, 0, L, d_offset=d_offset)
        fig.add_trace(go.Scatter(
            x=x_cl, y=y_cl, mode='lines',
            line=dict(color=col, width=lw, dash=dash),
            name=seg_key,
            legendgroup=legend_group,
            showlegend=True,
            hovertemplate=f'{seg_key}<extra></extra>',
        ))
 
        # Segment key label at midpoint (grouped, no own legend entry)
        mid_t = float(np.interp(L / 2, cum_dist, unew))
        xm, ym = splev(mid_t, tck)
        fig.add_trace(go.Scatter(
            x=[xm + d_offset * 0.8], y=[ym + d_offset * 0.8],
            mode='text',
            text=[seg_key],
            textfont=dict(size=9, color=col),
            legendgroup=legend_group,
            showlegend=False,
            hoverinfo='skip',
        ))
 
        # s_change tick — primary handoff boundary (grouped, no own entry)
        t_sc = float(np.interp(s_change, cum_dist, unew))
        xsc, ysc = splev(t_sc, tck)
        fig.add_trace(go.Scatter(
            x=[xsc], y=[ysc], mode='markers',
            marker=dict(color=col, size=11, symbol='triangle-down'),
            legendgroup=legend_group,
            showlegend=False,
            hovertemplate=f'{seg_key} s_change<extra></extra>',
        ))
 
        # Extra s_change_* ticks
        for key in [k for k in geo if k.startswith('s_change_')]:
            sc_extra = geo[key]
            t_sc2    = float(np.interp(sc_extra, cum_dist, unew))
            xsc2, ysc2 = splev(t_sc2, tck)
            fig.add_trace(go.Scatter(
                x=[xsc2], y=[ysc2], mode='markers',
                marker=dict(color=col, size=9, symbol='triangle-down',
                            opacity=0.6),
                legendgroup=legend_group,
                showlegend=False,
                hovertemplate=f'{seg_key} {key}<extra></extra>',
            ))
 
        # Validity polygon
        if show_validity_polygons:
            poly = entry.get('validity_polygon')
            if poly is not None and not poly.is_empty:
                try:
                    px, py = poly.exterior.xy
                    fig.add_trace(go.Scatter(
                        x=list(px), y=list(py), mode='lines',
                        fill='toself',
                        fillcolor=_to_rgba_str(col, 0.10),
                        line=dict(color=col, width=0.8, dash='dot'),
                        opacity=0.5,
                        legendgroup=legend_group,
                        showlegend=False,
                        hoverinfo='skip',
                    ))
                except Exception:
                    pass  # MultiPolygon edge case — skip
 
    # ── Turn segments ────────────────────────────────────────────────────
    turn_entries = {k: v for k, v in segment_registry.items()
                    if v['type'] == 'turn'}
 
    approach_groups = defaultdict(list)
    for tk, te in turn_entries.items():
        approach_groups[te['approach_seg']].append(tk)
 
    turn_d_offsets = {}
    for app_seg, keys in approach_groups.items():
        n = len(keys)
        offsets = np.linspace(-(n - 1) / 2, (n - 1) / 2, n) * turn_offset_m
        for tk, d_off in zip(keys, offsets):
            turn_d_offsets[tk] = d_off
 
    try:
        cmap_turn = mpl.colormaps['tab20']
    except AttributeError:
        cmap_turn = cm.get_cmap('tab20')  # older matplotlib fallback
 
    turn_list = list(turn_entries.keys())
    turn_colors = {tk: cmap_turn(i / max(len(turn_list), 1))
                   for i, tk in enumerate(turn_list)}
 
    for turn_key, te in turn_entries.items():
        geom     = geometry_store[turn_key]
        tck_t, unew_t, cum_t = geom['spline']
        L_t      = geom['total_length']
        col_t    = _to_rgba_str(turn_colors[turn_key])
        method   = geom.get('method', '?')
        dash_t   = 'solid' if method == 'clothoid' else 'dash'
        d_off    = turn_d_offsets.get(turn_key, 0.0)
 
        legend_group = turn_key
 
        x_t, y_t = _spline_xy(tck_t, unew_t, cum_t, 0, L_t, d_offset=d_off)
        fig.add_trace(go.Scatter(
            x=x_t, y=y_t, mode='lines',
            line=dict(color=col_t, width=2.5, dash=dash_t),
            name=f'{turn_key} [{method}]',
            legendgroup=legend_group,
            showlegend=True,
            hovertemplate=f'{turn_key} [{method}]<extra></extra>',
        ))
 
        # Start / end markers at true spline endpoints
        xs_t, ys_t = splev(unew_t[[0, -1]], tck_t)
        fig.add_trace(go.Scatter(
            x=[xs_t[0]], y=[ys_t[0]], mode='markers',
            marker=dict(color=col_t, size=11, symbol='circle'),
            legendgroup=legend_group,
            showlegend=False,
            hovertemplate=f'{turn_key} start<extra></extra>',
        ))
        fig.add_trace(go.Scatter(
            x=[xs_t[1]], y=[ys_t[1]], mode='markers',
            marker=dict(color=col_t, size=11, symbol='diamond'),
            legendgroup=legend_group,
            showlegend=False,
            hovertemplate=f'{turn_key} end<extra></extra>',
        ))
 
        # Validity polygon
        if show_validity_polygons:
            poly = te.get('validity_polygon')
            if poly is not None and not poly.is_empty:
                try:
                    px, py = poly.exterior.xy
                    fig.add_trace(go.Scatter(
                        x=list(px), y=list(py), mode='lines',
                        fill='toself',
                        fillcolor=_to_rgba_str(turn_colors[turn_key], 0.06),
                        line=dict(color=col_t, width=0.6, dash='dot'),
                        opacity=0.4,
                        legendgroup=legend_group,
                        showlegend=False,
                        hoverinfo='skip',
                    ))
                except Exception:
                    pass
    
    # Add bike lanes
    from tools_utils import add_bike_lane_boundaries_plotly
    
    add_bike_lane_boundaries_plotly(fig, geometry_store, segment_registry)
 
    # ── Style-guide legend entries (static reference, not toggleable data) ─
    style_guide = [
        # dict(name='Lane — forward direction', mode='lines',
        #      line=dict(color='dimgray', width=3, dash='solid')),
        # dict(name='Lane — reverse direction', mode='lines',
        #      line=dict(color='dimgray', width=2, dash='dash')),
        dict(name='Validity polygon', mode='markers',
             marker=dict(color='dimgray', size=14, symbol='square', opacity=0.15)),
        dict(name='s_change boundary', mode='markers',
             marker=dict(color='dimgray', size=10, symbol='triangle-down')),
        # dict(name='Turn spline (clothoid)', mode='lines',
        #      line=dict(color='dimgray', width=2.5, dash='solid')),
        # dict(name='Turn spline (Hermite)', mode='lines',
        #      line=dict(color='dimgray', width=2.5, dash='dash')),
        dict(name='Turn start', mode='markers',
             marker=dict(color='dimgray', size=10, symbol='circle')),
        dict(name='Turn end', mode='markers',
             marker=dict(color='dimgray', size=10, symbol='diamond')),
        dict(name='Bike lane band', mode='markers',
         marker=dict(color='mediumseagreen', size=14, symbol='square', opacity=0.2)),
    ]
    for i, sg in enumerate(style_guide):
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode=sg['mode'],
            line=sg.get('line'),
            marker=sg.get('marker'),
            name=sg['name'],
            legendgroup=f'style_guide_{i}',
            showlegend=True,
            hoverinfo='skip',
        ))
 
    # ── Layout ───────────────────────────────────────────────────────────
    fig.update_layout(
        title=dict(
            text=(
                'Segment registry — Phase 2/3<br>'
                '<sup>Click a legend entry to toggle that segment\'s '
                'centerline, validity polygon, and s_change ticks together'
                '</sup>'
            ),
            font=dict(size=14),
        ),
        xaxis_title='X local [m]',
        yaxis_title='Y local [m]',
        width=width,
        height=height,
        legend=dict(
            groupclick='togglegroup',  # click one item -> toggles whole group
            font=dict(size=9),
            itemsizing='constant',
        ),
        template='plotly_white',
        hovermode='closest',
    )
    # Equal aspect ratio, like ax.set_aspect('equal')
    fig.update_yaxes(scaleanchor='x', scaleratio=1)
    fig.update_xaxes(showgrid=True, gridcolor='rgba(0,0,0,0.1)')
    fig.update_yaxes(showgrid=True, gridcolor='rgba(0,0,0,0.1)')
 
    return fig


