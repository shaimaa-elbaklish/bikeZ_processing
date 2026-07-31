"""
tools_utils.py
-----------------------
Some miscellaneous tools for plotting, etc.

Authors : ETH Zürich IVT
"""

# =============================================================================
# IMPORTS
# =============================================================================
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import plotly.graph_objects as go

from collections import defaultdict
from scipy.interpolate import splev

# =============================================================================
# HELPERS
# =============================================================================

# def _is_axis_entry(geom_key, geo):
#     """Road-axis entries (as opposed to turn splines) in geometry_store.
#     # ASSUMPTION: axis entries carry an 's_change' key; turn entries have
#     # 's_stop' set to None (per your snippet). Adjust if your convention
#     # differs, e.g. geo.get('kind') == 'axis'."""
#     return geo.get('s_change') is not None

def _is_axis_entry(key, val):
    """True if geometry_store entry is a spline-based axis dict."""
    if key in ('x_offset', 'y_offset'):
        return False
    if key.startswith('intersection_area') or key.startswith('__'):
        return False
    return isinstance(val, dict)


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


def _spline_xy_variable_offset(tck, unew, cum_dist, s_vals, d_vals):
    """
    Like _spline_xy, but accepts an array of per-point lateral offsets
    (d_vals) instead of a single scalar d_offset — needed when the offset
    itself varies along s, e.g. a bike lane boundary spline.

    s_vals, d_vals : arrays of the same length
    Returns (x, y) arrays in local EPSG:2056 coords.
    """
    t_vals = np.interp(s_vals, cum_dist, unew)
    x_c,  y_c  = splev(t_vals, tck, der=0)
    dx_c, dy_c = splev(t_vals, tck, der=1)
    tang = np.sqrt(dx_c**2 + dy_c**2)
    tang = np.where(tang > 1e-12, tang, 1.0)
    nx = -dy_c / tang   # left normal
    ny =  dx_c / tang
    x_c = x_c + d_vals * nx
    y_c = y_c + d_vals * ny
    return x_c, y_c

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


def _to_rgba_str(color, alpha=1.0):
    """Convert any matplotlib-compatible color (name, hex, or RGBA tuple)
    into a Plotly-compatible 'rgba(r,g,b,a)' string."""
    r, g, b, a = mcolors.to_rgba(color, alpha=alpha)
    return f'rgba({int(r * 255)},{int(g * 255)},{int(b * 255)},{a:.3f})'

# =============================================================================
# SITE GEOMETRY LAYER
# =============================================================================
def add_road_axes_plotly(fig, geometry_store, color_map=None, offset_m=1.5,
                         alpha=0.4):
    """Adds road centerline splines (fwd solid + rev dashed) to fig."""
    if color_map is None:
        color_map = _build_color_map(geometry_store)

    for geom_key, geo in geometry_store.items():
        if not _is_axis_entry(geom_key, geo):
            continue
        if geo.get('s_stop') is None:
            continue  # skip turn entries stored alongside axes

        tck, unew, cum_dist = geo['spline']
        L = geo['total_length']
        s_change = geo['s_change']
        positive_dir = geo.get('positive_dir', '')
        col = color_map[geom_key]

        # Forward direction — solid, offset left
        x_fwd, y_fwd = _spline_xy(tck, unew, cum_dist, 0, L, d_offset=+offset_m)
        fig.add_trace(go.Scatter(
            x=x_fwd, y=y_fwd, mode='lines',
            line=dict(color=col, width=2, dash='dash'),
            opacity=alpha,
            name=f'{geom_key} ({positive_dir}\u2191)',
            legendgroup=f'axis-{geom_key}',
            hovertemplate=f'{geom_key} fwd<extra></extra>',
        ))

        # s_change marker(s)
        for key in [k for k in geo if k == 's_change' or k.startswith('s_change_')]:
            sc = geo[key]
            t_sc = float(np.interp(sc, cum_dist, unew))
            xsc, ysc = splev(t_sc, tck)
            fig.add_trace(go.Scatter(
                x=[xsc], y=[ysc], mode='markers',
                marker=dict(color=col, size=10, symbol='triangle-down'),
                opacity=alpha,
                name=f'{geom_key} {key}',
                legendgroup=f'axis-{geom_key}',
                showlegend=False,
                hovertemplate=f'{geom_key} {key}={sc:.1f}m<extra></extra>',
            ))
    return fig, color_map


def add_bike_lane_boundaries_plotly(fig, geometry_store, segment_registry,
                                     n_pts=50, color='mediumseagreen',
                                     alpha=1, legend_group='bike_lanes'):
    """
    Adds bike lane boundary bands to an existing Plotly figure. One shared
    legend entry ('Bike lanes') toggles all bike lane bands/boundaries at
    once — per-segment traces stay grouped under it but don't get their
    own legend entries.

    Parameters
    ----------
    fig : go.Figure — figure to add traces to (mutated in place)
    geometry_store, segment_registry : same as plot_segment_registry_plotly
    n_pts        : samples along each bike lane's s_domain
    color        : fill/line color for all bike lane bands
    legend_group : shared legendgroup key

    Returns
    -------
    fig : same figure, for chaining
    """
    first_trace = True
    fillcolor =  _to_rgba_str(color, 0.20)
    if alpha < 1:
        color = _to_rgba_str(color, alpha)

    for seg_key, entry in segment_registry.items():
        if entry['type'] != 'lane':
            continue

        bike_lane = entry.get('bike_lane')
        if bike_lane is None or 'd_boundary_spline' not in bike_lane:
            continue

        geom_key = entry['geometry_key']
        geo = geometry_store[geom_key]
        tck, unew, cum_dist = geo['spline']

        d_bnd_spl    = bike_lane['d_boundary_spline']
        w_bike       = bike_lane['w_bike']
        side         = bike_lane['side']
        s_min, s_max = bike_lane['s_domain']

        s_bl  = np.linspace(s_min, s_max, n_pts)
        d_bnd = d_bnd_spl(s_bl)
        d_far = d_bnd + side * w_bike

        x_bnd, y_bnd = _spline_xy_variable_offset(tck, unew, cum_dist, s_bl, d_bnd)
        x_far, y_far = _spline_xy_variable_offset(tck, unew, cum_dist, s_bl, d_far)

        # Filled band between near and far boundary
        x_band = np.concatenate([x_bnd, x_far[::-1], x_bnd[:1]])
        y_band = np.concatenate([y_bnd, y_far[::-1], y_bnd[:1]])

        fig.add_trace(go.Scatter(
            x=x_band, y=y_band, mode='lines',
            fill='toself',
            fillcolor=fillcolor,
            line=dict(color=color, width=1.0),
            name='Bike lanes',
            legendgroup=legend_group,
            showlegend=first_trace,   # only the first trace shows in legend
            hoveron='points',
            hovertemplate=f'{seg_key} bike lane (w={w_bike:.2f} m)<extra></extra>',
        ))

        # Near boundary line (crisper edge on top of the fill)
        fig.add_trace(go.Scatter(
            x=x_bnd, y=y_bnd, mode='lines',
            line=dict(color=color, width=1.5, dash='dot'),
            legendgroup=legend_group,
            showlegend=False,
            hoverinfo='skip',
        ))

        first_trace = False

    return fig


def add_turn_centerline_plotly(fig, geometry_store, segment_registry,
                                turn_keys=None, turn_offset_m=1.0, alpha=0.4,
                                show_validity_polygons=False):
    """Adds turn centerline(s) to fig. If turn_keys is given, only those
    turns are drawn; otherwise all turns in segment_registry are drawn."""
    turn_entries = {k: v for k, v in segment_registry.items()
                    if v['type'] == 'turn'}
    if turn_keys is not None:
        turn_entries = {k: v for k, v in turn_entries.items() if k in turn_keys}

    approach_groups = defaultdict(list)
    for tk, te in turn_entries.items():
        approach_groups[te['approach_seg']].append(tk)

    turn_d_offsets = {}
    for app_seg, keys in approach_groups.items():
        n = len(keys)
        offsets = np.linspace(-(n - 1) / 2, (n - 1) / 2, n) * turn_offset_m
        for tk, d_off in zip(keys, offsets):
            turn_d_offsets[tk] = d_off

    palette = ['#e377c2', '#7f7f7f', '#bcbd22', '#d62728', '#17becf']
    turn_colors = {tk: palette[i % len(palette)]
                   for i, tk in enumerate(turn_entries.keys())}

    for turn_key, te in turn_entries.items():
        geom = geometry_store[turn_key]
        tck_t, unew_t, cum_t = geom['spline']
        L_t = geom['total_length']
        col_t = turn_colors[turn_key]
        method = geom.get('method', '?')
        d_off = turn_d_offsets.get(turn_key, 0.0)
        legend_group = turn_key

        x_t, y_t = _spline_xy(tck_t, unew_t, cum_t, 0, L_t, d_offset=d_off)
        fig.add_trace(go.Scatter(
            x=x_t, y=y_t, mode='lines',
            line=dict(color=col_t, width=2, dash='dash'),
            opacity=alpha,
            name=f'{turn_key} [{method}]',
            legendgroup=legend_group,
            showlegend=True,
            hovertemplate=f'{turn_key} [{method}]<extra></extra>',
        ))

        xs_t, ys_t = splev(unew_t[[0, -1]], tck_t)
        fig.add_trace(go.Scatter(
            x=[xs_t[0]], y=[ys_t[0]], mode='markers',
            marker=dict(color=col_t, size=11, symbol='circle'),
            opacity=alpha,
            legendgroup=legend_group, showlegend=False,
            hovertemplate=f'{turn_key} start<extra></extra>',
        ))
        fig.add_trace(go.Scatter(
            x=[xs_t[1]], y=[ys_t[1]], mode='markers',
            marker=dict(color=col_t, size=11, symbol='diamond'),
            opacity=alpha,
            legendgroup=legend_group, showlegend=False,
            hovertemplate=f'{turn_key} end<extra></extra>',
        ))

        if show_validity_polygons:
            poly = te.get('validity_polygon')
            if poly is not None and not poly.is_empty:
                try:
                    px, py = poly.exterior.xy
                    fig.add_trace(go.Scatter(
                        x=list(px), y=list(py), mode='lines',
                        fill='toself',
                        fillcolor=_to_rgba_str(col_t, 0.06),
                        line=dict(color=col_t, width=0.6, dash='dot'),
                        opacity=0.3,
                        legendgroup=legend_group, showlegend=False,
                        hoverinfo='skip',
                    ))
                except Exception:
                    pass
    return fig, turn_colors


# =============================================================================
# EXTRACT ALL MISSING GAPS
# =============================================================================
def extract_all_gaps(veh_df, include_datetime=False):
    df = veh_df.sort_values(['veh_id', 'time']).reset_index(drop=True)
    n = len(df)

    veh_id = df['veh_id'].to_numpy()
    time = df['time'].to_numpy()
    missing = df['missing'].to_numpy()

    new_block = (veh_id != np.roll(veh_id, 1)) | (missing != np.roll(missing, 1))
    new_block[0] = True
    block_id = np.cumsum(new_block)

    # restrict to missing rows only
    mblock = block_id[missing]
    midx = np.flatnonzero(missing)          # positions of missing rows in df
    mveh = veh_id[missing]

    # find block boundaries via first/last occurrence per block_id (blocks are contiguous)
    # since rows are sorted and blocks contiguous, boundaries = where mblock changes
    boundary_start = np.r_[True, mblock[1:] != mblock[:-1]]
    boundary_end = np.r_[mblock[1:] != mblock[:-1], True]

    start_idx = midx[boundary_start]          # first row-index of each gap block
    end_idx = midx[boundary_end]               # last row-index of each gap block
    block_veh = mveh[boundary_start]
    n_points = np.diff(np.flatnonzero(np.r_[boundary_start, True]))  # count per block

    # candidate neighbor indices
    prev_idx = start_idx - 1
    next_idx = end_idx + 1

    prev_valid = (prev_idx >= 0)
    prev_valid[prev_valid] &= (veh_id[prev_idx[prev_valid]] == block_veh[prev_valid])
    prev_valid[prev_valid] &= (~missing[prev_idx[prev_valid]])

    next_valid = (next_idx < n)
    next_valid[next_valid] &= (veh_id[next_idx[next_valid]] == block_veh[next_valid])
    next_valid[next_valid] &= (~missing[next_idx[next_valid]])

    start_time = np.where(prev_valid, time[np.clip(prev_idx, 0, n - 1)], time[start_idx])
    end_time = np.where(next_valid, time[np.clip(next_idx, 0, n - 1)], time[end_idx])

    gaps = pd.DataFrame({
        'veh_id': block_veh,
        'start_time': start_time,
        'end_time': end_time,
        'n_points': n_points,
        'duration': end_time - start_time,
    })
    
    if include_datetime:
        dt = df['datetime'].to_numpy()
        start_datetime = np.where(prev_valid, dt[np.clip(prev_idx, 0, n - 1)], dt[start_idx])
        end_datetime = np.where(next_valid, dt[np.clip(next_idx, 0, n - 1)], dt[end_idx])
        gaps['start_datetime'] = start_datetime
        gaps['end_datetime'] = end_datetime
    
    return gaps