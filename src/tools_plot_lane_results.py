"""
tools_plot_results.py
---------------------
Plotting functions for examining tools_lane_coords_V4 output.

Functions
---------
plot_trajectory_map(bike_df, geometry_store, segment_registry, …)
    Folium map — one trajectory coloured by segment_id, with validity
    polygons, s_change markers, and match_quality info in tooltips.

plot_lane_coords(bike_df, …)
    Matplotlib — time-series panel of s, d, s_dot, d_dot, s_ddot, d_ddot.
    Segments are shaded by role with match_quality labelled.

plot_fleet_summary(output_df, …)
    Matplotlib — fleet-level diagnostics: match_quality distribution,
    is_fallback rate, movement_key counts, is_reverse rate.

plot_debug_panel(bike_df, geometry_store, segment_registry, …)
    Matplotlib 2×2 panel for one vehicle:
      (0,0) X-Y path with lane centerlines and turn splines
      (0,1) Cumulative s vs d, coloured by role
      (1,0) speed_ekf, s_dot, d_dot vs time
      (1,1) a_ekf, s_ddot, d_ddot vs time

Authors : ETH Zürich IVT
"""

# =============================================================================
# IMPORTS
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
from matplotlib.lines import Line2D
from collections import defaultdict

import folium
from pyproj import Transformer
from scipy.interpolate import splev


# =============================================================================
# SHARED HELPERS
# =============================================================================

def _is_axis_entry(key, val):
    """True if geometry_store entry is a spline-based axis dict."""
    if key in ('x_offset', 'y_offset'):
        return False
    if key.startswith('intersection_area') or key.startswith('__'):
        return False
    return isinstance(val, dict)
_PALETTE = [
    '#4878d0', '#ee854a', '#6acc65', '#d65f5f',
    '#956cb4', '#8c613c', '#dc7ec0', '#797979',
    '#d5bb67', '#82c6e2', '#e45858', '#56b4e9',
]
_ROLE_COLORS = {
    'approach':   '#2196F3',   # blue
    'turn':       '#FF9800',   # orange
    'departure':  '#4CAF50',   # green
    None:         '#BBBBBB',
}
_QUALITY_ALPHA = {
    'good':      1.0,
    'poor':      0.55,
    'fallback':  0.35,
    'unmatched': 0.20,
}
_proj_fwd = Transformer.from_crs("EPSG:2056", "EPSG:4326", always_xy=True)


def _local_to_latlon(x_arr, y_arr, x_offset, y_offset):
    lon, lat = _proj_fwd.transform(
        np.asarray(x_arr) + x_offset,
        np.asarray(y_arr) + y_offset,
    )
    return list(zip(lat, lon))


def _seg_color_map(segment_registry):
    """Assign a hex color to each seg_key from _PALETTE."""
    keys = [k for k, v in segment_registry.items()
            if v['type'] == 'lane']
    cmap = {k: _PALETTE[i % len(_PALETTE)] for i, k in enumerate(keys)}
    for k, v in segment_registry.items():
        if k not in cmap:
            cmap[k] = '#888888'
    return cmap


def _spline_latlon(tck, unew, cum_dist, s0, s1, x_offset, y_offset, n=100):
    s_vals = np.linspace(s0, s1, n)
    t_vals = np.interp(s_vals, cum_dist, unew)
    x, y   = splev(t_vals, tck)
    return _local_to_latlon(x, y, x_offset, y_offset)


# =============================================================================
# FUNCTION 1 — FOLIUM TRAJECTORY MAP
# =============================================================================

def plot_trajectory_map(bike_df,
                         geometry_store,
                         segment_registry,
                         color_by='segment_id',
                         show_validity_polygons=True,
                         show_s_change=True,
                         zoom_start=19,
                         save_path=None):
    """
    Plot one vehicle trajectory on a swisstopo folium map.

    Trajectory points are coloured by `color_by`:
        'segment_id'    — each matched segment gets a distinct color
        'role'          — approach=blue, turn=orange, departure=green
        'match_quality' — good=green, poor=yellow, fallback=orange,
                          unmatched=red

    Validity polygons for matched segments are shown as translucent fills.
    s_change markers (white circle) are shown on each segment centerline.
    Unmatched points are shown as small gray dots.

    Parameters
    ----------
    bike_df             : DataFrame — output of to_lane_coordinates()
    geometry_store      : dict
    segment_registry    : dict
    color_by            : str — 'segment_id' | 'role' | 'match_quality'
    show_validity_polygons : bool
    show_s_change       : bool
    zoom_start          : int
    save_path           : str | None

    Returns
    -------
    m : folium.Map
    """
    x_offset  = geometry_store['x_offset']
    y_offset  = geometry_store['y_offset']

    # ── Per-trajectory colour map keyed by seg_key (not geom_key) ────────────
    # Segments in chain order so colours are consistent across panels.
    _SEG_PALETTE = [
        '#4878d0', '#ee854a', '#6acc65', '#d65f5f',
        '#956cb4', '#8c613c', '#dc7ec0', '#2ec4b6',
        '#d5bb67', '#82c6e2', '#e45858', '#56b4e9',
    ]
    matched_segs_ordered = list(dict.fromkeys(
        s for s in bike_df['segment_id']
        if s is not None and s == s and s in segment_registry
    ))
    seg_col = {sk: _SEG_PALETTE[i % len(_SEG_PALETTE)]
               for i, sk in enumerate(matched_segs_ordered)}

    # ── Color function ────────────────────────────────────────────────────────
    QUALITY_COL = {
        'good':      '#2ecc71',
        'poor':      '#f39c12',
        'fallback':  '#e67e22',
        'unmatched': '#e74c3c',
    }

    def _row_color(row):
        if color_by == 'segment_id':
            seg = row['segment_id'] if 'segment_id' in row.index else None
            return seg_col.get(seg, '#BBBBBB') if seg and seg == seg else '#BBBBBB'
        elif color_by == 'role':
            role = row['segment_role'] if 'segment_role' in row.index else None
            return _ROLE_COLORS.get(role, '#BBBBBB')
        elif color_by == 'match_quality':
            q = row['match_quality'] if 'match_quality' in row.index else None
            return QUALITY_COL.get(q, '#BBBBBB')
        return '#888888'

    # ── Centre map ────────────────────────────────────────────────────────────
    xy     = bike_df[['x_ekf', 'y_ekf']].dropna().to_numpy()
    lon_c, lat_c = _proj_fwd.transform(
        float(np.median(xy[:, 0])) + x_offset,
        float(np.median(xy[:, 1])) + y_offset,
    )
    m = folium.Map(location=[lat_c, lon_c], zoom_start=zoom_start,
                   tiles=None, control_scale=True)
    folium.TileLayer(
        tiles=("https://wmts.geo.admin.ch/1.0.0/"
               "ch.swisstopo.swissimage/default/current/3857/{z}/{x}/{y}.jpeg"),
        attr="© swisstopo",
        name="swissimage",
        max_zoom=25,
    ).add_to(m)

    veh_id = bike_df['veh_id'].iloc[0] \
             if 'veh_id' in bike_df.columns else '?'

    # ── Validity polygons for matched segments ────────────────────────────────
    if show_validity_polygons:
        fg_poly = folium.FeatureGroup(name='Validity polygons', show=True)
        for seg_key in matched_segs_ordered:
            entry = segment_registry.get(seg_key)
            if entry is None:
                continue
            poly = entry.get('validity_polygon')
            if poly is None or poly.is_empty:
                continue
            col = seg_col.get(seg_key, '#888888')
            try:
                px, py      = poly.exterior.xy
                coords_poly = _local_to_latlon(px, py, x_offset, y_offset)
                folium.Polygon(
                    locations=coords_poly,
                    color=col, weight=1.5, opacity=0.6,
                    fill=True, fill_color=col, fill_opacity=0.12,
                    tooltip=seg_key,
                ).add_to(fg_poly)
            except Exception:
                pass
        fg_poly.add_to(m)

    # ── s_change markers ──────────────────────────────────────────────────────
    if show_s_change:
        fg_sc = folium.FeatureGroup(name='s_change markers', show=True)
        for seg_key in matched_segs_ordered:
            entry = segment_registry.get(seg_key)
            if entry is None or entry['type'] == 'turn':
                continue
            geom_key = entry['geometry_key']
            geo      = geometry_store.get(geom_key, {})
            s_change = geo.get('s_change')
            if s_change is None:
                continue
            tck, unew, cum_dist = geo['spline']
            t_sc     = float(np.interp(s_change, cum_dist, unew))
            x_sc, y_sc = splev(t_sc, tck)
            lon_sc, lat_sc = _proj_fwd.transform(
                float(x_sc) + x_offset, float(y_sc) + y_offset
            )
            col = seg_col.get(seg_key, '#888888')
            folium.CircleMarker(
                location=(lat_sc, lon_sc),
                radius=7, color=col, fill=True,
                fill_color='white', fill_opacity=1.0, weight=3,
                tooltip=f'{seg_key}  s_change={s_change:.2f} m',
            ).add_to(fg_sc)

            # Extra s_change_* markers
            for key in [k for k in geo
                        if k.startswith('s_') and
                        k not in ('s_stop', 's_yield', 's_change')]:
                sc_val = geo[key]
                t_ec   = float(np.interp(sc_val, cum_dist, unew))
                x_ec, y_ec = splev(t_ec, tck)
                lon_ec, lat_ec = _proj_fwd.transform(
                    float(x_ec) + x_offset, float(y_ec) + y_offset
                )
                folium.CircleMarker(
                    location=(lat_ec, lon_ec),
                    radius=5, color=col, fill=True,
                    fill_color='yellow', fill_opacity=1.0, weight=2,
                    tooltip=f'{seg_key}  {key}={sc_val:.2f} m',
                ).add_to(fg_sc)
        fg_sc.add_to(m)

    # ── Centerlines for matched segments only ────────────────────────────────
    fg_cl = folium.FeatureGroup(name='Centerlines', show=True)
    for seg_key in matched_segs_ordered:
        entry    = segment_registry.get(seg_key)
        if entry is None:
            continue
        geom_key = entry['geometry_key']
        geo      = geometry_store.get(geom_key)
        if geo is None or not isinstance(geo, dict) or 'spline' not in geo:
            continue
        col              = seg_col.get(seg_key, '#888888')
        tck, unew, cum_dist = geo['spline']
        L                = geo['total_length']
        is_forward       = entry['is_forward']
        # Offset slightly so opposing directions are visually separated
        d_off = 1.5 if is_forward else -1.5
        s_v   = np.linspace(0, L, 200)
        t_v   = np.interp(s_v, cum_dist, unew)
        from scipy.interpolate import splev as _splev_cl
        xc, yc   = _splev_cl(t_v, tck, der=0)
        dxc, dyc = _splev_cl(t_v, tck, der=1)
        tang = np.sqrt(dxc**2 + dyc**2)
        tang = np.where(tang > 1e-12, tang, 1.0)
        nx = -dyc / tang;  ny = dxc / tang
        xp = xc + d_off * nx;  yp = yc + d_off * ny
        latlon_cl = _local_to_latlon(xp, yp, x_offset, y_offset)
        dash = None if is_forward else '6 4'
        folium.PolyLine(
            locations=latlon_cl,
            color=col, weight=3, opacity=0.6,
            dash_array=dash,
            tooltip=f'{seg_key}  ({"fwd" if is_forward else "rev"})',
        ).add_to(fg_cl)
    fg_cl.add_to(m)

    # ── Trajectory points ─────────────────────────────────────────────────────
    fg_traj = folium.FeatureGroup(
        name=f'Trajectory — veh {veh_id}  ({color_by})', show=True
    )

    for _, row in bike_df.iterrows():
        if np.isnan(row['x_ekf']) or np.isnan(row['y_ekf']):
            continue
        lon_p, lat_p = _proj_fwd.transform(
            row['x_ekf'] + x_offset,
            row['y_ekf'] + y_offset,
        )
        col     = _row_color(row)
        quality = row.get('match_quality') or 'unmatched'
        alpha   = _QUALITY_ALPHA.get(quality, 0.5)
        radius  = 4 if quality not in (None, 'unmatched') else 2

        seg_id  = row.get('segment_id')  or 'none'
        role    = row.get('segment_role') or 'none'
        rev     = bool(row.get('is_reverse', False))
        fb      = bool(row.get('is_fallback', False))
        s_val   = row.get('s', float('nan'))
        d_val   = row.get('d', float('nan'))

        tooltip = (
            f"seg={seg_id}  role={role}  q={quality}<br>"
            f"s={s_val:.2f}m  d={d_val:.2f}m<br>"
            f"reverse={rev}  fallback={fb}"
        )
        folium.CircleMarker(
            location=(lat_p, lon_p),
            radius=radius,
            color=col, fill=True, fill_color=col,
            fill_opacity=alpha, weight=1,
            tooltip=tooltip,
        ).add_to(fg_traj)

    # Start / end markers
    first_valid = bike_df[bike_df['x_ekf'].notna()].iloc[0]
    last_valid  = bike_df[bike_df['x_ekf'].notna()].iloc[-1]
    for row_end, icon_col, icon_name in [
        (first_valid, 'green', 'play'),
        (last_valid,  'red',   'stop'),
    ]:
        lon_e, lat_e = _proj_fwd.transform(
            row_end['x_ekf'] + x_offset,
            row_end['y_ekf'] + y_offset,
        )
        folium.Marker(
            location=(lat_e, lon_e),
            icon=folium.Icon(color=icon_col, icon=icon_name, prefix='fa'),
            tooltip='START' if icon_col == 'green' else 'END',
        ).add_to(fg_traj)

    fg_traj.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)

    if save_path:
        m.save(save_path)
        print(f"Map saved: {save_path}")
    return m


# =============================================================================
# FUNCTION 2 — LANE COORDINATE TIME SERIES
# =============================================================================

def plot_lane_coords(bike_df,
                     variables=('s', 'd', 's_dot', 'd_dot', 's_ddot', 'd_ddot'),
                     time_col='t',
                     save_path=None):
    """
    Time-series panel plot of lane coordinate outputs for one vehicle.

    Each variable gets its own subplot. Segment regions are shaded by role:
        approach  = light blue
        turn      = light orange
        departure = light green
    Match quality is annotated on the shaded region:
        good → solid shade
        poor → hatched
        fallback → hatched + 'fb' label
        unmatched → no shade

    Reverse-traversal rows are marked with a red dot on the s panel.

    Parameters
    ----------
    bike_df    : DataFrame — output of to_lane_coordinates()
    variables  : tuple of str — columns to plot
    time_col   : str — time column name
    save_path  : str | None
    """
    ROLE_SHADE = {
        'approach':  '#cce5ff',
        'turn':      '#ffe8cc',
        'departure': '#ccf0cc',
    }

    n_vars = len(variables)
    fig, axes = plt.subplots(n_vars, 1, figsize=(14, 2.8 * n_vars),
                              sharex=True)
    if n_vars == 1:
        axes = [axes]

    veh_id   = bike_df['veh_id'].iloc[0] \
               if 'veh_id' in bike_df.columns else '?'
    mov_key  = bike_df['movement_key'].dropna().mode()
    mov_key  = mov_key.iloc[0] if len(mov_key) else 'unmatched'
    fig.suptitle(
        f'Lane coordinates — veh {veh_id}   movement: {mov_key}',
        fontsize=11, y=1.01
    )

    # Reset index so t[i] indexing is consistent with iterrows()
    bike_df = bike_df.reset_index(drop=True)

    t = bike_df[time_col].to_numpy() if time_col in bike_df.columns \
        else np.arange(len(bike_df))

    # ── Identify segment regions ──────────────────────────────────────────────
    # Build a list of (start_t, end_t, role, quality, is_reverse, is_fallback)
    regions = []
    current_seg = None
    for i, row in bike_df.iterrows():
        seg = row.get('segment_id')
        if seg != current_seg:
            if current_seg is not None:
                regions[-1]['end_t'] = t[i - 1] if i > 0 else t[0]
            if seg is not None:
                regions.append({
                    'start_t':   t[i],
                    'end_t':     t[min(i + 1, len(t) - 1)],
                    'role':      row.get('segment_role'),
                    'quality':   row.get('match_quality', 'unmatched'),
                    'seg_key':   seg,
                    'is_reverse':bool(row.get('is_reverse', False)),
                    'fallback':  bool(row.get('is_fallback', False)),
                })
            current_seg = seg
    if regions:
        regions[-1]['end_t'] = t[-1]

    # ── Plot each variable ────────────────────────────────────────────────────
    y_labels = {
        's':     's [m]',
        'd':     'd [m]',
        's_dot': 'ṡ [m/s]',
        'd_dot': 'd̈ [m/s]',
        's_ddot':'s̈ [m/s²]',
        'd_ddot':'d̈ [m/s²]',
    }

    for ax, var in zip(axes, variables):
        if var not in bike_df.columns:
            ax.set_ylabel(var)
            ax.text(0.5, 0.5, f'column "{var}" not found',
                    transform=ax.transAxes, ha='center', va='center',
                    color='red')
            continue

        y = bike_df[var].to_numpy(dtype=float)

        # ── Shade segment regions ─────────────────────────────────────────────
        for reg in regions:
            role    = reg['role']
            quality = reg['quality']
            shade   = ROLE_SHADE.get(role, '#eeeeee')
            hatch   = '//' if quality in ('poor', 'fallback') else ''
            ax.axvspan(reg['start_t'], reg['end_t'],
                       color=shade, alpha=0.5, hatch=hatch, zorder=0)
            # Label at region midpoint (only on top axis)
            if ax is axes[0]:
                t_mid   = 0.5 * (reg['start_t'] + reg['end_t'])
                label   = reg['seg_key']
                if reg['is_reverse']:
                    label += ' ↩'
                if reg['fallback']:
                    label += ' (fb)'
                ax.text(t_mid, ax.get_ylim()[1] if ax.get_ylim()[1] != 0 else 1,
                        label, fontsize=6, ha='center', va='bottom',
                        color='#333333', clip_on=True, rotation=45)

        # ── Plot signal ───────────────────────────────────────────────────────
        ax.plot(t, y, color='#222222', linewidth=1.2, zorder=3)

        # Mark reverse-traversal points on s panel
        if var == 's' and 'is_reverse' in bike_df.columns:
            rev_mask = bike_df['is_reverse'].astype(bool)
            ax.scatter(t[rev_mask], y[rev_mask],
                       color='red', s=12, zorder=5,
                       label='is_reverse')
            if rev_mask.any():
                ax.legend(fontsize=7, loc='upper right')

        # Zero line for d, d_dot, d_ddot
        if var in ('d', 'd_dot', 'd_ddot', 's_ddot'):
            ax.axhline(0, color='#999999', linewidth=0.8, linestyle='--',
                       zorder=1)

        ax.set_ylabel(y_labels.get(var, var), fontsize=9)
        ax.grid(True, alpha=0.25)

    axes[-1].set_xlabel(time_col if time_col in bike_df.columns
                        else 'frame index', fontsize=9)

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_patches = [
        mpatches.Patch(facecolor=ROLE_SHADE['approach'],
                       label='approach', alpha=0.6),
        mpatches.Patch(facecolor=ROLE_SHADE['turn'],
                       label='turn', alpha=0.6),
        mpatches.Patch(facecolor=ROLE_SHADE['departure'],
                       label='departure', alpha=0.6),
        mpatches.Patch(facecolor='#dddddd', hatch='//',
                       label='poor / fallback', alpha=0.5),
    ]
    fig.legend(handles=legend_patches, loc='lower center',
               ncol=4, fontsize=8, framealpha=0.9,
               bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()


# =============================================================================
# FUNCTION 3 — FLEET SUMMARY DIAGNOSTICS
# =============================================================================

def plot_fleet_summary(output_df, save_path=None):
    """
    Fleet-level diagnostic plots for a batch of processed trajectories.

    Four panels:
      1. match_quality distribution (bar chart, per-vehicle and total)
      2. movement_key frequency (horizontal bar — top 15)
      3. is_fallback and is_reverse rates per segment_id
      4. Matched point fraction per vehicle (sorted)

    Parameters
    ----------
    output_df : DataFrame — pd.concat of all to_lane_coordinates() outputs
    save_path : str | None
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Fleet summary — lane coordinate transform results',
                 fontsize=13)

    # ── Panel 1: match_quality distribution ──────────────────────────────────
    ax = axes[0, 0]
    quality_order = ['good', 'poor', 'fallback', 'unmatched']
    quality_cols  = {
        'good':      '#2ecc71',
        'poor':      '#f39c12',
        'fallback':  '#e67e22',
        'unmatched': '#e74c3c',
    }
    counts = output_df['match_quality'].value_counts()
    bars   = [counts.get(q, 0) for q in quality_order]
    x_pos  = np.arange(len(quality_order))
    rects  = ax.bar(x_pos, bars,
                    color=[quality_cols[q] for q in quality_order],
                    edgecolor='white', linewidth=0.8)
    ax.bar_label(rects, padding=3, fontsize=9)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(quality_order, fontsize=10)
    ax.set_ylabel('Number of trajectory points')
    ax.set_title('Match quality — all points')
    ax.grid(axis='y', alpha=0.3)

    total = len(output_df)
    for q, c in zip(quality_order, bars):
        ax.text(quality_order.index(q), c + total * 0.002,
                f'{100*c/max(total,1):.1f}%',
                ha='center', va='bottom', fontsize=8, color='#333333')

    # ── Panel 2: movement_key frequency ──────────────────────────────────────
    ax = axes[0, 1]
    mov_counts = (
        output_df[output_df['movement_key'].notna()]
        .groupby('veh_id')['movement_key']
        .first()
        .value_counts()
        .head(15)
    )
    if len(mov_counts):
        cmap_tab = plt.cm.tab20
        colors   = [cmap_tab(i / len(mov_counts))
                    for i in range(len(mov_counts))]
        ax.barh(mov_counts.index[::-1], mov_counts.values[::-1],
                color=colors[::-1], edgecolor='white')
        ax.set_xlabel('Number of vehicles')
        ax.set_title('Movement key frequency (top 15)')
        ax.grid(axis='x', alpha=0.3)
        for i, (val, label) in enumerate(
                zip(mov_counts.values[::-1], mov_counts.index[::-1])):
            ax.text(val + 0.2, i, str(val), va='center', fontsize=8)
    else:
        ax.text(0.5, 0.5, 'No matched movements',
                transform=ax.transAxes, ha='center', va='center',
                color='gray')
        ax.set_title('Movement key frequency')

    # ── Panel 3: fallback and reverse rates per segment_id ───────────────────
    ax = axes[1, 0]
    matched = output_df[output_df['segment_id'].notna()].copy()
    if len(matched):
        seg_stats = matched.groupby('segment_id').agg(
            n_pts      =('segment_id', 'count'),
            n_fallback =('is_fallback', 'sum'),
            n_reverse  =('is_reverse',  'sum'),
        )
        seg_stats['fallback_rate'] = seg_stats['n_fallback'] / seg_stats['n_pts']
        seg_stats['reverse_rate']  = seg_stats['n_reverse']  / seg_stats['n_pts']
        seg_stats = seg_stats.sort_values('n_pts', ascending=True)

        y   = np.arange(len(seg_stats))
        ax.barh(y - 0.2, seg_stats['fallback_rate'], height=0.35,
                color='#e67e22', alpha=0.8, label='fallback rate')
        ax.barh(y + 0.2, seg_stats['reverse_rate'],  height=0.35,
                color='#9b59b6', alpha=0.8, label='reverse rate')
        ax.set_yticks(y)
        ax.set_yticklabels(seg_stats.index, fontsize=8)
        ax.set_xlabel('Fraction of points')
        ax.set_title('Fallback and reverse rates per segment')
        ax.set_xlim(0, 1)
        ax.legend(fontsize=8)
        ax.grid(axis='x', alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No matched segments',
                transform=ax.transAxes, ha='center', va='center',
                color='gray')
        ax.set_title('Fallback / reverse rates per segment')

    # ── Panel 4: matched point fraction per vehicle ───────────────────────────
    ax = axes[1, 1]
    if 'veh_id' in output_df.columns:
        veh_stats = output_df.groupby('veh_id').agg(
            total    =('x_ekf', 'count'),
            matched  =('segment_id', lambda x: x.notna().sum()),
        )
        veh_stats['match_frac'] = veh_stats['matched'] / veh_stats['total'].clip(1)
        veh_stats = veh_stats.sort_values('match_frac')

        colors_veh = [
            '#2ecc71' if f >= 0.8 else
            '#f39c12' if f >= 0.5 else
            '#e74c3c'
            for f in veh_stats['match_frac']
        ]
        ax.barh(np.arange(len(veh_stats)), veh_stats['match_frac'],
                color=colors_veh, edgecolor='white', height=0.7)
        ax.axvline(0.8, color='#2ecc71', linestyle='--',
                   linewidth=1.2, label='80% threshold')
        ax.axvline(0.5, color='#e74c3c', linestyle='--',
                   linewidth=1.2, label='50% threshold')
        ax.set_xlabel('Matched point fraction')
        ax.set_title(f'Match fraction per vehicle  (n={len(veh_stats)})')
        ax.set_xlim(0, 1)
        ax.legend(fontsize=8)
        ax.grid(axis='x', alpha=0.3)

        # Summary text
        n_good = (veh_stats['match_frac'] >= 0.8).sum()
        n_poor = ((veh_stats['match_frac'] >= 0.5) &
                  (veh_stats['match_frac'] < 0.8)).sum()
        n_bad  = (veh_stats['match_frac'] < 0.5).sum()
        ax.text(0.98, 0.02,
                f'≥80%: {n_good}  |  50-80%: {n_poor}  |  <50%: {n_bad}',
                transform=ax.transAxes, ha='right', va='bottom',
                fontsize=8, color='#333333',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    else:
        ax.text(0.5, 0.5, 'No veh_id column',
                transform=ax.transAxes, ha='center', va='center',
                color='gray')
        ax.set_title('Match fraction per vehicle')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()


# =============================================================================
# FUNCTION 4 — 2×2 DEBUG PANEL
# =============================================================================

def plot_debug_panel(bike_df,
                     geometry_store,
                     segment_registry,
                     time_col='t',
                     xy_offset=True,
                     save_path=None):
    """
    2×2 debug panel for one vehicle trajectory.

    Panel layout
    ------------
    (0,0) X-Y path
        Trajectory coloured by segment_id — each matched segment gets a
        distinct colour. Lane centerlines drawn as background. Turn splines
        drawn in matching colour. s_change ticks and start/end markers.

    (0,1) s vs d  (global s, not reset per segment)
        x-axis: s [m], continuous across the full trajectory.
        y-axis: d [m].
        Points coloured by segment_id.
        Bike lane boundary shown if available (dashed cyan lines).

    (1,0) Speed decomposition vs time
        speed_ekf (black), s_dot and d_dot coloured by segment_id.
        Background shaded by segment colour (light, alpha=0.15).
        Hatched if match_quality is poor/fallback.
        Cyan shading where in_bike_lane == True.

    (1,1) Acceleration decomposition vs time
        a_ekf (black), s_ddot and d_ddot coloured by segment_id.
        Same background shading as (1,0).

    Parameters
    ----------
    bike_df          : DataFrame — output of to_lane_coordinates()
    geometry_store   : dict
    segment_registry : dict
    time_col         : str — time column name (default 't')
    xy_offset        : bool — subtract first valid x/y so path starts at (0,0)
    save_path        : str | None

    Returns
    -------
    fig : matplotlib Figure
    """
    from scipy.interpolate import splev as _splev

    # Reset index so numpy array indexing (x_plt[i], t[i], etc.) is
    # consistent with iterrows() — bike_df may be a slice of a larger df
    bike_df = bike_df.reset_index(drop=True)

    x_offset_gs = geometry_store['x_offset']
    y_offset_gs = geometry_store['y_offset']

    veh_id  = bike_df['veh_id'].iloc[0] \
              if 'veh_id' in bike_df.columns else '?'
    mov_key = bike_df['movement_key'].dropna().mode()
    mov_key = mov_key.iloc[0] if len(mov_key) else 'unmatched'

    t = bike_df[time_col].to_numpy(dtype=float) \
        if time_col in bike_df.columns \
        else np.arange(len(bike_df), dtype=float)

    x_raw = bike_df['x_ekf'].to_numpy(dtype=float)
    y_raw = bike_df['y_ekf'].to_numpy(dtype=float)
    x0    = x_raw[np.isfinite(x_raw)][0] if xy_offset else 0.0
    y0    = y_raw[np.isfinite(y_raw)][0] if xy_offset else 0.0
    x_plt = x_raw - x0
    y_plt = y_raw - y0

    # ── Matched segments in chain order ──────────────────────────────────────
    matched_segs = list(dict.fromkeys(
        s for s in bike_df['segment_id']
        if s is not None and s == s and s in segment_registry
    ))

    # ── Per-segment colour palette (distinct per seg_key, not geom_key) ───────
    _SEG_PALETTE = [
        '#4878d0', '#ee854a', '#6acc65', '#d65f5f',
        '#956cb4', '#8c613c', '#dc7ec0', '#2ec4b6',
        '#d5bb67', '#82c6e2', '#e45858', '#56b4e9',
    ]
    seg_col = {sk: _SEG_PALETTE[i % len(_SEG_PALETTE)]
               for i, sk in enumerate(matched_segs)}
    seg_col[None] = '#BBBBBB'

    def _row_col(seg):
        return seg_col.get(seg, '#BBBBBB')

    # ── Segment time regions (for background shading in time panels) ──────────
    regions = []
    current_seg = None
    for i, row in bike_df.iterrows():
        seg = row.get('segment_id')
        if seg != current_seg:
            if current_seg is not None and regions:
                regions[-1]['end_t'] = t[i - 1] if i > 0 else t[0]
            if seg is not None:
                regions.append({
                    'start_t': t[i],
                    'end_t':   t[min(i + 1, len(t) - 1)],
                    'seg_key': seg,
                    'quality': row.get('match_quality', 'unmatched'),
                })
            current_seg = seg
    if regions:
        regions[-1]['end_t'] = t[-1]

    def _shade_seg_regions(ax):
        """Light segment-coloured background + hatch for poor/fallback."""
        for reg in regions:
            col   = _row_col(reg['seg_key'])
            hatch = '//' if reg['quality'] in ('poor', 'fallback') else ''
            ax.axvspan(reg['start_t'], reg['end_t'],
                       color=col, alpha=0.12, hatch=hatch, zorder=0)

    def _shade_bike_lane(ax, y_arr):
        """Cyan shading where in_bike_lane == True."""
        if 'in_bike_lane' not in bike_df.columns:
            return
        bl = bike_df['in_bike_lane'].to_numpy(dtype=float)
        in_lane = np.isfinite(bl) & (bl > 0.5)
        if not in_lane.any():
            return
        # Find contiguous runs
        changes = np.diff(in_lane.astype(int), prepend=0, append=0)
        starts  = np.where(changes ==  1)[0]
        ends    = np.where(changes == -1)[0]
        for s_i, e_i in zip(starts, ends):
            ax.axvspan(t[s_i], t[min(e_i, len(t)-1)],
                       color='cyan', alpha=0.25, zorder=1,
                       label='in_bike_lane' if s_i == starts[0] else '_nolegend_')

    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(6*2, 4*2))
    fig.suptitle(
        f'Debug panel — veh {veh_id}   movement: {mov_key}',
        fontsize=12, #y=1.01
    )

    # =========================================================================
    # (0,0) X-Y PATH
    # =========================================================================
    ax = axes[0, 0]

    # ── Lane centerlines as background ────────────────────────────────────────
    LANE_OFFSET_M = 2.5
    for geom_key, geo in geometry_store.items():
        if not _is_axis_entry(geom_key, geo) or geo.get('s_stop') is None:
            continue
        tck, unew, cum_dist = geo['spline']
        L     = geo['total_length']
        col   = '#BBBBBB'  # background axes always gray

        for is_fwd, d_off in [(True, +LANE_OFFSET_M), (False, -LANE_OFFSET_M)]:
            s_v  = np.linspace(0, L, 200)
            t_v  = np.interp(s_v, cum_dist, unew)
            xc, yc   = _splev(t_v, tck, der=0)
            dxc, dyc = _splev(t_v, tck, der=1)
            tang = np.sqrt(dxc**2 + dyc**2)
            tang = np.where(tang > 1e-12, tang, 1.0)
            nx = -dyc / tang;  ny = dxc / tang
            xp = (xc + d_off * nx) - x0
            yp = (yc + d_off * ny) - y0
            ls = '-' if is_fwd else '--'
            ax.plot(xp, yp, color=col, linewidth=1.8, linestyle=ls,
                    alpha=0.4, zorder=2, solid_capstyle='round')

        # s_change tick
        s_change = geo.get('s_change')
        if s_change is not None:
            t_sc     = float(np.interp(s_change, cum_dist, unew))
            xsc, ysc = _splev(t_sc, tck)
            ax.scatter(xsc - x0, ysc - y0,
                       color=col, s=60, marker='v', zorder=5, alpha=0.8)

        # Extra s_change_*
        for key in [k for k in geo if k.startswith('s_')
                    and k not in ('s_stop', 's_yield', 's_change')]:
            sc_val   = geo[key]
            t_ec     = float(np.interp(sc_val, cum_dist, unew))
            xec, yec = _splev(t_ec, tck)
            ax.scatter(xec - x0, yec - y0,
                       color=col, s=40, marker='v', zorder=5,
                       alpha=0.5, facecolors='yellow', edgecolors=col)

    # ── Turn splines ──────────────────────────────────────────────────────────
    for seg_key in matched_segs:
        if segment_registry[seg_key]['type'] != 'turn':
            continue
        geo_t           = geometry_store.get(seg_key, {})
        if 'spline' not in geo_t:
            continue
        tck_t, unew_t, cum_t = geo_t['spline']
        L_t  = geo_t['total_length']
        col  = _row_col(seg_key)
        t_v  = np.linspace(0, 1, 150)
        xt, yt = _splev(t_v, tck_t)
        ax.plot(xt - x0, yt - y0, color=col, linewidth=2.5,
                linestyle='-', alpha=0.7, zorder=3)

    # ── Trajectory coloured by segment_id ────────────────────────────────────
    prev_seg = None
    seg_x, seg_y = [], []

    def _flush(seg_x, seg_y, col):
        if len(seg_x) > 1:
            ax.plot(seg_x, seg_y, color=col, linewidth=2,
                    zorder=6, solid_capstyle='round')

    for i, row in bike_df.iterrows():
        xi = x_plt[i];  yi = y_plt[i]
        if not (np.isfinite(xi) and np.isfinite(yi)):
            continue
        seg = row.get('segment_id')
        col = _row_col(seg)
        if seg != prev_seg:
            _flush(seg_x, seg_y,
                   _row_col(prev_seg))
            seg_x, seg_y = [xi], [yi]
            prev_seg = seg
        else:
            seg_x.append(xi);  seg_y.append(yi)
    _flush(seg_x, seg_y,
           _row_col(prev_seg))

    # Unmatched points — small gray dots
    unmatched_mask = bike_df['segment_id'].isna()
    ax.scatter(x_plt[unmatched_mask], y_plt[unmatched_mask],
               color='#BBBBBB', s=6, zorder=5, label='unmatched')

    # Start / end arrows
    valid_idx = np.where(np.isfinite(x_plt) & np.isfinite(y_plt))[0]
    if len(valid_idx) >= 2:
        i0, i1 = valid_idx[0], valid_idx[-1]
        ax.annotate('', xy=(x_plt[i0], y_plt[i0]),
                    xytext=(x_plt[i0] - 0.5, y_plt[i0] - 0.5),
                    arrowprops=dict(arrowstyle='->', color='green', lw=2))
        ax.scatter(x_plt[i1], y_plt[i1], color='red',
                   s=80, marker='s', zorder=9, label='end')
        ax.scatter(x_plt[i0], y_plt[i0], color='green',
                   s=80, marker='o', zorder=9, label='start')

    ax.set_xlabel('X local [m]', fontsize=9)
    ax.set_ylabel('Y local [m]', fontsize=9)
    ax.set_title('X-Y path  (coloured by segment)', fontsize=10)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.25)

    # Legend: one entry per matched segment
    legend_handles = [
        Line2D([0], [0], color=_row_col(sk), linewidth=2.5, label=sk)
        for sk in matched_segs
    ]
    legend_handles.append(
        Line2D([0], [0], color='#BBBBBB', linewidth=1.5, label='unmatched')
    )
    ax.legend(handles=legend_handles, fontsize=7, loc='best',
              framealpha=0.85)

    # =========================================================================
    # (0,1) S vs D  (global s, not reset per segment)
    # =========================================================================
    ax = axes[0, 1]

    # ── Continuous s across segment boundaries ───────────────────────────────
    # Raw s resets at each segment boundary (each segment has its own spline).
    # Stitch segments together by offsetting each so it starts where the
    # previous one ended — giving a continuous s axis across the full chain.
    s_raw  = bike_df['s'].to_numpy(dtype=float)
    segs   = bike_df['segment_id'].to_numpy()
    s_plot = np.full_like(s_raw, np.nan)

    offset      = 0.0
    prev_s_last = np.nan
    i = 0
    while i < len(s_raw):
        seg_i = segs[i]
        j = i
        while j < len(s_raw) and segs[j] == seg_i:
            j += 1
        if seg_i is not None:
            s_win = s_raw[i:j]
            valid = np.isfinite(s_win)
            if valid.any():
                if np.isfinite(prev_s_last):
                    # Shift so this segment continues from last valid s
                    offset = prev_s_last - s_win[np.argmax(valid)]
                s_plot[i:j] = np.where(valid, s_win + offset, np.nan)
                prev_s_last = s_plot[i:j][valid][-1]
        i = j

    d_plot = bike_df['d'].to_numpy(dtype=float)

    # ── Build per-segment s offset map for bike boundary alignment ────────────
    # Recompute the offset applied to each segment's s so the bike lane
    # boundary can be shifted by the same amount onto the continuous s axis.
    seg_s_offset = {}   # {seg_key: offset applied during stitching}
    _offset      = 0.0
    _prev_last   = np.nan
    _i = 0
    while _i < len(s_raw):
        _seg_i = segs[_i]
        _j = _i
        while _j < len(s_raw) and segs[_j] == _seg_i:
            _j += 1
        if _seg_i is not None:
            _win   = s_raw[_i:_j]
            _valid = np.isfinite(_win)
            if _valid.any():
                if np.isfinite(_prev_last):
                    _offset = _prev_last - _win[np.argmax(_valid)]
                seg_s_offset[_seg_i] = _offset
                _prev_s_cont = np.where(_valid, _win + _offset, np.nan)
                _prev_last   = _prev_s_cont[_valid][-1]
        _i = _j

    # ── Lines coloured by segment_id ─────────────────────────────────────────
    unm = np.array([s is None or s != s for s in segs])

    for sk in matched_segs:
        mask = np.array([s == sk for s in segs]) & np.isfinite(s_plot) & np.isfinite(d_plot)
        if mask.any():
            ax.plot(s_plot[mask], d_plot[mask],
                    color=_row_col(sk), linewidth=1.8, alpha=0.85,
                    zorder=4, label=sk, solid_capstyle='round')

    unm_mask = unm & np.isfinite(d_plot) & np.isfinite(s_plot)
    if unm_mask.any():
        ax.plot(s_plot[unm_mask], d_plot[unm_mask],
                color='#BBBBBB', linewidth=1.0, alpha=0.5,
                zorder=3, label='unmatched')

    # ── Bike lane boundaries on continuous s axis ────────────────────────────
    # Uses s_native from the dataframe (direct projection arc-length, no
    # directed offset). Evaluates d_boundary_spline on a uniform native s
    # grid (smooth) and maps onto the continuous s axis via the stitching
    # offset — which is now correctly computed in native s space.
    first_bl = True
    if 's_native' in bike_df.columns:
        s_native_df = bike_df['s_native'].to_numpy(dtype=float)
        for seg_key in matched_segs:
            entry     = segment_registry[seg_key]
            bike_lane = entry.get('bike_lane')
            if bike_lane is None or 'd_boundary_spline' not in bike_lane:
                continue
            d_bnd_spl = bike_lane['d_boundary_spline']
            s_domain  = bike_lane.get('s_domain', (0, 1))
            w_bike    = bike_lane.get('w_bike', 0)
            side      = bike_lane.get('side', -1)

            # Rows for this segment with valid s_native and continuous s
            seg_mask = (np.array([s == seg_key for s in segs]) &
                        np.isfinite(s_native_df) & np.isfinite(s_plot))
            if not seg_mask.any():
                continue

            # Build native→continuous mapping from actual data points
            # (linear: continuous = native + offset, constant per segment)
            s_nat_pts  = s_native_df[seg_mask]
            s_cont_pts = s_plot[seg_mask]
            # Offset = continuous - native (should be constant; take median)
            nat2cont_offset = float(np.median(s_cont_pts - s_nat_pts))

            # Uniform native s grid → smooth boundary
            s_nat_grid = np.linspace(s_domain[0], s_domain[1], 120)
            d_bnd      = np.array([float(d_bnd_spl(si)) for si in s_nat_grid])
            d_far      = d_bnd + side * w_bike
            s_cont_grid = s_nat_grid + nat2cont_offset

            ax.plot(s_cont_grid, d_bnd, color='cyan', linewidth=1.8,
                    linestyle='-',  zorder=5, alpha=0.9,
                    label='bike boundary (inner)' if first_bl else '_nolegend_')
            ax.plot(s_cont_grid, d_far, color='cyan', linewidth=1.2,
                    linestyle='--', zorder=5, alpha=0.7,
                    label='bike boundary (outer)' if first_bl else '_nolegend_')
            first_bl = False

    ax.axhline(0, color='#999999', linewidth=0.8, linestyle='--', zorder=1)
    ax.set_xlabel('s [m]  (continuous across chain)', fontsize=9)
    ax.set_ylabel('d [m]  (+ = left of travel)', fontsize=9)
    ax.set_title('s vs d  (coloured by segment)', fontsize=10)
    ax.legend(fontsize=7, loc='best', framealpha=0.85)
    ax.grid(True, alpha=0.25)

    # =========================================================================
    # (1,0) SPEED DECOMPOSITION vs TIME
    # =========================================================================
    ax = axes[1, 0]
    # _shade_seg_regions(ax)
    _shade_bike_lane(ax, None)

    speed_raw = bike_df['speed_ekf'].to_numpy(dtype=float) \
                if 'speed_ekf' in bike_df.columns else np.full(len(t), np.nan)
    s_dot_arr = bike_df['s_dot'].to_numpy(dtype=float) \
                if 's_dot' in bike_df.columns else np.full(len(t), np.nan)
    d_dot_arr = bike_df['d_dot'].to_numpy(dtype=float) \
                if 'd_dot' in bike_df.columns else np.full(len(t), np.nan)
    seg_ids   = bike_df['segment_id'].to_numpy()
    unm       = np.array([s is None or s != s for s in seg_ids])

    ax.plot(t, speed_raw, color='black', linewidth=1.6,
            zorder=4, label='speed_ekf [km/h]')

    for sk in matched_segs:
        mask = np.array([s == sk for s in seg_ids])
        if not mask.any():
            continue
        col = _row_col(sk)
        ax.plot(np.where(mask, t, np.nan),
                np.where(mask, s_dot_arr, np.nan),
                color=col, linewidth=1.6, zorder=5, label=f's_dot [{sk}]')
        ax.plot(np.where(mask, t, np.nan),
                np.where(mask, d_dot_arr, np.nan),
                color=col, linewidth=1.2, linestyle='--', zorder=5,
                label=f'd_dot [{sk}]')
    if unm.any():
        ax.plot(np.where(unm, t, np.nan), np.where(unm, s_dot_arr, np.nan),
                color='#BBBBBB', linewidth=1.0, zorder=3)
        ax.plot(np.where(unm, t, np.nan), np.where(unm, d_dot_arr, np.nan),
                color='#BBBBBB', linewidth=0.8, linestyle='--', zorder=3)

    if 'is_reverse' in bike_df.columns:
        rev_mask = bike_df['is_reverse'].astype(bool).to_numpy()
        if rev_mask.any():
            ax.scatter(t[rev_mask], s_dot_arr[rev_mask],
                       color='red', s=15, zorder=7, label='is_reverse')

    ax.axhline(0, color='#999999', linewidth=0.8, linestyle='--', zorder=1)
    ax.set_xlabel(time_col if time_col in bike_df.columns
                  else 'frame index', fontsize=9)
    ax.set_ylabel('[km/h]', fontsize=9)
    ax.set_ylim([-25, 35])
    ax.set_title('Speed decomposition vs time', fontsize=10)
    ax.legend(fontsize=7, loc='upper right', framealpha=0.85, ncol=2)
    ax.grid(True, alpha=0.25)

    # =========================================================================
    # (1,1) ACCELERATION DECOMPOSITION vs TIME
    # =========================================================================
    ax = axes[1, 1]
    # _shade_seg_regions(ax)
    _shade_bike_lane(ax, None)

    a_raw  = bike_df['a_ekf'].to_numpy(dtype=float) \
             if 'a_ekf' in bike_df.columns else np.full(len(t), np.nan)
    s_ddot = bike_df['s_ddot'].to_numpy(dtype=float) \
             if 's_ddot' in bike_df.columns else np.full(len(t), np.nan)
    d_ddot = bike_df['d_ddot'].to_numpy(dtype=float) \
             if 'd_ddot' in bike_df.columns else np.full(len(t), np.nan)

    ax.plot(t, a_raw, color='black', linewidth=1.6,
            zorder=4, label='a_ekf [m/s²]')

    for sk in matched_segs:
        mask = np.array([s == sk for s in seg_ids])
        if not mask.any():
            continue
        col = _row_col(sk)
        ax.plot(np.where(mask, t, np.nan),
                np.where(mask, s_ddot, np.nan),
                color=col, linewidth=1.6, zorder=5, label=f's_ddot [{sk}]')
        ax.plot(np.where(mask, t, np.nan),
                np.where(mask, d_ddot, np.nan),
                color=col, linewidth=1.2, linestyle='--', zorder=5,
                label=f'd_ddot [{sk}]')
    if unm.any():
        ax.plot(np.where(unm, t, np.nan), np.where(unm, s_ddot, np.nan),
                color='#BBBBBB', linewidth=1.0, zorder=3)
        ax.plot(np.where(unm, t, np.nan), np.where(unm, d_ddot, np.nan),
                color='#BBBBBB', linewidth=0.8, linestyle='--', zorder=3)

    ax.axhline(0, color='#999999', linewidth=0.8, linestyle='--', zorder=1)
    ax.set_xlabel(time_col if time_col in bike_df.columns
                  else 'frame index', fontsize=9)
    ax.set_ylabel('[m/s²]', fontsize=9)
    ax.set_ylim([-4, 4])
    ax.set_title('Acceleration decomposition vs time', fontsize=10)
    ax.legend(fontsize=7, loc='upper right', framealpha=0.85, ncol=2)
    ax.grid(True, alpha=0.25)

    # ── Bottom legend: segment colours + indicators ───────────────────────────
    bottom_handles = [
        Line2D([0], [0], color=_row_col(sk), linewidth=2.5, label=sk)
        for sk in matched_segs
    ] + [
        mpatches.Patch(facecolor='cyan',    alpha=0.3, label='in bike lane'),
        mpatches.Patch(facecolor='#BBBBBB', hatch='//',
                       alpha=0.4, label='poor / fallback'),
    ]
    fig.legend(handles=bottom_handles,
               loc='lower center',
               ncol=min(len(bottom_handles), 5),
               fontsize=8, framealpha=0.9,
               bbox_to_anchor=(0.5, -0.02))

    # plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    # plt.show()
    return fig