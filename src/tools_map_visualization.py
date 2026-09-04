"""
TITLE
-------------------------------------------
Authors:        Shaimaa K. El-Baklish
Organization:   ETH Zürich, Switzerland, IVT - Institute for Transportation Planning and Systems
Development:    2025-2026
Submitted to:   JOURNAL
-------------------------------------------
"""

###############################################################################
# IMPORTS
###############################################################################
import os
import sys
import folium

import numpy as np

from shapely.geometry import LineString
from collections import defaultdict
from shapely.ops import transform
from scipy.interpolate import splev
# from shapely.ops import transform as shp_transform

from tools_utils import _PROJ_2056_TO_LONLAT 
from tools_utils import _is_axis_entry, _build_color_map, _local_to_latlon
from tools_utils import _spline_xy_to_latlon, _spline_xy_variable_offset_to_latlon
from tools_utils import w_bike_at, w_bike_label
from tools_utils import _polygon_rings_latlon, _is_ring_entry, _ring_arc_latlon

###############################################################################
# CONSTANTS: Projection
###############################################################################
project_xy2056_to_lonlat = lambda x, y, z=None: _PROJ_2056_TO_LONLAT.transform(x, y)


###############################################################################
# METHODS
###############################################################################
def create_swisstopo_map(center_lat, center_lon, zoom_start=20, add_layer_control=True):
    m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom_start, tiles=None, control_scale=True)
    # Add swisstopo basemap
    folium.TileLayer(
        tiles="https://wmts.geo.admin.ch/1.0.0/ch.swisstopo.pixelkarte-farbe/default/current/3857/{z}/{x}/{y}.jpeg",
        attr="© swisstopo / geo.admin.ch",
        name="swisstopo.pixelkarte-farbe",
        overlay=False,
        control=True,
        max_zoom=25,
        min_zoom=0,
        subdomains=None,
        tms=False
    ).add_to(m)
    # Optional: add orthophoto as another layer
    folium.TileLayer(
        tiles="https://wmts.geo.admin.ch/1.0.0/ch.swisstopo.swissimage/default/current/3857/{z}/{x}/{y}.jpeg",
        attr="© swisstopo / geo.admin.ch",
        name="swisstopo.swissimage",
        overlay=True,
        control=True,
        max_zoom=25
    ).add_to(m)
    if add_layer_control:
        # Add a layer control so you can toggle
        folium.LayerControl().add_to(m)
    return m


def create_gis_zh_map(center_lat, center_lon, zoom_start=20, add_layer_control=True):
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom_start,
        tiles=None,
        control_scale=True,
        max_zoom=25,          # <-- allow the map itself to zoom further
    )

    folium.raster_layers.WmsTileLayer(
        url="https://wms.zh.ch/OGDOrthoZH",
        layers="ortho_s_2024",
        fmt="image/jpeg",
        transparent=False,
        version="1.3.0",
        attr="© Kanton Zürich, GIS-ZH",
        name="Orthofoto ZH 2024/25",
        overlay=False,
        control=True,
        max_zoom=25,           # <-- allow this layer to render past zoom 18
    ).add_to(m)

    if add_layer_control:
        folium.LayerControl().add_to(m)
    return m


def plot_line_xy_2056(m, x_pts, y_pts, label, linecolor='black', lineweight=5, 
                      linealpha=0.8, linedashed=True, start_point=False):
    line_xy = LineString(np.column_stack((x_pts, y_pts)))
    centerline = transform(project_xy2056_to_lonlat, line_xy)
    centerline = [(lat, lon) for lon, lat in centerline.coords]
    folium.PolyLine(
        locations=centerline,
        color=linecolor,
        weight=lineweight,
        opacity=linealpha,
        dash_array="10, 20" if linedashed else None,
        tooltip=label
    ).add_to(m)
    if start_point:
        folium.Marker(
            location=centerline[0],
            icon=folium.Icon(color=linecolor),
            tooltip=f"START {label}"
        ).add_to(m)
    return m


def plot_spline_xy_2056(m, spl_rep, label, linecolor='black', lineweight=5, 
                        linealpha=0.8, linedashed=True, num_pts=50, start_point=False):
    tck = spl_rep[0]
    x_spline, y_spline = splev(np.linspace(0, 1, num_pts), tck)
    line_xy = LineString(np.column_stack((x_spline, y_spline)))
    
    centerline = transform(project_xy2056_to_lonlat, line_xy)
    centerline = [(lat, lon) for lon, lat in centerline.coords]
    folium.PolyLine(
        locations=centerline,
        color=linecolor,
        weight=lineweight,
        opacity=linealpha,
        dash_array="10, 20" if linedashed else None,
        tooltip=label
    ).add_to(m)
    if start_point:
        folium.Marker(
            location=centerline[0],
            icon=folium.Icon(color=linecolor),
            tooltip=f"START {label}"
        ).add_to(m)
    return m


def plot_line_latlon(m, latlon_pts, label, linecolor='black', lineweight=5, 
                     linealpha=0.8, linedashed=True, start_point=False):
    folium.PolyLine(
        locations=latlon_pts,
        color=linecolor,
        weight=lineweight,
        opacity=linealpha,
        dash_array="10, 20" if linedashed else None,
        tooltip=label
    ).add_to(m)
    if start_point:
        folium.Marker(
            location=latlon_pts[0],
            icon=folium.Icon(color=linecolor),
            tooltip=f"START {label}"
        ).add_to(m)
    return m
        

def plot_all_centerlines_splines_xy_2056(m, splines_dict, colors_dict=None, linedashed=True, add_layer_control=True):
    if colors_dict is None:
        colors_dict = {
            'N': 'lightblue', 'S': 'orange', 'W': 'green', 'E': 'pink',
        }
    splines_by_start = defaultdict(dict)
    for key, spline in splines_dict.items():
        start, end = key.split("_2_")
        splines_by_start[start].update({key: spline})
    splines_by_start = dict(splines_by_start)
    
    for start_label, spline_dict in splines_by_start.items():
        fg = folium.FeatureGroup(name=start_label, show=False)  # hide by default
        color = colors_dict.get(start_label, "gray")
        for traj_label, spline in spline_dict.items():
            plot_spline_xy_2056(
                fg,
                spline,
                label=traj_label,
                linecolor=color,
                linedashed=linedashed,
                start_point=True
            )    
        fg.add_to(m)
    
    if add_layer_control:
        # Add layer control to toggle visibility
        folium.LayerControl(collapsed=False).add_to(m)
    return m
        

def plot_bicycles_trajectories_xy_2056(m, traj_df, linecolor='black', 
                                       lineweight=5, linealpha=0.8, 
                                       linedashed=False, add_layer_control=False,
                                       ekf=True):
    df = traj_df.copy()
    if add_layer_control:
        fg = folium.FeatureGroup(name="Trajectories", show=False)  # hide by default
    if ekf:
        if 'lat_ekf' not in df.columns or 'lon_ekf' not in df.columns:
            df["lon_ekf"], df["lat_ekf"] = project_xy2056_to_lonlat(df["x_act_ekf"].values, df["y_act_ekf"].values)
        for bike_id in df['veh_id'].unique():
            traj = df[(df["veh_id"] == bike_id)]
            plot_line_latlon(fg if add_layer_control else m, 
                             traj[['lat_ekf', 'lon_ekf']].values.tolist(), f"Bicycle {bike_id}", 
                             linecolor, lineweight, linealpha, linedashed, start_point=False)
    else:
        for bike_id in df['veh_id'].unique():
            traj = df[(df["veh_id"] == bike_id)]
            plot_line_latlon(fg if add_layer_control else m, 
                             traj[['lat', 'lon']].values.tolist(), f"Bicycle {bike_id}", 
                             linecolor, lineweight, linealpha, linedashed, start_point=False)
        
    if add_layer_control:
        fg.add_to(m)
        folium.LayerControl(collapsed=False).add_to(m)
    return m


# =============================================================================
# PROJECTION HELPERS
# =============================================================================
def _kml_geom_to_latlon(geometry):
    """WGS84 KML geometry → [(lat, lon), …]."""
    coords = list(geometry.coords)
    return [(c[1], c[0]) for c in coords]
 
 
# =============================================================================
# CREATE REGISTRY MAP
# =============================================================================
def create_registry_map(geometry_store, segment_registry, movement_registry,
                         gdf_swisstopo=None,
                         center_lat=None, center_lon=None,
                         zoom_start=19,
                         base_map_src='swisstopo',
                         save_path=None):
    """
    Visualize geometry_store, segment_registry, and movement_registry
    on an interactive folium / swisstopo map.
 
    If center_lat/center_lon are None, the map is centred automatically
    from the first lane geometry spline midpoint.
 
    Layer groups (all toggleable via LayerControl):
        KML — stop lines, yield lines, bike lane boundaries, intersection box
        Axis: {name}  — road centerline + s_change markers
        Lane: {seg_key} — validity polygon + spline + s_change
        Turn: {app} → {dep} — validity polygon + turn spline
        Movement: {key} — full path
 
    Parameters
    ----------
    geometry_store   : dict
    segment_registry : dict
    movement_registry: dict
    gdf_swisstopo    : GeoDataFrame
    center_lat/lon   : float | None
    zoom_start       : int
    base_map_src     : str | 'swisstopo' — sets base map to use ('swisstopo', 'gis-zh')
    save_path        : str | None — saves HTML if provided
 
    Returns
    -------
    m : folium.Map
    """
    x_offset  = geometry_store['x_offset']
    y_offset  = geometry_store['y_offset']
    skip      = {'x_offset', 'y_offset'}
    color_map = _build_color_map(geometry_store)
 
    # ── Auto-centre from first lane axis midpoint ─────────────────────────────
    if center_lat is None or center_lon is None:
        for k, v in geometry_store.items():
            if k in skip or v.get('s_stop') is None:
                continue
            tck, unew, _ = v['spline']
            x_m, y_m     = splev(unew[len(unew) // 2], tck)
            lon_c, lat_c = _PROJ_2056_TO_LONLAT.transform(x_m + x_offset, y_m + y_offset)
            center_lat, center_lon = lat_c, lon_c
            break
    
    if base_map_src == 'gis-zh':
        m = create_gis_zh_map(center_lat, center_lon,
                                  zoom_start=zoom_start,
                                  add_layer_control=False)
    else:
        m = create_swisstopo_map(center_lat, center_lon,
                                  zoom_start=zoom_start,
                                  add_layer_control=False)
    
    
 
    # =========================================================================
    # GROUP 0 — KML overlays
    # =========================================================================
    if gdf_swisstopo is not None:
        DIRECTION_SUFFIXES = ('_NB', '_SB', '_EB', '_WB', '_NE', '_SW')
     
        fg_poly  = folium.FeatureGroup(name='KML — Intersection area',  show=False)
        fg_stop  = folium.FeatureGroup(name='KML — Stop lines',         show=True)
        fg_yield = folium.FeatureGroup(name='KML — Yield lines',        show=True)
        fg_bike  = folium.FeatureGroup(name='KML — Bike lane boundaries', show=False)
     
        for _, row in gdf_swisstopo.iterrows():
            desc = row['Description']
            geom = row.geometry
     
            if desc == 'Intersection_Area':
                coords = _kml_geom_to_latlon(geom.exterior)
                folium.Polygon(
                    locations=coords,
                    color='gold', weight=2,
                    fill=True, fill_color='yellow', fill_opacity=0.15,
                    tooltip='Intersection area',
                ).add_to(fg_poly)
     
            elif desc.endswith('_Stop'):
                folium.PolyLine(
                    locations=_kml_geom_to_latlon(geom),
                    color='red', weight=3, opacity=0.9,
                    tooltip=desc,
                ).add_to(fg_stop)
                # Label at midpoint
                coords = list(geom.coords)
                mid    = coords[len(coords) // 2]
                folium.Marker(
                    location=(mid[1], mid[0]),
                    tooltip=desc,
                    icon=folium.DivIcon(
                        html=f'<div style="font-size:9px;color:darkred;'
                             f'font-weight:bold;white-space:nowrap">{desc}</div>',
                        icon_size=(120, 16),
                    ),
                ).add_to(fg_stop)
     
            elif desc.endswith('_Yield'):
                folium.PolyLine(
                    locations=_kml_geom_to_latlon(geom),
                    color='darkorange', weight=3, opacity=0.9,
                    dash_array='8 5',
                    tooltip=desc,
                ).add_to(fg_yield)
                coords = list(geom.coords)
                mid    = coords[len(coords) // 2]
                folium.Marker(
                    location=(mid[1], mid[0]),
                    tooltip=desc,
                    icon=folium.DivIcon(
                        html=f'<div style="font-size:9px;color:darkorange;'
                             f'font-weight:bold;white-space:nowrap">{desc}</div>',
                        icon_size=(120, 16),
                    ),
                ).add_to(fg_yield)
     
            elif any(desc.endswith(s) for s in DIRECTION_SUFFIXES):
                folium.PolyLine(
                    locations=_kml_geom_to_latlon(geom),
                    color='cyan', weight=3, opacity=0.85,
                    tooltip=desc,
                ).add_to(fg_bike)
     
        fg_poly.add_to(m)
        fg_stop.add_to(m)
        fg_yield.add_to(m)
        fg_bike.add_to(m)
    
    # =========================================================================
    # GROUP 0b — Bike Lane Corridors
    # =========================================================================
    add_bike_lane_layer_folium(m, geometry_store, segment_registry,
                                x_offset, y_offset)
    
    # =========================================================================
    # GROUP 0c — Car Lane Corridors
    # =========================================================================
    add_car_lane_layer_folium(m, geometry_store, segment_registry,
                           x_offset, y_offset)
    
    # =========================================================================
    # GROUP 0c — Intersection area polygons (from geometry_store)
    # One FeatureGroup per intersection_area_* key.
    # These are the computed polygons (built from s_change normal lines),
    # distinct from the KML Intersection_Area overlay above.
    # =========================================================================
    int_area_keys = [k for k in geometry_store
                     if k.startswith('intersection_area')]
 
    _IA_COLORS = ['#FF6B6B', '#FFD93D', '#6BCB77', '#4D96FF',
                  '#C77DFF', '#FF9F1C', '#2EC4B6', '#E71D36']
 
    for ia_idx, ia_key in enumerate(sorted(int_area_keys)):
        ia_poly = geometry_store[ia_key]
        ia_col  = _IA_COLORS[ia_idx % len(_IA_COLORS)]
        label   = ia_key.replace('intersection_area_', '')
 
        fg_ia = folium.FeatureGroup(
            name=f'Intersection box: {label}', show=True
        )
        try:
            px, py      = ia_poly.exterior.xy
            coords_ia   = _local_to_latlon(px, py, x_offset, y_offset)
            folium.Polygon(
                locations=coords_ia,
                color=ia_col, weight=2.5, opacity=0.9,
                fill=True, fill_color=ia_col, fill_opacity=0.20,
                tooltip=f'Intersection box: {label}',
                dash_array=None,
            ).add_to(fg_ia)
            # Label at centroid
            cx_local = float(ia_poly.centroid.x)
            cy_local = float(ia_poly.centroid.y)
            lon_c, lat_c = _PROJ_2056_TO_LONLAT.transform(
                cx_local + x_offset, cy_local + y_offset
            )
            folium.Marker(
                location=(lat_c, lon_c),
                tooltip=f'{label}  area={ia_poly.area:.0f} m²',
                icon=folium.DivIcon(
                    html=(f'<div style="font-size:10px;font-weight:bold;'
                          f'color:{ia_col};white-space:nowrap;'
                          f'text-shadow:0 0 3px white">{label}</div>'),
                    icon_size=(160, 20),
                    icon_anchor=(80, 10),
                ),
            ).add_to(fg_ia)
        except Exception:
            pass
        fg_ia.add_to(m)
 
    # =========================================================================
    # GROUP 1 — Road axes (one FeatureGroup per geometry key)
    # =========================================================================
    for geom_key, geo in geometry_store.items():
        if not _is_axis_entry(geom_key, geo):
            continue
        if geo.get('s_stop') is None and not _is_ring_entry(geo):
            continue
 
        tck, unew, cum_dist = geo['spline']
        L                   = geo['total_length']
        s_change            = geo.get('s_change')
        col                 = color_map[geom_key]
 
        fg = folium.FeatureGroup(
            name=f'Axis: {geom_key}', show=True
        )
 
        # Full centerline
        latlon = _spline_xy_to_latlon(
            tck, unew, cum_dist, 0, L, x_offset, y_offset
        )
        folium.PolyLine(
            locations=latlon,
            color=col, weight=4, opacity=0.7,
            tooltip=f'{geom_key}  (positive_dir={geo.get("positive_dir")}  L={L:.1f}m)',
        ).add_to(fg)
 
        # s=0 start marker
        folium.CircleMarker(
            location=latlon[0],
            radius=5, color='red', fill=True, fill_color='red',
            tooltip=f'{geom_key} s=0 (start)',
        ).add_to(fg)
 
        # s=L end marker
        folium.CircleMarker(
            location=latlon[-1],
            radius=5, color='black', fill=True, fill_color='black',
            tooltip=f'{geom_key} s=L={L:.1f}m (end)',
        ).add_to(fg)
 
        # Primary s_change marker
        if geo.get('s_change') is not None:
            latlon_sc = _spline_xy_to_latlon(tck, unew, cum_dist, s_change, s_change,
                                 x_offset, y_offset, n=1)[0]
            folium.CircleMarker(
                location=latlon_sc,
                radius=7, color=col, fill=True, fill_color='white',
                fill_opacity=0.9, weight=3,
                tooltip=f'{geom_key}  s_change={s_change:.2f} m',
            ).add_to(fg)
 
        # Extra s_change_* markers (secondary junctions)
        for key in [k for k in geo if k.startswith('s_')
                    and k not in ('s_stop', 's_yield', 's_change')]:
            sc_val     = geo[key]
            latlon_ec  = _spline_xy_to_latlon(tck, unew, cum_dist, sc_val, sc_val,
                                 x_offset, y_offset, n=1)[0]
            folium.CircleMarker(
                location=latlon_ec,
                radius=6, color=col, fill=True, fill_color='yellow',
                fill_opacity=0.9, weight=2,
                tooltip=f'{geom_key}  {key}={sc_val:.2f} m',
            ).add_to(fg)
 
        fg.add_to(m)
 
    # =========================================================================
    # GROUP 2 — Lane segments
    # =========================================================================
    for seg_key, entry in segment_registry.items():
        if entry['type'] not in ('lane', 'ring'):
            continue
 
        geom_key            = entry['geometry_key']
        is_forward          = entry['is_forward']
        d_max               = entry.get('d_max', '?')
        validity_poly       = entry.get('validity_polygon')
        col                 = color_map.get(geom_key, '#888888')
 
        geo                 = geometry_store[geom_key]
        tck, unew, cum_dist = geo['spline']
        L                   = geo['total_length']
        s_change            = geo.get('s_change')
 
        fg = folium.FeatureGroup(name=f'Lane: {seg_key}', show=False)
 
        # Validity polygon
        for ring in _polygon_rings_latlon(validity_poly, x_offset, y_offset):
            folium.Polygon(
                locations=ring,
                color=col, weight=1, opacity=0.5,
                fill=True, fill_color=col, fill_opacity=0.12,
                tooltip=f'{seg_key}  d_max={d_max} m',
            ).add_to(fg)
 
        # Full centerline — direction encoded by dash style
        dash = None if is_forward else '6 4'
        latlon = _spline_xy_to_latlon(
            tck, unew, cum_dist, 0, L, x_offset, y_offset
        )
        fwd_str = 'fwd' if is_forward else 'rev'
        folium.PolyLine(
            locations=latlon,
            color=col, weight=5, opacity=0.75,
            dash_array=dash,
            tooltip=f'{seg_key} [{fwd_str}]',
        ).add_to(fg)
 
        # s_change boundary marker
        if s_change is not None:
            latlon_sc = _spline_xy_to_latlon(tck, unew, cum_dist, s_change, s_change,
                                 x_offset, y_offset, n=1)[0]
            folium.CircleMarker(
                location=latlon_sc,
                radius=6, color=col, fill=True, fill_color='white',
                fill_opacity=1.0, weight=3,
                tooltip=f'{seg_key}  s_change={s_change:.2f} m',
            ).add_to(fg)
 
        # Extra s_change_* markers
        for key in [k for k in geo if k.startswith('s_')
                    and k not in ('s_stop', 's_yield', 's_change')]:
            sc_val    = geo[key]
            latlon_ec = _spline_xy_to_latlon(tck, unew, cum_dist, sc_val, sc_val,
                                 x_offset, y_offset, n=1)[0]
            folium.CircleMarker(
                location=latlon_ec,
                radius=5, color=col, fill=True, fill_color='yellow',
                fill_opacity=1.0, weight=2,
                tooltip=f'{seg_key}  {key}={sc_val:.2f} m',
            ).add_to(fg)
 
        fg.add_to(m)
 
    # =========================================================================
    # GROUP 3 — Turn segments
    # =========================================================================
    for turn_key, turn_entry in segment_registry.items():
        if turn_entry['type'] != 'turn':
            continue
 
        app_seg      = turn_entry['approach_seg']
        dep_seg      = turn_entry['departure_seg']
        d_max        = turn_entry.get('d_max', '?')
        validity_poly= turn_entry.get('validity_polygon')
 
        # Color from approach segment's geometry
        app_geom_key = segment_registry[app_seg]['geometry_key']
        col          = color_map.get(app_geom_key, '#888888')
 
        geo          = geometry_store[turn_key]
        tck, unew, cum_dist = geo['spline']
        L            = geo['total_length']
        method       = geo.get('method', '?')
        dash         = None if method == 'clothoid' else '5 4'
 
        label = f'{app_seg} → {dep_seg}'
        fg    = folium.FeatureGroup(name=f'Turn: {label}', show=False)
 
        # Validity polygon
        for ring in _polygon_rings_latlon(validity_poly, x_offset, y_offset):
            folium.Polygon(
                locations=ring,
                color=col, weight=1, opacity=0.4,
                fill=True, fill_color=col, fill_opacity=0.10,
                tooltip=f'{turn_key}  d_max={d_max} m',
            ).add_to(fg)
 
        # Turn spline
        latlon = _spline_xy_to_latlon(
            tck, unew, cum_dist, 0, L, x_offset, y_offset
        )
        folium.PolyLine(
            locations=latlon,
            color=col, weight=4, opacity=0.85,
            dash_array=dash,
            tooltip=f'{label}  [{method}]  L={L:.1f}m',
        ).add_to(fg)
 
        # Start / end markers
        folium.CircleMarker(
            location=latlon[0], radius=5,
            color=col, fill=True, fill_color='white',
            fill_opacity=1.0, weight=3,
            tooltip=f'START: {label}',
        ).add_to(fg)
        folium.CircleMarker(
            location=latlon[-1], radius=5,
            color=col, fill=True, fill_color=col,
            fill_opacity=1.0, weight=3,
            tooltip=f'END: {label}',
        ).add_to(fg)
 
        fg.add_to(m)
 
    # =========================================================================
    # GROUP 4 — Movements
    # Lane segments trimmed to their relevant s_change boundary so the
    # plotted path shows only the portion the cyclist actually traverses.
    # s_change keys are read from the adjacent turn segment entry.
    # =========================================================================
    for mov_key, sequence in movement_registry.items():
        app_seg      = sequence[0][0]
        app_geom_key = segment_registry[app_seg]['geometry_key']
        col          = color_map.get(app_geom_key, '#888888')
 
        fg = folium.FeatureGroup(
            name=f'Movement: {mov_key}', show=False
        )
 
        # Boundaries come from the ADJACENT turn, not the first turn in the
        # sequence. A through-ring chain has TWO turns, so "first" would pick
        # the entry turn and then look up its departure key on the departure
        # leg's geometry, where it does not exist.
        def _neighbour_turn(i, step):
            """The turn immediately before (-1) or after (+1) position i."""
            j = i + step
            if 0 <= j < len(sequence):
                sk = sequence[j][0]
                if segment_registry[sk]['type'] == 'turn':
                    return segment_registry[sk]
            return None
        
        first_latlon = None
        latlon       = None
        
        for i, (seg_key, role) in enumerate(sequence):
            entry               = segment_registry[seg_key]
            geom_key            = entry['geometry_key']
            geo                 = geometry_store[geom_key]
            tck, unew, cum_dist = geo['spline']
            L                   = geo['total_length']
            is_fwd              = entry['is_forward']
            arc_txt             = ''
        
            if entry['type'] == 'ring':
                # The arc actually ridden, not the whole circle. Both bounds
                # are gates on the ring's own geometry, and the adjacent turn
                # defs already name them:
                #   entry turn  departure_s_change_key -> 's_entry_<leg>'
                #   exit  turn  approach_s_change_key  -> 's_exit_<leg>'
                t_in  = _neighbour_turn(i, -1)
                t_out = _neighbour_turn(i, +1)
                k_in  = t_in.get('departure_s_change_key') if t_in else None
                k_out = t_out.get('approach_s_change_key') if t_out else None
                s_in  = geo.get(k_in)  if k_in  else None
                s_out = geo.get(k_out) if k_out else None
        
                if s_in is None or s_out is None:
                    latlon = _spline_xy_to_latlon(
                        tck, unew, cum_dist, 0.0, L, x_offset, y_offset)
                else:
                    latlon = _ring_arc_latlon(
                        geo, s_in, s_out, x_offset, y_offset)
                    arc_txt = f'  arc={(s_out - s_in) % L:.1f} m'
        
            else:
                if entry['type'] == 'turn':
                    s_start, s_end = 0.0, L
                else:
                    t   = _neighbour_turn(i, +1 if role == 'approach' else -1)
                    key = (t or {}).get(
                        'approach_s_change_key' if role == 'approach'
                        else 'departure_s_change_key', 's_change')
                    s_bnd = geo.get(key, geo.get('s_change'))
                    if s_bnd is None:
                        s_start, s_end = 0.0, L
                    elif role == 'approach':
                        s_start, s_end = (0.0, s_bnd) if is_fwd else (s_bnd, L)
                    else:
                        s_start, s_end = (s_bnd, L) if is_fwd else (0.0, s_bnd)
        
                latlon = _spline_xy_to_latlon(
                    tck, unew, cum_dist, s_start, s_end, x_offset, y_offset)
        
            # Reverse reverse-direction segments so latlon runs in travel order.
            # A ring is always is_forward=True, and _ring_arc_latlon already
            # returns travel order, so this leaves it alone.
            if not is_fwd:
                latlon = latlon[::-1]
        
            folium.PolyLine(
                locations=latlon,
                color=col, weight=6, opacity=0.85,
                tooltip=f'{mov_key} — {seg_key} [{role}]{arc_txt}',
            ).add_to(fg)
         
            if first_latlon is None:
                first_latlon = latlon[0]
 
        # Start / end markers
        if first_latlon:
            folium.Marker(
                location=first_latlon,
                tooltip=f'START: {mov_key}',
                icon=folium.Icon(color='green', icon='play', prefix='fa'),
            ).add_to(fg)
        if latlon:
            folium.Marker(
                location=latlon[-1],
                tooltip=f'END: {mov_key}',
                icon=folium.Icon(color='red', icon='stop', prefix='fa'),
            ).add_to(fg)
 
        fg.add_to(m)
 
    # ── Layer control ─────────────────────────────────────────────────────────
    folium.LayerControl(collapsed=False).add_to(m)
 
    # ── Save ──────────────────────────────────────────────────────────────────
    if save_path:
        m.save(save_path)
        print(f"Map saved to {save_path}")
 
    return m


def add_bike_lane_layer_folium(m, geometry_store, segment_registry,
                                x_offset, y_offset,
                                n_pts=50, color='#00CC96', show=True):
    """
    Adds a 'Bike lanes (computed)' FeatureGroup to an existing folium map,
    drawing each lane segment's bike lane as a filled corridor between its
    near and far boundary splines — mirrors add_bike_lane_boundaries_plotly
    but in lat/lon for folium.

    Parameters
    ----------
    m : folium.Map — map to add the layer to (mutated in place)
    geometry_store, segment_registry : same as create_registry_map
    x_offset, y_offset : from geometry_store
    n_pts  : samples along each bike lane's s_domain
    color  : corridor fill/line color
    show   : whether the layer is visible by default

    Returns
    -------
    m : same map, for chaining
    """
    fg_bike_computed = folium.FeatureGroup(
        name='Bike lanes (computed)', show=show
    )

    for seg_key, entry in segment_registry.items():
        if entry['type'] != 'lane':
            continue

        bike_lane = entry.get('bike_lane')
        if bike_lane is None or 'd_boundary_spline' not in bike_lane:
            continue

        geom_key = entry['geometry_key']
        geo      = geometry_store[geom_key]
        tck, unew, cum_dist = geo['spline']

        d_bnd_spl    = bike_lane['d_boundary_spline']
        w_bike_lbl   = w_bike_label(bike_lane)
        side         = bike_lane['side']
        s_min, s_max = bike_lane['s_domain']

        s_bl  = np.linspace(s_min, s_max, n_pts)
        d_bnd = d_bnd_spl(s_bl)
        d_far = d_bnd + side * w_bike_at(bike_lane, s_bl)

        latlon_bnd = _spline_xy_variable_offset_to_latlon(
            tck, unew, cum_dist, s_bl, d_bnd, x_offset, y_offset
        )
        latlon_far = _spline_xy_variable_offset_to_latlon(
            tck, unew, cum_dist, s_bl, d_far, x_offset, y_offset
        )

        # Filled corridor band
        band = latlon_bnd + latlon_far[::-1] + [latlon_bnd[0]]
        folium.Polygon(
            locations=band,
            color=color, weight=1, opacity=0.6,
            fill=True, fill_color=color, fill_opacity=0.20,
            tooltip=f'{seg_key} bike lane (w={w_bike_lbl})',
        ).add_to(fg_bike_computed)

        # Near boundary line (crisper edge)
        folium.PolyLine(
            locations=latlon_bnd,
            color=color, weight=2, opacity=0.9,
            dash_array='4 3',
            tooltip=f'{seg_key} bike lane boundary',
        ).add_to(fg_bike_computed)

    fg_bike_computed.add_to(m)
    return m


def add_car_lane_layer_folium(m, geometry_store, segment_registry,
                               x_offset, y_offset,
                               n_pts=50, color='#636EFA', show=False):
    """
    Adds a 'Car lanes (defined)' FeatureGroup to an existing folium map,
    shading each car lane in segment_registry[...]['car_lane_d_bnd'].

    Each lane_idx entry can be EITHER:
        (d_lb, d_ub)                  — constant-width tuple, in metres,
                                         NATIVE SPLINE COORDINATES (the same
                                         raw left-normal frame used by
                                         bike lane boundary splines — NOT the
                                         travel-direction-relative frame
                                         used by d_left/d_right; no sign
                                         flip needed regardless of
                                         is_forward). Band is reconstructed
                                         by sweeping the centerline.
    OR
        {'polygon': Polygon, ...}     — dict carrying an actual KML-derived
                                         footprint (local XY, 2056-offset
                                         frame, as produced by
                                         add_car_lane_boundaries). Plotted
                                         directly — no reconstruction from
                                         d bounds, so it renders correctly
                                         even for lanes at an angle to the
                                         centerline. If 'polygon' is missing
                                         or empty, falls back to 'd_bounds'
                                         (d_lb(s), d_ub(s)) sampled over the
                                         segment's s-domain, if present.

    Parameters
    ----------
    m : folium.Map — map to add the layer to (mutated in place)
    geometry_store, segment_registry : same as create_registry_map
    x_offset, y_offset : from geometry_store
    n_pts  : samples along each segment's s-domain (constant/function cases)
    color  : lane fill/line color
    show   : whether the layer is visible by default
    Returns
    -------
    m : same map, for chaining
    """
    fg_car_lanes = folium.FeatureGroup(name='Car lanes (defined)', show=show)
    for seg_key, entry in segment_registry.items():
        if entry['type'] != 'lane':
            continue
        car_lane_d_bnd = entry.get('car_lane_d_bnd')
        if not car_lane_d_bnd:
            continue
        geom_key            = entry['geometry_key']
        geo                 = geometry_store[geom_key]
        tck, unew, cum_dist = geo['spline']
        L                   = geo['total_length']
        s_change            = geo.get('s_change')
        # same s-domain rule as build_segment_registry (choose longer arm)
        if s_change is not None:
            s_start, s_end = (0.0, s_change) if s_change >= L - s_change \
                              else (s_change, L)
        else:
            s_start, s_end = 0.0, L

        for lane_idx, lane_val in car_lane_d_bnd.items():
            tooltip_prefix = f'{seg_key} lane {lane_idx}'

            # ── Option A: dict carrying a KML-derived polygon ──────────────
            if isinstance(lane_val, dict):
                poly_local = lane_val.get('polygon')
                if poly_local is not None and not poly_local.is_empty:
                    latlon = _local_to_latlon(
                        *poly_local.exterior.xy, x_offset, y_offset)
                    folium.Polygon(
                        locations=latlon,
                        color=color, weight=1, opacity=0.6,
                        fill=True, fill_color=color, fill_opacity=0.15,
                        tooltip=f'{tooltip_prefix}  (KML polygon)',
                    ).add_to(fg_car_lanes)
                    continue

                # No polygon stored — fall back to d_bounds functions, if any.
                d_lb_fn, d_ub_fn = lane_val.get('d_bounds', (None, None))
                if d_lb_fn is None or d_ub_fn is None:
                    print(f"  [WARN] {tooltip_prefix}: no polygon and no "
                          f"resolved d_bounds — skipped")
                    continue
                s_min, s_max = lane_val.get('s_domain', (s_start, s_end))
                s_vals   = np.linspace(s_min, s_max, n_pts)
                d_lb_arr = d_lb_fn(s_vals)
                d_ub_arr = d_ub_fn(s_vals)
                tooltip  = f'{tooltip_prefix}  (d(s) bounds)'

            # ── Option B: plain (d_lb, d_ub) constant tuple ────────────────
            else:
                d_lb, d_ub = lane_val
                s_vals   = np.linspace(s_start, s_end, n_pts)
                d_lb_arr = np.full_like(s_vals, d_lb)
                d_ub_arr = np.full_like(s_vals, d_ub)
                tooltip  = f'{tooltip_prefix}  d∈[{d_lb:.2f}, {d_ub:.2f}] m'

            latlon_lb = _spline_xy_variable_offset_to_latlon(
                tck, unew, cum_dist, s_vals, d_lb_arr, x_offset, y_offset
            )
            latlon_ub = _spline_xy_variable_offset_to_latlon(
                tck, unew, cum_dist, s_vals, d_ub_arr, x_offset, y_offset
            )
            band = latlon_lb + latlon_ub[::-1] + [latlon_lb[0]]
            folium.Polygon(
                locations=band,
                color=color, weight=1, opacity=0.6,
                fill=True, fill_color=color, fill_opacity=0.15,
                tooltip=tooltip,
            ).add_to(fg_car_lanes)

    fg_car_lanes.add_to(m)
    return m

