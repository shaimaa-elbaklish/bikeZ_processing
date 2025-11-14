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
import folium

import numpy as np

from pyproj import Transformer
from shapely.geometry import LineString
from collections import defaultdict
from shapely.ops import transform
from scipy.interpolate import splev

###############################################################################
# CONSTANTS: Projection
###############################################################################
transformer_xy2056_to_lonlat = Transformer.from_crs("EPSG:2056", "EPSG:4326", always_xy=True)
project_xy2056_to_lonlat = lambda x, y, z=None: transformer_xy2056_to_lonlat.transform(x, y)


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
                                       linedashed=False, add_layer_control=False):
    df = traj_df.copy()
    if 'lat_ekf' not in df.columns or 'lon_ekf' not in df.columns:
        df["lon_ekf"], df["lat_ekf"] = project_xy2056_to_lonlat(df["x_act_ekf"].values, df["y_act_ekf"].values)
    if add_layer_control:
        fg = folium.FeatureGroup(name="Trajectories", show=False)  # hide by default
    for bike_id in df['veh_id'].unique():
        traj = df[(df["veh_id"] == bike_id)]
        plot_line_latlon(fg if add_layer_control else m, 
                         traj[['lat_ekf', 'lon_ekf']].values.tolist(), f"Bicycle {bike_id}", 
                         linecolor, lineweight, linealpha, linedashed, start_point=False)
        
    if add_layer_control:
        fg.add_to(m)
        folium.LayerControl(collapsed=False).add_to(m)
    return m
    
    


