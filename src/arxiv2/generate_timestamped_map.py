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
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from pyproj import Transformer
from folium.plugins import TimestampedGeoJson

from _constants import BikeZ_Config
from tools_map_visualization import create_swisstopo_map
from tools_map_visualization import plot_all_centerlines_splines_xy_2056
from tools_map_visualization import plot_bicycles_trajectories_xy_2056

# #############################################################################
# CONSTANTS
# #############################################################################
# Configuration
BikeZ_Config = BikeZ_Config()

# Specify Trajectory File
date = BikeZ_Config.avail_dates[0]
campaign = f"Zurich_2025{date[5:7]}" # June or September

intersection, code = BikeZ_Config.avail_intersections[date][0]
timeslot = BikeZ_Config.avail_timeslots[date][(intersection, code)][3] # 'AM1'

XY_2056_Bounds = BikeZ_Config.XY_2056_Bounds[date][(intersection, code)]
X_2056_offset = XY_2056_Bounds[0][0]
Y_2056_offset = XY_2056_Bounds[1][0]


mode = BikeZ_Config.avail_modes[0] # Bike
data_root = BikeZ_Config.data_root[campaign][mode]
filename = f"trajectories_bikes_{date}_{intersection}_{timeslot}_{code}-1-ekf.csv"
df_bik = pd.read_csv(data_root + f"{date}/{intersection}/{filename}")
df_bik = df_bik.dropna()
df_bik['datetime'] = pd.to_datetime(df_bik['datetime'], format='ISO8601')
df_bik['x_act_ekf'] = df_bik['x_ekf'] + X_2056_offset
df_bik['y_act_ekf'] = df_bik['y_ekf'] + Y_2056_offset
transformer = Transformer.from_crs("EPSG:2056", "EPSG:4326", always_xy=True)
df_bik["lon_ekf"], df_bik["lat_ekf"] = transformer.transform(df_bik["x_act_ekf"].values, df_bik["y_act_ekf"].values)
center_lat, center_lon = df_bik["lat_ekf"].mean(), df_bik["lon_ekf"].mean()

mode = BikeZ_Config.avail_modes[1] # Vehicle
data_root = BikeZ_Config.data_root[campaign][mode]
filename = f"trajectories_vehicles_{date}_{intersection}_{timeslot}_{code}-1-ekf.csv"
df_veh = pd.read_csv(data_root + f"{date}/{intersection}/{filename}")
df_veh = df_veh.dropna()
df_veh['datetime'] = pd.to_datetime(df_veh['datetime'], format='ISO8601')
df_veh['x_act_ekf'] = df_veh['x_ekf'] + X_2056_offset
df_veh['y_act_ekf'] = df_veh['y_ekf'] + Y_2056_offset
df_veh["lon_ekf"], df_veh["lat_ekf"] = transformer.transform(df_veh["x_act_ekf"].values, df_veh["y_act_ekf"].values)


m = create_swisstopo_map(center_lat=center_lat, center_lon=center_lon, add_layer_control=False)

features = []

# Bikes — black circles
for bike_id, grp in df_bik.groupby("veh_id"):
    for _, row in grp.iterrows():
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [row["lon_ekf"], row["lat_ekf"]],
            },
            "properties": {
                "time": row["datetime"].strftime("%Y-%m-%dT%H:%M:%S"),
                "popup": f"{row['veh_type']} {bike_id}",
                "icon": "circle",
                "iconstyle": {
                    "fillColor": "blue",
                    "fillOpacity": 0.8,
                    "stroke": "false",
                    "radius": 3
                }
            }
        })

# Vehicles — red, slightly larger circles
for veh_id, grp in df_veh.groupby("veh_id"):
    for _, row in grp.iterrows():
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [row["lon_ekf"], row["lat_ekf"]],
            },
            "properties": {
                "time": row["datetime"].strftime("%Y-%m-%dT%H:%M:%S"),
                "popup": f"{row['veh_type']} {veh_id}",
                "icon": "circle",
                "iconstyle": {
                    "fillColor": "red",
                    "fillOpacity": 0.6,
                    "stroke": "false",
                    "radius": 5
                }
            }
        })

features.sort(key=lambda f: f["properties"]["time"])

TimestampedGeoJson(
    {
        "type": "FeatureCollection",
        "features": features,
    },
    period="PT1S",
    duration="PT1S",
    transition_time=40,
    loop=False,
).add_to(m)

# 2. CSS/JavaScript fix — preserves per-feature fill color on history points
css = """
<style>
.leaflet-zoom-animated path.leaflet-interactive[fill="blue"] {
    stroke: blue !important;
    fill: blue !important;
    fill-opacity: 0.15 !important;
    stroke-opacity: 0.15 !important;
    stroke-width: 1px !important;
}
.leaflet-zoom-animated path.leaflet-interactive[fill="red"] {
    stroke: red !important;
    fill: red !important;
    fill-opacity: 0.15 !important;
    stroke-opacity: 0.15 !important;
    stroke-width: 1px !important;
}
</style>
"""
m.get_root().html.add_child(folium.Element(css))
js_fix = """
<script>
document.addEventListener("DOMContentLoaded", function () {
    // Poll until the map renders, then shrink history point radii
    var interval = setInterval(function () {
        var paths = document.querySelectorAll(
            '.leaflet-zoom-animated path.leaflet-interactive[fill="blue"], ' +
            '.leaflet-zoom-animated path.leaflet-interactive[fill="red"]'
        );
        paths.forEach(function (p) {
            // Only shrink history points (low opacity), not active dots
            var opacity = parseFloat(p.getAttribute('fill-opacity') || 1);
            if (opacity < 0.5) {
                p.setAttribute('r', '1.5');
            }
        });
    }, 100);  // check every 100ms as animation runs
});
</script>
"""
m.get_root().html.add_child(folium.Element(js_fix))

m.save(f"../maps/timestamped_trajectories_ALL_map_{date}_{intersection}_{timeslot}_{code}.html")

sys.exit(1)


# # Flags
# EKF = True
# CENTERLINES = False


# # #############################################################################
# # MAIN: Load Data (Trajectories)
# # #############################################################################
# if EKF:
#     filename = f"trajectories_bikes_{date}_{intersection}_{timeslot}_{code}-1-ekf.csv"
#     df = pd.read_csv(data_root + f"{date}/{intersection}/{filename}")
#     df = df.dropna()
#     df['datetime'] = pd.to_datetime(df['datetime'], format='ISO8601')
    
#     # Convert from EPSG:2056 to EPSG:4326 (lat, lon)
#     df['x_act_ekf'] = df['x_ekf'] + BikeZ_Config.X_2056_Bounds[0]
#     df['y_act_ekf'] = df['y_ekf'] + BikeZ_Config.Y_2056_Bounds[0]
#     transformer = Transformer.from_crs("EPSG:2056", "EPSG:4326", always_xy=True)
#     df["lon_ekf"], df["lat_ekf"] = transformer.transform(df["x_act_ekf"].values, df["y_act_ekf"].values)
#     center_lat, center_lon = df["lat_ekf"].mean(), df["lon_ekf"].mean()
# else:
#     filename = f"trajectories_bikes_{date}_{intersection}_{timeslot}_{code}-1.csv"
#     df = pd.read_csv(data_root + f"{date}/{intersection}/{filename}")
#     df = df.dropna()

#     df['missing'] = (df['speed(km/h)'] == -1)
#     df = df.rename(columns={
#         'speed(km/h)': 'speed',
#         'a(m/s2)': 'a',
#         'time(s)': 'time',
#         'X_2056(m)': 'x_act',
#         'Y_2056(m)': 'y_act',
#         'longitude': 'lon',
#         'latitude': 'lat'
#     })
#     df['x'] = df['x_act'] - X_2056_offset
#     df['y'] = df['y_act'] - Y_2056_offset
#     df['datetime'] = pd.to_datetime(df['datetime'], format='ISO8601')
#     ref_datetime = df['datetime'].min()
#     ref_time = df.loc[(df['datetime'] == ref_datetime) & (df['time'] >= 0), 'time'].unique()[0]
#     df['time'] = df['datetime'].apply(lambda x: np.round((x - ref_datetime).total_seconds() + ref_time, decimals=3))
#     df = df.sort_values(by=['veh_id', 'time'], ascending=True)
#     df = df[~df['missing']]
#     df["lon_ekf"] = df["lon"]
#     df["lat_ekf"] = df["lat"]


# # centerlines
# if CENTERLINES:
#     with open(f"../data/centerlines_splines_{date}_{intersection}.pkl", "rb") as f:
#         centerlines_spl_dict = pickle.load(f)

# # #############################################################################
# # MAIN: Map Timestamped Visualization
# # #############################################################################
# m = create_swisstopo_map(center_lat=df["lat_ekf"].mean(), center_lon=df["lon_ekf"].mean(), add_layer_control=False)
# if CENTERLINES:
#     plot_all_centerlines_splines_xy_2056(m, centerlines_spl_dict, add_layer_control=True)
# # plot_bicycles_trajectories_xy_2056(m, df, linecolor='black', linealpha=0.1, add_layer_control=True)

# # Build GeoJSON features
# features = []
# for bike_id, grp in df.groupby("veh_id"):
#     for _, row in grp.iterrows():
#         features.append({
#             "type": "Feature",
#             "geometry": {
#                 "type": "Point",
#                 "coordinates": [row["lon_ekf"], row["lat_ekf"]],
#             },
#             "properties": {
#                 "time": row["datetime"].strftime("%Y-%m-%dT%H:%M:%S"),  # ISO format
#                 "popup": f"Bicycle {bike_id}",
#                 "icon": "circle",
#                 "iconstyle": {
#                     "fillColor": "black",
#                     "fillOpacity": 0.8,
#                     "stroke": "true",
#                     "radius": 5
#                 }
#             }
#         })

# TimestampedGeoJson(
#     {
#         "type": "FeatureCollection",
#         "features": features,
#     },
#     period="PT1S",                # animation time step, already lowest setting
#     duration="PT1S",              # history length visible
#     transition_time=40,           # smooth animation, 40 ms = 25 FPS
#     loop=False,
# ).add_to(m)


# m.save(f"../maps/timestamped_trajectories_map_{date}_{intersection}_{timeslot}_{code}.html")

