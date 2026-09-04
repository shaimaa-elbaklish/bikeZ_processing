"""
TITLE OF PAPAER
-------------------------------------------
Authors:        Shaimaa El-Baklish
Organization:   ETH Zürich, Switzerland, IVT - Institute for Transportation Planning and Systems
Development:    2025-2026
Submitted to:   JOURNAL
-------------------------------------------
"""

# #############################################################################
# IMPORTS
# #############################################################################
import sys
import folium
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from folium.plugins import TimestampedGeoJson

from _constants import BikeZ_Config
from tools_utils import _PROJ_2056_TO_LONLAT
from tools_map_visualization import create_swisstopo_map

# #############################################################################
# CONSTANTS
# #############################################################################
# Configuration
BikeZ_Config = BikeZ_Config()

parser = argparse.ArgumentParser(description="Timestamped Map Visualization for BikeZ trajectories")
parser.add_argument("date",           type=str, help="Date string, e.g. 2025-06-16")
parser.add_argument("intersection",   type=str, help="Intersection ID, e.g. D3")
parser.add_argument("code",           type=str, help="Code letter, e.g. E")
parser.add_argument("timeslot",       type=str, help="Timeslot, e.g. AM1")
parser.add_argument("is_subsampled",  type=str, help="Which files to use: True or False")
parser.add_argument("history_length", type=int, nargs='?', default=0,
                     help="Number of history seconds to show (0 = current position only), must be integer")
args = parser.parse_args()

date         = args.date
intersection = args.intersection
code         = args.code
timeslot     = args.timeslot
SUBSAMPLED   = args.is_subsampled.lower() == "true"
history_len  = args.history_length
history_len  = int(max(0, history_len))

campaign  = f"Zurich_2025{date[5:7]}"
subsampled_data_root = BikeZ_Config.subsampled_data_root
#"C:/Users/ShaimaaElBaklish/OneDrive - ETH Zurich/BikeZ-Subsampled/"
loc_num = BikeZ_Config.location_map[(date[5:7], intersection, code)]

XY_2056_Bounds = BikeZ_Config.XY_2056_Bounds[date][(intersection, code)]
X_2056_offset = XY_2056_Bounds[0][0]
Y_2056_offset = XY_2056_Bounds[1][0]

# #############################################################################
# MAIN: Load Data (Bikes + Vehicles)
# #############################################################################
if SUBSAMPLED:
    mode = BikeZ_Config.avail_modes[0] # Bike
    # filename = f"location_{loc_num}/{loc_num}_{mode}s_{date}_{timeslot}.csv"
    # df_bik = pd.read_csv(subsampled_data_root + filename)
    filename = f"location_{loc_num}/{loc_num}_{mode}s_{date}_{timeslot}.parquet"
    df_bik = pd.read_parquet(subsampled_data_root + filename)
    df_bik['datetime'] = pd.to_datetime(df_bik['datetime'], format='ISO8601')
    center_lat, center_lon = df_bik["lat_ekf"].mean(), df_bik["lon_ekf"].mean()
    
    mode = BikeZ_Config.avail_modes[1] # Vehicle
    # filename = f"location_{loc_num}/{loc_num}_{mode}s_{date}_{timeslot}.csv"
    # df_veh = pd.read_csv(subsampled_data_root + filename)
    filename = f"location_{loc_num}/{loc_num}_{mode}s_{date}_{timeslot}.parquet"
    df_veh = pd.read_parquet(subsampled_data_root + filename)
    df_veh['datetime'] = pd.to_datetime(df_veh['datetime'], format='ISO8601')
    
    if history_len == 0:
        df_bik['datetime'] = df_bik['datetime'].dt.round('100ms')
        df_veh['datetime'] = df_veh['datetime'].dt.round('100ms')
else:
    mode = BikeZ_Config.avail_modes[0] # Bike
    data_root = BikeZ_Config.data_root[campaign][mode]
    filename = f"trajectories_bikes_{date}_{intersection}_{timeslot}_{code}-1-ekf"
    # df_bik = pd.read_csv(data_root + f"{date}/{intersection}/{filename}.csv")
    # df_bik['datetime'] = pd.to_datetime(df_bik['datetime'], format='ISO8601')
    df_bik = pd.read_parquet(data_root + f"{date}/{intersection}/{filename}.parquet")
    df_bik = df_bik.dropna()
    df_bik['x_act_ekf'] = df_bik['x_ekf'] + X_2056_offset
    df_bik['y_act_ekf'] = df_bik['y_ekf'] + Y_2056_offset
    df_bik["lon_ekf"], df_bik["lat_ekf"] = _PROJ_2056_TO_LONLAT.transform(
        df_bik["x_act_ekf"].values, df_bik["y_act_ekf"].values)
    center_lat, center_lon = df_bik["lat_ekf"].mean(), df_bik["lon_ekf"].mean()
    
    mode = BikeZ_Config.avail_modes[1] # Vehicle
    data_root = BikeZ_Config.data_root[campaign][mode]
    filename = f"trajectories_vehicles_{date}_{intersection}_{timeslot}_{code}-1-ekf"
    # df_veh = pd.read_csv(data_root + f"{date}/{intersection}/{filename}.csv")
    # df_veh['datetime'] = pd.to_datetime(df_veh['datetime'], format='ISO8601')
    df_veh = pd.read_parquet(data_root + f"{date}/{intersection}/{filename}.parquet")
    df_veh = df_veh.dropna()
    df_veh['x_act_ekf'] = df_veh['x_ekf'] + X_2056_offset
    df_veh['y_act_ekf'] = df_veh['y_ekf'] + Y_2056_offset
    df_veh["lon_ekf"], df_veh["lat_ekf"] = _PROJ_2056_TO_LONLAT.transform(
        df_veh["x_act_ekf"].values, df_veh["y_act_ekf"].values)


# start_time = df_bik['datetime'].min()
# end_time = start_time + pd.Timedelta(seconds=10)
# df_bik = df_bik[(df_bik['datetime'] >= start_time) & (df_bik['datetime'] < end_time)]
# df_veh = df_veh[(df_veh['datetime'] >= start_time) & (df_veh['datetime'] < end_time)]


# #############################################################################
# MAIN: Create Map
# #############################################################################
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
                "time": row["datetime"].strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3],
                "popup": f"{row['veh_type']} {bike_id} - {row['datetime'].strftime('%H:%M:%S.%f')[:-3]}",
                "icon": "circle",
                "iconstyle": {
                    "fillColor": "blue",
                    "fillOpacity": 0.6 if history_len > 0 else 0.8,
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
                "time": row["datetime"].strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3],
                "popup": f"{row['veh_type']} {veh_id} - {row['datetime'].strftime('%H:%M:%S.%f')[:-3]}",
                "icon": "circle",
                "iconstyle": {
                    "fillColor": "red",
                    "fillOpacity": 0.4 if history_len > 0 else 0.6,
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
    duration=f"PT{history_len}S",
    transition_time=40,
    loop=False,
).add_to(m)


# 2. CSS/JavaScript fix — preserves per-feature fill color on history points
css = """
<style>
.leaflet-zoom-animated path.leaflet-interactive[fill="blue"] {
    stroke: blue !important;
    fill: blue !important;
    stroke-width: 1px !important;
}
.leaflet-zoom-animated path.leaflet-interactive[fill="red"] {
    stroke: red !important;
    fill: red !important;
    stroke-width: 1px !important;
}
</style>
"""
m.get_root().html.add_child(folium.Element(css))
    # fill-opacity: 0.15 !important;
    # stroke-opacity: 0.15 !important;

# js_fix = """
# <script>
# document.addEventListener("DOMContentLoaded", function () {
#     // Poll until the map renders, then shrink history point radii
#     var interval = setInterval(function () {
#         var paths = document.querySelectorAll(
#             '.leaflet-zoom-animated path.leaflet-interactive[fill="blue"], ' +
#             '.leaflet-zoom-animated path.leaflet-interactive[fill="red"]'
#         );
#         paths.forEach(function (p) {
#             // Only shrink history points (low opacity), not active dots
#             var opacity = parseFloat(p.getAttribute('fill-opacity') || 1);
#             if (opacity < 0.5) {
#                 p.setAttribute('r', '1.5');
#             }
#         });
#     }, 100);  // check every 100ms as animation runs
# });
# </script>
# """
# m.get_root().html.add_child(folium.Element(js_fix))


m.save(f"../maps/timestamped_trajectories_ALL_map_{date}_{intersection}_{timeslot}_{code}_history{history_len}s.html")
