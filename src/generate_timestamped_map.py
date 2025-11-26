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
date = BikeZ_Config.avail_dates[0]
intersection = BikeZ_Config.avail_intersections[2]
time_slot = 'AM1'
code= 'E'

# #############################################################################
# MAIN: Load Data (Trajectories)
# #############################################################################
filename = f"trajectories_bikes_{date}_{intersection}_{time_slot}_{code}-1-ekf.csv"
df = pd.read_csv(BikeZ_Config.data_root + f"{date}/{intersection}/{filename}")
df = df.dropna()
df['datetime'] = pd.to_datetime(df['datetime'], format='ISO8601')

# Convert from EPSG:2056 to EPSG:4326 (lat, lon)
df['x_act_ekf'] = df['x_ekf'] + BikeZ_Config.X_2056_Bounds[0]
df['y_act_ekf'] = df['y_ekf'] + BikeZ_Config.Y_2056_Bounds[0]
transformer = Transformer.from_crs("EPSG:2056", "EPSG:4326", always_xy=True)
df["lon_ekf"], df["lat_ekf"] = transformer.transform(df["x_act_ekf"].values, df["y_act_ekf"].values)
center_lat, center_lon = df["lat_ekf"].mean(), df["lon_ekf"].mean()


# centerlines
with open(f"../data/centerlines_splines_{date}_{intersection}.pkl", "rb") as f:
    centerlines_spl_dict = pickle.load(f)

# #############################################################################
# MAIN: Map Timestamped Visualization
# #############################################################################
m = create_swisstopo_map(center_lat=df["lat_ekf"].mean(), center_lon=df["lon_ekf"].mean(), add_layer_control=False)
plot_all_centerlines_splines_xy_2056(m, centerlines_spl_dict, add_layer_control=True)
# plot_bicycles_trajectories_xy_2056(m, df, linecolor='black', linealpha=0.1, add_layer_control=True)

# Build GeoJSON features
features = []
for bike_id, grp in df.groupby("veh_id"):
    for _, row in grp.iterrows():
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [row["lon_ekf"], row["lat_ekf"]],
            },
            "properties": {
                "time": row["datetime"].strftime("%Y-%m-%dT%H:%M:%S"),  # ISO format
                "popup": f"Bicycle {bike_id}",
                "icon": "circle",
                "iconstyle": {
                    "fillColor": "black",
                    "fillOpacity": 0.8,
                    "stroke": "true",
                    "radius": 5
                }
            }
        })

TimestampedGeoJson(
    {
        "type": "FeatureCollection",
        "features": features,
    },
    period="PT1S",                # animation time step, already lowest setting
    duration="PT1S",              # history length visible
    transition_time=40,           # smooth animation, 40 ms = 25 FPS
    loop=False,
).add_to(m)


m.save(f"../maps/timestamped_trajectories_map_{date}_{intersection}_{time_slot}_{code}.html")

