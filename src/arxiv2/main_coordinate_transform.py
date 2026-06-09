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
import os
import sys
import pickle
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from matplotlib.backends.backend_pdf import PdfPages

from _constants import BikeZ_Config
from _logger import Logger

# #############################################################################
# CONSTANTS
# #############################################################################
# Configuration
BikeZ_Config = BikeZ_Config()

parser = argparse.ArgumentParser(description="Coordinate transform for BikeZ trajectories")
parser.add_argument("date",          type=str, help="Date string, e.g. 2025-06-16")
parser.add_argument("mode",          type=str, help="Mode: bike or vehicle")
parser.add_argument("intersection",  type=str, help="Intersection ID, e.g. D3")
parser.add_argument("code",          type=str, help="Code letter, e.g. E")
parser.add_argument("timeslot",      type=str, help="Timeslot, e.g. AM1")
parser.add_argument("debug_plotting",type=str, help="Enable debug plots: True or False")
args = parser.parse_args()

date         = args.date
mode         = args.mode
intersection = args.intersection
code         = args.code
timeslot     = args.timeslot
DEBUG_PLOT   = args.debug_plotting.lower() == "true"

campaign  = f"Zurich_2025{date[5:7]}"
data_root = BikeZ_Config.data_root[campaign][mode]

XY_2056_Bounds = BikeZ_Config.XY_2056_Bounds[date][(intersection, code)]
X_2056_offset = XY_2056_Bounds[0][0]
Y_2056_offset = XY_2056_Bounds[1][0]

# OPP_DIRECTIONS = {"N": "S", "S": "N", "W": "E", "E": "W"}

# bike_lane_tol = 0.4

log = Logger(date, intersection, code, timeslot, f"CT_{mode}")

# #############################################################################
# MAIN: Load data
# #############################################################################
# trajectories after EKF
if mode == "bike":
    filename = f"trajectories_bikes_{date}_{intersection}_{timeslot}_{code}-1-ekf.csv"
else:
    filename = f"trajectories_vehicles_{date}_{intersection}_{timeslot}_{code}-1-ekf.csv"
df = pd.read_csv(data_root + f"{date}/{intersection}/{filename}")
df = df.dropna()
df['x_act_ekf'] = df['x_ekf'] + X_2056_offset
df['y_act_ekf'] = df['y_ekf'] + Y_2056_offset
center_lat, center_lon = df.loc[~df['missing'], "lat"].mean(), df.loc[~df['missing'], "lon"].mean()

# Load geometry, segment, and movement registries
registry_path = f"../data/registry_{date}_{intersection}_{code}.pkl"
registry = pickle.load(open(registry_path, 'rb'))
geometry_store    = registry['geometry_store']
segment_registry  = registry['segment_registry']
movement_registry = registry['movement_registry']
max_chain_length  = registry['metadata'].get('max_chain_length', 3)


# #############################################################################
# MAIN: Perform Coordinate Transform for ALL Bicycles
# #############################################################################
from tools_lane_coords_V3 import to_lane_coordinates, build_registry_luts
# from tools_lane_coords_V2 import _build_segment_bboxes
from tools_lane_coords_V3 import POLYGON_ENTRY_TOLERANCE

# One-time setup — do this once before your vehicle loop
build_registry_luts(geometry_store)
# seg_bboxes = _build_segment_bboxes(segment_registry, geometry_store)

# Pre-expand validity polygons once — avoids calling .buffer() inside the loop
for entry in segment_registry.values():
    poly = entry.get('validity_polygon')
    if poly is not None and not poly.is_empty:
        entry['_validity_polygon_expanded'] = poly.buffer(POLYGON_ENTRY_TOLERANCE)
        

mod_df = None
unique_ids = df['veh_id'].unique()
for bike_id in tqdm(unique_ids, desc="Processing Coordinate Transform on Bicycles"):
    bike_df = df[df["veh_id"] == bike_id].copy()
    
    bike_df = to_lane_coordinates(
        bike_df, movement_registry,
        segment_registry, geometry_store,
        max_chain_length=max_chain_length,
        # seg_bboxes=seg_bboxes, 
        log=log, verbose=DEBUG_PLOT
    )

    if mod_df is None:
        mod_df = bike_df.copy()
    else:
        mod_df = pd.concat((mod_df, bike_df), ignore_index=True)

if mode == "bike":
    filename = f"trajectories_bikes_{date}_{intersection}_{timeslot}_{code}-1-ekf-lane.csv"
else:
    filename = f"trajectories_vehicles_{date}_{intersection}_{timeslot}_{code}-1-ekf-lane.csv"
mod_df.to_csv(data_root + f"{date}/{intersection}/{filename}", index=False)


# #############################################################################
# MAIN: Plot Coordinate Transform for ALL Bicycles
# #############################################################################
save_path = os.path.join("../debugging/", f"{date}-{intersection}")
os.makedirs(save_path, exist_ok=True)

if DEBUG_PLOT:
    from tools_plotting import build_lane_color_map, plot_lane_coord_debug
    
    # Build once per registry — consistent colors across all plots
    lane_color_map = build_lane_color_map(geometry_store)
    pdf_path = os.path.join(save_path, f'{timeslot}_{code}_lane_coordinates_all_bicycles.pdf')
    
    plt.ioff()
    with PdfPages(pdf_path) as pdf:
        for bike_id in tqdm(unique_ids, desc="Plotting Lane Coordinates"):
            bike_df = mod_df[mod_df["veh_id"] == bike_id].copy()
            fig = plot_lane_coord_debug(
                bike_df, segment_registry, geometry_store,
                XY_2056_Bounds, bike_id,
                lane_color_map=lane_color_map,
                save_path=None          # don't save individual PNGs
            )
            pdf.savefig(fig, dpi=72)   # low dpi for compact file
            plt.close(fig)

    print(f"Saved PDF: {pdf_path}")
    
    
    # for bike_id in tqdm(unique_ids, desc="Plotting Lane Coordinates on Bicycles"):
    #     bike_df = mod_df[(mod_df["veh_id"] == bike_id)].copy()
    #     plot_lane_coord_debug(
    #         bike_df, segment_registry, geometry_store,
    #         XY_2056_Bounds, bike_id,
    #         lane_color_map=lane_color_map,
    #         save_path=os.path.join(save_path, f'{timeslot}_{code}_lane_coordinates_bicycle_{bike_id}.png')
    #     )
    #     plt.close('all')
  
sys.exit(1)


