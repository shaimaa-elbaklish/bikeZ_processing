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
import gc
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
parser.add_argument("is_subsampled", type=str, help="Which files to transform: True or False")
parser.add_argument("debug_plotting",type=str, help="Enable debug plots: True or False")
args = parser.parse_args()

date         = args.date
mode         = args.mode
intersection = args.intersection
code         = args.code
timeslot     = args.timeslot
SUBSAMPLED   = args.is_subsampled.lower() == "true"
DEBUG_PLOT   = args.debug_plotting.lower() == "true"

campaign  = f"Zurich_2025{date[5:7]}"
data_root = BikeZ_Config.data_root[campaign][mode]

XY_2056_Bounds = BikeZ_Config.XY_2056_Bounds[date][(intersection, code)]
X_2056_offset = XY_2056_Bounds[0][0]
Y_2056_offset = XY_2056_Bounds[1][0]

loc_num = BikeZ_Config.location_map[(date[5:7], intersection, code)]
subsampled_data_root = BikeZ_Config.subsampled_data_root
#"C:/Users/ShaimaaElBaklish/OneDrive - ETH Zurich/BikeZ-Subsampled/"

log = Logger(date, intersection, code, timeslot, f"CT_{mode}")

# #############################################################################
# MAIN: Load data
# #############################################################################
# trajectories after EKF
if SUBSAMPLED:
    filename = f"location_{loc_num}/{loc_num}_{mode}s_{date}_{timeslot}.csv"
    df = pd.read_csv(subsampled_data_root + filename)
else:
    if mode == "bike":
        filename = f"trajectories_bikes_{date}_{intersection}_{timeslot}_{code}-1-ekf"
    else:
        filename = f"trajectories_vehicles_{date}_{intersection}_{timeslot}_{code}-1-ekf"
    # df = pd.read_csv(data_root + f"{date}/{intersection}/{filename}.csv")
    df = pd.read_parquet(data_root + f"{date}/{intersection}/{filename}.parquet")
    df = df.dropna()
    df['x_act_ekf'] = df['x_ekf'] + X_2056_offset
    df['y_act_ekf'] = df['y_ekf'] + Y_2056_offset

# Load geometry, segment, and movement registries
registry_path = f"../data/registry_{date}_{intersection}_{code}.pkl"
registry = pickle.load(open(registry_path, 'rb'))
geometry_store    = registry['geometry_store']
segment_registry  = registry['segment_registry']
movement_registry = registry['movement_registry']
max_chain_length  = registry['metadata'].get('max_chain_length', 3)

# forced transforms if any
forced_transforms_df = pd.read_csv("../data/forced_transforms.csv")
forced_transforms_df = forced_transforms_df[(forced_transforms_df['date'] == date) &
                                            (forced_transforms_df['intersection'] == intersection) &
                                            (forced_transforms_df['code'] == code) &
                                            (forced_transforms_df['timeslot'] == timeslot) &
                                            (forced_transforms_df['mode'] == mode)].copy()
forced_transforms_ids = forced_transforms_df['veh_id'].unique()
forced_transforms_df['forced_chain_list'] = forced_transforms_df['forced_chain'].apply(
    lambda s: [x.strip() for x in s.strip('[]').split(',')]
)

# #############################################################################
# MAIN: Perform Coordinate Transform for ALL Bicycles
# #############################################################################
from tools_lane_coords_V4 import to_lane_coordinates, setup_registry
from tools_lane_coords_V4 import to_lane_coordinates_forced
from tools_lane_coords_V4 import add_car_lane_membership


setup_registry(geometry_store, segment_registry)

mod_df = None
unique_ids = df['veh_id'].unique()
for bike_id in tqdm(unique_ids, desc=f"Processing Coordinate Transform on {mode}s"):
    bike_df = df[df["veh_id"] == bike_id].copy()
    
    if bike_id in forced_transforms_ids:
        chain = forced_transforms_df.loc[forced_transforms_df['veh_id'] == bike_id, 'forced_chain_list'].item()
        bike_df = to_lane_coordinates_forced(
            bike_df,
            forced_chain=chain,
            segment_registry=segment_registry,
            geometry_store=geometry_store,
            movement_registry=movement_registry,
            verbose=DEBUG_PLOT, log=log
        )
    else:
        bike_df = to_lane_coordinates(
            bike_df, movement_registry,
            segment_registry, geometry_store,
            max_chain_length=max_chain_length,
            agent_mode=mode,
            verbose=DEBUG_PLOT, log=log
        )

    if mod_df is None:
        mod_df = bike_df.copy()
    else:
        mod_df = pd.concat((mod_df, bike_df), ignore_index=True)

# Assign car lane membership
mod_df = add_car_lane_membership(mod_df, segment_registry, tol=0.25)

if SUBSAMPLED:
    filename = f"location_{loc_num}/{loc_num}_{mode}s_{date}_{timeslot}_lane.csv"
    save_mod_df = mod_df.copy()
    # retain only s_native/d_native
    save_mod_df = save_mod_df.drop(columns=['s', 'd'])
    # convert speeds from km/h to m/s
    speed_cols = ['speed_ekf', 's_dot', 'd_dot']
    save_mod_df[speed_cols] = save_mod_df[speed_cols] / 3.6
    save_mod_df.to_csv(subsampled_data_root + filename, index=False)
    del save_mod_df
    gc.collect()
else:
    if mode == "bike":
        filename = f"trajectories_bikes_{date}_{intersection}_{timeslot}_{code}-1-ekf-lane"
    else:
        filename = f"trajectories_vehicles_{date}_{intersection}_{timeslot}_{code}-1-ekf-lane"
    # mod_df.to_csv(data_root + f"{date}/{intersection}/{filename}.csv", index=False)
    mod_df.to_parquet(data_root + f"{date}/{intersection}/{filename}.parquet", compression='zstd', index=False)


# #############################################################################
# MAIN: Plot Coordinate Transform for ALL Bicycles
# #############################################################################
save_path = os.path.join("../debugging/", f"{date}-{intersection}")
os.makedirs(save_path, exist_ok=True)

if DEBUG_PLOT:
    from tools_plot_lane_results import plot_debug_panel
    
    if mode == "bike":
        pdf_path = os.path.join(save_path, f'{timeslot}_{code}_lane_coordinates_all_bicycles.pdf')
    else:
        pdf_path = os.path.join(save_path, f'{timeslot}_{code}_lane_coordinates_all_vehicles.pdf')
    
    plt.ioff()
    with PdfPages(pdf_path) as pdf:
        for bike_id in tqdm(unique_ids, desc="Plotting Lane Coordinates"):
            bike_df = mod_df[mod_df["veh_id"] == bike_id].copy()
            fig = plot_debug_panel(
                bike_df,
                geometry_store,
                segment_registry,
                time_col='time',
                xy_offset=True,     # set False to use raw EPSG:2056 coords
                save_path=None
            )
            fig.tight_layout()
            pdf.savefig(fig, dpi=72)   # low dpi for compact file
            plt.close(fig)

    print(f"Saved PDF: {pdf_path}")
  



