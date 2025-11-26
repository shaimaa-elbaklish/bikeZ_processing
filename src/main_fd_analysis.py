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
import gc
import sys
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from scipy.interpolate import splprep, splev

from _constants import BikeZ_Config
from tools_bfd import determine_leader
from tools_bfd import compute_pseudo_states_pfd_N2
from tools_bfd import aggregate_FD

# #############################################################################
# CONSTANTS
# #############################################################################
date = BikeZ_Config.avail_dates[0]
intersection = BikeZ_Config.avail_intersections[2]
# all_time_slots = [f'AM{i}' for i in range(1, 7)]
all_time_slots = ['PM2', 'PM3', 'PM4', 'PM5']
code= 'E'

X_2056_offset = BikeZ_Config.X_2056_Bounds[0]
Y_2056_offset = BikeZ_Config.Y_2056_Bounds[0]

min_longitudinal_dist = 0.25 # m
max_lookahead_dist = 50.0 # m
bike_width = 0.8 # m
max_lane_width = 3.0 # m

COMPUTE_PSEUDO_STATES = True

# #############################################################################
# MAIN: Load data
# #############################################################################
# bike lane boundaries
with open(f"../data/bike_lane_boundaries_splines_{date}_{intersection}.pkl", "rb") as f:
    lane_boundaries_spl_dict = pickle.load(f) 
    

# #############################################################################
# MAIN: Process all time slots
# #############################################################################
if COMPUTE_PSEUDO_STATES:
    pfd_df_all = None
    for time_slot in all_time_slots:
        # trajectories after EKF and Lane Coordinate Transformation
        filename = f"trajectories_bikes_{date}_{intersection}_{time_slot}_{code}-1-ekf-lane.csv"
        df = pd.read_csv(BikeZ_Config.data_root + f"{date}/{intersection}/{filename}")
        df = df.dropna()
        
        # Extract Leader-Follower Pairs
        df = determine_leader(df, bike_width, min_longitudinal_dist, max_lookahead_dist, max_lane_width)
        
        # BFD method
        pfd_df = compute_pseudo_states_pfd_N2(
            df, lane_width=2.5, fps=BikeZ_Config.fps, in_bike_lane=True
        )
        pfd_df['time_slot'] = time_slot
        pfd_df['date'] = date
        if pfd_df_all is None:
            pfd_df_all = pfd_df.copy()
        else:
            pfd_df_all = pd.concat((pfd_df_all, pfd_df), ignore_index=True)
        del df, pfd_df
        gc.collect()
    
    pfd_df_all.to_csv(f"../data/BFD_TS_{date}_{intersection}_PM_{code}.csv", index=False)
else:
    pfd_df_all = pd.read_csv(f"../data/BFD_TS_{date}_{intersection}_PM_{code}.csv")    

# #############################################################################
# MAIN: Aggregate
# #############################################################################
agg_df = aggregate_FD(pfd_df_all, max_density=400.0, bin_width=0.3, min_observations=50)

fig, axs = plt.subplots(1, 2, figsize=(8, 4))
axs[0].scatter(pfd_df_all['Density'], pfd_df_all['Flow'], alpha=0.1, s=2, label='Pseudo-traffic States')
axs[0].scatter(agg_df['Density'], agg_df['Flow'], label='Aggregated')
axs[0].set(xlim=[0, 400], xlabel='Density [bic/km]',
           ylim=[0, 3000], ylabel='Flow [bic/h]')
# axs[0].legend()

axs[1].scatter(pfd_df_all['Density'], pfd_df_all['Speed'], alpha=0.1, s=2, label='Pseudo-traffic States')
axs[1].scatter(agg_df['Density'], agg_df['Speed'], label='Aggregated')
axs[1].set(xlim=[0, 400], xlabel='Density [bic/km]',
           ylim=[0, 30], ylabel='Speed [km/h]')
axs[1].legend(loc='upper right')

fig.tight_layout()


# for time_slot in all_time_slots:
#     # trajectories after EKF and Lane Coordinate Transformation
#     filename = f"trajectories_bikes_{date}_{intersection}_{time_slot}_{code}-1.csv"
#     df = pd.read_csv(BikeZ_Config.data_root + f"{date}/{intersection}/{filename}")
#     df = df.dropna()
#     df['datetime'] = pd.to_datetime(df['datetime'], format='ISO8601')
#     print('Time Slot = ', time_slot)
#     print(df['datetime'].min(), df['datetime'].max())
#     print()