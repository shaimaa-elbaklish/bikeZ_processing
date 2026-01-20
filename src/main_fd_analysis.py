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
import random
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from _constants import BikeZ_Config
from tools_bfd import determine_leader
from tools_bfd import compute_pseudo_states_pfd_N2
from tools_bfd import aggregate_FD, calibrate_FD

# #############################################################################
# CONSTANTS
# #############################################################################
# Configuration
BikeZ_Config = BikeZ_Config()

# Specify Trajectory File
date = BikeZ_Config.avail_dates[0]
campaign = f"Zurich_2025{date[5:7]}" # June or September
mode = BikeZ_Config.avail_modes[0] # Bike
data_root = BikeZ_Config.data_root[campaign][mode]

intersection, code = BikeZ_Config.avail_intersections[date][3]
time_slot_code = 'AM'
all_time_slots = [f'AM{i}' for i in range(1, 7)]
# time_slot_code = 'PM'
# all_time_slots = ['PM2', 'PM3', 'PM4', 'PM5', 'PM6']
# timeslot = BikeZ_Config.avail_timeslots[date][(intersection, code)][11] # 'AM1'

XY_2056_Bounds = BikeZ_Config.XY_2056_Bounds[date][(intersection, code)]
X_2056_offset = XY_2056_Bounds[0][0]
Y_2056_offset = XY_2056_Bounds[1][0]

min_longitudinal_dist = 0.25 # m
max_lookahead_dist = 50.0 # m
bike_width = 0.8 # m
max_lane_width = 3.0 # m

COMPUTE_PSEUDO_STATES = False

random.seed(0)
np.random.seed(0)

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
        df = pd.read_csv(data_root + f"{date}/{intersection}/{filename}")
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
    
    pfd_df_all.to_csv(f"../data/BFD_TS_{date}_{intersection}_{time_slot_code}_{code}.csv", index=False)
else:
    pfd_df_all = pd.read_csv(f"../data/BFD_TS_{date}_{intersection}_{time_slot_code}_{code}.csv")    

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
#     df = pd.read_csv(data_root + f"{date}/{intersection}/{filename}")
#     df = df.dropna()
#     df['datetime'] = pd.to_datetime(df['datetime'], format='ISO8601')
#     print('Time Slot = ', time_slot)
#     print(df['datetime'].min(), df['datetime'].max())
#     print()

# sys.exit(1)

# #############################################################################
# MAIN: Aggregate per Bike Lane
# #############################################################################
pfd_df_all_AM = pd.read_csv(f"../data/BFD_TS_{date}_{intersection}_AM_{code}.csv") 
pfd_df_all_PM = pd.read_csv(f"../data/BFD_TS_{date}_{intersection}_PM_{code}.csv") 
pfd_df_all = pd.concat((pfd_df_all_AM, pfd_df_all_PM), ignore_index=True)
del pfd_df_all_AM, pfd_df_all_PM
gc.collect()

unique_bike_lanes = ['S_NB', 'S_SB', 'W_EB', 'W_WB'] # 'N_SB'
bike_lane_width_dict = {
    'S_NB': 3.5, 
    'S_SB': 1.5,
    # 'N_SB': 1.5, 
    'W_EB': 1.5, 
    'W_WB': 1.8, 
}
agg_params_dict = {
    'S_NB': (0.3, 75, 'plum', 'purple'), 
    'S_SB': (0.3, 50, 'lightcoral', 'maroon'),
    # 'N_SB': (0.3, 50, 'tab:red'), 
    'W_EB': (0.3, 50, 'mediumseagreen', 'darkgreen'), 
    'W_WB': (0.3, 25, 'skyblue', 'midnightblue'), 
}

fig, axs = plt.subplots(1, 2, figsize=(8, 4))
for bl in unique_bike_lanes:
    print()
    print(f"Bike Lane ID = {bl}")
    bl_pfd_df = pfd_df_all[pfd_df_all['Bike_Lane_ID'] == bl].copy()
    if bl == 'S_NB':
        bl_pfd_df = bl_pfd_df[bl_pfd_df['Position_Longitudinal'] <= bl_pfd_df['Position_Longitudinal'].max() - 3.0]
    lw = bike_lane_width_dict[bl]
    bl_pfd_df['Density'] = bl_pfd_df['Density'] / lw
    bl_pfd_df['Flow'] = bl_pfd_df['Flow'] / lw
    
    bw, min_obs, scol, lcol = agg_params_dict[bl]
    
    agg_df = aggregate_FD(bl_pfd_df, max_density=200.0, bin_width=bw, min_observations=min_obs)
    axs[0].scatter(agg_df['Density'], agg_df['Flow'], color=scol, alpha=0.5,
                   label=f"{bl} ($lw$ = {lw} m)")
    axs[1].scatter(agg_df['Density'], agg_df['Speed'], color=scol, alpha=0.5,
                   label=f"{bl} ($lw$ = {lw} m)")
    
    
    jam_density = None
    loss_fn = 'HuberLoss'
    Ks = agg_df["Density"].to_numpy().astype(np.float32)
    Qs = agg_df["Flow"].to_numpy().astype(np.float32)
    Vs = agg_df["Speed"].to_numpy().astype(np.float32)
    res = calibrate_FD(Ks=Ks, Qs=Qs, Vs=Vs, FD_form='WuFreeFD', 
                       loss_fn=loss_fn, k_jam_est=jam_density)
    k_FD, q_FD, v_FD = res[0], res[1], res[2]
    # WuFreeFD returns: K_test, Q_pred_free, V_pred_free, vf, v_crit, delta, k_crit
    k_crit = res[-1]
    k_FD, q_FD, v_FD = res[0], res[1], res[2]
    Q_cap = np.amax(q_FD)
    k_cap = k_FD[q_FD == Q_cap]
    axs[0].plot(k_FD[k_FD <= k_cap*1.1], q_FD[k_FD <= k_cap*1.1], linestyle="dashed", color=lcol, alpha=0.75)
    axs[1].plot(k_FD[k_FD <= k_cap*1.1], v_FD[k_FD <= k_cap*1.1], linestyle="dashed", color=lcol, alpha=0.75)
    
    if bl in ['W_EB', 'S_NB']:
        k_cong_ratio = 0.95
        if jam_density is None:
            jam_density = k_crit * np.power(res[-4]/(res[-4]-res[-3]), 1/res[-2]) # np.power(vf/(vf-v_cr), 1/delta) * k_crit
        cong_idxs = (Ks >= k_cap*k_cong_ratio) & (Qs <= Q_cap)
        res = calibrate_FD(Ks=Ks[cong_idxs], Qs=Qs[cong_idxs], Vs=Vs[cong_idxs], FD_form='WuCongFD', 
                           loss_fn=loss_fn, k_jam_est=jam_density)
        # WuCongFD returns: K_test, Q_pred_cong, V_pred_cong, k_jam, w
        k_FD, q_FD, v_FD = res[0], res[1], res[2]
        axs[0].plot(k_FD[k_FD >= k_cap*k_cong_ratio], q_FD[k_FD >= k_cap*k_cong_ratio], linestyle="dashed", color=lcol, alpha=0.75)
        axs[1].plot(k_FD[k_FD >= k_cap*k_cong_ratio], v_FD[k_FD >= k_cap*k_cong_ratio], linestyle="dashed", color=lcol, alpha=0.75)
 


axs[0].set(xlim=[0, 200], xlabel='Density [bic/km/m]',
           ylim=[0, 2500], ylabel='Flow [bic/h/m]')
axs[1].set(xlim=[0, 200], xlabel='Density [bic/km/m]',
           ylim=[0, 25], ylabel='Speed [km/h]')
h, l = axs[1].get_legend_handles_labels()
fig.legend(h, l, bbox_to_anchor=(0.5, -0.05), loc='lower center', ncol=4, bbox_transform=fig.transFigure)
fig.tight_layout()
fig.savefig(f"../figures/BFD_BikeZ_{date}_{intersection}_{code}.pdf", dpi=300, bbox_inches='tight')