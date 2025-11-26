"""
The Bicycle Fundamental Diagram: Empirical Insights into Bicycle Flow for Sustainable Urban Mobility
-------------------------------------------
Authors:        Shaimaa K. El-Baklish, Ying-Chuan Ni, Kevin Riehl, Anastasios Kouvelas, Michail A. Makridis
Organization:   ETH Zürich, Switzerland, IVT - Institute for Transportation Planning and Systems
Development:    2025
Submitted to:   JOURNAL
-------------------------------------------
"""

# #############################################################################
# IMPORTS
# #############################################################################
import sys
import ast

import numpy as np
import pandas as pd
import lmfit as lm
import matplotlib.pyplot as plt

from dataclasses import dataclass
from typing import Tuple


# #############################################################################
# FUNCTIONS
# #############################################################################
def determine_leader(df, bike_width, min_longitudinal_dist, max_lookahead_dist, max_lane_width):
    df[['Preceding', 'Space_Hdwy', 'Time_Hdwy', 'Lateral_Gap', 'Cartesian_Space_Hdwy']] = -1.0
    df['Preceding'] = df['Preceding'].astype(int)
    grouped = df.groupby(by=['time', 'Centerline_ID'])
    for (t, c_id), group_df in grouped:
        if len(group_df) <= 1:
            continue
        group_df = group_df.sort_values(by='Position_Longitudinal', ascending=True).reset_index()
        for idx, row in group_df.iterrows():
            possible_leaders = group_df[group_df['Position_Longitudinal'] >= row['Position_Longitudinal']].copy()
            possible_leaders = possible_leaders[group_df['veh_id'] != row['veh_id']]
            if possible_leaders.empty:
                continue
            # print(row['veh_id'], possible_leaders.empty, len(possible_leaders))
            possible_leaders['Lateral_Diff'] = (possible_leaders['Position_Lateral'] - row['Position_Lateral']).abs()
            possible_leaders['Longitudinal_Diff'] = possible_leaders['Position_Longitudinal'] - row['Position_Longitudinal']
            
            # Problem (1)
            possible_leaders['Condition'] = (possible_leaders['Lateral_Diff'] <= 0.5*bike_width) & \
                (possible_leaders['Longitudinal_Diff'] >= min_longitudinal_dist)
            prec_idx = possible_leaders['Condition'].idxmax()
            if possible_leaders.loc[prec_idx, 'Longitudinal_Diff'] > max_lookahead_dist:
                # Go to problem (2)
                possible_leaders['Combined_Dist'] = possible_leaders['Lateral_Diff']/max_lane_width + \
                                                    possible_leaders['Longitudinal_Diff']/max_lookahead_dist + \
                                                    1000*(possible_leaders['Longitudinal_Diff'] <= min_longitudinal_dist)
                prec_idx = possible_leaders['Combined_Dist'].idxmin()
            
            # print(possible_leaders[['veh_id', 'Lateral_Diff', 'Longitudinal_Diff']])
            # print(prec_idx)
            # print()
            group_df.loc[idx, "Preceding"] = group_df.loc[prec_idx, "veh_id"]
            space_hdwy = (group_df.loc[prec_idx, "Position_Longitudinal"] - group_df.loc[idx, "Position_Longitudinal"])
            speed = group_df.loc[idx, "Speed_Longitudinal"]
            if speed < 0:
                speed = group_df.loc[idx, "speed_ekf"]
            time_hdwy = space_hdwy / (speed/3.6) # speed from km/h to m/s
            cart_space_hdwy = np.sqrt((group_df.loc[prec_idx, "x_ekf"] - group_df.loc[idx, "x_ekf"])**2 + (group_df.loc[prec_idx, "y_ekf"] - group_df.loc[idx, "y_ekf"])**2)
            
            df.loc[row['index'], "Preceding"] = group_df.loc[idx, "Preceding"]
            df.loc[row['index'], "Space_Hdwy"] = space_hdwy
            df.loc[row['index'], "Time_Hdwy"] = time_hdwy
            group_df.loc[idx, "Space_Hdwy"] = space_hdwy
            group_df.loc[idx, "Lateral_Gap"] = abs(group_df.loc[prec_idx, "Position_Lateral"] - group_df.loc[idx, "Position_Lateral"])
            df.loc[row['index'], "Cartesian_Space_Hdwy"] = cart_space_hdwy
            df.loc[row['index'], "Lateral_Gap"] = abs(group_df.loc[prec_idx, "Position_Lateral"] - group_df.loc[idx, "Position_Lateral"])
    
    return df


def compute_pseudo_states_pfd_N2(df: pd.DataFrame, lane_width: float, fps: float, in_bike_lane: bool = True) -> pd.DataFrame:
    """
    This function computes the pseudo-traffic states based on the BFD method.

    Parameters
    ----------
    df : pd.DataFrame
        Bicycles trajectories.
    lane_width : float
        Lane width setting in meters.
    config : dataclass, optional
        Dataset configuration. The default is CRB_Config.

    Returns
    -------
    grouped : TYPE
        Pseudo-traffic states dataframe.
    """
    if 'Space_Hdwy' not in df.columns:
        print("ERROR: Please make sure you leader-follower pairs are identified at first and Space_Hdwy is computed.")
        sys.exit(1)
        return
    subdf = df.copy()
    # subdf = determine_leader(subdf)
    subdf = subdf.sort_values(by=['veh_id', 'time'], ascending=True)
    subdf = subdf.reset_index().drop(columns='index')
    subdf[['Next_Position_Longitudinal', 'Next_Position_Lateral']] = subdf.groupby('veh_id')[['Position_Longitudinal', 'Position_Lateral']].shift(-1)
    subdf[['Next_Space_Hdwy', 'Next_Cartesian_Space_Hdwy', 'Next_Lateral_Gap']] = subdf.groupby('veh_id')[['Space_Hdwy', 'Cartesian_Space_Hdwy', 'Lateral_Gap']].shift(-1)
    subdf = subdf.drop(subdf[subdf["time"]==subdf["time"].max()].index)
    subdf = subdf.dropna()
    subdf = subdf[subdf['Preceding'] != -1] # only leader-follower pairs
    if in_bike_lane:
        subdf = subdf[subdf["In_Bike_Lane"]]
    
    
    num_vehicles = 2
    avg_bike_length = 1.8 # m 
    grouped = subdf[["time", "veh_id", "Preceding", "Time_Hdwy", "Space_Hdwy", 
                     "Next_Space_Hdwy", "Lateral_Gap", "Next_Lateral_Gap",
                     "Position_Longitudinal", "Next_Position_Longitudinal", 
                     "Position_Lateral", "Next_Position_Lateral",
                     "Cartesian_Space_Hdwy", "Next_Cartesian_Space_Hdwy",
                     "Bike_Lane_ID", "Centerline_ID"]].copy()
    grouped["TTT"] =  (1/fps) * (num_vehicles-1) / 3600.0 # hour
    # grouped["x0"] = grouped["Lane_Y"]
    # grouped["xt"] = grouped["Next_Lane_Y"]
    grouped["x0"] = grouped["Position_Longitudinal"]
    grouped["xt"] = grouped["Next_Position_Longitudinal"]
    
    grouped["xL0"] = grouped["x0"] + grouped["Space_Hdwy"]
    grouped["xLt"] = grouped["xt"] + grouped["Next_Space_Hdwy"]
    
    grouped["TTD"] = abs(grouped["xt"]-grouped["x0"]) / 1000.0 # km
    grouped["Area"] = 0.5*(1/fps/3600.0)*(grouped["xL0"]-grouped["x0"] + grouped["xLt"]-grouped["xt"] + 2*avg_bike_length)/1000.0 # km.h
    grouped["Area"] = grouped["Area"].astype(np.float64).clip(lower=0)
    grouped = grouped[grouped['Area'] > 0]
    grouped["Density"] = grouped["TTT"] / grouped["Area"]
    grouped["Flow"] = grouped["TTD"] / grouped["Area"]
    grouped["Speed"] = grouped["Flow"] / grouped["Density"]
    grouped["Vehicle_IDs"] = grouped[["Preceding", "veh_id"]].values.tolist()
    
    grouped = grouped[["time", "Vehicle_IDs", "Density", "Flow", "Speed", "Area",
                       "TTT", "TTD", "Position_Lateral", "Position_Longitudinal", 
                       "Time_Hdwy", "Space_Hdwy", "Cartesian_Space_Hdwy", 
                       "Lateral_Gap", "Bike_Lane_ID", "Centerline_ID"]]
    
    grouped = grouped.dropna()
    return grouped


def aggregate_FD(ts_df, max_density=180.0, bin_width=0.3, min_observations=15):
    ts_df['Density_Bin'] = pd.cut(x=ts_df['Density'], bins=np.arange(0, max_density, bin_width))
    agg_df = ts_df.groupby(["Density_Bin"], observed=False).agg({
        "Density": "mean", 
        "Flow": "mean",
        "Speed": "mean",
        "Density_Bin": "count"
    })
    agg_df = agg_df.rename(
        columns={"Density_Bin": "Num_Observations"}
    )
    agg_df = agg_df.dropna()
    print(agg_df["Num_Observations"].min(), agg_df["Num_Observations"].max())
    print(agg_df["Num_Observations"].mean(), agg_df["Num_Observations"].median())
    agg_df = agg_df[agg_df["Num_Observations"] >= min_observations]
    return agg_df