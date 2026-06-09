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


def _expFD(Ks, vf, alpha, k_crit):
    V_pred = vf * np.exp(-np.power(Ks/k_crit, alpha)/alpha)
    Q_pred = Ks * V_pred
    return V_pred, Q_pred


def _WuFreeFD(Ks, vf, v_crit, delta, k_crit):
    V_pred_free = np.maximum(0, vf - (vf-v_crit)*np.power(Ks/k_crit, delta))
    Q_pred_free = Ks * V_pred_free
    return V_pred_free, Q_pred_free


def _WuCongFD(Ks, w, k_jam):
    Q_pred_cong = np.maximum(0, w*(Ks - k_jam))
    V_pred_cong = Q_pred_cong / Ks
    return V_pred_cong, Q_pred_cong


def _nrmse(params, Ks, Qs, Vs, FD_form):
    if FD_form == "ExpFD":
        vf, alpha = params['vf'], params['alpha']
        k_crit = params['k_crit']
        V_pred, Q_pred = _expFD(Ks, vf, alpha, k_crit)
    elif FD_form == "WuFreeFD":
        vf, delta = params['vf'], params['delta']
        k_crit, v_crit = params['k_crit'], params['v_crit']
        V_pred, Q_pred = _WuFreeFD(Ks, vf, v_crit, delta, k_crit,)
    elif FD_form == "WuCongFD":
        k_jam, w = params['k_jam'], params['w']
        V_pred, Q_pred = _WuCongFD(Ks, w, k_jam)
    else:
        raise NotImplementedError()
    rmse_Q = np.sqrt(np.mean(np.square(Q_pred - Qs)))
    rmse_V = np.sqrt(np.mean(np.square(V_pred - Vs)))
    obj = rmse_Q/np.mean(Qs) + rmse_V/np.mean(Vs)
    return obj


def _huberLoss(params, Ks, Qs, Vs, FD_form, deltaH_Q=10.0, deltaH_V=1.0):
    if FD_form == "ExpFD":
        vf, alpha = params['vf'], params['alpha']
        k_crit = params['k_crit']
        V_pred, Q_pred = _expFD(Ks, vf, alpha, k_crit)
    elif FD_form == "WuFreeFD":
        vf, delta = params['vf'], params['delta']
        k_crit, v_crit = params['k_crit'], params['v_crit']
        V_pred, Q_pred = _WuFreeFD(Ks, vf, v_crit, delta, k_crit)
    elif FD_form == 'WuCongFD':
        k_jam, w = params['k_jam'], params['w']
        V_pred, Q_pred = _WuCongFD(Ks, w, k_jam)
    else:
        raise NotImplementedError()
    abs_diff_Q = np.abs(Q_pred - Qs)
    loss_Q = deltaH_Q * (abs_diff_Q - 0.5*deltaH_Q)
    loss_Q[abs_diff_Q <= deltaH_Q] = 0.5*np.square(abs_diff_Q[abs_diff_Q <= deltaH_Q])
    
    abs_diff_V = np.abs(V_pred - Vs)
    loss_V = deltaH_V * (abs_diff_V - 0.5*deltaH_V)
    loss_V[abs_diff_V <= deltaH_V] = 0.5*np.square(abs_diff_V[abs_diff_V <= deltaH_V])
    
    obj = np.sum(loss_Q)/np.mean(Qs) + np.sum(loss_V)/np.mean(Vs)
    return obj


def calibrate_FD(Ks, Qs, Vs, FD_form = "ExpFD", loss_fn = "NRMSE", k_jam_est=None, log_results=False):
    if FD_form == "ExpFD":
        params = lm.create_params(
            vf = {'value': 2, 'min': 1e-05, 'max': 20},
            alpha = {'value': 5, 'min': 1e-05, 'max': 50},
            k_crit = {'value': 100, 'min': 1e-05, 'max': 120}
        )
    elif FD_form == "WuFreeFD":
        if k_jam_est is None:
            params = lm.create_params(
                vf = {'value': 2, 'min': 1e-05, 'max': 20},
                delta = {'value': 0.5, 'min': 1e-05, 'max': 10},
                k_crit = {'value': 70, 'min': 50.0, 'max': 120},
                v_crit = {'value': 2, 'min': 1e-05, 'max': 20},
            )
        else:
            params = lm.create_params(
                vf = {'value': 2, 'min': 1e-05, 'max': 20},
                delta = {'value': 0.5, 'min': 1e-05, 'max': 10},
                k_crit = {'value': 70, 'min': 50.0, 'max': k_jam_est-10}, #min(120, k_jam_est-10)
                v_crit = {'value': 2, 'min': 1e-05, 'max': 20},
            )
    elif FD_form == "WuCongFD":
        if k_jam_est is None:
            params = lm.create_params(
                k_jam = {'value': 120, 'min': 50, 'max': 200},
                w = {'value': -2, 'min': -40, 'max': -1e-05},
            )
        else:
            params = lm.create_params(
                k_jam = {'value': k_jam_est, 'vary': False},
                w = {'value': -2, 'min': -40, 'max': -1e-05},
            )
    else:
        raise NotImplementedError()
    if loss_fn == "NRMSE":
        res = lm.minimize(_nrmse, params, args=(Ks, Qs, Vs, FD_form), method='differential_evolution')
    elif loss_fn == "HuberLoss":
        res = lm.minimize(_huberLoss, params, args=(Ks, Qs, Vs, FD_form), method='differential_evolution')
    else:
        raise NotImplementedError()
    print(lm.fit_report(res.params))
    if FD_form == "ExpFD":
        vf, alpha = res.params['vf'].value, res.params['alpha'].value
        k_crit = res.params['k_crit'].value
        K_test = np.linspace(0, 2*k_crit, 200)
        V_pred, Q_pred = _expFD(K_test, vf, alpha, k_crit)
        return K_test, Q_pred, V_pred, vf, alpha, k_crit
    elif FD_form == "WuFreeFD":
        vf, delta = res.params['vf'].value, res.params['delta'].value
        k_crit, v_crit = res.params['k_crit'].value, res.params['v_crit'].value
        K_test = np.linspace(0, 2*k_crit, 200)
        V_pred_free, Q_pred_free = _WuFreeFD(K_test, vf, v_crit, delta, k_crit)
        return K_test, Q_pred_free, V_pred_free, vf, v_crit, delta, k_crit
    elif FD_form == "WuCongFD":
        k_jam, w = res.params['k_jam'].value, res.params['w'].value
        K_test = np.linspace(0, k_jam, 200)
        V_pred_cong, Q_pred_cong = _WuCongFD(K_test, w, k_jam)
        return K_test, Q_pred_cong, V_pred_cong, k_jam, w
    else:
        raise NotImplementedError()