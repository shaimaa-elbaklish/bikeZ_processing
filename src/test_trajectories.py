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
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from _constants import BikeZ_Config

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

intersection, code = BikeZ_Config.avail_intersections[date][-1]
all_timeslots = BikeZ_Config.avail_timeslots[date][(intersection, code)]
# timeslot = BikeZ_Config.avail_timeslots[date][(intersection, code)][0] # 'AM1' or 'PM1

XY_2056_Bounds = BikeZ_Config.XY_2056_Bounds[date][(intersection, code)]

PLOTTING = True

# #############################################################################
# MAIN
# #############################################################################
for timeslot in all_timeslots[:1]:
    print(timeslot)
    filename = f"trajectories_bikes_{date}_{intersection}_{timeslot}_{code}-1.csv"
    df = pd.read_csv(data_root + f"{date}/{intersection}/{filename}")
    # COLUMNS: ['veh_id', 'veh_type', 'speed(km/h)', 'a(m/s2)', 'time(s)', 'X_2056(m)', 'Y_2056(m)', 'longitude', 'latitude', 'datetime']
    # add a column as a missing flag
    df['missing'] = (df['speed(km/h)'] == -1)
    # print(df.loc[df['missing'], 'veh_id'].unique())
    # IDs with missing values: 22,  72, 152, 161
    
    df = df.rename(columns={
        'speed(km/h)': 'speed', 
        'a(m/s2)': 'a', 
        'time(s)': 'time', 
        'X_2056(m)': 'x_act', 
        'Y_2056(m)': 'y_act',
        'longitude': 'lon', 
        'latitude': 'lat'
    })
    df['datetime'] = pd.to_datetime(df['datetime'], format='ISO8601')
    
    # Fix time = -1 issues
    # Find ref. datetime (i.e. datetime when time == 0)
    ref_datetime = df['datetime'].min()
    ref_time = df.loc[(df['datetime'] == ref_datetime) & (df['time'] >= 0), 'time'].unique()[0]
    df['time'] = df['datetime'].apply(lambda x: np.round((x - ref_datetime).total_seconds() + ref_time, decimals=3))
    
    df = df.sort_values(by=['veh_id', 'time'], ascending=True)
    
    
    if PLOTTING:
        plt.figure(timeslot, figsize=(4, 4))
        grouped = df.groupby(by=['veh_id'])
        for (bike_id,), bike_df in grouped:
            bike_df = bike_df[~bike_df['missing']]
            plt.plot(bike_df['x_act'], bike_df['y_act'], color='blue')
        if XY_2056_Bounds is not None and len(XY_2056_Bounds) == 2:
            plt.xlim(XY_2056_Bounds[0])
            plt.ylim(XY_2056_Bounds[1])
        plt.tight_layout()
    
    
    # Checking Frame Number and Uniform Delta Time
    grouped = df.groupby(by=['veh_id'])
    for (bike_id,), bike_df in grouped:
        bike_df['frame_nr'] = np.round(bike_df['time'] * BikeZ_Config.fps + 1e-05, decimals=0)
        bike_df['frame_nr'] = bike_df['frame_nr'].astype(int)
        if not bike_df['frame_nr'].is_monotonic_increasing:
            print("Non-montonic frames, ID = ", bike_id)
            sys.exit(1)
        if bike_df['frame_nr'].duplicated().any():
            print("Duplicated frames, ID = ", bike_id)
            sys.exit(1)
