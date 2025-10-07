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
import datetime
import warnings
warnings.simplefilter('ignore', RuntimeWarning) # Ignore all RuntimeWarnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from _constants import BIKE_Z_DATAROOT, BIKE_Z_FILES

# #############################################################################
# MAIN: BIKE-Z
# #############################################################################
input_file = BIKE_Z_FILES[0]
print(f"... Processing {input_file.split('/')[-1]}.")
print(f"... Date: {input_file.split('/')[0]}")
print(f"... Location: {input_file.split('.')[0].split('_')[-1]}")

input_filepath = os.path.join(BIKE_Z_DATAROOT, input_file)
df = pd.read_csv(input_filepath)
df['datetime'] = pd.to_datetime(df['datetime'], format='ISO8601')

print(f"Number of unique bicycles = {df['veh_id'].nunique()}")
print(f"Start time: {datetime.datetime.strftime(df['datetime'].min(), '%d-%m-%Y %H:%M:%S')}")
print(f"End time: {datetime.datetime.strftime(df['datetime'].max(), '%d-%m-%Y %H:%M:%S')}")

print(f"Range in X = {df.loc[df['X_2056(m)']>=0, 'X_2056(m)'].max()-df.loc[df['X_2056(m)']>=0, 'X_2056(m)'].min()} m")
print(f"Range in Y = {df.loc[df['Y_2056(m)']>=0, 'Y_2056(m)'].max()-df.loc[df['Y_2056(m)']>=0, 'Y_2056(m)'].min()} m")

all_times = df['datetime'].copy()
all_times = all_times.drop_duplicates().sort_values()
delta_times = all_times.diff(1).dropna()
# dt is 0.04 seconds (i.e. 25 FPS)

dt = np.median(delta_times)
dt = dt.astype('timedelta64[ms]')