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
import shutil
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from _constants import BikeZ_Config

# #############################################################################
# CONSTANTS
# #############################################################################
# Configuration
BikeZ_Config = BikeZ_Config()
all_dates_list = BikeZ_Config.avail_dates
all_modes = BikeZ_Config.avail_modes
# data_root = BikeZ_Config.data_root[campaign][mode]

# #############################################################################
# Main
# #############################################################################
for date in all_dates_list:
    campaign = f"Zurich_2025{date[5:7]}"
    for mode in all_modes:
        data_root = BikeZ_Config.data_root[campaign][mode]
        all_intersections_list = BikeZ_Config.avail_intersections[date]
        for intersection, code in all_intersections_list:
            all_timeslots = BikeZ_Config.avail_timeslots[date][(intersection, code)]
            for timeslot in all_timeslots:
                filename = f"trajectories_{mode}s_{date}_{intersection}_{timeslot}_{code}-1-ekf-lane.csv"
                csv_path = data_root + f"{date}/{intersection}/{filename}"
                exists = os.path.exists(csv_path)
                
                if exists:
                    # 1. Read CSV and fix datetime and speed_cols
                    df = pd.read_csv(csv_path)
                    df['datetime'] = pd.to_datetime(df['datetime'], format='ISO8601')
                    try:
                        speed_cols = ['speed_ekf', 's_dot', 'd_dot']
                        df[speed_cols] = df[speed_cols] / 3.6
                    except KeyError:
                        continue
                    
                    # 2. Write Parquet
                    savename = f"trajectories_{mode}s_{date}_{intersection}_{timeslot}_{code}-1-ekf-lane.parquet"
                    parquet_path = data_root + f"{date}/{intersection}/{savename}"
                    df.to_parquet(parquet_path, compression='zstd')
                    
                    # 3. Verify before deleting anything irreversible
                    df_check = pd.read_parquet(parquet_path)
                    if not df.equals(df_check):
                        raise ValueError(f"Round-trip mismatch, NOT deleting CSV: {csv_path}")
                    
                    # 4. Delete the original CSV only after verification passes
                    os.remove(csv_path)
                    
                    # 5. Copy parquet to the other location
                    dest_root = "C:/Users/ShaimaaElBaklish/OneDrive - ETH Zurich/BikeZ-NEW/"
                    if date[5:7] == "06":
                        foldername = f"June-{intersection}"
                    else:
                        foldername = f"Sep-{intersection}-{code}"
                    os.makedirs(os.path.dirname(dest_root + foldername), exist_ok=True)
                    dest_path = dest_root + foldername + f"/{savename}"
                    shutil.copy2(parquet_path, dest_path)
                
                # sys.exit(1)
            print(f"Done for {mode}_{date}_{intersection}_{code}")




