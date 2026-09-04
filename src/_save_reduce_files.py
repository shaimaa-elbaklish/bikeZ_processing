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

def human(n):
    for unit in ["B", "KB", "MB", "GB"]:
        if n < 1024:
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} TB"

# #############################################################################
# MAIN: Creating Zip Files for Upload
# #############################################################################
import zipfile
from pathlib import Path


subsampled_data_root = BikeZ_Config.subsampled_data_root
locations_list = [2, 3, 4, 5, 8, 9, 10, 11, 12, 13]

total_csv_size = 0
for loc_num in locations_list:
    avail_dates_timeslots = BikeZ_Config.get_available_dates_and_timeslots(loc_num)
    
    loc_dir = Path(subsampled_data_root) / f"location_{loc_num}"
    zip_path = Path(subsampled_data_root) / f"location_{loc_num}.zip"
    
    # Collect all CSVs for this location (bikes, vehicles, lane files, etc.)
    csv_files = sorted(loc_dir.glob("*_lane.csv"))

    if not csv_files:
        print(f"No CSVs found for location_{loc_num}, skipping.")
        continue

    total_csv_size = sum(f.stat().st_size for f in csv_files)

    # Remove any existing zip so we don't append to a stale one
    if zip_path.exists():
        zip_path.unlink()

    # compresslevel=9 is DEFLATE's max compression, equivalent to `zip -9`
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for f in csv_files:
            zf.write(f, arcname=f.name)  # arcname=f.name flattens paths, like -j

    zip_size = zip_path.stat().st_size

    print(f"location_{loc_num}: {len(csv_files)} files")
    print(f"  Total CSV size: {human(total_csv_size)}")
    print(f"  Zip size:       {human(zip_size)}")
    print(f"  Ratio:          {total_csv_size / zip_size:.2f}x smaller")
    print()
    
    # sys.exit(1)

sys.exit(1)

# # #############################################################################
# # MAIN: Subsampled files
# # #############################################################################
# subsampled_data_root = BikeZ_Config.subsampled_data_root
# locations_list = [i for i in range(2, 14)]

# total_csv_size = 0
# total_parquet_size = 0
# n_files = 0
# n_errors = 0
# for loc_num in locations_list:
#     avail_dates_timeslots = BikeZ_Config.get_available_dates_and_timeslots(loc_num)
#     for date, timeslots in avail_dates_timeslots.items():
#         for timeslot in timeslots:
#             # --- Load Trajectories: Bicycles ---
#             mode = 'bike'
#             filename = f"location_{loc_num}/{loc_num}_{mode}s_{date}_{timeslot}.csv"
#             csv_path = subsampled_data_root + filename
#             parquet_path = csv_path.replace('.csv', '.parquet')
#             try:
#                 df_bik = pd.read_csv(csv_path)
#                 df_bik['datetime'] = pd.to_datetime(df_bik['datetime'], format='ISO8601')

#                 # veh_type: object -> category (low cardinality string)
#                 df_bik['veh_type'] = df_bik['veh_type'].astype('category')

#                 # Write parquet
#                 df_bik.to_parquet(parquet_path, engine="pyarrow", compression="zstd", compression_level=15)

#                 # Verify parquet was written and is non-empty before deleting original
#                 if not os.path.exists(parquet_path) or os.path.getsize(parquet_path) == 0:
#                     raise IOError(f"Parquet write failed or produced empty file: {parquet_path}")

#                 csv_size = os.path.getsize(csv_path)
#                 parquet_size = os.path.getsize(parquet_path)
                
#                 # Verify before deleting anything irreversible
#                 df_check = pd.read_parquet(parquet_path)
#                 if not df_bik.equals(df_check):
#                     raise ValueError(f"Round-trip mismatch, NOT deleting CSV: {csv_path}")
                
#                 # Delete original CSV only after confirming parquet is good
#                 os.remove(csv_path)

#                 total_csv_size += csv_size
#                 total_parquet_size += parquet_size
#                 n_files += 1

#             except Exception as e:
#                 n_errors += 1
#                 print(f"ERROR processing {csv_path}: {e}")
#                 # Clean up partial parquet file if it exists, so we don't leave junk behind
#                 if os.path.exists(parquet_path):
#                     os.remove(parquet_path)
#                 continue

# print()
# print(f"Files converted:     {n_files}")
# if n_errors:
#     print(f"Files with errors:    {n_errors}")
# print(f"Total CSV size:       {human(total_csv_size)}")
# print(f"Total Parquet size:   {human(total_parquet_size)}")
# if total_parquet_size > 0:
#     print(f"Overall compression:  {total_csv_size / total_parquet_size:.2f}x smaller")


# # #############################################################################
# # MAIN: 25 fps data
# # #############################################################################
# for date in all_dates_list:
#     campaign = f"Zurich_2025{date[5:7]}"
#     for mode in all_modes:
#         data_root = BikeZ_Config.data_root[campaign][mode]
#         all_intersections_list = BikeZ_Config.avail_intersections[date]
#         for intersection, code in all_intersections_list:
#             all_timeslots = BikeZ_Config.avail_timeslots[date][(intersection, code)]
#             for timeslot in all_timeslots:
#                 filename = f"trajectories_{mode}s_{date}_{intersection}_{timeslot}_{code}-1-ekf-lane.csv"
#                 csv_path = data_root + f"{date}/{intersection}/{filename}"
#                 exists = os.path.exists(csv_path)
                
#                 if exists:
#                     # 1. Read CSV and fix datetime and speed_cols
#                     df = pd.read_csv(csv_path)
#                     df['datetime'] = pd.to_datetime(df['datetime'], format='ISO8601')
#                     try:
#                         speed_cols = ['speed_ekf', 's_dot', 'd_dot']
#                         df[speed_cols] = df[speed_cols] / 3.6
#                     except KeyError:
#                         continue
                    
#                     # 2. Write Parquet
#                     savename = f"trajectories_{mode}s_{date}_{intersection}_{timeslot}_{code}-1-ekf-lane.parquet"
#                     parquet_path = data_root + f"{date}/{intersection}/{savename}"
#                     df.to_parquet(parquet_path, compression='zstd')
                    
#                     # 3. Verify before deleting anything irreversible
#                     df_check = pd.read_parquet(parquet_path)
#                     if not df.equals(df_check):
#                         raise ValueError(f"Round-trip mismatch, NOT deleting CSV: {csv_path}")
                    
#                     # 4. Delete the original CSV only after verification passes
#                     os.remove(csv_path)
                    
#                     # 5. Copy parquet to the other location
#                     dest_root = "C:/Users/ShaimaaElBaklish/OneDrive - ETH Zurich/BikeZ-NEW/"
#                     if date[5:7] == "06":
#                         foldername = f"June-{intersection}"
#                     else:
#                         foldername = f"Sep-{intersection}-{code}"
#                     os.makedirs(os.path.dirname(dest_root + foldername), exist_ok=True)
#                     dest_path = dest_root + foldername + f"/{savename}"
#                     shutil.copy2(parquet_path, dest_path)
                
#                 # sys.exit(1)
#             print(f"Done for {mode}_{date}_{intersection}_{code}")




