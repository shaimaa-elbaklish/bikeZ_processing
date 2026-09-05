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

sys.exit(1)
# #############################################################################
# MAIN: Creating Zip Files for Upload
# #############################################################################
import zipfile
from pathlib import Path


subsampled_data_root = BikeZ_Config.subsampled_data_root
locations_list = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

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
    # compresslevel=6 is the default
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for f in csv_files:
            zf.write(f, arcname=f.name)  # arcname=f.name flattens paths, like -j

    zip_size = zip_path.stat().st_size

    print(f"location_{loc_num}: {len(csv_files)} files")
    print(f"  Total CSV size: {human(total_csv_size)}")
    print(f"  Zip size:       {human(zip_size)}")
    print(f"  Ratio:          {total_csv_size / zip_size:.2f}x smaller")
    print()

sys.exit(1)

# #############################################################################
# MAIN: 25 fps data, upload to onedrive
# #############################################################################
from collections import defaultdict


copy_dir_root = "C:/Users/ShaimaaElBaklish/OneDrive - ETH Zurich/BikeZ-NEW/"
locations_list = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

# track which files landed in which folder, so we can zip per-folder afterward
folder_files = defaultdict(list)

for loc_num in locations_list:
    avail_dates_timeslots = BikeZ_Config.get_available_dates_and_timeslots(loc_num)
    
    for date, all_timeslots in avail_dates_timeslots.items():
        for timeslot in all_timeslots:
            intersection, code = BikeZ_Config.get_intersection_code(date, loc_num, timeslot)
            for mode in all_modes:
                campaign = f"Zurich_2025{date[5:7]}"
                data_root = BikeZ_Config.data_root[campaign][mode]
                filename = f"trajectories_{mode}s_{date}_{intersection}_{timeslot}_{code}-1-ekf-lane.parquet"
                parquet_path = data_root + f"{date}/{intersection}/{filename}"
                exists = os.path.exists(parquet_path)
                if not exists:
                    print(filename, "does not exist!")
                    sys.exit(1)
                
                # copy to onedrive
                if date[5:7] == '06':
                    folder = f'June-{intersection}-Location{loc_num}'
                elif date[5:7] == '09':
                    folder = f'Sep-{intersection}-{code}-Location{loc_num}'
                else:
                    print('Unrecognized date!')
                    sys.exit(1)
                
                # copy if needed to copy_dir_path / folder
                # dest_dir = os.path.join(copy_dir_root, folder)
                # os.makedirs(dest_dir, exist_ok=True)
                # dest_path = os.path.join(dest_dir, filename)
                # shutil.copy2(parquet_path, dest_path)
                # folder_files[folder].append(dest_path)
                
                # just record the source path and its arcname for later zipping
                folder_files[folder].append((parquet_path, filename))
                
    print(f'Done for location {loc_num}.')

import py7zr

# zip all parquets for each folder directly from source into a single archive per folder
for folder, files in folder_files.items():
    archive_path = os.path.join(copy_dir_root, f"{folder}.7z")
    total_parquet_size = sum(os.path.getsize(f) for f, _ in files)

    # LZMA2 filter, max-ish compression level (0-9 preset scale mapped internally)
    filters = [{'id': py7zr.FILTER_LZMA2, 'preset': 9}]

    with py7zr.SevenZipFile(archive_path, 'w', filters=filters) as archive:
        for parquet_path, filename in files:
            archive.write(parquet_path, arcname=filename)

    archive_size = os.path.getsize(archive_path)
    ratio = (1 - archive_size / total_parquet_size) * 100 if total_parquet_size else 0
    print(f"7z'd {len(files)} files into {archive_path}")
    print(f"  Parquet total size: {human(total_parquet_size)}")
    print(f"  7z size:            {human(archive_size)}")
    print(f"  Space saved:        {ratio:.1f}%")
    print()


