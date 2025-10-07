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
import pytz
import numpy as np


# #############################################################################
# CONSTANTS: GLOBAL
# #############################################################################
TIMEZONE = pytz.timezone('Europe/Berlin')

# #############################################################################
# CONSTANTS: BIKE-Z
# #############################################################################
BIKE_Z_DATAROOT = "C:/Users/ShaimaaElBaklish/Documents/Datasets/BikeZ/Zurich_202506/bike_trajectories/v2/"
BIKE_Z_FILES =[
    "2025-06-16/merged_gessnerbrücke/merged_bikes_2025-06-16_AM1_gessnerbrucke.csv",
    "2025-06-16/merged_gessnerbrücke/merged_bikes_2025-06-16_AM2_gessnerbrucke.csv",
    "2025-06-16/merged_gessnerbrücke/merged_bikes_2025-06-16_AM3_gessnerbrucke.csv",
    "2025-06-16/merged_gessnerbrücke/merged_bikes_2025-06-16_AM4_gessnerbrucke.csv",
    "2025-06-16/merged_gessnerbrücke/merged_bikes_2025-06-16_AM5_gessnerbrucke.csv",
    "2025-06-16/merged_gessnerbrücke/merged_bikes_2025-06-16_AM6_gessnerbrucke.csv",
    "2025-06-16/merged_gessnerbrücke/merged_bikes_2025-06-16_PM1_gessnerbrucke.csv",
    "2025-06-16/merged_gessnerbrücke/merged_bikes_2025-06-16_PM2_gessnerbrucke.csv",
    "2025-06-16/merged_gessnerbrücke/merged_bikes_2025-06-16_PM3_gessnerbrucke.csv",
    "2025-06-16/merged_gessnerbrücke/merged_bikes_2025-06-16_PM4_gessnerbrucke.csv",
    "2025-06-16/merged_gessnerbrücke/merged_bikes_2025-06-16_PM5_gessnerbrucke.csv",
    "2025-06-16/merged_gessnerbrücke/merged_bikes_2025-06-16_PM6_gessnerbrucke.csv",
    "2025-06-16/merged_zollstrasse/merged_bikes_2025-06-16_AM1_zollstrasse.csv",
    "2025-06-16/merged_zollstrasse/merged_bikes_2025-06-16_AM2_zollstrasse.csv",
    "2025-06-16/merged_zollstrasse/merged_bikes_2025-06-16_AM3_zollstrasse.csv",
    "2025-06-16/merged_zollstrasse/merged_bikes_2025-06-16_AM4_zollstrasse.csv",
    "2025-06-16/merged_zollstrasse/merged_bikes_2025-06-16_AM5_zollstrasse.csv",
    "2025-06-16/merged_zollstrasse/merged_bikes_2025-06-16_AM6_zollstrasse.csv",
    "2025-06-17/merged_gessnerbrücke/merged_bikes_2025-06-17_AM1_gessnerbrucke",
    "2025-06-17/merged_gessnerbrücke/merged_bikes_2025-06-17_AM2_gessnerbrucke",
    "2025-06-17/merged_gessnerbrücke/merged_bikes_2025-06-17_AM3_gessnerbrucke",
    "2025-06-17/merged_gessnerbrücke/merged_bikes_2025-06-17_AM4_gessnerbrucke",
    "2025-06-17/merged_gessnerbrücke/merged_bikes_2025-06-17_AM5_gessnerbrucke",
    "2025-06-17/merged_gessnerbrücke/merged_bikes_2025-06-17_AM6_gessnerbrucke",
    "2025-06-17/merged_gessnerbrücke/merged_bikes_2025-06-17_PM1_gessnerbrucke",
    "2025-06-17/merged_gessnerbrücke/merged_bikes_2025-06-17_PM2_gessnerbrucke",
    "2025-06-17/merged_gessnerbrücke/merged_bikes_2025-06-17_PM3_gessnerbrucke",
    "2025-06-17/merged_gessnerbrücke/merged_bikes_2025-06-17_PM4_gessnerbrucke",
    "2025-06-17/merged_gessnerbrücke/merged_bikes_2025-06-17_PM5_gessnerbrucke",
    "2025-06-17/merged_gessnerbrücke/merged_bikes_2025-06-17_PM6_gessnerbrucke",
    "2025-06-17/merged_zollstrasse/merged_bikes_2025-06-17_PM1_zollstrasse.csv",
    "2025-06-17/merged_zollstrasse/merged_bikes_2025-06-17_PM2_zollstrasse.csv",
    "2025-06-17/merged_zollstrasse/merged_bikes_2025-06-17_PM3_zollstrasse.csv",
    "2025-06-17/merged_zollstrasse/merged_bikes_2025-06-17_PM4_zollstrasse.csv",
    "2025-06-17/merged_zollstrasse/merged_bikes_2025-06-17_PM5_zollstrasse.csv",
    "2025-06-17/merged_zollstrasse/merged_bikes_2025-06-17_PM6_zollstrasse.csv",
]

# #############################################################################
# CONSTANTS: TUMDOT-MUC
# #############################################################################
TUMDOT_MUC_DATAROOT = "C:/Users/ShaimaaElBaklish/polybox/Datasets/TUMDOT-MUC/Trajectory Data/"
TUMDOT_MUC_FILES = [f"tumdot_muc_part_{i}" for i in range(1, 23)]