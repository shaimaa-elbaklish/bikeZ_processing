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

from dataclasses import dataclass
from typing import Tuple

# #############################################################################
# CONSTANTS: BIKE-Z
# #############################################################################
@dataclass
class BikeZ_Config:
    data_root: str = "C:/Users/ShaimaaElBaklish/Documents/Datasets/BikeZ/Zurich_202506/bike_trajectories/v2/"
    avail_dates: Tuple[str] = ("2025-06-16", "2025-06-17")
    avail_intersections: Tuple[str] = ("D1", "D2", "D3", "D4")
    
    timezone = pytz.timezone('Europe/Berlin')
    
    X_2056_Bounds: Tuple[float] = (2682700, 2682860)
    Y_2056_Bounds: Tuple[float] = (1247820, 1247960)
    
    fps: float = 25.0

# #############################################################################
# CONSTANTS: Filtering
# #############################################################################
SPEED_ESTIMATION_HORIZON = 15
VEHICLE_DIMENSION_MOVING_AVERAGE_WINDOW_LENGTH = 10
ANGLE_VELOCITY_THRESHOLD = 10/360*2*np.pi
SKIP_KALMAN_FILTERING_MAX_GAP = 25
KALMAN_INITIAL_ESTIMATION_WINDOW_LENGTH = 25
KALMAN_TRANSIENT_PERIOD = 25
POST_FILTERING_KERNEL_A = 0.01
POST_FILTERING_KERNEL_B = 20

PROCESSING_MAX_VELOCITY = 16
PROCESSING_THR_VELOCITY = 0.5
