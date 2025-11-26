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

from dataclasses import dataclass, field
from typing import Tuple, Dict

# #############################################################################
# CONSTANTS: BIKE-Z
# #############################################################################
@dataclass
class BikeZ_Config:
    timezone = pytz.timezone('Europe/Berlin')
    fps: float = 25.0
    dir_root: str = "C:/Users/ShaimaaElBaklish/Documents/Datasets/BikeZ/Zurich_202506/bike_trajectories/v2/"
    avail_modes: Tuple[str] = ("bike", "vehicle")
    data_root: Dict = field(default_factory=lambda: {
        "Zurich_202506": {
            "bike": "C:/Users/ShaimaaElBaklish/Documents/Datasets/BikeZ/Zurich_202506/bike_trajectories/v2/",
            "vehicle": "C:/Users/ShaimaaElBaklish/Documents/Datasets/BikeZ/Zurich_202506/vehicle_trajectories/"
            },
        "Zurich_202509": {
            "bike": "C:/Users/ShaimaaElBaklish/Documents/Datasets/BikeZ/Zurich_202509/bike_trajectories/",
            "vehicle": "C:/Users/ShaimaaElBaklish/Documents/Datasets/BikeZ/Zurich_202509/vehicle_trajectories/"
            }
    })
    avail_dates: Tuple[str] = ("2025-06-16", "2025-06-17", "2025-09-29", "2025-09-30")
    avail_intersections: Dict = field(default_factory=lambda: {
        "2025-06-16": [("D1", "A"), ("D2", "G"), ("D2", "C"), ("D3", "E"), ("D4", "F")],
        "2025-06-17": [("D1", "A"), ("D1", "B"), ("D2", "C"), ("D3", "E"), ("D4", "F")],
        "2025-09-29": [("D1", "A"), ("D1", "C"), ("D2", "B"), ("D2", "E")],
        "2025-09-30": [("D1", "G"), ("D1", "H"), ("D2", "F"), ("D2", "I")]
    })    
    avail_timeslots: Dict = field(default_factory=lambda: {
        "2025-06-16": {
            ("D1", "A"): [f"AM{i}" for i in range(1, 7)] + [f"PM{i}" for i in range(1, 7)], 
            ("D2", "G"): [f"AM{i}" for i in range(1, 7)], 
            ("D2", "C"): [f"PM{i}" for i in range(1, 7)], 
            ("D3", "E"): [f"AM{i}" for i in range(1, 7)] + [f"PM{i}" for i in range(1, 7)], 
            ("D4", "F"): [f"AM{i}" for i in range(1, 7)] + [f"PM{i}" for i in range(1, 7)]
        },
        "2025-06-17": {
            ("D1", "A"): [f"AM{i}" for i in range(1, 7)], 
            ("D1", "B"): [f"PM{i}" for i in range(1, 7)], 
            ("D2", "C"): [f"AM{i}" for i in range(1, 7)] + [f"PM{i}" for i in range(1, 7)], 
            ("D3", "E"): [f"AM{i}" for i in range(1, 7)] + [f"PM{i}" for i in range(1, 7)], 
            ("D4", "F"): [f"AM{i}" for i in range(1, 7)] + [f"PM{i}" for i in range(1, 7)]
        }, 
        "2025-09-29": {
            ("D1", "A"): [f"AM{i}" for i in range(1, 7)], 
            ("D1", "C"): [f"PM{i}" for i in range(1, 7)], 
            ("D2", "B"): [f"AM{i}" for i in range(1, 7)], 
            ("D2", "E"): [f"PM{i}" for i in range(1, 7)]
        }, 
        "2025-09-30": {
            ("D1", "G"): [f"AM{i}" for i in range(1, 7)], 
            ("D1", "H"): [f"PM{i}" for i in range(1, 4)], 
            ("D2", "F"): [f"AM{i}" for i in range(1, 7)], 
            ("D2", "I"): [f"PM{i}" for i in range(1, 4)]
        }
    })
    XY_2056_Bounds: Dict = field(default_factory=lambda: {
        "2025-06-16": {
            ("D1", "A"): [(2682300, 2682480), (1248410, 1248530)], 
            ("D2", "G"): [(2682410, 2682575), (1248370, 1248480)], 
            ("D2", "C"): [(2682440, 2682610), (1248350, 1248480)], 
            ("D3", "E"): [(2682700, 2682860), (1247820, 1247960)], 
            ("D4", "F"): [(2682780, 2682950), (1247780, 1247890)]
        },
        "2025-06-17": {
            ("D1", "A"): [(2682300, 2682480), (1248410, 1248530)], 
            ("D1", "B"): [(2682300, 2682510), (1248400, 1248530)], 
            ("D2", "C"): [(2682440, 2682610), (1248350, 1248490)], 
            ("D3", "E"): [(2682710, 2682880), (1247780, 1247940)], 
            ("D4", "F"): [(2682740, 2682940), (1247770, 1247915)]
        }, 
        "2025-09-29": {
            ("D1", "A"): [], 
            ("D1", "C"): [], 
            ("D2", "B"): [], 
            ("D2", "E"): []
        }, 
        "2025-09-30": {
            ("D1", "G"): [], 
            ("D1", "H"): [], 
            ("D2", "F"): [], 
            ("D2", "I"): []
        }
    })
    

# #############################################################################
# CONSTANTS: Filtering
# #############################################################################
SPEED_ESTIMATION_HORIZON = 15
VEHICLE_DIMENSION_MOVING_AVERAGE_WINDOW_LENGTH = 10
ANGLE_VELOCITY_THRESHOLD = 10/360*2*np.pi
SKIP_KALMAN_FILTERING_MAX_GAP = 25
KALMAN_INITIAL_ESTIMATION_WINDOW_LENGTH = 15
KALMAN_TRANSIENT_PERIOD = 25
POST_FILTERING_KERNEL_A = 0.01
POST_FILTERING_KERNEL_B = 20

PROCESSING_MAX_VELOCITY = 16
PROCESSING_THR_VELOCITY = 0.5

