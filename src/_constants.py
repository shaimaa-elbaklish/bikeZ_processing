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

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Tuple, Dict

# #############################################################################
# CONSTANTS: BIKE-Z
# #############################################################################


@dataclass
class BikeZ_Config:
    timezone = pytz.timezone('Europe/Berlin')
    fps: float = 25.0
    dir_root: str = "C:/Users/ShaimaaElBaklish/Documents/Datasets/BikeZ/"
    avail_modes: Tuple[str] = ("bike", "vehicle")
    data_root: Dict = field(default_factory=lambda: {
        "Zurich_202506": {
            "bike": "C:/Users/ShaimaaElBaklish/Documents/Datasets/BikeZ/Zurich_202506/bike_trajectories/v2/",
            # "bike": "C:/Users/tramseier/Documents/SVT/Data/BikeZ/MobilLysis_bicycle_trajectory/",
            "vehicle": "C:/Users/ShaimaaElBaklish/Documents/Datasets/BikeZ/Zurich_202506/vehicle_trajectories/"
        },
        "Zurich_202509": {
            "bike": "C:/Users/ShaimaaElBaklish/Documents/Datasets/BikeZ/Zurich_202509/bike_trajectories/",
            # "bike": "C:/Users/tramseier/Documents/SVT/Data/BikeZ/MobilLysis_bicycle_trajectory/",
            "vehicle": "C:/Users/ShaimaaElBaklish/Documents/Datasets/BikeZ/Zurich_202509/vehicle_trajectories/"
        }
    })
    subsampled_data_root: str = "C:/Users/ShaimaaElBaklish/OneDrive - ETH Zurich/BikeZ-Subsampled/"
    avail_dates: Tuple[str] = (
        "2025-06-16", "2025-06-17", "2025-09-29", "2025-09-30")
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
            ("D1", "A"): [(2682300, 2682510), (1248400, 1248530)],
            ("D2", "G"): [(2682380, 2682610), (1248350, 1248490)],
            ("D2", "C"): [(2682380, 2682610), (1248350, 1248490)],
            ("D3", "E"): [(2682700, 2682880), (1247780, 1247960)],
            ("D4", "F"): [(2682740, 2682950), (1247770, 1247915)]
        },
        "2025-06-17": {
            ("D1", "A"): [(2682300, 2682510), (1248400, 1248530)],
            ("D1", "B"): [(2682300, 2682510), (1248400, 1248530)],
            ("D2", "C"): [(2682380, 2682610), (1248350, 1248490)],
            ("D3", "E"): [(2682700, 2682880), (1247780, 1247960)],
            ("D4", "F"): [(2682740, 2682950), (1247770, 1247915)]
        },
        "2025-09-29": {
            ("D1", "A"): [(2683290, 2683500), (1246745, 1246850)],
            ("D1", "C"): [(2680735, 2680840), (1248765, 1248960)],
            ("D2", "B"): [(2681270, 2681450), (1248150, 1248280)],
            ("D2", "E"): [(2680490, 2680650), (1248550, 1248670)]
        },
        "2025-09-30": {
            ("D1", "G"): [(2680180, 2680360), (1246930, 1247050)],
            ("D1", "H"): [(2680065, 2680265), (1248755, 1248880)],
            ("D2", "F"): [(2680820, 2680995), (1247060, 1247160)],
            ("D2", "I"): [(2679830, 2680015), (1248850, 1248990)] 
        }
    })
    location_map: Dict = field(default_factory=lambda: {
        ('06', 'D1', 'A'): 4, ('06', 'D1', 'B'): 4,
        ('06', 'D2', 'G'): 5, ('06', 'D2', 'C'): 5,
        ('06', 'D3', 'E'): 2,
        ('06', 'D4', 'F'): 3,
    
        ('09', 'D1', 'A'): 6,
        ('09', 'D1', 'C'): 8,
        ('09', 'D1', 'G'): 11,
        ('09', 'D1', 'H'): 12,
        ('09', 'D2', 'B'): 7,
        ('09', 'D2', 'E'): 9,
        ('09', 'D2', 'F'): 10,
        ('09', 'D2', 'I'): 13,
    })
    avail_timeslots_by_location: Dict = field(default_factory=lambda: {
        2: {
            "AM": [f"AM{i}" for i in range(1, 7)],
            "PM": [f"PM{i}" for i in range(1, 7)],
        },
        3: {
            "AM": [f"AM{i}" for i in range(1, 7)],
            "PM": [f"PM{i}" for i in range(1, 7)],
        },
        4: {
            "A": [f"AM{i}" for i in range(1, 7)] + [f"PM{i}" for i in range(1, 7)],
            "B": [f"PM{i}" for i in range(1, 7)],
        },
        5: {
            "G": [f"AM{i}" for i in range(1, 7)],
            "C": [f"AM{i}" for i in range(1, 7)] + [f"PM{i}" for i in range(1, 7)],
        },
        6: {"A": [f"AM{i}" for i in range(1, 7)]},
        7: {"B": [f"AM{i}" for i in range(1, 7)]},
        8: {"C": [f"PM{i}" for i in range(1, 7)]},
        9: {"E": [f"PM{i}" for i in range(1, 7)]},
        10: {"F": [f"AM{i}" for i in range(1, 7)]},
        11: {"G": [f"AM{i}" for i in range(1, 7)]},
        12: {"H": [f"PM{i}" for i in range(1, 4)]},
        13: {"I": [f"PM{i}" for i in range(1, 4)]},
    })
    date_location_timeslot_map: Dict[Tuple[str, int, str], Tuple[str, str]] = field(
        init=False, default_factory=dict
    )

    def __post_init__(self):
        for date, intersections in self.avail_timeslots.items():
            month = date[5:7]  # "2025-06-16" -> "06"
            for (intersection, code), timeslots in intersections.items():
                location = self.location_map[(month, intersection, code)]
                for slot in timeslots:
                    key = (date, location, slot)
                    if key in self.date_location_timeslot_map:
                        existing = self.date_location_timeslot_map[key]
                        raise ValueError(
                            f"Duplicate mapping for {key}: "
                            f"{existing} vs {(intersection, code)}"
                        )
                    self.date_location_timeslot_map[key] = (intersection, code)

    def get_intersection_code(
        self, date: str, location: int, timeslot: str
    ) -> Tuple[str, str]:
        """Look up (intersection, code) for a given date, location, and timeslot."""
        try:
            return self.date_location_timeslot_map[(date, location, timeslot)]
        except KeyError:
            raise KeyError(
                f"No intersection/code found for "
                f"date={date!r}, location={location!r}, timeslot={timeslot!r}"
            )


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
