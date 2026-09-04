"""
TITLE OF PAPAER
-------------------------------------------
Authors:        Shaimaa El-Baklish
Organization:   ETH Zürich, Switzerland, IVT - Institute for Transportation Planning and Systems
Development:    2025-2026
Submitted to:   JOURNAL
-------------------------------------------

Central configuration and constants for the BikeZ trajectory-processing pipeline.
"""

# #############################################################################
# IMPORTS
# #############################################################################
import pytz
import numpy as np

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Tuple, Dict, List

# #############################################################################
# CONSTANTS: BIKE-Z
# #############################################################################


@dataclass
class BikeZ_Config:
    """
    Dataset paths, recording-session catalog, and spatial bounds for BikeZ.
    Holds everything a script needs to locate raw/subsampled data on disk
    and to validate a (date, intersection, code, timeslot) combination
    against what was actually recorded. 
    Provides functions `get_intersection_code(date, location, timeslot)` 
    and `get_available_dates_and_timeslots(location)`.
    """
    timezone = pytz.timezone('Europe/Berlin')
    fps: float = 25.0
    dir_root: str = "C:/Users/ShaimaaElBaklish/Documents/Datasets/BikeZ/"
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
            ("D1", "A"): [f"AM{i}" for i in range(1, 8)] + [f"PM{i}" for i in range(1, 7)],
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
    
    def get_available_dates_and_timeslots(self, location: int) -> Dict[str, List[str]]:
        """Return {date: [timeslots]} for all dates a given location was recorded."""
        result: Dict[str, List[str]] = {}
        for (date, loc, slot) in self.date_location_timeslot_map:
            if loc == location:
                result.setdefault(date, []).append(slot)
    
        if not result:
            raise KeyError(f"No dates found for location={location!r}")
    
        # sort timeslots within each date, and sort dates for stable output
        return {date: sorted(result[date]) for date in sorted(result)}


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


# #############################################################################
# CONSTANTS: Visualization
# #############################################################################
_GEOM_PALETTE = [
    'steelblue', 'tomato', 'mediumpurple', 'darkorange',
    'seagreen',  'crimson', 'goldenrod',   'teal',
    'slategray', 'orchid',  'sienna',      'cornflowerblue',
    'deeppink',  'olive',   'peru',        'dodgerblue',
]
_GEOM_PALETTE_FALLBACK = 'dimgray'


_SEG_PALETTE = [
    '#4878d0', '#ee854a', '#6acc65',  # royalblue, coral, mediumseagreen
    '#d65f5f', '#956cb4', '#8c613c',  # indianred, mediumpurple, sienna (brown)
    '#dc7ec0', '#2ec4b6', '#d5bb67',  # orchid (pink), lightseagreen (teal),  darkkhaki (gold)
    '#82c6e2', '#e45858', '#56b4e9',  # skyblue, indianred (brighter red), cornflowerblue (bright blue)
    '#9bcb4d', '#5b4dcb', '#c14dcb',  # yellowgreen, slateblue (indigo), mediumorchid 
    '#8a2834',  # brown (dark maroon)
]
_SEG_PALETTE_FALLBACK = '#888888'  # gray