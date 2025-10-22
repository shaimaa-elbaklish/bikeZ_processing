# BikeZ Analysis

## Mobilysis Data Processing: EKF

### How to Run
1. Change the path to the BikeZ data. In `_constant.py`, change the `data_root` attribute in the `BikeZ_Config` dataclass.
```
@dataclass
class BikeZ_Config:
    data_root: str = "/path/to/BikeZ/Zurich_202506/bike_trajectories/v2/" # <-- CHANGE HERE!
    avail_dates: Tuple[str] = ("2025-06-16", "2025-06-17")
    avail_intersections: Tuple[str] = ("D1", "D2", "D3", "D4")
    timezone = pytz.timezone('Europe/Berlin')
    X_2056_Bounds: Tuple[float] = (2682700, 2682860)
    Y_2056_Bounds: Tuple[float] = (1247820, 1247960)
    fps: float = 25.0
```
2. In `main_kf.py`, you need to change the inputs in the CONSTANTS section. These inputs specify which data file you want to process via EKF.
```
date = BikeZ_Config.avail_dates[0]
intersection = BikeZ_Config.avail_intersections[2]
time_slot = 'AM1'
code = 'E'
``` 
3. Run `main_kf.py`.

### Outputs
The output is a csv file saved in the same location as the original file. It has the naming convention:
```
<original-filename>-ekf.csv
Example Original File: trajectories_bikes_2025-06-16_D3_AM1_E-1.csv
---> Output File: trajectories_bikes_2025-06-16_D3_AM1_E-1-ekf.csv
```

It has the following columns.
- `veh_id`, `veh_type`: same as original.
- `time`: time (seconds) from when the drone started recording.
- `datetime`: global timestamp string.
- `x_act`, `y_act`: original X and Y position (meters) in the EPSG:2056 Projected coordinate system.
- `speed`, `a`: original estimated speed (km/h) and acceleration (m/s<sup>2</sup>).
- `lat`, `lon`: original latitude and longitude of bike position.
- `x`, `y`: original X and Y offset positions (meters), <br> i.e. `x = x_act - x_offset` and `y = y - y_offset`; <br> where x_offset = 2682700 and y_offset = 1247820. This is done for numerical stability.
- `missing`: boolean flag of whether this row constituted missing data in the original file.
- `x_ekf`, `y_ekf`: EKF-smoothed X and Y offset positions (meters)
- `speed_ekf`, `a_ekf`: EKF-smoothed speed (km/h) and acceleration (m/s<sup>2</sup>).
- `angle_ekf`: EKF-estimated heading angle (rad) of the bicycle.

