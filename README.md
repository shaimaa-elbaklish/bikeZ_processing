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



## Mobilysis Data Processing: Lane Coordinate Transformation

## Steps
This is performed on Gessnerbrücke (D3 location).

1. **Centerlines, Stoplines and Lane Boundaries Extraction via OSMNX and SwissTopo**:
    - *Relevant code*: `maps_gessnerbrucke.py`
    - *Using OSMNX*:
        1. Get all features in `Zürich, Switzerland`.
        ```
        place = "Zürich, Switzerland" # Define your area
        tags = {"highway": True} # Download all features with highway tag
        gdf_main = ox.features.features_from_place(place, tags=tags)
        ```
        2. Filter for each road to extract all possible centerlines.
        ```
        road_name = "Lagerstrasse"  # Filter for road name
        gdf = gdf_main[gdf_main['name'] == road_name]
        gdf = gdf[(gdf.geometry.type == "LineString") & (gdf['highway'].isin(main_road_types))]
        merged_centerline = linemerge(list(gdf.geometry))
        # Extract coordinates (handle both LineString and MultiLineString)
        if merged_centerline.geom_type == 'LineString':
            coords = list(merged_centerline.coords)
        elif merged_centerline.geom_type == 'MultiLineString':
            coords = []
            for line in merged_centerline.geoms:
                coords.extend(list(line.coords))
        # Convert (lon, lat) to (lat, lon) for folium
        lagerstrasse_branch = [(lat, lon) for lon, lat, _ in coords] # <--- This is the extracted centerline
        ```
    - *Using SwissTopo*:
        1. Add remaining centerlines using the draw tool on SwissTopo. For example, see: https://s.geo.admin.ch/jkfynb8vzf5w
        2. Download the drawing as `kml` file.
        3. Extract the relevant geometries according to the `Description` you have added to each drawn element.
        ```
        kml_path = "../maps/from_swisstopo/gessnerbrucke.kml"
        gdf_swisstopo = gpd.read_file(kml_path, driver='KML')
        row = gdf_swisstopo[gdf_swisstopo['Description'] == 'Stadttunnel_Centerline'].copy()
        centerline = row.geometry.item()
        stadttunnel_branch = [(c[1], c[0]) for c in centerline.coords] <--- This is the extracted centerline
        ```
2. **B-Splines Fitting for All Possible Connections**:
    - *Relevant code*: `maps_gessnerbrucke.py`
    - *Relevant tools*: `tools_coordinateTransform.py` and `tools_map_visualization.py`
    - Now, our centerlines maybe continuous or disconnected. Also, we need to extract centerlines for all possible motions within the intersection. For the Gessnerbrücke location, we have a 4 way intersection with 8 legs. This means 12 possible motions. Therefore, we need to have 12 centerline splines.
    - Splines are expressed in **XY:2056 coordinate system**.
    - Finally, all splines need to be saved for future use (via a dictionary). Path: `../data/centerlines_splines_<date>_<location>.pkl`. For example, `../data/centerlines_splines_2025-06-16_D3.pkl` for Gessnerbrücke (D3 location).
    - *Through Motions with Continuous Centerlines:*
    ```
    # Example for North to South Motion
    north_south_spl = fit_roadway_centerline_spline(kasernenstrasse_NS_branch) # tuple (tck, unew, cum_dist)
    splines_dict['N_2_S'] = north_south_spl # <--- Saving
    ```
    - Disconnected centerlines (whether through or turning) need to be connected smoothly (i.e. C2 contiinuity). For this purpose, I use clothoids (i.e. Euler spirals) that connect the disconnected parts via their tangents and curvatures to ensure smoothness. If clothoids fail (happens when disconnected parts are (near-)collinear), a hermitian connection is used (i.e. provides C1 continuity but is sufficient for (near-)collinear connections).
    - Also, for turning connections, we need to cut the involved centerlines at their respective stoplines so that we can make the connections.
    - *Turning Motions*:
    ```
    # Example for South to East Motion
    # Cut kasernenstrasse_SN_branch at stopline
    centerline = LineString([(lon, lat) for lat, lon in kasernenstrasse_SN_branch])
    centerline = cut_line_at_stop(centerline, kasernenstrasse_south_stopline, choose='first')
    kasernenstrasse_SN_branch = [(lat, lon) for lon, lat in centerline.coords]
    # Get its spline in XY:2056 coordinates
    tmp_spl = fit_roadway_centerline_spline(kasernenstrasse_SN_branch)
    tck = tmp_spl[0]
    x_spline, y_spline = splev(np.linspace(0, 1, 50), tck)
    xy_kasernenstrasse_south = np.column_stack((x_spline, y_spline))
    # Make clothoid connection
    south_east_merged_coords, _, _= connect_lines(xy_kasernenstrasse_south, xy_gessnerbrucke_east, n_connector=120, verbose=True)
    south_east_spl = fit_roadway_centerline_spline(south_east_merged_coords, coordsys='2056')
    plot_spline_xy_2056(m, south_east_spl, label="Turning Centerline (S->E)", 
                        linecolor=colors_dict['south'], linedashed=True, start_point=True)
    splines_dict['S_2_E'] = south_east_spl <--- Saving
    ```
    - We can visualize all centerlines on map via `folium`.
    ```
    m = create_swisstopo_map(center_lat=df["lat_ekf"].mean(), center_lon=df["lon_ekf"].mean(), add_layer_control=False)
    plot_all_centerlines_splines_xy_2056(m, splines_dict, add_layer_control=True) <--- Layer Control must be True in last layer only (if exists)

    m.save(f"../maps/road_centerlines_map_{date}_{intersection}.html") <--- Map with only centerlines

    m = create_swisstopo_map(center_lat=df["lat_ekf"].mean(), center_lon=df["lon_ekf"].mean(), add_layer_control=False)
    plot_all_centerlines_splines_xy_2056(m, splines_dict, add_layer_control=False)
    plot_bicycles_trajectories_xy_2056(m, df, linecolor='black', linealpha=0.25, add_layer_control=True) <--- Layer Control must be True in last layer only (if exists)

    m.save(f"../maps/trajectories_map_{date}_{intersection}_{time_slot}_{code}.html") <--- Map with only centerlines + all bicycles trajectories
    ```
    - Same thing is done for bike lane boundaries (if exists). They are saved in `../data/bike_lane_boundaries_splines_<date>_<location>.pkl`. Their purpose is to determine whether the bike is driving within its dedicated lane and where that lane ends (i.e. stopline).
3. Projection onto Roadway-Aligned (i.e. Lane) Coordinates:
    - *Relevant code*: `main_coordinate_transform.py`
    - *Relevant tools*: `tools_coordinateTransform.py`
    - *Note*: This part I am still developing to be generalized to all bicycles. It is working on an individual bicycle level.
    - *Core Idea*:
        1. Determine the relevant motion for the bicycle (i.e. `N_2_S`, `W_2_S`, etc.) in order to use the appropriate centerline. 
        ```
        centerline_id = match_bicycle_to_centerline(bike_df, centerlines_start_end_pts_dict)
        centerline_spl = centerlines_spl_dict[centerline_id]
        tck, unew, cum_dist = centerline_spl
        ```
        This matching is via start and end points of centerlines and bicycle trajectories. This works in most cases. However, when cyclists start their trajectories in the middle of the intersection, the selection may not be appropriate.
        2. Obtain the corresponding bike lane boundaries (if exists), their validity regions and their lateral offset with respect to the matched centerline.
        ```
        centerline_start, centerline_end = centerline_id.split('_2_')
        lb_keys = [
            f"{centerline_start}_{OPP_DIRECTIONS[centerline_start]}B",
            f"{centerline_end}_{centerline_end}B"
        ]
        lane_boundary_spl = [
            lane_boundaries_spl_dict[lb_keys[0]],
            lane_boundaries_spl_dict[lb_keys[1]],
        ]
        lane_boundary_info <--- This is where we store the validity regions and lateral offsets (code omitted here for brevity)
        ```
        3. Perform the coordinate transformation.
        ```
        roadway_out = bike_df.apply(lambda row: convert_xy2056_to_roadway_coordinates([row['x_act_ekf'], row['y_act_ekf']], tck, unew, cum_dist), axis=1)
        ```
        The `roadway_out` and other relevant data are then properly matched in the following columns.
        - `Position_Longitudinal` and `Position_Lateral`: <br> longitudinal and lateral positions relative to the matched centerline (in meters).
        - `Speed_Longitudinal` and `Speed_Lateral`: <br> longitudinal and lateral speeds (in km/h).
        - `Accel_Longitudinal` and `Accel_Lateral`: <br> longitudinal and lateral accelerations (in m/s<sup>2</sup>).
        - `Bike_Lane_ID`: <br> Where valid, it is the ID of the dedicated bike lane (e.g. `S_NB` means south leg on the northbound direction). Default is `None`.
        - `In_Bike_Lane`: <br> Where valid, `Bike_Lane_ID` is not `None`. `In_Bike_Lane` is a boolean flag where `True` means that the cyclist is driving inside the dedicated bike lane. This is currently done within a tolerance of 0.25 meters. Default is `None`.

**Pertaining Questions/Steps:**
1. Do locations differ per date?
2. Bike lane widths per location.
3. Signalized or nnsignalized intersection per location.
4. XY:2056 Bounds per location.
5. Extending to other locations (D1, D2, D4).

