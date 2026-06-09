"""
TITLE
-------------------------------------------
Authors:        Shaimaa K. El-Baklish
Organization:   ETH Zürich, Switzerland, IVT - Institute for Transportation Planning and Systems
Development:    2025
Submitted to:   JOURNAL
-------------------------------------------
"""

###############################################################################
# IMPORTS
###############################################################################
import os
import sys

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

from shapely.geometry import Point, Polygon
from shapely.affinity import translate
from shapely.ops import transform
from pedpy import PEDPY_BLUE, PEDPY_ORANGE, DENSITY_COL
from pedpy import load_trajectory, plot_trajectories, plot_measurement_setup
from pedpy import WalkableArea, MeasurementArea, MeasurementLine, Cutoff
from pedpy import compute_individual_voronoi_polygons, compute_voronoi_density
from pedpy import compute_individual_speed, compute_voronoi_speed
from pedpy import SpeedCalculation, plot_speed, plot_density, plot_voronoi_cells
from pedpy.methods.speed_calculator import compute_species
from pedpy.methods.speed_calculator import compute_line_speed
from pedpy.methods.density_calculator import compute_line_density
from pedpy.methods.flow_calculator import compute_line_flow

# #############################################################################
# FUNCTIONS
# #############################################################################
def prepare_data_pedpy(df, fps, walkable_area):
    df['frame_nr'] = np.round(df['time'] * fps, decimals=0)
    df['frame_nr'] = df['frame_nr'].astype(int)
    df = df[["veh_id", "frame_nr", "x_ekf", "y_ekf"]]
    df = df.rename(columns={
        "veh_id": "ID",
        "frame_nr": "frame",
        "x_ekf": "X",
        "y_ekf": "Y"
    })
    df = df.astype({"ID": "int"})
    df["Z"] = 1.1 # average bicycle height
    df = df[["ID", "frame", "X", "Y", "Z"]]
    # drop points outside of walkable area
    df["Outside"] = df.apply(lambda row: walkable_area.polygon.intersection(Point(row['X'], row['Y'])).is_empty, axis=1)
    df = df[~df["Outside"]]
    df = df.drop(columns='Outside')
    return df


def define_measurement_setup(kml_path, X_2056_offset, Y_2056_offset):
    gdf_swisstopo = gpd.read_file(kml_path, driver='KML')
    gdf_swisstopo = gdf_swisstopo.to_crs(2056)

    row = gdf_swisstopo[gdf_swisstopo['Description'] == 'Observed_Area'].copy()
    observed_area_polygon = transform(lambda x, y, z=None: (x, y), row.geometry.item())
    observed_area_polygon = translate(observed_area_polygon, 
                                      xoff=-X_2056_offset, yoff=-Y_2056_offset)
    walkable_area = WalkableArea(observed_area_polygon)
    
    measurement_areas = []
    measurement_lines = []
    for _, row in gdf_swisstopo.iterrows():
        if row['Description'][-3:] == '_MA':
            ma_polygon = transform(lambda x, y, z=None: (x, y), row.geometry)
            ma_polygon = ma_polygon.convex_hull
            ma_polygon = translate(ma_polygon, xoff=-X_2056_offset, yoff=-Y_2056_offset)
            measurement_areas.append(MeasurementArea(ma_polygon))
        
        if row['Description'][-9:] == '_Stopline':
            ma_polygon = transform(lambda x, y, z=None: (x, y), row.geometry)
            ma_polygon = translate(ma_polygon, xoff=-X_2056_offset, yoff=-Y_2056_offset)
            measurement_lines.append(MeasurementLine([ma_polygon.coords[0], ma_polygon.coords[-1]]))
    
    return walkable_area, measurement_areas, measurement_lines


def compute_voronoi_states(traj, walkable_area, measurement_areas, return_full=False):
    try:
        individual = compute_individual_voronoi_polygons(
            traj_data=traj, walkable_area=walkable_area,
            cut_off=Cutoff(radius=(1.8+1.5)/2, quad_segments=1)
        )
    except IndexError:      
        plt.figure()
        plot_measurement_setup(
            walkable_area=walkable_area, traj=traj, traj_alpha=0.5, traj_width=1,
            measurement_areas=measurement_areas, ma_line_width=2, ma_alpha=0.5,
        ).set_aspect("equal")
        plt.show()
        sys.exit(1)
    
    individual_speed_single_sided = compute_individual_speed(
        traj_data=traj, frame_step=int(traj.frame_rate), compute_velocity=False,
        speed_calculation=SpeedCalculation.BORDER_SINGLE_SIDED,
    )
    individual_joined = individual_speed_single_sided.merge(individual, on=['id', 'frame'], how='inner')
    individual_joined['flow'] = individual_joined['density'] * individual_joined['speed'] # bic/m^2 * m/s = bic/s/m
    # switch flow and speed columns
    individual_joined['temp'] = individual_joined['flow']
    individual_joined['flow'] = individual_joined['speed']
    individual_joined['speed'] = individual_joined['temp']
    individual_joined = individual_joined.drop(columns=['temp'])
    
    voronoi_density_areas, voronoi_speed_areas = [], []
    for ma in measurement_areas:
        density_voronoi, intersecting = compute_voronoi_density(
            individual_voronoi_data=individual, measurement_area=ma
        )
        voronoi_density_areas.append(density_voronoi)
    
        voronoi_speed = compute_voronoi_speed(
            traj_data=traj, individual_voronoi_intersection=intersecting,
            individual_speed=individual_joined, measurement_area=ma,
        )
        voronoi_speed_areas.append(voronoi_speed)
    
    voronoi_states_areas = []
    for i in range(len(measurement_areas)):
        voronoi_states = voronoi_density_areas[i].join(voronoi_speed_areas[i], how="inner")
        voronoi_states = voronoi_states.rename(columns={'speed': 'flow'}) # switch back
        voronoi_states['density'] = voronoi_states['density']*1000.0 # bic/km/m
        voronoi_states['flow'] = voronoi_states['flow']*3600.0 # bic/h/m
        voronoi_states['speed'] = voronoi_states['flow'] / voronoi_states['density'] # bic/h/m / bic/km/m = km/h
        voronoi_states_areas.append(voronoi_states)
    
    voronoi_states_all = pd.concat(voronoi_states_areas)
    voronoi_states_all = voronoi_states_all.rename(columns={
        'density': 'Density', 'flow': 'Flow', 'speed': 'Speed'
    })
    if return_full:
        return voronoi_states_all, voronoi_states_areas, individual_joined
    return voronoi_states_all


def compute_voronoi_states_lines(traj, walkable_area, measurement_lines, return_full=False):
    try:
        individual = compute_individual_voronoi_polygons(
            traj_data=traj, walkable_area=walkable_area,
            cut_off=Cutoff(radius=(1.8+1.5)/2, quad_segments=1)
        )
    except IndexError:      
        plt.figure()
        plot_measurement_setup(
            walkable_area=walkable_area, traj=traj, traj_alpha=0.5, traj_width=1,
            measurement_lines=measurement_lines, ma_line_width=2, ma_alpha=0.5,
        ).set_aspect("equal")
        plt.show()
        sys.exit(1)
    # A species represents a set of pedestrians that encouters the measurement line frome the same side.
    voronoi_states_lines = []
    for ml in measurement_lines:
        species = compute_species(
            individual_voronoi_polygons=individual, measurement_line=ml, 
            trajectory_data=traj, frame_step=int(traj.frame_rate)
        )
        individual_speed_single_sided = compute_individual_speed(
            traj_data=traj, frame_step=int(traj.frame_rate), compute_velocity=True,
            speed_calculation=SpeedCalculation.BORDER_SINGLE_SIDED,
        )
        speeds = compute_line_speed(
            individual_voronoi_polygons=individual,
            measurement_line=ml,
            individual_speed=individual_speed_single_sided,
            species=species,
        )
        densities = compute_line_density(
            individual_voronoi_polygons=individual,
            measurement_line=ml,
            species=species,
        )
        flows = compute_line_flow(
            individual_voronoi_polygons=individual,
            measurement_line=ml,
            species=species,
            individual_speed=individual_speed_single_sided,
        )
        ts_df = densities[['frame', 'density']].copy()
        ts_df = ts_df.merge(flows[['frame', 'flow']], on=['frame'], how='inner')
        ts_df = ts_df.merge(speeds[['frame', 'speed']], on=['frame'], how='inner')
        ts_df['density'] = ts_df['density']*1000.0 # bic/km/m
        ts_df['flow'] = ts_df['flow']*3600.0 # bic/h/m
        ts_df['speed'] = ts_df['speed'] * 3.6 # km/h
        ts_df['speed_cl'] = ts_df['flow'] / ts_df['density'] # bic/h/m / bic/km/m = km/h
        voronoi_states_lines.append(ts_df)
    
    voronoi_states_all = pd.concat(voronoi_states_lines)
    voronoi_states_all = voronoi_states_all.rename(columns={
        'density': 'Density', 'flow': 'Flow', 'speed': 'Speed', 'speed_cl': 'Speed_CL'
    })
    if return_full:
        return voronoi_states_all, voronoi_states_lines, individual
    return voronoi_states_all