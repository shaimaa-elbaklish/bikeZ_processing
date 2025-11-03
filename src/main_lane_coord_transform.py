
from tools_coordinateTransform import (extract_roadway_centerline, project_point_onto_spline,
                                       convert_utm_to_roadway_coordinates, convert_roadway_to_utm_coordinates)

tck, unew, cum_dist = extract_roadway_centerline("BIKEZ")

for loop ...
t_star, closest_point = project_point_onto_spline(point, tck)
t_star, tangent, normal, s, d = convert_utm_to_roadway_coordinates(point, tck, unew, cum_dist)
