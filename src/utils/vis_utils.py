import os
import cv2
import imageio
import numpy as np
from matplotlib import pyplot as plt
from habitat.utils.visualizations import maps


AGENT_SPRITE = imageio.imread(
    os.path.join(
        "/nethome/mhahn30/Simulator/habitat-lab/habitat/utils/visualizations",
        "assets",
        "maps_topdown_agent_sprite",
        "100x100.png",
    )
)
AGENT_SPRITE = np.ascontiguousarray(np.flipud(AGENT_SPRITE))


# convert 3d points to 2d topdown coordinates
def convert_points_to_topdown(pathfinder, points, meters_per_pixel):
    points_topdown = []
    bounds = pathfinder.get_bounds()
    for point in points:
        # convert 3D x,z to topdown x,y
        px = (point[0] - bounds[0][0]) / meters_per_pixel
        py = (point[2] - bounds[0][2]) / meters_per_pixel
        points_topdown.append(np.array([px, py]))
    return points_topdown


# display a topdown map with matplotlib
def display_map(topdown_map):
    plt.figure(figsize=(12, 8))
    ax = plt.subplot(1, 1, 1)
    ax.axis("off")
    plt.imshow(topdown_map)


def grid_map(sim, sim_topdown_map, point, switch=False):
    color_nav = (138, 43, 226)
    if switch:
        color_nav = (75, 0, 130)
    point_padding = 5
    a_x, a_y = maps.to_grid(
        point[2],
        point[0],
        sim_topdown_map.shape[0:2],
        sim=sim,
    )
    sim_topdown_map = cv2.circle(
        sim_topdown_map,
        (a_y, a_x),
        color=color_nav,
        radius=point_padding,
        thickness=-1,
    )
    return sim_topdown_map


def start_grid_map(sim, start_pose, goal_pose):
    # Params
    point_padding = 5

    # This map is a 2D boolean array
    sim_topdown_map = maps.get_topdown_map_from_sim(
        sim, map_resolution=480, draw_border=True
    )

    a_x, a_y = maps.to_grid(
        start_pose[2],
        start_pose[0],
        sim_topdown_map.shape[0:2],
        sim=sim,
    )
    sim_topdown_map[
        a_x - point_padding : a_x + point_padding + 1,
        a_y - point_padding : a_y + point_padding + 1,
    ] = maps.MAP_SOURCE_POINT_INDICATOR
    a_x, a_y = maps.to_grid(
        goal_pose[2],
        goal_pose[0],
        sim_topdown_map.shape[0:2],
        sim=sim,
    )
    sim_topdown_map[
        a_x - point_padding : a_x + point_padding + 1,
        a_y - point_padding : a_y + point_padding + 1,
    ] = maps.MAP_TARGET_POINT_INDICATOR
    uncolored_top_down = sim_topdown_map.copy()
    sim_topdown_map = maps.colorize_topdown_map(sim_topdown_map)
    return uncolored_top_down, sim_topdown_map
