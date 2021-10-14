import numpy as np
import skimage
import quaternion
from src.functions.validity_func.fmm_planner import FMMPlanner
import cv2


def cartesian_to_polar(x, y):
    rho = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)
    return rho, phi


def quaternion_rotate_vector(quat: np.quaternion, v: np.array) -> np.array:
    r"""Rotates a vector by a quaternion
    Args:
        quaternion: The quaternion to rotate by
        v: The vector to rotate
    Returns:
        np.array: The rotated vector
    """
    vq = np.quaternion(0, 0, 0, 0)
    vq.imag = v
    return (quat * vq * quat.inverse()).imag


def get_relative_location(source_position, source_rotation, goal_position):
    direction_vector = np.asarray(goal_position) - np.asarray(source_position)
    direction_vector_agent = quaternion_rotate_vector(
        quaternion.from_float_array(source_rotation).inverse(), direction_vector
    )
    rho, phi = cartesian_to_polar(-direction_vector_agent[2], direction_vector_agent[0])
    return rho, -phi


def get_l2_distance(x1, x2, y1, y2):
    """
    Computes the L2 distance between two points.
    """
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


def get_sim_location(as_pos, as_rot):
    """
    Input:
        as_pos: agent_state position (x,y,z)
        as_rot: agent_state rotation (4D quaternion)
    Output:
        sim_pose: 3-dof sim pose
    """
    x = as_pos[2]
    y = as_pos[0]
    o = quaternion.as_rotation_vector(as_rot)[1]
    sim_pose = (x, y, o)
    return sim_pose


def get_rel_pose_change(pos2, pos1):
    x1, y1, o1 = pos1
    x2, y2, o2 = pos2

    theta = np.arctan2(y2 - y1, x2 - x1) - o1
    dist = get_l2_distance(x1, x2, y1, y2)
    dx = dist * np.cos(theta)
    dy = dist * np.sin(theta)
    do = o2 - o1

    return dx, dy, do


def get_new_pose(pose, rel_pose_change):
    x, y, o = pose
    dx, dy, do = rel_pose_change

    global_dx = dx * np.sin(o) + dy * np.cos(o)
    global_dy = dx * np.cos(o) - dy * np.sin(o)
    x += global_dy
    y += global_dx
    o += do
    o = o % (2 * np.pi)

    return x, y, o


def get_sim_pose_from_mapper_coords(
    map_resolution, mapper_coords, mapper_size, sim_origin
):
    """
    Input:
        mapper_coords: mapper coordinates (x, y)
        mapper_size: (l, w)
        sim_origin: 3-dof sim_pose which cooresponds to
                    (l/2, w/2) in mapper coordinates
    Output:
        sim_pose:  3-dof sim pose
    """

    pos1 = [
        mapper_size[0] * map_resolution / 200.0,
        mapper_size[1] * map_resolution / 200.0,
        0.0,
    ]

    dx = (mapper_coords[0] - mapper_size[0] / 2.0) * map_resolution / 100.0
    dy = (mapper_coords[1] - mapper_size[0] / 2.0) * map_resolution / 100.0

    pos2 = [
        mapper_size[0] * map_resolution / 200.0 - dy,
        mapper_size[1] * map_resolution / 200.0 - dx,
        0.0,
    ]

    rel_pose_change = get_rel_pose_change(pos2, pos1)

    sim_pose = get_new_pose(sim_origin, rel_pose_change)
    return sim_pose


def get_scene_map_coords_from_sim_pose(map_dict, map_resolution, sim_pose):
    rel_pos = (
        sim_pose[0] * 100.0 - map_dict["min_xy"][0],
        sim_pose[1] * 100.0 - map_dict["min_xy"][1],
    )
    coords = (int(rel_pos[1] / map_resolution), int(rel_pos[0] / map_resolution))
    return coords


def sector_mask(shape, centre, radius, angle_range):
    """
    From: https://stackoverflow.com/questions/18352973/mask-a-circular-sector-in-a-numpy-array
    Return a boolean mask for a circular sector. The start/stop angles in
    `angle_range` should be given in clockwise order.
    """

    x, y = np.ogrid[: shape[0], : shape[1]]
    cx, cy = centre
    tmin, tmax = np.deg2rad(angle_range)

    # ensure stop angle > start angle
    if tmax < tmin:
        tmax += 2 * np.pi

    # convert cartesian --> polar coordinates
    r2 = (x - cx) * (x - cx) + (y - cy) * (y - cy)
    theta = np.arctan2(x - cx, y - cy) - tmin

    # wrap angles between 0 and 2*pi
    theta %= 2 * np.pi

    # circular mask
    circmask = r2 <= radius * radius

    # angular mask
    anglemask = theta <= (tmax - tmin)

    # return circmask * anglemask
    return circmask


import math


def get_patch_coordinates(angles, goal, pano_radius):
    dist = pano_radius
    coords = []
    # 120 FOV -> crop to 110 get the following angles
    # angles = [-55, -45, -30, -15, 0, 15, 30, 45, 55]
    for ang in angles:
        ang = math.radians(ang)
        y = int(dist * np.cos(ang) + goal[0])
        x = int(dist * np.sin(ang) + goal[1])
        coords.append([x, y])
    return coords


def get_panorama_and_projection(pano_mapper, curr_depth_img, map_size_cm, reset=True):
    if reset:
        pano_mapper.reset_map()
    mapper_pose = (map_size_cm / 2.0, map_size_cm / 2.0, 0)
    fp_proj, proj_map, fp_explored, exp_map, wall_map = pano_mapper.update_map(
        curr_depth_img[:, :, 0] * 1000.0, mapper_pose
    )
    return proj_map, exp_map, wall_map


def get_labels(
    radius,
    angles,
    proj_map,
    exp_map,
    as_position,
    as_rotation,
    map_size_cm,
    map_resolution,
):
    selem = skimage.morphology.disk(5 / map_resolution)
    labels = []

    small_pano_radius = (map_size_cm / map_resolution) / 2.0 / 3
    goal_init = [map_size_cm / map_resolution / 2.0, map_size_cm / map_resolution / 2.0]

    traversible = skimage.morphology.binary_dilation(proj_map, selem) != True
    planner = FMMPlanner(traversible, 360 // 10, 1)
    planner.set_goal(goal_init)

    coords1 = get_patch_coordinates(angles, goal_init, goal_init[0])
    coords2 = get_patch_coordinates(angles, goal_init, small_pano_radius)

    for coord1, coord2 in zip(coords1, coords2):
        if planner.fmm_dist[coord1[0], coord1[1]] < goal_init[0] * 1.05:
            label = 1.0
        elif planner.fmm_dist[coord2[0], coord2[1]] < small_pano_radius * 1.025:
            label = 1.0
        else:
            label = 0.0
        labels.append(label)
    return labels
