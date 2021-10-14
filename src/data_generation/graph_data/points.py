from habitat_sim import ShortestPath
from habitat_sim.utils.common import quat_from_angle_axis
from habitat_sim.geo import UP
import random
import quaternion
from numpy.linalg import inv, norm
import numpy as np

from src.utils.sim_utils import se3_to_mat
from src.data_generation.graph_data.graph import *


def newPoint(nodeid, A, B, pathfinder):
    newPos = B[0:3, 3]
    # newRot = B[0:3, 0:3]
    edge_attr_1 = inv(A) @ B
    edge_attr_2 = inv(B) @ A

    if not pathfinder.is_navigable(newPos):  # if its not navigable
        return None

    path = ShortestPath()
    path.requested_start = A[0:3, 3]
    path.requested_end = newPos
    pathfinder.find_path(path)
    geodesic_distance = path.geodesic_distance
    eculidean_distance = norm(np.asarray(A[0:3, 3]) - np.asarray(newPos)) + 0.00001
    if geodesic_distance / eculidean_distance > 1.1:
        return None

    return [nodeid, B, edge_attr_1, edge_attr_2]


"""VAILD POINT"""


def check_points(node, edgeDistance, pathfinder):
    nav = []
    stateA = se3_to_mat(quaternion.from_float_array(node.rot), np.asarray(node.pos))
    forward = se3_to_mat(
        quaternion.from_float_array([1, 0, 0, 0]), np.asarray([0, 0, -edgeDistance])
    )
    stateForward = stateA @ forward
    p = newPoint(node.nodeid, stateA, stateForward, pathfinder)
    if p is not None:
        p[3] = np.deg2rad(0)
        nav.append(p)

    right = se3_to_mat(
        quat_from_angle_axis(np.deg2rad(-1 * -55), UP), np.asarray([0, 0, 0])
    )
    stateRight = stateA @ right @ forward
    p = newPoint(node.nodeid, stateA, stateRight, pathfinder)
    if p is not None:
        p[3] = np.deg2rad(-60)
        nav.append(p)

    right = se3_to_mat(
        quat_from_angle_axis(np.deg2rad(-1 * 45), UP), np.asarray([0, 0, 0])
    )
    stateRight = stateA @ right @ forward
    p = newPoint(node.nodeid, stateA, stateRight, pathfinder)
    if p is not None:
        p[3] = np.deg2rad(-40)
        nav.append(p)

    right = se3_to_mat(
        quat_from_angle_axis(np.deg2rad(-1 * 20), UP), np.asarray([0, 0, 0])
    )
    stateRight = stateA @ right @ forward
    p = newPoint(node.nodeid, stateA, stateRight, pathfinder)
    if p is not None:
        p[3] = np.deg2rad(-20)
        nav.append(p)

    left = se3_to_mat(quat_from_angle_axis(np.deg2rad(60), UP), np.asarray([0, 0, 0]))
    stateLeft = stateA @ left @ forward
    p = newPoint(node.nodeid, stateA, stateLeft, pathfinder)
    if p is not None:
        p[3] = np.deg2rad(60)
        nav.append(p)

    left = se3_to_mat(quat_from_angle_axis(np.deg2rad(40), UP), np.asarray([0, 0, 0]))
    stateLeft = stateA @ left @ forward
    p = newPoint(node.nodeid, stateA, stateLeft, pathfinder)
    if p is not None:
        p[3] = np.deg2rad(40)
        nav.append(p)

    left = se3_to_mat(quat_from_angle_axis(np.deg2rad(20), UP), np.asarray([0, 0, 0]))
    stateLeft = stateA @ left @ forward
    p = newPoint(node.nodeid, stateA, stateLeft, pathfinder)
    if p is not None:
        p[3] = np.deg2rad(20)
        nav.append(p)

    backward = se3_to_mat(
        quat_from_angle_axis(np.deg2rad(180), UP), np.asarray([0, 0, 0])
    )
    stateBackward = stateA @ backward
    p = newPoint(node.nodeid, stateA, stateBackward, pathfinder)
    if p is not None:
        p[3] = np.deg2rad(180)
        nav.append(p)

    return nav


def generate_new_points(graph, pathfinder):
    navigable_points = {}

    for n in graph.nodes:
        edgeDistance = 1
        nav = check_points(n, edgeDistance, pathfinder)
        navigable_points[n.nodeid] = nav

    return navigable_points


"""non_ground truth VAILD POINT"""
#
from src.functions.validity_func.validity_utils import *
from src.functions.validity_func.map_builder import build_mapper


def find_point(node, sim):
    nav = []
    # Get Depth image
    edgeDistance = 1.5
    angles = [-55, -45, -30, -15, 0, 15, 30, 45, 55]
    map_size_cm = 1200
    map_resolution = 5
    as_position = np.asarray(node.pos)
    as_rotation = quaternion.from_float_array(node.rot)
    obs = sim.get_observations_at(as_position, as_rotation)
    curr_depth_img = obs["depth"]

    # Get projection map
    pano_mapper = build_mapper()
    proj_map, exp_map, wall_map = get_panorama_and_projection(
        pano_mapper, curr_depth_img, map_size_cm
    )
    # Validity based on depth
    radius = 1.6
    labels = get_labels(
        radius,
        angles,
        proj_map,
        wall_map,
        exp_map,
        as_position,
        as_rotation,
        map_size_cm,
        map_resolution,
    )
    A = se3_to_mat(quaternion.from_float_array(node.rot), np.asarray(node.pos))
    forward = se3_to_mat(
        quaternion.from_float_array([1, 0, 0, 0]), np.asarray([0, 0, -edgeDistance])
    )
    forward = A @ forward
    for ang, lab in zip(angles, labels):
        if lab:
            if ang > 0:
                left = se3_to_mat(
                    quat_from_angle_axis(np.deg2rad(-1 * ang), UP),
                    np.asarray([0, 0, 0]),
                )
                B = A @ left @ forward
            elif ang < 0:
                left = se3_to_mat(
                    quat_from_angle_axis(np.deg2rad(-1 * ang), UP),
                    np.asarray([0, 0, 0]),
                )
                B = A @ left @ forward
            else:
                A @ forward
            edge_attr_1 = inv(A) @ B
            nav.append([node.nodeid, B, edge_attr_1, ang])
    return nav


def gen_points(graph, sim):
    navigable_points = {}
    for n in graph.nodes:
        nav = find_point(n, sim)
        navigable_points[n.nodeid] = nav
    return navigable_points
