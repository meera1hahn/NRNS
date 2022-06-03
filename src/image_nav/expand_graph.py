import quaternion
from numpy.linalg import norm
import numpy as np
import torch

from habitat_sim import ShortestPath
from habitat_sim.utils.common import quat_from_angle_axis
from habitat_sim.geo import UP

from src.utils.sim_utils import se3_to_mat, get_edge_attr
from src.functions.validity_func.map_builder import build_mapper
from src.functions.validity_func.validity_utils import (
    get_panorama_and_projection,
    get_labels,
)


"""
Determine explorable areas and then expand the graph with unexplored nodes (ghost ndoes) at these areas
"""


def expand(agent):
    if agent.args.use_gt_exporable_area:
        # using simulator
        ghost_nodes = get_gt_validity(agent, addbackwards=True)
    else:
        # using depth
        ghost_nodes = predict_validity(agent, addbackwards=True)

    add_ghosts(agent, ghost_nodes)

    if len(ghost_nodes) <= 2:
        prev_depth = agent.depth_img.copy()
        prev_pos = agent.current_pos.clone().detach()
        prev_rot = agent.current_rot.clone().detach()
        agent_turn(agent, 30)
        agent.current_pos = prev_pos
        agent.current_rot = prev_rot
        agent.depth_img = prev_depth
        agent_turn(agent, -30)
        agent.current_pos = prev_pos
        agent.current_rot = prev_rot
        agent.depth_img = prev_depth
        agent.steps += 8


"""
If there are no valid areas in front of the agent (i.e. agent is infront of a wall). 
Agent turns 30 degrees and searches for explorable area.
"""


def agent_turn(agent, degrees):
    stateA = se3_to_mat(
        quaternion.from_float_array(agent.current_rot.numpy()),
        agent.current_pos.numpy(),
    )
    stateB = stateA @ se3_to_mat(
        quat_from_angle_axis(np.deg2rad(degrees), UP), np.asarray([0, 0, 0])
    )
    prev_rot = agent.current_rot.clone().detach()
    agent.current_rot = torch.tensor(
        quaternion.as_float_array(quaternion.from_rotation_matrix(stateB[0:3, 0:3]))
    )
    obs = agent.sim.get_observations_at(
        agent.current_pos.numpy().tolist(),
        quaternion.from_float_array(agent.current_rot.numpy()),
    )
    agent.depth_img = obs["depth"]
    if agent.args.use_gt_exporable_area:
        # using simulator
        ghost_nodes = get_gt_validity(agent, addbackwards=False)
    else:
        # using depth
        ghost_nodes = predict_validity(agent, addbackwards=False)

    agent.current_rot = prev_rot
    add_ghosts(agent, ghost_nodes)


"""
Add unexplored nodes at areas deemed to be explorable
"""


def add_ghosts(agent, ghost_nodes):
    addEdge = []
    addNode = []
    for node in ghost_nodes:
        node_rotation = quaternion.as_float_array(
            quaternion.from_rotation_matrix(node[0:3, 0:3])
        )
        # snap to mesh cause of so many holes floor in mp3d, can be removed
        navigable, node[0:3, 3] = testPoint(node[0:3, 3], agent)
        if not navigable:
            continue
        localized, location = agent.localize(node[0:3, 3], node_rotation)
        if localized:
            addEdge.append(location)
        else:
            addNode.append(node)
    for e in set(addEdge):
        add_ghost_edge(agent, e)
    for n in addNode:
        add_ghost_node(agent, n)


"""
Add edges from explored nodes to unexplored nodes
"""


def add_ghost_edge(agent, location):
    edge_attr, delta_rot = get_edge_attr(
        agent.current_pos.numpy(),
        agent.current_rot.numpy(),
        agent.node_poses[location].numpy(),
        agent.node_rots[location].numpy(),
    )
    agent.add_edge([agent.current_node, location], edge_attr, delta_rot)


"""
Add unexplored nodes in agents graph
"""


def add_ghost_node(agent, node):
    feat = None
    agent.add_node(node[0:3, 3], node[0:3, 0:3], feat)
    edge_attr, delta_rot = get_edge_attr(
        agent.current_pos.numpy(),
        agent.current_rot.numpy(),
        node[0:3, 3],
        quaternion.as_float_array(quaternion.from_rotation_matrix(node[0:3, 0:3])),
    )
    agent.add_edge([agent.current_node, agent.total_nodes - 1], edge_attr, delta_rot)


"""
Predict geometrically explorable areas using depth
"""


def predict_validity(agent, addbackwards):
    # Test angles
    angles = [0, -15, 15, -30, 30, -45, 45, -55, 55]
    # Get projection map
    pano_mapper = build_mapper()
    proj_map, exp_map, wall_map = get_panorama_and_projection(
        pano_mapper, agent.depth_img.copy(), agent.map_size_cm
    )

    # Validity based on depth
    radius = 1.5
    labels = get_labels(
        radius,
        angles,
        wall_map,
        proj_map,
        agent.current_pos.detach().clone().numpy(),
        quaternion.from_float_array(agent.current_rot.detach().clone().numpy()),
        agent.map_size_cm,
        agent.map_resolution,
    )

    # Using simulator for teleportation
    stateA = se3_to_mat(
        quaternion.from_float_array(agent.current_rot.numpy()),
        agent.current_pos.numpy(),
    )
    forward = se3_to_mat(
        quaternion.from_float_array([1, 0, 0, 0]),
        np.asarray([0, 0, -agent.edge_length]),
    )
    ghost_nodes = []
    angles = [0, -15, 15, -30, 30, -45, 45, -60, 60]
    for ang, label in zip(angles, labels):
        if label == 0.0:
            continue
        stateB = (
            stateA
            @ se3_to_mat(
                quat_from_angle_axis(np.deg2rad(ang), UP), np.asarray([0, 0, 0])
            )
            @ forward
        )
        ghost_nodes.append(stateB)

    # Add Backward
    if addbackwards:
        stateB = stateA @ se3_to_mat(
            quat_from_angle_axis(np.deg2rad(180), UP), np.asarray([0, 0, 0])
        )
        ghost_nodes.append(stateB)

    return ghost_nodes


"""
Next functions for: Get geometrically explorable areas using habitat and the mesh itself
"""


def get_gt_validity(agent, addbackwards):
    stateA = se3_to_mat(
        quaternion.from_float_array(agent.current_rot.numpy()),
        agent.current_pos.numpy(),
    )
    middle_points = move_dist(stateA, agent.pathfinder, agent.edge_length, addbackwards)
    return middle_points


def testPoint(newPos, agent):
    currPos = agent.current_pos.numpy()
    if not agent.pathfinder.is_navigable(newPos):
        tempPos = agent.pathfinder.snap_point(newPos)
        snapDist = norm(newPos[[0, 2]] - np.asarray(tempPos)[[0, 2]])
        newPos = tempPos
        if snapDist > 0.15:
            return False, None

    path = ShortestPath()
    path.requested_start = currPos
    path.requested_end = newPos
    agent.pathfinder.find_path(path)
    geodesic_distance = path.geodesic_distance
    eculidean_distance = norm(currPos - np.asarray(newPos))
    if geodesic_distance / eculidean_distance > 1.2:
        return False, newPos
    return True, newPos


def newPoint(A, B, pathfinder):
    newPos = B[0:3, 3]
    if not pathfinder.is_navigable(newPos):  # if its not navigable
        tempPos = pathfinder.snap_point(newPos)
        snapDist = norm(newPos[[0, 2]] - np.asarray(tempPos)[[0, 2]])
        B[0:3, 3] = tempPos
        if snapDist > 0.15:
            return False, None

    path = ShortestPath()
    path.requested_start = A[0:3, 3]
    path.requested_end = B[0:3, 3]
    pathfinder.find_path(path)
    geodesic_distance = path.geodesic_distance
    eculidean_distance = norm(np.asarray(A[0:3, 3]) - np.asarray(B[0:3, 3]))
    if geodesic_distance / eculidean_distance > 1.2:
        return False, B
    return True, B


def ghost(degrees, stateA, forward, pathfinder, middle_points):
    stateB = (
        stateA
        @ se3_to_mat(
            quat_from_angle_axis(np.deg2rad(degrees), UP), np.asarray([0, 0, 0])
        )
        @ forward
    )
    p = newPoint(stateA, stateB, pathfinder)
    if p[0]:
        middle_points.append(p[1])
    return middle_points


def move_dist(stateA, pathfinder, edgeDistance, addbackwards):
    edgeDistance = 1.5
    middle_points = []
    forward = se3_to_mat(
        quaternion.from_float_array([1, 0, 0, 0]), np.asarray([0, 0, -edgeDistance])
    )

    stateForward = stateA @ forward

    p = newPoint(stateA, stateForward, pathfinder)
    if p[0]:
        middle_points.append(p[1])

    middle_points = ghost(-15, stateA, forward, pathfinder, middle_points)
    middle_points = ghost(15, stateA, forward, pathfinder, middle_points)
    middle_points = ghost(-30, stateA, forward, pathfinder, middle_points)
    middle_points = ghost(30, stateA, forward, pathfinder, middle_points)
    middle_points = ghost(-45, stateA, forward, pathfinder, middle_points)
    middle_points = ghost(45, stateA, forward, pathfinder, middle_points)
    middle_points = ghost(-60, stateA, forward, pathfinder, middle_points)
    middle_points = ghost(60, stateA, forward, pathfinder, middle_points)

    if addbackwards:
        forward = se3_to_mat(
            quaternion.from_float_array([1, 0, 0, 0]), np.asarray([0, 0, -0])
        )
        middle_points = ghost(180, stateA, forward, pathfinder, middle_points)

    return middle_points
