from src.utils.sim_utils import diff_rotation_signed
from src.image_nav.expand_graph import expand
from src.image_nav.target_prediction import predict_end_exploration
from src.image_nav.distance_prediction import predict_distances
from src.functions.validity_func.validity_utils import (
    get_relative_location,
    get_sim_location,
)
from src.functions.validity_func.local_nav import LocalAgent

import numpy as np
import quaternion
import math

turn_angle = 15
step_size = 0.25


def single_image_nav(agent, visualizer, args):
    if agent.visualize:
        visualizer.set_start_images(agent.rgb_img, agent.goal_img)
        visualizer.current_graph(agent)

    """While not reached goal location OR exceed max steps:"""
    while True:
        """
        1) check if should switch to local nav & do local nav
        """
        reached_goal = predict_end_exploration(args, agent, visualizer)
        if reached_goal:
            break

        """
        2) Generate new ghost nodes
        """
        expand(agent)

        """ 
        3) End the navigation if no more nodes to explore 
        """
        if len(agent.unexplored_nodes) == 0:
            print("no more to explore")
            break

        """
        4) Predicts distance from end node to all ghost nodes
                # adds pred dist to dist from cur node to ghost nodes
                # select node with lowest cost
        """
        next_node = predict_distances(agent)

        """
        5) Go to Subgoal
            # back tracks if necessary
            # updates agent vars
            # updates feats with actual observation
            # adds steps to counter
        """
        agent.steps += 1  # add steps once select subgoal
        closest_dist = math.inf
        closest_connected = None
        angle_connected = None

        next_pos = agent.node_poses[next_node]
        next_rot = agent.node_rots[next_node]
        for edge in [list(e) for e in agent.graph.edges]:
            if agent.graph.nodes[edge[0]]["status"] == "explored":
                if edge[1] == next_node:
                    closest_pose = agent.node_poses[edge[0]].numpy()
                    edge_distance = np.linalg.norm(closest_pose - next_pos.numpy())
                    if edge_distance < closest_dist:
                        closest_connected = edge[0]
                        closest_dist = edge_distance

        closest_pose = agent.node_poses[closest_connected].numpy()
        closest_rot = quaternion.from_float_array(
            agent.node_rots[closest_connected].numpy()
        )

        if closest_connected != agent.current_node:
            closest_pose, closest_rot = backtrack(agent, closest_connected, visualizer)
            if closest_pose is None:
                break

        angle_connected = round(
            diff_rotation_signed(
                closest_rot,
                quaternion.from_float_array(next_rot.numpy()),
            ).item()
        )
        pose = None
        turns = abs(math.ceil(angle_connected / turn_angle))
        forward = round(closest_dist / step_size)
        for _ in range(turns):
            if angle_connected >= 0:
                obs, pose, rotation = agent.take_step("left")
            else:
                obs, pose, rotation = agent.take_step("right")
            if agent.visualize:
                visualizer.update(agent, obs)
        for _ in range(forward):
            obs, pose, rotation = agent.take_step("forward")
            if agent.visualize:
                visualizer.update(agent, obs)

        if pose is None:
            obs, pose, rotation = agent.take_step("forward")
            if agent.visualize:
                visualizer.update(agent, obs)
        agent.update_agent(next_node, pose, rotation, obs)
        agent.localize_ue()

        """
        6) Check if we reached the max steps
        """
        if agent.steps >= args.max_steps:
            break


def backtrack(agent, closest_connected, visualizer):
    start_pos = agent.current_pos.numpy()
    start_rot = agent.current_rot.numpy()
    goal_pos = agent.node_poses[closest_connected].numpy()

    local_agent = LocalAgent(
        actuation_noise=False,
        pose_noise=False,
        curr_pos=start_pos,
        curr_rot=start_rot,
        map_size_cm=1200,
        map_resolution=5,
    )
    prev_pos = start_pos
    terminate_local = 0
    delta_dist, delta_rot = get_relative_location(start_pos, start_rot, goal_pos)
    agent.prev_poses.append([start_pos, start_rot])
    local_agent.update_local_map(
        np.expand_dims(agent.sim.get_sensor_observations()["depth"], axis=2)
    )
    local_agent.set_goal(delta_dist, delta_rot)
    try:
        action, terminate_local = local_agent.navigate_local()
        for _ in range(50):
            obs = agent.sim.step(action)
            if agent.visualize:
                visualizer.update(agent, obs)
            curr_depth_img = obs["depth"]
            curr_pose = agent.sim.get_agent_state().position
            curr_rot = quaternion.as_float_array(agent.sim.get_agent_state().rotation)
            delta_dist, delta_rot = get_relative_location(curr_pose, curr_rot, goal_pos)
            local_agent.new_sim_origin = get_sim_location(
                curr_pose, agent.sim.get_agent_state().rotation
            )
            local_agent.update_local_map(curr_depth_img)
            action, terminate_local = local_agent.navigate_local()
            agent.steps += 1
            agent.length_taken += np.linalg.norm(curr_pose - prev_pos)
            prev_pos = curr_pose
            if terminate_local == 1:
                break
    except:
        print("ERROR: local navigation through error")
        return None, None


    closest_pose = curr_pose
    closest_rot = quaternion.from_float_array(curr_rot)
    return closest_pose, closest_rot
