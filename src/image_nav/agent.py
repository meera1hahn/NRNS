import torch
import quaternion
import numpy as np
from numpy.linalg import inv
from src.utils.vis_utils import start_grid_map
from src.utils.model_utils import get_res_feats
from src.utils.sim_utils import diff_rotation, se3_to_mat, NoisySensor
from habitat.sims.habitat_simulator.actions import HabitatSimActions
import networkx as nx


class Agent:
    def __init__(
        self,
        args,
        sim,
        pathfinder,
        resnet,
        scan_name,
        model_goal=None,
        model_switch=None,
        model_feat_pred=None,
        model_action=None,
    ):
        # Params
        self.args = args
        self.use_gt_distances = args.use_gt_distances
        self.pose_noise = args.pose_noise
        self.actuation_noise = args.actuation_noise
        self.node_feat_size = 512
        self.dist_thresh = 0.25
        self.rot_thres = 45
        self.edge_length = 1
        self.map_size_cm = 1200
        self.map_resolution = 5
        self.visualize = args.visualize
        # Habitat Simulator
        self.sim = sim
        self.pathfinder = pathfinder
        self.resnet = resnet
        self.scan_name = scan_name
        # Current State
        self.rgb_img = None
        self.depth_img = None
        self.current_pos, self.current_rot = None, None
        self.goal_pos, self.goal_rot, self.goal_feat = None, None, None
        self.goal_pos_pred = None
        self.length_taken = 0
        self.steps = 0
        self.actions = [4]
        self.prev_poses = []
        # GRAPH
        self.graph = None
        self.current_node = 0
        # Graph Nodes
        self.unexplored_nodes = []  # ids
        self.total_nodes = None
        self.node_feats = None
        self.node_poses = None
        self.node_rots = None
        # Visualization
        self.uncolored_grid, self.topdown_grid = None, None
        self.map_images = []
        self.switch_index = 500
        # Models
        self.goal_model = model_goal
        self.goal_model.eval()
        self.switch_model = model_switch
        self.switch_model.eval()
        self.feat_model = model_feat_pred
        self.feat_model.eval()
        self.action_model = model_action
        self.action_model.eval()

    def reset_agent(self, start_position, start_rotation, goal_position, goal_rotation):
        self.graph = nx.Graph()
        self.current_node = 0
        self.graph.add_node(0)
        self.graph.nodes[0]["status"] = "explored"
        self.node_poses = torch.tensor([start_position], dtype=torch.float)
        self.node_rots = torch.tensor([start_rotation], dtype=torch.float)
        self.node_feats = self.get_feat(start_position, start_rotation)
        self.current_pos = torch.tensor(start_position)
        self.current_rot = torch.tensor(start_rotation)
        self.total_nodes = 1
        self.self_loop(0)
        if self.pose_noise:
            self.noisy_sensor = NoisySensor(noise_level=1.0)
        self.goal_pos = np.asarray(goal_position)
        self.goal_rot = np.asarray(goal_rotation)
        self.goal_feat = self.get_feat(goal_position, goal_rotation)
        self.sim.set_agent_state(
            start_position, quaternion.from_float_array(start_rotation)
        )
        obs = self.sim.get_observations_at(
            start_position,
            quaternion.from_float_array(start_rotation),
        )
        self.rgb_img = obs["rgb"][:, :, :3]
        self.depth_img = obs["depth"]
        self.goal_img = self.sim.get_observations_at(
            self.goal_pos.tolist(),
            quaternion.from_float_array(self.goal_rot),
        )["rgb"][:, :, :3]
        self.goal_feat = get_res_feats(self.goal_img.copy(), self.resnet)
        self.create_visuals()

    def create_visuals(self):
        self.uncolored_grid, self.topdown_grid = start_grid_map(
            self.sim, self.current_pos.numpy(), self.goal_pos
        )
        self.start_image = np.float32(self.rgb_img.copy())

    def self_loop(self, node_id):
        stateA = se3_to_mat(
            quaternion.from_float_array(self.node_rots[node_id]),
            np.asarray(self.node_poses[node_id]),
        )
        edge_attr = inv(stateA) @ stateA
        edge_attr = (
            torch.flatten(torch.tensor(edge_attr, dtype=torch.float))
            .unsqueeze(0)
            .unsqueeze(0)
        )
        self.graph.add_edge(node_id, node_id)
        self.graph.edges[node_id, node_id]["attr"] = edge_attr.squeeze(0).squeeze(0)
        self.graph.edges[node_id, node_id]["rotation"] = 0.0

    def add_node(
        self, postion, rotation, feat=None
    ):  # position is numpy, rotation in rotation matrix
        self.graph.add_node(self.total_nodes)
        self.graph.nodes[self.total_nodes]["status"] = "unexplored"
        self.unexplored_nodes.append(self.total_nodes)
        self.total_nodes += 1
        self.node_poses = torch.cat(
            [self.node_poses, torch.tensor(postion, dtype=torch.float).unsqueeze(0)],
            dim=0,
        )
        rotation = quaternion.as_float_array(quaternion.from_rotation_matrix(rotation))
        self.node_rots = torch.cat(
            [self.node_rots, torch.tensor(rotation, dtype=torch.float).unsqueeze(0)],
            dim=0,
        )
        if feat is not None:
            self.node_feats = torch.cat([self.node_feats, feat], dim=0)
        else:
            feat = torch.Tensor((np.zeros(self.node_feat_size) + 0.5)).unsqueeze(0)
            self.node_feats = torch.cat([self.node_feats, feat], dim=0)

    def add_edge(self, edge, attr, rot):  # edge is a list, attribute is list
        if edge in [list(e) for e in self.graph.edges]:
            return

        self.graph.add_edge(
            *edge,
        )
        self.graph.edges[edge]["rotation"] = rot
        self.graph.edges[edge]["attr"] = torch.flatten(
            torch.tensor(attr, dtype=torch.float)
        )

    # localize ghost node in map using postion and rotation
    def localize(self, test_position, test_rotation, skip=[], full_graph=False):
        localized = False
        location = None
        total_close = 0
        for i in self.graph:
            if i in skip:
                continue
            pos = self.node_poses[i].numpy()
            rot = self.node_rots[i].numpy()
            euc_dist = np.linalg.norm(pos - test_position)
            rot_dist = diff_rotation(
                quaternion.from_float_array(rot),
                quaternion.from_float_array(test_rotation),
            )
            if (euc_dist < 0.9 and (0 <= rot_dist < 29)) or (
                full_graph and euc_dist < 0.25 and (0 <= rot_dist < 30)
            ):
                return True, i

            if not full_graph and euc_dist < 0.5:
                total_close += 1

        return localized, location  # , total_close

    # localize unexplored in map using postion and rotation
    def localize_ue(self):
        removed = []
        for ue in self.unexplored_nodes:
            test_position = self.node_poses[ue].numpy()
            test_rotation = self.node_rots[ue].numpy()
            skip = removed + [ue]
            localized, _ = self.localize(
                test_position, test_rotation, skip, full_graph=True
            )
            if localized:
                removed.append(ue)

        for r in removed:
            self.graph.remove_node(r)
            self.unexplored_nodes.remove(r)

    def update_agent(self, next_node, pose=None, rotation=None, obs=None):
        self.current_node = next_node
        if pose is not None:
            self.node_poses[self.current_node] = torch.tensor(pose)
        if rotation is not None:
            self.node_rots[self.current_node] = torch.tensor(rotation)
        self.current_pos, self.current_rot = (
            self.node_poses[self.current_node],
            self.node_rots[self.current_node],
        )
        if obs is None:
            self.sim.set_agent_state(
                self.current_pos.numpy().tolist(),
                quaternion.from_float_array(self.current_rot.numpy()),
            )
            obs = self.sim.get_observations_at(
                self.current_pos.numpy().tolist(),
                quaternion.from_float_array(self.current_rot.numpy()),
            )
        self.rgb_img = obs["rgb"][:, :, :3]
        self.depth_img = obs["depth"]
        self.node_feats[self.current_node, :] = get_res_feats(self.rgb_img, self.resnet)
        self.graph.nodes[self.current_node]["status"] = "explored"
        self.unexplored_nodes.remove(self.current_node)

    def take_step(self, action):
        previous_state = self.sim.get_agent_state()
        previous_pose = previous_state.position
        previous_rotation = quaternion.as_float_array(previous_state.rotation)
        if action == "forward":
            action = 1
            if self.actuation_noise:
                obs = self.sim.step(HabitatSimActions.NOISY_FORWARD)  # noisy forward
            else:
                obs = self.sim.step(HabitatSimActions.MOVE_FORWARD)
        elif action == "left":
            action = 2
            if self.actuation_noise:
                obs = self.sim.step(HabitatSimActions.NOISY_LEFT)  # noisy left
            else:
                obs = self.sim.step(HabitatSimActions.TURN_LEFT)  # left
        elif action == "right":
            action = 3
            if self.actuation_noise:
                obs = self.sim.step(HabitatSimActions.NOISY_RIGHT)  # noisy right
            else:
                obs = self.sim.step(HabitatSimActions.TURN_RIGHT)  # right
        else:
            raise Exception("no valid action given to agent")
        pose = self.sim.get_agent_state().position
        rotation = quaternion.as_float_array(self.sim.get_agent_state().rotation)
        if self.pose_noise:
            pose = self.noisy_sensor.get_noisy_pose(action, previous_pose, pose)
        self.length_taken += np.linalg.norm(previous_pose - pose)
        self.steps += 1
        self.prev_poses.append([previous_pose, previous_rotation])
        return obs, pose, rotation

    def get_feat(self, pos, rot):
        obs = self.sim.get_observations_at(
            pos,
            quaternion.from_float_array(rot),
        )
        img = obs["rgb"][:, :, :3]
        return get_res_feats(img.copy(), self.resnet)
