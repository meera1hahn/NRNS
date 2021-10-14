import numpy as np
import cv2, imageio
import os
import matplotlib.pyplot as plt
import skimage
import src.functions.validity_func.depth_utils as du


def build_mapper(camera_height=1.25):
    params = {}
    camera_height = 1.25
    map_size_cm = 1200
    params["frame_width"] = 640
    params["frame_height"] = 480
    params["fov"] = 120
    params["resolution"] = 5
    params["map_size_cm"] = map_size_cm
    params["agent_min_z"] = 25
    params["agent_medium_z"] = 100
    params["agent_max_z"] = 150
    params["agent_height"] = camera_height * 100
    params["agent_view_angle"] = np.pi
    params["du_scale"] = 1
    params["vision_range"] = 64
    params["use_mapper"] = 1
    params["visualize"] = 0
    params["maze_task"] = 1
    params["obs_threshold"] = 1
    mapper = MapBuilder(params)
    return mapper


class MapBuilder(object):
    def __init__(self, params):
        self.params = params
        frame_width = params["frame_width"]
        frame_height = params["frame_height"]
        fov = params["fov"]
        self.camera_matrix = du.get_camera_matrix(frame_width, frame_height, fov)
        self.vision_range = params["vision_range"]
        self.map_size_cm = params["map_size_cm"]
        self.resolution = params["resolution"]
        agent_min_z = params["agent_min_z"]
        agent_medium_z = params["agent_medium_z"]
        agent_max_z = params["agent_max_z"]
        self.z_bins = [agent_min_z, agent_medium_z, agent_max_z]
        self.du_scale = params["du_scale"]
        self.use_mapper = params["use_mapper"]
        self.visualize = params["visualize"]
        self.maze_task = params["maze_task"]
        self.obs_threshold = params["obs_threshold"]

        self.map = np.zeros(
            (
                self.map_size_cm // self.resolution + 1,
                self.map_size_cm // self.resolution + 1,
                len(self.z_bins) + 1,
            ),
            dtype=np.float32,
        )
        self.agent_height = params["agent_height"]
        self.agent_view_angle = params["agent_view_angle"]

    def update_map(self, depth, current_pose):
        mask2 = depth > 9999.0
        depth[mask2] = 0.0

        for i in range(depth.shape[1]):
            depth[:, i][depth[:, i] == 0.0] = depth[:, i].max()

        mask1 = depth == 0
        depth[mask1] = np.NaN

        point_cloud = du.get_point_cloud_from_z(
            depth, self.camera_matrix, scale=self.du_scale
        )

        agent_view = du.transform_camera_view(
            point_cloud, self.agent_height, self.agent_view_angle
        )

        shift_loc = [self.vision_range * self.resolution // 2, 0, np.pi / 2.0]
        agent_view_centered = du.transform_pose(agent_view, shift_loc)

        agent_view_flat, is_valids = du.bin_points(
            agent_view_centered, self.vision_range, self.z_bins, self.resolution
        )

        x1 = self.map_size_cm // (self.resolution * 2) - self.vision_range // 2
        x2 = x1 + self.vision_range
        y1 = self.map_size_cm // (self.resolution * 2)
        y2 = y1 + self.vision_range
        agent_view_cropped = agent_view_flat[:, :, 1] + agent_view_flat[:, :, 2]

        agent_view_cropped = agent_view_cropped / self.obs_threshold
        agent_view_cropped[agent_view_cropped >= 0.5] = 1.0
        agent_view_cropped[agent_view_cropped < 0.5] = 0.0

        agent_view_explored = agent_view_flat.sum(2)
        agent_view_explored[agent_view_explored > 0] = 1.0

        geocentric_pc = du.transform_pose(agent_view, current_pose)

        geocentric_flat, is_valids = du.bin_points(
            geocentric_pc, self.map.shape[0], self.z_bins, self.resolution
        )

        self.map = self.map + geocentric_flat

        map_gt = (self.map[:, :, 1] + self.map[:, :, 2]) // self.obs_threshold
        map_gt[map_gt >= 0.5] = 1.0
        map_gt[map_gt < 0.5] = 0.0

        wall_map_gt = self.map[:, :, 2] // self.obs_threshold
        map_gt[map_gt >= 0.5] = 1.0
        map_gt[map_gt < 0.5] = 0.0

        explored_gt = self.map.sum(2)
        explored_gt[explored_gt > 1] = 1.0

        return agent_view_cropped, map_gt, agent_view_explored, explored_gt, wall_map_gt

    def get_st_pose(self, current_loc):
        loc = [
            -(
                current_loc[0] / self.resolution
                - self.map_size_cm // (self.resolution * 2)
            )
            / (self.map_size_cm // (self.resolution * 2)),
            -(
                current_loc[1] / self.resolution
                - self.map_size_cm // (self.resolution * 2)
            )
            / (self.map_size_cm // (self.resolution * 2)),
            90 - np.rad2deg(current_loc[2]),
        ]
        return loc

    def reset_map(self):
        self.map = np.zeros(
            (
                self.map_size_cm // self.resolution + 1,
                self.map_size_cm // self.resolution + 1,
                len(self.z_bins) + 1,
            ),
            dtype=np.float32,
        )

    def get_map(self):
        return self.map
