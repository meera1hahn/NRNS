import numpy as np
import os
import imageio
import csv
import cv2
from scipy import ndimage
import quaternion
from typing import Tuple

from habitat.utils.visualizations import maps


class Visualizer:
    def __init__(
        self,
        args,
    ):
        self.dataset = args.dataset
        self.font_size = (210, 35)
        self.font_color = (255, 255, 255)
        self.font_type = cv2.FONT_HERSHEY_SIMPLEX
        self.args = args
        self.render_configs = {}
        self.floor = "0"
        self.dimensions = []
        if self.dataset == "mp3d":
            self.floor_reader()
        self.start_images = []
        self.seen_images = []
        self.map_images = []
        self.prev_poses = []

    def set_start_images(self, start_img, goal_img):
        self.start_images = []
        self.seen_images = []
        self.map_images = []
        self.seen_images.append(start_img)
        self.start_images = [start_img, goal_img]

    def get_floor(self, end, scan_name):
        end_point = np.asarray(end)
        z = end_point[1]
        self.floor = "0"
        for key, confs in self.render_configs[scan_name].items():
            z_min = float(confs[2])
            z_max = float(confs[5])
            if z < z_min:
                break
            self.floor = key
            if z < z_max:
                break

    def update(self, agent, obs):
        if self.dataset == "mp3d":
            self.seen_images.append(obs["rgb"])
            self.current_graph(agent)
        else:
            self.prev_poses.append(
                [agent.current_pos.numpy(), agent.current_rot.numpy()]
            )

    def floor_reader(self):
        with open(self.args.floorplan_dir + "render_config.csv") as csvfile:
            reader = csv.DictReader(csvfile)
            for item in reader:
                if item["level"] == "0":
                    self.render_configs[item["scanId"]] = {}
                self.render_configs[item["scanId"]][item["level"]] = [
                    item["x_low"],
                    item["y_low"],
                    item["z_low"],
                    item["x_high"],
                    item["y_high"],
                    item["z_high"],
                    item["width"],
                    item["height"],
                ]

    def single_edge(self, e1, e2, habitat_map):
        (
            world_min_width,
            world_min_height,
            worldWidth,
            worldHeight,
            imgWidth,
            imgHeight,
        ) = self.dimensions
        e1_gx = int((e1[0] - world_min_width) / worldWidth * imgWidth)
        e1_gy = int((-e1[2] - world_min_height) / worldHeight * imgHeight)
        e2_gx = int((e2[0] - world_min_width) / worldWidth * imgWidth)
        e2_gy = int((-e2[2] - world_min_height) / worldHeight * imgHeight)
        p1 = (e1_gx, e1_gy)
        p2 = (e2_gx, e2_gy)
        habitat_map = cv2.line(cv2.UMat(habitat_map), p1, p2, (0, 0, 0), thickness=2)
        habitat_map = cv2.UMat.get(habitat_map)
        habitat_map = habitat_map[::-1, :, :].astype(np.uint8)
        return habitat_map

    def single_point(self, pos, ori, habitat_map, point_color):
        (
            world_min_width,
            world_min_height,
            worldWidth,
            worldHeight,
            imgWidth,
            imgHeight,
        ) = self.dimensions
        heading = -quaternion.as_rotation_vector(ori)[1]
        h_gx = round((pos[0] - world_min_width) / worldWidth * imgWidth)
        h_gy = round((-pos[2] - world_min_height) / worldHeight * imgHeight)
        habitat_map = self.draw_agent(
            habitat_map[::-1, :, :], (h_gy, h_gx), (heading), agent_radius_px=10
        )

        habitat_map = np.float32(habitat_map)
        habitat_map = cv2.circle(
            cv2.UMat(habitat_map),
            (int(h_gx), int(h_gy)),
            10,
            point_color,
            thickness=2,
        )
        habitat_map = cv2.UMat.get(habitat_map)
        habitat_map = habitat_map[::-1, :, :].astype(np.uint8)
        return habitat_map, (int(h_gx), int(h_gy))

    def draw_agent(
        self,
        image: np.ndarray,
        agent_center_coord: Tuple[int, int],
        agent_rotation: float,
        agent_radius_px: int = 5,
    ) -> np.ndarray:
        AGENT_SPRITE = imageio.imread(
            os.path.join(
                self.args.floorplan_dir,
                "100x100.png",
            )
        )
        AGENT_SPRITE = np.ascontiguousarray(np.flipud(AGENT_SPRITE))

        # Rotate before resize to keep good resolution.
        rotated_agent = ndimage.interpolation.rotate(
            AGENT_SPRITE, agent_rotation * 180 / np.pi
        )
        # Rescale because rotation may result in larger image than original, but
        # the agent sprite size should stay the same.
        initial_agent_size = AGENT_SPRITE.shape[0]
        new_size = rotated_agent.shape[0]
        agent_size_px = max(1, int(agent_radius_px * 2 * new_size / initial_agent_size))
        resized_agent = cv2.resize(
            rotated_agent,
            (agent_size_px, agent_size_px),
            interpolation=cv2.INTER_LINEAR,
        )

        background = image
        foreground = resized_agent
        location = agent_center_coord
        foreground_size = foreground.shape[:2]
        half_size1 = foreground_size[0] // 2
        half_size2 = foreground_size[1] // 2
        min_pad = (
            int(max(0, half_size1 - location[0])),
            int(max(0, half_size2 - location[1])),
        )

        max_pad = (
            int(
                max(
                    0,
                    (location[0] + (foreground_size[0] - half_size1))
                    - background.shape[0],
                )
            ),
            int(
                max(
                    0,
                    (location[1] + (foreground_size[1] - half_size2))
                    - background.shape[1],
                )
            ),
        )

        background_patch = background[
            int(location[0] - foreground_size[0] // 2 + min_pad[0]) : int(
                location[0]
                + (foreground_size[0] - foreground_size[0] // 2)
                - max_pad[0]
            ),
            int(location[1] - half_size2 + min_pad[1]) : int(
                location[1] + (foreground_size[1] - half_size2) - max_pad[1]
            ),
        ]

        foreground = foreground[
            min_pad[0] : int(foreground.shape[0] - max_pad[0]),
            min_pad[1] : int(foreground.shape[1] - max_pad[1]),
        ]
        if foreground.size == 0 or background_patch.size == 0:
            # Nothing to do, no overlap.
            return background

        if foreground.shape[2] == 4:
            # Alpha blending
            foreground = (
                background_patch.astype(np.int32) * (255 - foreground[:, :, [3]])
                + foreground[:, :, :3].astype(np.int32) * foreground[:, :, [3]]
            ) // 255
        background_patch[:] = foreground
        return background

    def get_dimensions(self, scan_name):
        world_min_width = float(self.render_configs[scan_name][self.floor][0])
        world_max_width = float(self.render_configs[scan_name][self.floor][3])
        world_min_height = float(self.render_configs[scan_name][self.floor][1])
        world_max_height = float(self.render_configs[scan_name][self.floor][4])
        worldWidth = abs(world_min_width) + abs(world_max_width)
        worldHeight = abs(world_min_height) + abs(world_max_height)
        imgWidth = round(float(self.render_configs[scan_name][self.floor][6]))
        imgHeight = round(float(self.render_configs[scan_name][self.floor][7]))
        self.dimensions = [
            world_min_width,
            world_min_height,
            worldWidth,
            worldHeight,
            imgWidth,
            imgHeight,
        ]

    def current_graph(self, agent, switch=False):
        if self.dataset != "mp3d":
            return
        habitat_map = self.get_graph(agent, switch)
        self.map_images.append(habitat_map.copy())

    def get_graph(self, agent, switch):
        self.get_floor(agent.goal_pos, agent.scan_name)
        self.get_dimensions(agent.scan_name)
        habitat_map = cv2.imread(
            "{}out_dir_rgb_png/output_{}_level_{}.0.png".format(
                self.args.floorplan_dir, agent.scan_name, self.floor
            )
        )
        black_pixels = np.where(
            (habitat_map[:, :, 0] == 0)
            & (habitat_map[:, :, 1] == 0)
            & (habitat_map[:, :, 2] == 0)
        )
        habitat_map[black_pixels] = [255, 255, 255]
        black_pixels = np.where(
            (habitat_map[:, :, 0] == 102)
            & (habitat_map[:, :, 1] == 102)
            & (habitat_map[:, :, 2] == 102)
        )
        habitat_map[black_pixels] = [255, 255, 255]

        """ Draw edges """
        for i in [list(e) for e in agent.graph.edges]:
            e1 = agent.node_poses[i[0], :].numpy()
            e2 = agent.node_poses[i[1], :].numpy()
            habitat_map = habitat_map[::-1, :, :].astype(np.uint8)
            habitat_map = self.single_edge(e1, e2, habitat_map)

        """ Draw nodes in topo graph """
        point_color = (255, 255, 255)
        for n in agent.graph:
            point = agent.node_poses[n, :].numpy()
            rot = quaternion.from_float_array(agent.node_rots[n, :].numpy())
            if agent.graph.nodes.data()[n]["status"] == "unexplored":
                point_color = (255, 255, 255)  # white cause its ghost-like
                habitat_map, _ = self.single_point(point, rot, habitat_map, point_color)
        for n in agent.graph:
            point = agent.node_poses[n, :].numpy()
            rot = quaternion.from_float_array(agent.node_rots[n, :].numpy())
            if agent.graph.nodes.data()[n]["status"] == "explored":
                point_color = (0, 255, 255)  # yellow
                habitat_map, _ = self.single_point(point, rot, habitat_map, point_color)
            if n == agent.current_node and switch == False:
                point_color = (255, 0, 0)  # blue
                habitat_map, _ = self.single_point(point, rot, habitat_map, point_color)

        if switch:
            point_color = (255, 0, 0)  # blue
            point = agent.current_pos.numpy()
            rot = quaternion.from_float_array(agent.current_rot.numpy())
            habitat_map, _ = self.single_point(point, rot, habitat_map, point_color)
        """ Draw green around START POINT"""
        habitat_map, _ = self.single_point(
            np.asarray(agent.node_poses[0, :].numpy()),
            quaternion.from_float_array(agent.node_rots[0, :].numpy()),
            habitat_map,
            (0, 255, 0),  # green
        )
        """ Draw red GOAL POINT"""
        habitat_map, _ = self.single_point(
            np.asarray(agent.goal_pos),
            quaternion.from_float_array(np.asarray(agent.goal_rot)),
            habitat_map,
            (0, 0, 255),  # red
        )
        habitat_map = cv2.cvtColor(habitat_map, cv2.COLOR_BGR2RGB)
        habitat_map = cv2.resize(habitat_map, (640, 480))
        return habitat_map

    def create_layout_mp3d(self, traj):
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        output_video_name = self.args.visualization_dir + traj + ".avi"
        vid_size = (1920, 480)
        vid = cv2.VideoWriter(output_video_name, fourcc, 4, vid_size)
        text = "Start Image"
        start = self.start_images[0].astype(np.uint8)
        cv2.putText(start, text, self.font_size, self.font_type, 1, (0, 0, 0), 6)
        cv2.putText(start, text, self.font_size, self.font_type, 1, self.font_color, 2)
        text = "Goal Image"
        goal = self.start_images[1].astype(np.uint8)
        goal[:, 625:, :] = [255, 255, 255]
        cv2.putText(goal, text, self.font_size, self.font_type, 1, (0, 0, 0), 6)
        cv2.putText(goal, text, self.font_size, self.font_type, 1, self.font_color, 2)
        im_top = cv2.hconcat((start, goal))
        im_top = im_top.astype(np.uint8)
        for i in range(len(self.seen_images)):
            one = self.seen_images[i].astype(np.uint8)
            one[:, :15, :] = [255, 255, 255]
            text = "Current Image"
            cv2.putText(one, text, self.font_size, self.font_type, 1, (0, 0, 0), 6)
            cv2.putText(
                one, text, self.font_size, self.font_type, 1, self.font_color, 2
            )
            two = self.map_images[i][:-50, :, :]
            two = cv2.copyMakeBorder(
                two,
                top=50,
                bottom=0,
                left=0,
                right=0,
                borderType=cv2.BORDER_CONSTANT,
                value=[255, 255, 255],
            )

            two = self.map_images[i].astype(np.uint8)
            text = "Current Graph"
            cv2.putText(two, text, self.font_size, self.font_type, 1, (0, 0, 0), 6)
            cv2.putText(
                two, text, self.font_size, self.font_type, 1, self.font_color, 2
            )
            im_bottom = cv2.hconcat((one, two))
            im_bottom = im_bottom.astype(np.uint8)

            im = cv2.hconcat((goal, im_bottom))
            im = im.astype("uint8")
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im = cv2.resize(im.copy(), vid_size)
            vid.write(im)
        vid.release()

        print(f"wrote video to {output_video_name}")

    def grid_map(self, sim, sim_topdown_map, point, switch=False):
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

    def create_layout(self, agent, traj):
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        output_video_name = self.args.visualization_dir + traj + ".avi"
        vid_size = (1280, 960)
        vid = cv2.VideoWriter(output_video_name, fourcc, 4, vid_size)
        text = "Start Image"
        start = self.start_images[0].astype(np.uint8)
        cv2.putText(start, text, self.font_size, self.font_type, 1, (0, 0, 0), 6)
        cv2.putText(start, text, self.font_size, self.font_type, 1, self.font_color, 2)
        text = "Goal Image"
        goal = self.start_images[1].astype(np.uint8)
        goal[:, 625:, :] = [255, 255, 255]
        cv2.putText(goal, text, self.font_size, self.font_type, 1, (0, 0, 0), 6)
        cv2.putText(goal, text, self.font_size, self.font_type, 1, self.font_color, 2)
        im_top = cv2.hconcat((start, goal))
        im_top = im_top.astype(np.uint8)
        two = agent.topdown_grid
        for enum, pose in enumerate(agent.prev_poses):
            one = np.float32(
                agent.sim.get_observations_at(
                    pose[0],
                    quaternion.from_float_array(pose[1]),
                )["rgb"][:, :, :3]
            ).astype(np.uint8)
            text = "Current Image"
            cv2.putText(one, text, self.font_size, self.font_type, 1, (0, 0, 0), 6)
            cv2.putText(
                one, text, self.font_size, self.font_type, 1, self.font_color, 2
            )
            if enum != 0:
                if enum >= agent.switch_index:
                    two = self.grid_map(agent.sim, agent.topdown_grid, pose[0], True)
                else:
                    two = self.grid_map(agent.sim, agent.topdown_grid, pose[0], False)

            padding = round(((two.shape[1] * 0.75) - 480) / 2)
            if padding > 0:
                two = cv2.copyMakeBorder(
                    two, padding, padding, 0, 0, cv2.BORDER_CONSTANT
                )
            if padding < 0:
                padding = round((640 - two.shape[1]) / 2)
                two = cv2.copyMakeBorder(
                    two,
                    0,
                    0,
                    padding,
                    padding,
                    cv2.BORDER_CONSTANT,
                    value=[255, 255, 255],
                )
            two = cv2.resize(two, (640, 480))
            im_bottom = cv2.hconcat((one, two))
            im = cv2.vconcat((im_top, im_bottom))
            im = im.astype("uint8")
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im = cv2.resize(im.copy(), vid_size)
            vid.write(im)
        vid.release()
        print(f"wrote video to {output_video_name}")
