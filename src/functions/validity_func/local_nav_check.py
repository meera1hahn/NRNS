import quaternion
import numpy as np
from habitat import get_config
from habitat.sims import make_sim
from habitat import get_config
import numpy as np
import quaternion
import cv2
import skimage

from src.functions.validity_func.validity_utils import *
from src.functions.validity_func.map_builder import build_mapper


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


def set_up_habitat(mp3d_data, scanName):
    scene = "{}{}/{}.glb".format(mp3d_data, scanName, scanName)
    config = get_config()
    config.defrost()
    config.SIMULATOR.SCENE = scene
    config.SIMULATOR.AGENT_0.SENSORS = ["RGB_SENSOR", "DEPTH_SENSOR"]
    config.SIMULATOR.RGB_SENSOR.HFOV = 120
    config.SIMULATOR.DEPTH_SENSOR.HFOV = 120
    config.SIMULATOR.TURN_ANGLE = 20
    # FORWARD_STEP_SIZE: 0.25
    # TURN_ANGLE: 10
    # TILT_ANGLE: 15

    config.freeze()
    sim = make_sim(id_sim=config.SIMULATOR.TYPE, config=config.SIMULATOR)
    return sim


class LocalAgent(object):
    def __init__(self, map_size_cm, map_resolution, sim, curr_pos, curr_rot):
        self.mapper = build_mapper()
        self.curr_pos = curr_pos
        self.curr_rot = curr_rot
        self.map_size_cm = map_size_cm
        self.map_resolution = map_resolution
        self.sim_origin = get_sim_location(
            self.curr_pos, quaternion.from_float_array(self.curr_rot)
        )
        self.collision = False
        self.initialize_local_map_pose()
        self.stg_x, self.stg_y = int(self.y_gt / map_resolution), int(
            self.x_gt / map_resolution
        )
        self.new_sim_origin = self.sim_origin
        self.reset_goal = True

        self.goal_pos = [0.9855455, 0.16325301, 0.18831685]
        self.goal_rot = [0.785535752773285, -0.0, 0.618816375732422, -0.0]

    def initialize_local_map_pose(self):
        self.mapper.reset_map()
        self.x_gt, self.y_gt, self.o_gt = (
            self.map_size_cm / 100.0 / 2.0,
            self.map_size_cm / 100.0 / 2.0,
            0.0,
        )
        x, y, o = self.x_gt, self.y_gt, self.o_gt
        self.reset_goal = True
        self.sim_origin = get_sim_location(
            self.curr_pos, quaternion.from_float_array(self.curr_rot)
        )
        self.local_locs = np.zeros(self.mapper.get_map()[:, :, 1].shape)

    def update_local_map(self, curr_depth_img):
        self.x_gt, self.y_gt, self.o_gt = self.get_mapper_pose_from_sim_pose(
            self.new_sim_origin,
            self.sim_origin,
        )

        x, y, o = self.x_gt, self.y_gt, self.o_gt
        _, self.local_map, _, self.local_exp_map, _ = self.mapper.update_map(
            curr_depth_img[:, :, 0] * 1000.0, (x, y, o)
        )

        if self.collision:
            self.mapper.map[self.stg_x, self.stg_y, 1] = 10.0
            self.collision = False

    def get_mapper_pose_from_sim_pose(self, sim_pose, sim_origin):
        x, y, o = get_rel_pose_change(sim_pose, sim_origin)
        return (
            self.map_size_cm - (x * 100.0 + self.map_size_cm / 2.0),
            self.map_size_cm - (y * 100.0 + self.map_size_cm / 2.0),
            o,
        )

    def navigate_local(self, delta_dist, delta_rot):
        traversible = (
            skimage.morphology.binary_dilation(
                self.local_map, skimage.morphology.disk(2)
            )
            != True
        )

        start = (
            int(self.y_gt / self.map_resolution),
            int(self.x_gt / self.map_resolution),
        )

        traversible[start[0] - 2 : start[0] + 3, start[1] - 2 : start[1] + 3] = 1
        planner = FMMPlanner(traversible, 360 // 10, 1)

        if self.reset_goal:
            goal = (
                start[0]
                + int(
                    delta_dist * np.sin(delta_rot + self.o_gt) * 100.0 / map_resolution
                ),
                start[1]
                + int(
                    delta_dist * np.cos(delta_rot + self.o_gt) * 100.0 / map_resolution
                ),
            )
            self.goal = goal
            planner.set_goal(self.goal, auto_improve=True)
            self.goal = planner.get_goal()
            self.reset_goal = False
        else:
            planner.set_goal(self.goal, auto_improve=True)

        stg_x, stg_y = start
        stg_x, stg_y, replan = planner.get_short_term_goal2((stg_x, stg_y))

        print(
            "pred dist", get_l2_distance(start[0], self.goal[0], start[1], self.goal[1])
        )

        if get_l2_distance(start[0], self.goal[0], start[1], self.goal[1]) < 3:
            terminate = 1
        else:
            terminate = 0

        agent_orientation = np.rad2deg(self.o_gt)
        action = planner.get_next_action(start, (stg_x, stg_y), agent_orientation)
        self.stg_x, self.stg_y = int(stg_x), int(stg_y)

        return action, terminate


def single_nav(local_agent, start_pos, start_rot, goal_pos):
    # get depth and rgb images
    obs = sim.get_observations_at(start_pos, quaternion.from_float_array(start_rot))
    sim.set_agent_state(start_pos, quaternion.from_float_array(start_rot))
    curr_depth_img = obs["depth"]
    terminate_local = 0
    delta_dist, delta_rot = get_relative_location(
        sim.get_agent_state().position,
        quaternion.as_float_array(sim.get_agent_state().rotation),
        goal_pos,
    )
    local_agent.update_local_map(curr_depth_img)

    action, terminate_local = local_agent.navigate_local(delta_dist, delta_rot)
    for _ in range(20):
        obs = sim.step(action)
        curr_depth_img = obs["depth"]
        delta_dist, delta_rot = get_relative_location(
            sim.get_agent_state().position,
            quaternion.as_float_array(sim.get_agent_state().rotation),
            goal_pos,
        )
        local_agent.new_sim_origin = get_sim_location(
            sim.get_agent_state().position, sim.get_agent_state().rotation
        )
        local_agent.update_local_map(curr_depth_img)
        action, terminate_local = local_agent.navigate_local(delta_dist, delta_rot)
        if terminate_local == 1:
            break


def test_single_nav(local_agent, start_pos, start_rot, goal_pos):
    # get depth and rgb images
    obs = sim.get_observations_at(start_pos, quaternion.from_float_array(start_rot))
    sim.set_agent_state(start_pos, quaternion.from_float_array(start_rot))
    curr_rgb_img = obs["rgb"]
    curr_depth_img = obs["depth"]
    cv2.imwrite("test_imgs/rgb_view_0.png", curr_rgb_img)

    terminate_local = 0
    previous_action = None
    delta_dist, delta_rot = get_relative_location(
        sim.get_agent_state().position,
        quaternion.as_float_array(sim.get_agent_state().rotation),
        goal_pos,
    )
    local_agent.update_local_map(curr_depth_img)

    action, terminate_local = local_agent.navigate_local(delta_dist, delta_rot)
    for i in range(50):
        print("step", i + 1)
        print("action", action)
        obs = sim.step(action)
        import ipdb

        ipdb.set_trace()
        curr_depth_img = obs["depth"]
        depthIm = curr_depth_img[:, :, 0] * 1000.0
        cv2.imwrite(f"test_imgs/BEEP_depth_view_{i+1}.png", depthIm)
        mapIm = (
            cv2.flip(local_agent.local_map + 0.5 * local_agent.local_exp_map, 0) * 255
        )
        cv2.imwrite(f"test_imgs/BEEP_depth_map_{i+1}.png", mapIm)
        mapIm = local_agent.local_map + 0.5 * local_agent.local_exp_map * 255
        mapIm = mapIm.astype("uint8")
        color = cv2.cvtColor(mapIm, cv2.COLOR_GRAY2RGB)
        # color[local_agent.goal[0], local_agent.goal[1], :] = [0, 255, 0]
        color[
            int(local_agent.stg_x) : int(local_agent.stg_x) + 4,
            int(local_agent.stg_y) : int(local_agent.stg_y) + 4,
            :,
        ] = [
            0,
            0,
            255,
        ]
        cv2.imwrite(f"test_imgs/BEEP_depth_map_color_{i+1}.png", color)

        delta_dist, delta_rot = get_relative_location(
            sim.get_agent_state().position,
            quaternion.as_float_array(sim.get_agent_state().rotation),
            goal_pos,
        )
        print("delta_dist", delta_dist)
        print("delta_rot", math.degrees(delta_rot))
        local_agent.new_sim_origin = get_sim_location(
            sim.get_agent_state().position, sim.get_agent_state().rotation
        )
        local_agent.update_local_map(curr_depth_img)
        action, terminate_local = local_agent.navigate_local(delta_dist, delta_rot)
        if terminate_local == 1:
            break
    print(sim.get_agent_state().position)
    print("real dist", delta_dist)
    print("read rot", math.degrees(delta_rot))


if __name__ == "__main__":
    map_size_cm = 1200
    map_resolution = 5
    mp3d_data = "/srv/share/datasets/habitat-sim-datasets/mp3d/"
    scanName = "2t7WUuJeko7"
    sim = set_up_habitat(mp3d_data, scanName)
    goal_pos = sim.sample_navigable_point()
    angle = np.random.uniform(0, 2 * np.pi)
    goal_rot = np.asarray([np.cos(angle / 2), 0, np.sin(angle / 2), 0])
    start_pos = [4.4579, 0.1633, 0.7829]
    start_rot = np.asarray([0.78553563, 0.0, 0.6188164, 0.0])
    goal_rgb = sim.get_observations_at(
        start_pos, quaternion.from_float_array(start_rot)
    )["rgb"]
    cv2.imwrite("test_imgs/start_view.png", goal_rgb)

    goal_rgb = sim.get_observations_at(goal_pos, quaternion.from_float_array(goal_rot))[
        "rgb"
    ]
    cv2.imwrite("test_imgs/goal_view.png", goal_rgb)

    local_agent = LocalAgent(map_size_cm, map_resolution, sim, start_pos, start_rot)
    test_single_nav(local_agent, start_pos, start_rot, goal_pos)