import numpy as np
from numpy.linalg import inv
import quaternion
import pickle
import habitat
from habitat import get_config
from habitat.sims import make_sim
from habitat import get_config
import habitat_sim
from habitat_sim import ShortestPath
from habitat.sims.habitat_simulator.actions import (
    HabitatSimActions,
    HabitatSimV1ActionSpaceConfiguration,
)
from src.utils.noisy_actions import CustomActionSpaceConfiguration

class NoisySensor:
    def __init__(self, noise_level):
        self.noise_level = noise_level
        self.noise_dir = "../../models/noise_models/"
        self.sensor_noise_fwd = pickle.load(
            open(self.noise_dir + "sensor_noise_fwd.pkl", "rb")
        )
        self.sensor_noise_right = pickle.load(
            open(self.noise_dir + "sensor_noise_right.pkl", "rb")
        )
        self.sensor_noise_left = pickle.load(
            open(self.noise_dir + "sensor_noise_left.pkl", "rb")
        )

    def get_l2_distance(self, x1, x2, y1, y2):
        """
        Computes the L2 distance between two points.
        """
        return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

    def get_rel_pose_change(self, pos2, pos1):
        x1, y1, o1 = pos1
        x2, y2, o2 = pos2
        theta = np.arctan2(y2 - y1, x2 - x1) - o1
        dist = self.get_l2_distance(x1, x2, y1, y2)
        dx = dist * np.cos(theta)
        dy = dist * np.sin(theta)
        do = o2 - o1
        return dx, dy, do

    def get_noisy_sensor_readings(self, action, gt_pose_change):
        dx_gt, dy_gt, do_gt = gt_pose_change
        if action == 1:  ## Forward
            x_err, y_err, o_err = self.sensor_noise_fwd.sample()[0][0]
        elif action == 2:  ## Left
            x_err, y_err, o_err = self.sensor_noise_left.sample()[0][0]
        elif action == 3:  ## Right
            x_err, y_err, o_err = self.sensor_noise_right.sample()[0][0]
        else:  ##Stop
            x_err, y_err, o_err = 0.0, 0.0, 0.0

        x_err = x_err * self.noise_level
        y_err = y_err * self.noise_level
        o_err = o_err * self.noise_level
        return dx_gt + x_err, dy_gt + y_err, do_gt + np.deg2rad(o_err)

    def get_new_pose(self, pose, rel_pose_change):
        x, y, o = pose
        dx, dy, do = rel_pose_change
        global_dx = dx * np.sin(o) + dy * np.cos(o)
        global_dy = dx * np.cos(o) - dy * np.sin(o)
        x += global_dy
        y += global_dx
        o += do
        return x, y, o

    def get_noisy_pose(self, action, previous_pose, pose):
        gt_pose_change = self.get_rel_pose_change(pose, previous_pose)
        noisy_pose_change = self.get_noisy_sensor_readings(action, gt_pose_change)
        noisy_pose = np.asarray(self.get_new_pose(previous_pose, noisy_pose_change))
        # import ipdb

        # ipdb.set_trace()
        return noisy_pose


def add_noise_actions_habitat():
    HabitatSimActions.extend_action_space("NOISY_FORWARD")
    HabitatSimActions.extend_action_space("NOISY_LEFT")
    HabitatSimActions.extend_action_space("NOISY_RIGHT")


def set_up_habitat_noise(scene, turn_angle=15):
    config = get_config()
    config.defrost()
    config.TASK.POSSIBLE_ACTIONS = config.TASK.POSSIBLE_ACTIONS + [
        "NOISY_FORWARD",
        "NOISY_LEFT",
        "NOISY_RIGHT",
    ]
    config.TASK.ACTIONS.NOISY_FORWARD = habitat.config.Config()
    config.TASK.ACTIONS.NOISY_FORWARD.TYPE = "NoisyForward"
    config.TASK.ACTIONS.NOISY_LEFT = habitat.config.Config()
    config.TASK.ACTIONS.NOISY_LEFT.TYPE = "NoisyLeft"
    config.TASK.ACTIONS.NOISY_RIGHT = habitat.config.Config()
    config.TASK.ACTIONS.NOISY_RIGHT.TYPE = "NoisyRight"
    config.SIMULATOR.ACTION_SPACE_CONFIG = "CustomActionSpaceConfiguration"
    config.SIMULATOR.SCENE = scene
    config.SIMULATOR.AGENT_0.SENSORS = ["RGB_SENSOR", "DEPTH_SENSOR"]
    config.SIMULATOR.RGB_SENSOR.HFOV = 120
    config.SIMULATOR.DEPTH_SENSOR.HFOV = 120
    config.SIMULATOR.TURN_ANGLE = turn_angle
    config.freeze()
    sim = make_sim(id_sim=config.SIMULATOR.TYPE, config=config.SIMULATOR)
    pathfinder = sim.pathfinder
    return sim, pathfinder


def set_up_habitat(scene):
    config = get_config()
    config.defrost()
    config.SIMULATOR.SCENE = scene
    config.SIMULATOR.AGENT_0.SENSORS = ["RGB_SENSOR", "DEPTH_SENSOR"]
    config.SIMULATOR.RGB_SENSOR.HFOV = 120
    config.SIMULATOR.DEPTH_SENSOR.HFOV = 120
    config.SIMULATOR.TURN_ANGLE = 15
    config.freeze()

    sim = make_sim(id_sim=config.SIMULATOR.TYPE, config=config.SIMULATOR)
    pathfinder = sim.pathfinder
    return sim, pathfinder


def get_geodesic_dist(pathfinder, start_pos, goal_pos):
    path = ShortestPath()
    path.requested_start = start_pos
    path.requested_end = goal_pos
    pathfinder.find_path(path)
    geodesic_distance = path.geodesic_distance
    return geodesic_distance


def get_num_steps(sim, start_pos, start_rot, goal_pos):
    try:
        assert type(start_rot) == np.quaternion
    except:
        raise RuntimeError("rotation was not a quaternion")
    sim.set_agent_state(start_pos, start_rot)
    greedy_follower = sim.make_greedy_follower(goal_radius=0.25)
    try:
        steps = greedy_follower.find_path(goal_pos)
        total_steps = len(steps) - 1
    except:
        print("Error: greedy follower could not find path!")
        total_steps = 20
    if total_steps > 50:
        total_steps = 20
    return total_steps


def get_steps(sim, start_pos, start_rot, goal_pos, radius=1):
    next_step = -1
    steps = []
    try:
        assert type(start_rot) == np.quaternion
    except:
        raise RuntimeError("rotation was not a quaternion")
    sim.set_agent_state(start_pos, start_rot)
    greedy_follower = sim.make_greedy_follower(goal_radius=radius)
    try:
        steps = greedy_follower.find_path(goal_pos)
        next_step = steps[0]
    except:
        print("Error: greedy follower could not find path!")
    return steps, next_step


def se3_to_mat(rotation: np.quaternion, translation: np.ndarray):
    mat = np.eye(4)
    mat[0:3, 0:3] = quaternion.as_rotation_matrix(rotation)
    mat[0:3, 3] = translation
    return mat


def diff_rotation(quat1: np.quaternion, quat2: np.quaternion):
    agent_rotation1 = -quaternion.as_rotation_vector(quat1)[1] * 180 / np.pi
    agent_rotation2 = -quaternion.as_rotation_vector(quat2)[1] * 180 / np.pi
    if agent_rotation1 < 0:
        agent_rotation1 = 360 + agent_rotation1
    if agent_rotation2 < 0:
        agent_rotation2 = 360 + agent_rotation2
    delta_rot = abs(agent_rotation1 - agent_rotation2) % 360
    if delta_rot > 180:
        delta_rot = 360 - delta_rot
    return delta_rot


def diff_rotation_signed(quat1: np.quaternion, quat2: np.quaternion):
    agent_rotation1 = -quaternion.as_rotation_vector(quat1)[1] * 180 / np.pi
    agent_rotation2 = -quaternion.as_rotation_vector(quat2)[1] * 180 / np.pi
    if agent_rotation1 < 0:
        agent_rotation1 = 360 + agent_rotation1
    if agent_rotation2 < 0:
        agent_rotation2 = 360 + agent_rotation2
    delta_rot = agent_rotation1 - agent_rotation2
    if delta_rot > 180:
        delta_rot -= 360
    if delta_rot < -180:
        delta_rot += 360
    return delta_rot


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


def get_relative_location(
    source_position: np.array, source_rotation: np.array, goal_position: np.array
):
    direction_vector = np.asarray(goal_position) - np.asarray(source_position)
    direction_vector_agent = quaternion_rotate_vector(
        quaternion.from_float_array(source_rotation).inverse(), direction_vector
    )
    rho, phi = cartesian_to_polar(-direction_vector_agent[2], direction_vector_agent[0])
    return rho, -phi


def get_edge_attr(
    positionA: np.array, rotationA: np.array, positionB: np.array, rotationB: np.array
):
    stateA = se3_to_mat(
        quaternion.from_float_array(rotationA),
        positionA,
    )
    stateB = se3_to_mat(
        quaternion.from_float_array(rotationB),
        positionB,
    )
    edge_attr = inv(stateA) @ stateB
    delta_rot = round(
        diff_rotation_signed(
            quaternion.from_float_array(rotationA),
            quaternion.from_float_array(rotationB),
        )
    )

    return edge_attr, delta_rot