import os
import cv2
import json
import numpy as np
from imageio import imwrite
from tqdm import tqdm
from os import environ
from habitat import get_config
from habitat.sims import make_sim
from habitat.utils.visualizations import maps
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.core.simulator import ShortestPathPoint
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from typing import Iterable, List, Union
from numpy import float64
import quaternion
import gzip

try:
    from habitat_sim.errors import GreedyFollowerError
except ImportError:
    GreedyFollower = BaseException
from src.utils.sim_utils import set_up_habitat, diff_rotation
from habitat.utils.geometry_utils import quaternion_to_list, quaternion_from_coeff


def on_level(point):
    if point[1] >= y_axes[0] and point[1] <= y_axes[1]:
        return True
    return False


max_steps = 500
near_dist = 1.5
far_dist = 10
ISLAND_RADIUS_LIMIT = 1.5


def get_action_shortest_path(
    sim: "HabitatSim",
    source_position: List[float],
    source_rotation: List[Union[int, float64]],
    goal_position: List[float],
    success_distance: float = 0.05,
    max_episode_steps: int = 500,
) -> List[ShortestPathPoint]:
    sim.reset()
    sim.set_agent_state(source_position, source_rotation)
    follower = ShortestPathFollower(sim, success_distance, False)

    easy_source = None
    shortest_path = []
    step_count = 0
    action = follower.get_next_action(goal_position)
    while action is not HabitatSimActions.STOP and step_count < max_episode_steps:
        state = sim.get_agent_state()
        shortest_path.append(
            ShortestPathPoint(
                state.position.tolist(),
                quaternion_to_list(state.rotation),
                action,
            )
        )
        sim.step(action)
        step_count += 1
        action = follower.get_next_action(goal_position)
        if step_count == 10:
            easy_source = [state.position.tolist(), quaternion_to_list(state.rotation)]

    goal_position = state.position.tolist()
    goal_rotation = quaternion_to_list(state.rotation)

    if step_count == max_episode_steps:
        shortest_path = None
    return shortest_path, easy_source, goal_position, goal_rotation


def get_next_point(pointA, pointName):
    for _ in range(500):
        pointB = sim.sample_navigable_point()
        if pointA == pointB:
            continue
        if sim.island_radius(pointB) < ISLAND_RADIUS_LIMIT:
            continue
        if np.abs(pointA[1] - pointB[1]) > 0.35:
            continue  # there is not a large difference in the z axis
        geo_dist = sim.geodesic_distance(pointA, pointB)
        if geo_dist == np.inf:
            continue
        if not near_dist <= geo_dist <= far_dist:
            continue
        euclid_dist = np.power(
            np.power(np.array(pointA) - np.array(pointB), 2).sum(0), 0.5
        )
        return pointB, geo_dist, euclid_dist
    return None, 0, 0


def build_single_path():
    for _ in range(500):
        source_position = sim.sample_navigable_point()
        if dataset == "mp3d":
            if not on_level(source_position):
                continue  # on correct level
        if sim.island_radius(source_position) < ISLAND_RADIUS_LIMIT:
            continue
        point2, geo_dist, euclid_dist = get_next_point(source_position, "2")
        if point2 is None:
            continue
        angle = np.random.uniform(0, 2 * np.pi)
        source_rotation = [0, np.sin(angle / 2), 0, np.cos(angle / 2)]
        try:
            (
                shortest_paths,
                easy_source,
                goal_position,
                goal_rotation,
            ) = get_action_shortest_path(
                sim,
                source_position=source_position,
                source_rotation=source_rotation,
                goal_position=point2,
                success_distance=0.05,
                max_episode_steps=500,
            )
        # Throws an error when it can't find a path
        except GreedyFollowerError:
            continue
        if shortest_paths is None:
            continue

        sources = []
        goal = [goal_position, goal_rotation]
        hard_source = [source_position, source_rotation]
        for enum, start in enumerate([easy_source, hard_source]):
            if start is None:
                continue
            geo_dist = sim.geodesic_distance(start[0], goal_position)
            if geo_dist == np.inf:
                continue
            if not near_dist <= geo_dist <= far_dist:
                continue
            euclid_dist = np.power(
                np.power(np.array(start[0]) - np.array(goal_position), 2).sum(0), 0.5
            )
            dist_ratio = geo_dist / (euclid_dist + 0.00001)
            quat1 = quaternion_from_coeff(np.asarray(start[1]))
            quat2 = quaternion_from_coeff(np.asarray(goal_rotation))
            rotation_diff = np.abs(diff_rotation(quat1, quat2))
            if enum == 0 and (dist_ratio >= 1.2 or rotation_diff > 35):
                continue
            sources.append([start, geo_dist, dist_ratio, rotation_diff])
        if len(sources) == 0:
            continue
        return sources, goal
    print("could not find full path")
    return None, None


def gather_paths_per_floor(house, paths_per_floor):
    if paths_per_floor is None:
        paths_per_floor = 300
    found = 0
    for _ in range(paths_per_floor):
        sources, goal = build_single_path()

        # move to next level
        if sources is None:
            break

        for source in sources:
            start, geo_dist, dist_ratio, rotation_diff = source
            quat1 = quaternion_from_coeff(np.asarray(start[1]))
            quat2 = quaternion_from_coeff(np.asarray(goal[1]))
            if dist_ratio < 1.2 and rotation_diff <= 35:
                if geo_dist < 3:
                    straight["easy"].append([house, start, goal, geo_dist])
                elif 3 <= geo_dist < 5:
                    straight["medium"].append([house, start, goal, geo_dist])
                else:
                    straight["hard"].append([house, start, goal, geo_dist])
            else:
                if geo_dist < 3:
                    curved["easy"].append([house, start, goal, geo_dist])
                elif 3 <= geo_dist < 5:
                    curved["medium"].append([house, start, goal, geo_dist])
                else:
                    curved["hard"].append([house, start, goal, geo_dist])
            found += 1
    print(f"found a total of {found} for this floor")


def generate_trajectories(houseList):
    # generate paths
    global pathfinder
    global sim
    global y_axes
    global straight
    global curved
    straight = {"easy": [], "medium": [], "hard": []}
    curved = {"easy": [], "medium": [], "hard": []}
    for enum, house in enumerate(tqdm(houseList)):
        print("current house:", house)
        if dataset == "mp3d":
            scene = "{}{}/{}.glb".format(sim_dir, house, house)
            sim, pathfinder = set_up_habitat(scene)
            for floor in range(int(scan_levels[house]["levels"])):
                print(
                    f"scan {house}, {floor + 1} out of {scan_levels[house]['levels']} floors"
                )
                y_axes = list(map(float, scan_levels[house]["z_axes"][floor]))
                gather_paths_per_floor(house, None)
        else:
            scene = "{}/{}.glb".format(sim_dir, house)
            sim, pathfinder = set_up_habitat(scene)
            paths_per_floor = (
                500  # round((scan_levels[house]["area"] * 1.0 / 500) * 300)
            )
            gather_paths_per_floor(house, paths_per_floor)
        pathfinder = None
        sim.close()


def save_data(data):
    episode_list = []
    for enum, d in enumerate(data):
        scan_name, start, goal, length_shortest = d
        if dataset == "mp3d":
            scene_id = f"mp3d/{scan_name}/{scan_name}.glb"
        else:
            scene_id = f"gibson/{scan_name}.glb"
        episode = {
            "scene_id": scene_id,
            "episode_id": scan_name + "_" + str(enum),
            "start_position": start[0],
            "start_rotation": start[1],
            "goals": [{"position": goal[0], "rotation": goal[1]}],
            "length_shortest": length_shortest,
        }
        episode_list.append(episode)
    print(len(episode_list))
    return {"episodes": episode_list}


if __name__ == "__main__":
    dataset = "gibson"  # mp3d
    sim_dir = "/srv/datasets/habitat-sim-datasets/"
    if dataset == "mp3d":
        sim_dir += "mp3d/"
    else:
        sim_dir += "gibson_train_val"
    base_dir = f"/srv/flash1/userid/topo_nav/{dataset}/"
    save_dir = base_dir + "image_nav_episodes/"
    data_splits = f"../../data_splits/{dataset}/"
    scan_levels = json.load(open(data_splits + f"{dataset}_scan_levels.json"))
    test_scene_file = data_splits + "scenes_test.txt"
    with open(test_scene_file) as f:
        test_scenes = sorted([line.rstrip() for line in f])

    generate_trajectories(test_scenes)

    data = save_data(straight["easy"])
    with gzip.open(save_dir + "straight/test_easy.json.gz", "wt") as f:
        json.dump(data, f)
    data = save_data(straight["medium"])
    with gzip.open(save_dir + "straight/test_medium.json.gz", "wt") as f:
        json.dump(data, f)
    data = save_data(straight["hard"])
    with gzip.open(save_dir + "straight/test_hard.json.gz", "wt") as f:
        json.dump(data, f)

    data = save_data(curved["easy"])
    with gzip.open(save_dir + "curved/test_easy.json.gz", "wt") as f:
        json.dump(data, f)
    data = save_data(curved["medium"])
    with gzip.open(save_dir + "curved/test_medium.json.gz", "wt") as f:
        json.dump(data, f)
    data = save_data(curved["hard"])
    with gzip.open(save_dir + "curved/test_hard.json.gz", "wt") as f:
        json.dump(data, f)