import json
import numpy as np
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.core.simulator import ActionSpaceConfiguration, ShortestPathPoint
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from typing import List, Union
from numpy import float64
import gzip
import submitit
import sys
from src.utils.model_utils import load_places_resnet, get_res_feats_batch

try:
    from habitat_sim.errors import GreedyFollowerError
except ImportError:
    GreedyFollower = BaseException
from src.utils.sim_utils import set_up_habitat
from habitat.utils.geometry_utils import quaternion_to_list

PATHS_PER_HOUSE = 1500
MAX_STEPS = 500
TEST_POINTS = 100
MIN_DIST = 1.5
MAX_DIST = 11
ISLAND_RADIUS_LIMIT = 1.5
RESNET = load_places_resnet()


def get_action_shortest_path(
    sim: "HabitatSim",
    source_position: List[float],
    source_rotation: List[Union[int, float64]],
    goal_position: List[float],
    success_distance: float = 0.05,
) -> List[ShortestPathPoint]:
    sim.reset()
    sim.set_agent_state(source_position, source_rotation)
    obs = sim.get_observations_at(source_position, source_rotation)
    images = [obs["rgb"]]
    follower = ShortestPathFollower(sim, success_distance, False)

    shortest_path = []
    step_count = 0
    action = follower.get_next_action(goal_position)
    while action is not HabitatSimActions.STOP and step_count < MAX_STEPS:
        state = sim.get_agent_state()
        shortest_path.append(
            ShortestPathPoint(
                state.position.tolist(),
                quaternion_to_list(state.rotation),
                action,
            )
        )
        obs = sim.step(action)
        images.append(obs["rgb"])
        step_count += 1
        action = follower.get_next_action(goal_position)

    goal_position = state.position.tolist()
    goal_rotation = quaternion_to_list(state.rotation)

    if step_count == MAX_STEPS:
        shortest_path, goal_position, goal_rotation, images = None, None, None, None

    return shortest_path, goal_position, goal_rotation, images


def get_next_point(pointA, sim, origin=None):
    for _ in range(TEST_POINTS):
        pointB = sim.sample_navigable_point()
        if pointA == pointB:
            continue
        if origin is not None:
            if origin == pointB:
                continue
        if sim.island_radius(pointB) < ISLAND_RADIUS_LIMIT:
            continue
        if np.abs(pointA[1] - pointB[1]) > 0.35:
            continue  # there is not a large difference in the z axis
        geo_dist = sim.geodesic_distance(pointA, pointB)
        if geo_dist == np.inf:
            continue
        if not MIN_DIST <= geo_dist <= MAX_DIST:
            continue
        return pointB
    return None, 0, 0


def build_single_path(sim, episode_id):
    for _ in range(TEST_POINTS):
        source_position = sim.sample_navigable_point()
        if sim.island_radius(source_position) < ISLAND_RADIUS_LIMIT:
            continue
        pointB = get_next_point(source_position, sim)
        if pointB is None:
            continue
        pointC = get_next_point(source_position, sim, origin=source_position)

        sampled_trajectory, goal_points, images = [], [], []
        angle = np.random.uniform(0, 2 * np.pi)
        source_rotation = [0, np.sin(angle / 2), 0, np.cos(angle / 2)]
        try:
            (
                shortest_pathAB,
                goal_positionAB,
                goal_rotationAB,
                imagesAB,
            ) = get_action_shortest_path(
                sim,
                source_position=source_position,
                source_rotation=source_rotation,
                goal_position=pointB,
                success_distance=0.05,
            )
            if shortest_pathAB is not None:
                sampled_trajectory.extend(shortest_pathAB)
                goal_points.append([goal_positionAB, goal_rotationAB])
                images.extend(imagesAB)
                if pointC is not None:
                    (
                        shortest_pathBC,
                        goal_positionBC,
                        goal_rotationBC,
                        imagesBC,
                    ) = get_action_shortest_path(
                        sim,
                        source_position=goal_positionAB,
                        source_rotation=goal_rotationAB,
                        goal_position=pointC,
                        success_distance=0.05,
                    )
                    if shortest_pathBC is not None:
                        sampled_trajectory.extend(shortest_pathBC)
                        goal_points.append([goal_positionBC, goal_rotationBC])
                        images.extend(imagesBC)

        # Throws an error when it can't find a path
        except GreedyFollowerError:
            continue
        if shortest_pathAB is None:
            continue

        # Extract Image Features
        filePath = save_dir + "feats/" + episode_id + ".pt"
        get_res_feats_batch(filePath, images, RESNET)

        source = [source_position, source_rotation]
        return source, sampled_trajectory, goal_points
    return None, None, None


def gather_paths_per_house(house, sim):
    found = 0
    episodes = []
    for _ in range(PATHS_PER_HOUSE):
        episode_id = house + "_" + str(found)
        source, sampled_trajectory, goal_points = build_single_path(sim, episode_id)
        if source is None:
            print("could not find any path")
            break
        episodes.append([source, sampled_trajectory, goal_points, episode_id])
        found += 1
    print(f"found a total of {found} paths for this house")
    return episodes


def save_data(data, scan_name):
    episode_list = []
    for item in data:
        source, sampled_trajectory, goal_points, episode_id = item
        if dataset == "mp3d":
            scene_id = f"mp3d/{scan_name}/{scan_name}.glb"
        else:
            scene_id = f"gibson/{scan_name}.glb"

        poses, rotations, actions = [], [], []
        for point in sampled_trajectory:
            poses.append(point.position)
            rotations.append(point.rotation)
            actions.append(point.action)
        episode = {
            "scene_id": scene_id,
            "episode_id": episode_id,
            "start_position": source[0],
            "start_rotation": source[1],
            "goals": goal_points,
            "poses": poses,
            "rotations": rotations,
            "actions": actions,
        }
        episode_list.append(episode)
    print(save_dir + scan_name + ".json.gz")
    with gzip.open(save_dir + scan_name + ".json.gz", "wt") as f:
        json.dump(episode_list, f)
    return {"episodes": episode_list}


def generate_trajectories(house):
    # generate paths
    episodes = []
    if dataset == "mp3d":
        scene = "{}{}/{}.glb".format(sim_dir, house, house)
        sim, _ = set_up_habitat(scene)
        episodes = gather_paths_per_house(house, sim)
    else:
        scene = "{}/{}.glb".format(sim_dir, house)
        sim, _ = set_up_habitat(scene)
        episodes = gather_paths_per_house(house, sim)
    sim.close()
    save_data(episodes, house)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        raise Exception("missing dataset argument-- Options: 'gibson' or 'mp3d'")
    print("dataset", sys.argv[1])
    dataset = sys.argv[1]
    sim_dir = "/srv/datasets/habitat-sim-datasets/"
    if dataset == "mp3d":
        sim_dir += "mp3d/"
    else:
        sim_dir += "gibson_train_val"
    base_dir = f"/srv/flash1/userid/topo_nav/{dataset}/"
    save_dir = base_dir + "trajectory_data/train_instances/"
    data_splits = f"../../data_splits/{dataset}/"
    scan_levels = json.load(open(data_splits + f"{dataset}_scan_levels.json"))
    train_scene_file = data_splits + "scenes_passive.txt"
    with open(train_scene_file) as f:
        train_scenes = sorted([line.rstrip() for line in f])

    submitit_log_dir = "/srv/flash1/userid/submitit/log_test"
    executor = submitit.AutoExecutor(folder=submitit_log_dir)
    executor.update_parameters(
        slurm_gres="gpu:1",
        slurm_cpus_per_task=6,
        slurm_time=24 * 60,
        slurm_partition="short",
        slurm_array_parallelism=30,
    )

    jobs = []

    with executor.batch():
        for house in train_scenes:
            print("current house:", house)
            jobs.append(
                executor.submit(
                    generate_trajectories,
                    house,
                )
            )

    print("jobs started", len(jobs))
    for job in jobs:
        print(job.results())
