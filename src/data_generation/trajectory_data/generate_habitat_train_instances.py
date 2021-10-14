import json
import numpy as np
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.core.simulator import ShortestPathPoint
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from typing import List, Union
from numpy import float64
import gzip
import submitit
import sys

try:
    from habitat_sim.errors import GreedyFollowerError
except ImportError:
    GreedyFollower = BaseException
from src.utils.sim_utils import set_up_habitat, diff_rotation
from habitat.utils.geometry_utils import quaternion_to_list, quaternion_from_coeff

PATHS_PER_HOUSE = 5000
MAX_STEPS = 400
TEST_POINTS = 100
MIN_DIST = 1.5
MAX_DIST = 10
ISLAND_RADIUS_LIMIT = 1.5


def get_action_shortest_path(
    sim: "HabitatSim",
    source_position: List[float],
    source_rotation: List[Union[int, float64]],
    goal_position: List[float],
    success_distance: float = 0.05,
) -> List[ShortestPathPoint]:
    sim.reset()
    sim.set_agent_state(source_position, source_rotation)
    follower = ShortestPathFollower(sim, success_distance, False)

    easy_source = None
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
        sim.step(action)
        step_count += 1
        action = follower.get_next_action(goal_position)
        if step_count == 10:
            easy_source = [state.position.tolist(), quaternion_to_list(state.rotation)]

    goal_position = state.position.tolist()
    goal_rotation = quaternion_to_list(state.rotation)

    if step_count == MAX_STEPS:
        shortest_path = None
    return shortest_path, easy_source, goal_position, goal_rotation


def get_next_point(pointA, sim):
    for _ in range(TEST_POINTS):
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
        if not MIN_DIST <= geo_dist <= MAX_DIST:
            continue
        euclid_dist = np.power(
            np.power(np.array(pointA) - np.array(pointB), 2).sum(0), 0.5
        )
        return pointB, geo_dist, euclid_dist
    return None, 0, 0


def build_single_path(sim):
    for _ in range(TEST_POINTS):
        source_position = sim.sample_navigable_point()
        if sim.island_radius(source_position) < ISLAND_RADIUS_LIMIT:
            continue
        point2, geo_dist, euclid_dist = get_next_point(source_position, sim)
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
            if not MIN_DIST <= geo_dist <= MAX_DIST:
                continue
            euclid_dist = np.power(
                np.power(np.array(start[0]) - np.array(goal_position), 2).sum(0), 0.5
            )
            dist_ratio = geo_dist / (euclid_dist + 0.00001)
            quat1 = quaternion_from_coeff(np.asarray(start[1]))
            quat2 = quaternion_from_coeff(np.asarray(goal_rotation))
            rotation_diff = np.abs(diff_rotation(quat1, quat2))
            if enum == 0 and (dist_ratio >= 1.2 and rotation_diff < 45):
                continue
            sources.append([start, geo_dist, dist_ratio, rotation_diff])
        if len(sources) == 0:
            continue
        return sources, goal
    print("could not any path")
    return None, None


def gather_paths_per_house(house, sim):
    found = 0
    straight = 0
    curved = 0
    episodes = []
    for _ in range(PATHS_PER_HOUSE):
        sources, goal = build_single_path(sim)
        if sources is None:
            break
        for source in sources:
            start, geo_dist, dist_ratio, rotation_diff = source
            if dist_ratio < 1.2 and rotation_diff < 45:
                straight += 1
            else:
                curved += 1
            episodes.append([house, start, goal, geo_dist])
            found += 1
    print(f"found a total of {found} for this house")
    print(f"found {straight} straight paths for this house")
    print(f"found {curved} curved paths for this house")
    return episodes


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
        }
        episode_list.append(episode)
    print(len(episode_list))
    with gzip.open(save_dir + scan_name + "_train.json.gz", "wt") as f:
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
    save_data(episodes)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        raise Exception("missing dataset argumennt-- Options: 'gibson' or 'mp3d'")
    print("dataset", sys.argv[1])
    dataset = sys.argv[1]
    sim_dir = "/srv/datasets/habitat-sim-datasets/"
    if dataset == "mp3d":
        sim_dir += "mp3d/"
    else:
        sim_dir += "gibson_train_val"
    base_dir = f"/srv/flash1/userid/topo_nav/{dataset}/"
    save_dir = base_dir + "image_nav_episodes/hab_train_data/"
    data_splits = f"../../data_splits/{dataset}/"
    scan_levels = json.load(open(data_splits + f"{dataset}_scan_levels.json"))
    train_scene_file = data_splits + "scenes_train.txt"
    with open(train_scene_file) as f:
        train_scenes = sorted([line.rstrip() for line in f])

    submitit_log_dir = "/srv/flash1/userid/submitit/log_test"
    executor = submitit.AutoExecutor(
        folder=submitit_log_dir
    )  # submission interface (logs are dumped in the folder)
    executor.update_parameters(
        slurm_gres="gpu:1",
        slurm_cpus_per_task=6,
        slurm_time=24 * 60,
        slurm_partition="short",
        slurm_array_parallelism=30,
    )  # timeout in min

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
