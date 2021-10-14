import json
import gzip
from tqdm import tqdm
import sys


def process(data):
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
    return episode_list


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

    all_data = []
    for enum, house in enumerate(tqdm(train_scenes)):
        print("current house:", house)
        jsonfilename = save_dir + house + "_train.json.gz"
        with gzip.open(jsonfilename, "r") as fin:
            data = json.loads(fin.read().decode("utf-8"))
            if dataset == "gibson":
                data = process(data)
            all_data.extend(data)
            print(len(all_data))
    all_data = {"episodes": all_data}
    with gzip.open(save_dir + "train.json.gz", "wt") as f:
        json.dump(all_data, f)
