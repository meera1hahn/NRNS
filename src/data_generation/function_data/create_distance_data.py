import numpy as np, re
import quaternion
import torch
import msgpack_numpy
import os
import argparse
from src.utils.sim_utils import diff_rotation
from src.utils.cfg import input_paths

"""Gets the data to train the Target function from the trajectories"""

"""Builds graph from all the passive videos, expects feats to already be calculated"""

def get_scenes(scenes, scanName):
    pairs = []
    traj_feats = {}
    for scene in scenes:
        """Load trajectory data"""
        if dataset == "mp3d":
            scan_name = scene.split("_")[0]
        else:
            scan_name = re.match(r"([a-z]+)([0-9]+)", scene, re.I).groups()[0]
        infoFile = trajectory_data_dir + "trajectoryInfo/" + scene + ".msg"
        info = msgpack_numpy.unpack(open(infoFile, "rb"), raw=False)
        states = info["states"]
        featFile = trajectory_data_dir + "trajectoryFeats/" + scene + ".pt"
        feats = torch.load(featFile).squeeze(-1).squeeze(-1)
        try:
            assert len(states) == feats.shape[0]
        except:
            print("error")
            continue

        """Loop over trajectory"""
        traj_feats[scene] = {}
        total = 0
        for n1 in range(0, len(states), 5):
            for n2 in range(n1 + 1, len(states), 5):
                # Get rotational difference between two nodes
                quat1 = quaternion.from_float_array(states[n1][1])
                quat2 = quaternion.from_float_array(states[n2][1])
                agent_rotation1 = -quaternion.as_rotation_vector(quat1)[1] * 180 / np.pi
                agent_rotation2 = -quaternion.as_rotation_vector(quat2)[1] * 180 / np.pi
                rotation_diff = diff_rotation(quat1, quat2)

                # Get distance between two nodes
                geodesic = 0
                for i in range(n1, n2):
                    geodesic += np.linalg.norm(
                        np.asarray(
                            np.asarray(states[i][pose_index])
                            - np.asarray(states[i + 1][pose_index])
                        )
                    )
                euclidean = np.linalg.norm(
                    np.asarray(
                        np.asarray(states[n1][pose_index])
                        - np.asarray(states[n2][pose_index])
                    )
                )

                # Save info and feats
                pairs.append(
                    {
                        "n1": n1,
                        "n2": n2,
                        "agent_position1": states[n1][pose_index],
                        "agent_position2": states[n2][pose_index],
                        "agent_rotation1": agent_rotation1,
                        "agent_rotation2": agent_rotation2,
                        "geodesic": geodesic,
                        "euclidean": euclidean,
                        "rotation_diff": rotation_diff,
                        "traj": scene,
                        # "floor": floor,
                        "scan_name": scan_name,
                    }
                )
                traj_feats[scene][str(n1)] = feats[n1].detach().numpy()
                traj_feats[scene][str(n2)] = feats[n2].detach().numpy()
                total += 1
                if geodesic > 10:
                    break  # go to next node

    outfile = distance_data_dir + scanName + "_graph_distance.msg"
    msgpack_numpy.pack(pairs, open(outfile, "wb"), use_bin_type=True)
    print("saved abridged feat data @", outfile)
    outfile = distance_data_dir + scanName + "_n_feats.msg"
    msgpack_numpy.pack(traj_feats, open(outfile, "wb"), use_bin_type=True)
    print("saved abridged feat data @", outfile)


def run_house(scanName):
    scenes = []
    for s in os.listdir(trajectory_feats_dir):
        if scanName in s:
            scenes.append(s[:-3])
    get_scenes(scenes, scanName)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = input_paths(parser)   
    args = parser.parse_args() 
    args.base_dir += f"{args.dataset}"
    args.data_splits += f"{args.dataset}/"


    dataset = "mp3d"  # "gibson"
    noise = False
    pose_index = 0
    if noise:
        pose_index = 2
    
    if noise:
        args.base_dir += f"{args.dataset}/noise/"
    else:
        args.base_dir += f"{args.dataset}/no_noise/"

    trajectory_data_dir = args.base_dir + "trajectory_data/"
    distance_data_dir = args.base_dir + "distance_data_straight/"
    trajectory_feats_dir = trajectory_data_dir + "trajectoryFeats/"
    passive_scene_file = args.base_dir + "scenes_train.txt"

    with open(passive_scene_file) as f:
        passive_scenes = sorted([line.rstrip() for line in f])
    houseList = sorted(passive_scenes)

    for enum, scanName in enumerate(houseList):
        print(f"Current Scan {scanName}")
        run_house(scanName)
