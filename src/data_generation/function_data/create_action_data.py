import numpy as np
import quaternion
import torch
import msgpack_numpy
import os
import argparse
from src.utils.cfg import input_paths

"""V1: Gets the data to train Behavioral Cloning from the trajectories"""
parser = argparse.ArgumentParser()
parser = input_paths(parser)   
args = parser.parse_args() 
args.base_dir += f"{args.dataset}/no_noise/"
args.data_splits += f"{args.dataset}/"

def get_angle_diff(states, n1, n2):
    # Get rotational difference between two nodes
    quat1 = quaternion.from_float_array(states[n1][1])
    agent_rotation1 = -quaternion.as_rotation_vector(quat1)[1] * 180 / np.pi
    quat2 = quaternion.from_float_array(states[n2][1])
    agent_rotation2 = -quaternion.as_rotation_vector(quat2)[1] * 180 / np.pi
    if agent_rotation1 < 0:
        agent_rotation1 = 360 + agent_rotation1
    if agent_rotation2 < 0:
        agent_rotation2 = 360 + agent_rotation2
    d = agent_rotation1 - agent_rotation2
    if d > 180:
        d -= 360
    if d < -180:
        d += 360
    return d, agent_rotation1, agent_rotation2


"""Builds graph from all the passive videos, expects feats to already be calculated"""


def get_scenes(scenes, scanName):
    ratio = []
    pairs = []
    traj_feats = {}
    for scene in scenes:

        """Load trajectory data"""
        scan_name = scene.split("_")[0]
        floor = scene.split("_")[1]
        infoFile = trajectory_data_dir + "trajectoryInfo/" + scene + ".msg"
        info = msgpack_numpy.unpack(open(infoFile, "rb"), raw=False)
        states = info["states"]
        actions = info["actions"]
        featFile = trajectory_data_dir + "trajectoryFeats/" + scene + ".pt"
        feats = torch.load(featFile).squeeze(-1).squeeze(-1)
        try:
            assert len(states) == feats.shape[0]
        except:
            import ipdb

            ipdb.set_trace()
            print("error")
            continue

        startIndex = 1
        """Loop over trajectory"""

        if startIndex == len(states) - 1:
            raise Exception("path is too short")
        traj_feats[scene] = {}
        total = 0
        forward = 0
        stop = 0
        for n1 in range(startIndex, len(states), 10):
            for n2 in range(n1 + 1, len(states), 2):
                previous_actions = actions[0:n1]
                next_actions = actions[n1:n2]
                if len(next_actions) <= 1:
                    next_step = 0
                else:
                    next_step = next_actions[0]

                # Save info and feats
                pairs.append(
                    {
                        "n1": n1,
                        "n2": n2,
                        "agent_position1": states[n1][0],
                        "agent_position2": states[n2][0],
                        "agent_rotation1": states[n1][1],
                        "agent_rotation2": states[n2][1],
                        "previous_actions": previous_actions,
                        "steps": next_actions,
                        "next_step": next_step,
                        "traj": scene,
                        "floor": floor,
                        "scan_name": scan_name,
                    }
                )
                traj_feats[scene][str(n1)] = feats[n1].detach().numpy()
                traj_feats[scene][str(n2)] = feats[n2].detach().numpy()
                traj_feats[scene]["prev"] = feats[0:n1].detach().numpy()
                total += 1
                if len(next_actions) > 25:
                    break  # go to next node
                if next_step == 0:
                    stop += 1
                if next_step == 1:
                    forward += 1

                if len(previous_actions) == 0:
                    ratio.append(0)
                else:
                    if next_step == previous_actions[-1]:
                        ratio.append(1)
                    else:
                        ratio.append(0)

            if total > 50:
                break
    print("ratio", np.mean(ratio))
    msgpack_numpy.pack(
        pairs,
        open(action_dir + scanName + "_action_data.msg", "wb"),
        use_bin_type=True,
    )
    print("saved abridged node data.")
    msgpack_numpy.pack(
        traj_feats,
        open(action_dir + scanName + "_action_feats.msg", "wb"),
        use_bin_type=True,
    )
    print("saved abridged feat data.")


def run_house(scanName):
    scenes = []
    for s in os.listdir(trajectory_feats_dir):
        if scanName in s:
            scenes.append(s[:-3])
    get_scenes(scenes, scanName)


if __name__ == "__main__":
    action_dir = args.base_dir + "behavioral_cloning/"
    trajectory_data_dir = args.base_dir + "trajectory_data/"
    trajectory_feats_dir = trajectory_data_dir + "trajectoryFeats/"
    passive_scene_file = args.base_dir + "scenes_passive.txt"
    with open(passive_scene_file) as f:
        passive_scenes = sorted([line.rstrip() for line in f])
    houseList = sorted(passive_scenes)
    for enum, scanName in enumerate(houseList):
        print(f"Current Scan {scanName}")
        run_house(scanName)
