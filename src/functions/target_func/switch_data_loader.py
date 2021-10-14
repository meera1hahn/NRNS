import numpy as np
import torch
from torch.utils.data import Dataset
import msgpack_numpy
import math
from src.utils.sim_utils import get_relative_location


class Loader:
    def __init__(self, args):
        self.datasets = {}
        self.args = args
        self.args.trajectory_info_dir = args.trajectory_data_dir + "trajectoryInfo/"
        self.args.distance_data_dir = args.base_dir + args.distance_data_dir

    def load_examples(self, data):
        scans = []
        trajs = []
        switches = []
        node_feat1 = []
        node_feat2 = []
        node_poses = []
        node_rots = []
        rot_diff = []
        for house in data:
            scan_name = house[0]["scan_name"]
            for d in house:
                switch = 0
                dist_ratio = d["geodesic"] / (d["euclidean"] + 0.00001)
                if (
                    (
                        abs(d["rotation_diff"]) <= 60
                        and d["geodesic"] <= 1
                        and d["euclidean"] <= 1
                        and dist_ratio <= 1.1
                    )
                    or (
                        abs(d["rotation_diff"]) <= 45
                        and d["geodesic"] <= 2.25
                        and d["euclidean"] <= 2.25
                        and dist_ratio <= 1.01
                    )
                    or (
                        abs(d["rotation_diff"]) <= 25
                        and d["geodesic"] <= 3.5
                        and d["euclidean"] <= 3.5
                        and dist_ratio <= 1.001
                    )
                ):
                    switch = 1
                scans.append(scan_name)
                trajs.append(d["traj"])
                switches.append(switch)
                node_feat1.append(str(d["n1"]))
                node_feat2.append(str(d["n2"]))
                node_poses.append([d["agent_position1"], d["agent_position2"]])
                node_rots.append([d["agent_rotation1"], d["agent_rotation2"]])
                rot_diff.append(d["rotation_diff"])

        return (
            scans,
            trajs,
            switches,
            node_feat1,
            node_feat2,
            node_poses,
            node_rots,
            rot_diff,
        )

    def build_dataset(self, split):
        print("Loading {} dataset...".format(self.args.dataset))
        splitFile = self.args.data_splits + "scenes_" + split + ".txt"
        splitScans = [x.strip() for x in open(splitFile, "r").readlines()]
        data = []
        for house in splitScans:
            houseFile = self.args.distance_data_dir + house + "_graph_distance.msg"
            if len(msgpack_numpy.unpack(open(houseFile, "rb"), raw=False)) != 0:
                data.append(msgpack_numpy.unpack(open(houseFile, "rb"), raw=False))

        data_size = len(data)
        print("[{}]: Using {} houses".format("data", data_size))

        (
            scans,
            trajs,
            switches,
            node_feat1,
            node_feat2,
            node_poses,
            node_rots,
            rot_diff,
        ) = self.load_examples(data)
        ratio = np.asarray(switches).sum() * 1.0 / len(switches)
        print("[{}]: Switch Ratio".format(ratio))
        dataset = DistanceDatset(
            self.args,
            scans,
            trajs,
            switches,
            node_feat1,
            node_feat2,
            node_poses,
            node_rots,
            rot_diff,
        )
        self.datasets[split] = dataset
        print("[{}]: Finish building dataset...".format(split))


class DistanceDatset(Dataset):
    def __init__(
        self,
        args,
        scans,
        trajs,
        switches,
        node_feat1,
        node_feat2,
        node_poses,
        node_rots,
        rot_diff,
    ):
        self.args = args
        self.scans = scans
        self.trajs = trajs
        self.switches = switches
        self.node_feat1 = node_feat1
        self.node_feat2 = node_feat2
        self.node_poses = node_poses
        self.node_rots = node_rots
        self.rot_diff = rot_diff

    def __getitem__(self, index):
        scan_name = self.scans[index]
        trajectory = self.trajs[index]

        featFile = self.args.distance_data_dir + scan_name + "_n_feats.msg"
        feats = msgpack_numpy.unpack(open(featFile, "rb"), raw=False)
        infoFile = self.args.trajectory_info_dir + trajectory + ".msg"
        states = msgpack_numpy.unpack(open(infoFile, "rb"), raw=False)["states"]

        switch = torch.tensor(self.switches[index], dtype=torch.float)
        node1 = torch.tensor(
            feats[trajectory][self.node_feat1[index]], dtype=torch.float
        )
        node2 = torch.tensor(
            feats[trajectory][self.node_feat2[index]], dtype=torch.float
        )
        angle_diff = torch.tensor(self.rot_diff[index], dtype=torch.float)
        angle_encoding = torch.tensor(
            [
                round(math.cos(math.radians(angle_diff)), 2),
                round(math.sin(math.radians(angle_diff)), 2),
            ],
            dtype=torch.float,
        )

        start_pos = states[int(self.node_feat1[index])][0]
        start_rot = states[int(self.node_feat1[index])][1]
        goal_pos = states[int(self.node_feat2[index])][0]
        rho, _ = get_relative_location(start_pos, start_rot, goal_pos)
        dist_score = torch.tensor(
            1 - min(rho, self.args.dist_max) / self.args.dist_max,
            dtype=torch.float,
        )

        return switch, node1, node2, angle_encoding, dist_score

    def __len__(self):
        return len(self.switches)
