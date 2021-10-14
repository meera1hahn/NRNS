import torch
from torch.utils.data import Dataset
import msgpack_numpy
import copy
import math
from src.utils.sim_utils import get_relative_location


class Loader:
    def __init__(self, args):
        self.datasets = {}
        self.args = args
        self.trajectory_info_dir = args.trajectory_data_dir + "trajectoryInfo/"
        self.args.distance_data_dir = args.base_dir + args.distance_data_dir

    def load_examples(self, splitScans):
        node_feat1 = []
        node_feat2 = []
        infos = []
        """ info to contain [floor,scan_name,traj,n1,n2]"""
        for house in splitScans:
            trajectory_info_dir = self.trajectory_info_dir
            distance_data_dir = self.args.distance_data_dir
            trajFile = distance_data_dir + house + "_graph_distance.msg"
            trajs = msgpack_numpy.unpack(open(trajFile, "rb"), raw=False)
            featFile = distance_data_dir + house + "_n_feats.msg"
            feats = msgpack_numpy.unpack(open(featFile, "rb"), raw=False)
            for d in trajs:
                dist_ratio = d["geodesic"] / (d["euclidean"] + 0.00001)

                if (
                    not (
                        abs(d["rotation_diff"]) <= 45
                        and d["geodesic"] <= 1
                        and d["euclidean"] <= 1
                        and dist_ratio <= 1.1
                    )
                    or not (
                        abs(d["rotation_diff"]) <= 25
                        and d["geodesic"] <= 2.25
                        and d["euclidean"] <= 2.25
                        and dist_ratio <= 1.01
                    )
                    or not (
                        abs(d["rotation_diff"]) <= 15
                        and d["geodesic"] <= 3.5
                        and d["euclidean"] <= 3.5
                        and dist_ratio <= 1.001
                    )
                ):
                    continue
                trajectory = d["traj"]

                node_feat1.append(feats[trajectory][str(d["n1"])])
                node_feat2.append(feats[trajectory][str(d["n2"])])

                infoFile = trajectory_info_dir + trajectory + ".msg"
                states = msgpack_numpy.unpack(open(infoFile, "rb"), raw=False)["states"]
                start_pos = states[d["n1"]][0]
                start_rot = states[d["n1"]][1]
                goal_pos = states[d["n2"]][0]
                rho, phi = get_relative_location(start_pos, start_rot, goal_pos)

                infos.append(
                    [
                        d["scan_name"],
                        d["traj"],
                        phi,
                        rho,
                        start_pos,
                        start_rot,
                        goal_pos,
                    ]
                )

        return (
            node_feat1,
            node_feat2,
            infos,
        )

    def build_dataset(self, split):
        splitFile = self.args.data_splits + "scenes_" + split + ".txt"
        splitScans = [x.strip() for x in open(splitFile, "r").readlines()]
        node_feat1, node_feat2, infos = self.load_examples(splitScans)
        print("[{}]: Using {} houses".format(split, len(splitScans)))
        dataset = DistanceDatset(self.args, node_feat1, node_feat2, infos)
        self.datasets[split] = dataset
        print("[{}]: Finish building dataset...".format(split))


class DistanceDatset(Dataset):
    def __init__(
        self,
        args,
        node_feat1,
        node_feat2,
        infos,
    ):
        self.args = args
        self.node_feat1 = node_feat1
        self.node_feat2 = node_feat2
        self.infos = infos

    def __getitem__(self, index):
        node1 = torch.tensor(copy.deepcopy(self.node_feat1[index]), dtype=torch.float)
        node2 = torch.tensor(copy.deepcopy(self.node_feat2[index]), dtype=torch.float)
        infos = self.infos[index]
        phi = torch.tensor(self.infos[index][2], dtype=torch.float)
        angle_encoding = torch.tensor(
            [
                round(math.cos(phi), 2),
                round(math.sin(phi), 2),
            ],
            dtype=torch.float,
        )

        geo_dist = self.infos[index][3]
        dist_score = geo_dist
        # torch.tensor(
        #     1 - min(geo_dist, self.args.dist_max) / self.args.dist_max,
        #     dtype=torch.float,
        # )
        start_poses = torch.tensor(infos[4], dtype=torch.float)
        start_rots = torch.tensor(infos[5], dtype=torch.float)
        goal_poses = torch.tensor(infos[6], dtype=torch.float)

        return (
            node1,
            node2,
            phi,
            angle_encoding,
            dist_score,
            start_poses,
            start_rots,
            goal_poses,
        )

    def __len__(self):
        return len(self.infos)
