import numpy as np
import torch
from torch.utils.data import Dataset
import msgpack_numpy
import torch.nn.functional as F


class Loader:
    def __init__(self, args):
        self.datasets = {}
        self.args = args
        self.bias = 5
        self.stop_bias = 15
        self.sim_dir = self.args.sim_dir
        self.map_dir = args.base_dir + "bc_metric_maps/"
        self.args.map_dir = self.map_dir

    def load_examples(self, splitScans):
        batch_size = 26
        goal_feats = []
        batch_feats = []
        prev_action_list = []
        next_action_list = []
        inflections = []

        infos = []  # [floor,scan_name,traj,n1,n2]
        max_seq_len = 0
        for scan in splitScans:
            directory = self.args.base_dir + self.args.action_dir + scan
            trajFile = directory + "_gru_action_data.msg"
            featFile = directory + "_gru_action_feats.msg"
            trajs = msgpack_numpy.unpack(open(trajFile, "rb"), raw=False)
            feats = msgpack_numpy.unpack(open(featFile, "rb"), raw=False)
            for d in trajs:
                trajectory = d["traj"]
                maps = np.asarray(d["maps"])
                maps = torch.from_numpy(np.repeat(maps[..., np.newaxis], 3, -1))
                torch.save(maps, self.map_dir + trajectory + ".pt")
                batch_feats.append(
                    torch.tensor(feats[trajectory][str(d["goal_index"])])
                )
                goal_feats.append(
                    torch.tensor(feats[trajectory][str(d["goal_index"])][-1]).repeat(
                        batch_size, 1
                    )
                )
                next_action_list.append(torch.tensor(np.asarray(d["next_actions"])))
                prev_action_list.append(
                    torch.tensor(np.asarray([4] + d["next_actions"][:-1]))
                )
                infos.append(
                    [
                        d["floor"],
                        d["scan_name"],
                        d["traj"],
                    ]
                )
                inflection_weight = [self.bias]
                for i in range(1, len(d["next_actions"]) - 1):
                    if d["next_actions"][i] != d["next_actions"][i - 1]:
                        inflection_weight.append(self.bias)
                    else:
                        inflection_weight.append(1)
                inflection_weight.append(self.stop_bias)
                inflections.append(torch.tensor(inflection_weight))
                max_seq_len += len(inflection_weight)
        print(max_seq_len)
        return (
            inflections,
            goal_feats,
            batch_feats,
            next_action_list,
            prev_action_list,
            infos,
        )

    def process_dataset(self, split):
        print("Loading {} dataset...".format(self.args.dataset))
        splitFile = self.args.data_splits + "scenes_" + split + ".txt"
        splitScans = [x.strip() for x in open(splitFile, "r").readlines()]
        print("[{}]: Using {} houses".format(split, len(splitScans)))

        (
            inflections,
            goal_feats,
            batch_feats,
            next_action_list,
            prev_action_list,
            infos,
        ) = self.load_examples(splitScans)

        dataset = ActionDataset(
            self.args,
            inflections,
            goal_feats,
            batch_feats,
            next_action_list,
            prev_action_list,
            infos,
        )
        return dataset

    def build_dataset(self, split):
        dataset = self.process_dataset(split)
        self.datasets[split] = dataset
        print("[{}]: Finish building dataset...".format(split))


class ActionDataset(Dataset):
    def __init__(
        self,
        args,
        inflections,
        goal_feats,
        batch_feats,
        next_action_list,
        prev_action_list,
        infos,
    ):
        self.args = args
        self.inflections = inflections
        self.goal_feats = goal_feats
        self.batch_feats = batch_feats
        self.next_action_list = next_action_list
        self.prev_action_list = prev_action_list
        self.infos = infos

    def __getitem__(self, index):
        inflection = self.inflections[index].clone()
        goal_feat = self.goal_feats[index].clone()
        batch_feat = self.batch_feats[index].clone()
        next_action = self.next_action_list[index].clone().long()
        prev_action = self.prev_action_list[index].clone().long()
        info = self.infos[index]
        maps = torch.load(self.args.map_dir + info[2] + ".pt")
        if len(inflection) < 26:
            # pad
            pad_amount = 26 - len(inflection)
            inflection = F.pad(inflection, (0, pad_amount), "constant", 0)
            batch_feat = F.pad(batch_feat, (0, 0, 0, pad_amount), "constant", 0)
            next_action = F.pad(next_action, (0, pad_amount), "constant", 0)
            prev_action = F.pad(prev_action, (0, pad_amount), "constant", 0)
            maps = F.pad(maps, (0, 0, 0, 0, 0, 0, 0, pad_amount), "constant", 0)

        return inflection, goal_feat, batch_feat, next_action, prev_action, maps, info

    def __len__(self):
        return len(self.infos)
