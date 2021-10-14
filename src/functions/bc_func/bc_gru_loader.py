import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import msgpack_numpy


class Loader:
    def __init__(self, args):
        self.datasets = {}
        self.args = args
        self.bias = 5
        self.stop_bias = 15

    def load_examples(self, splitScans):
        batch_size = 26
        goal_feats = []
        batch_feats = []
        prev_action_list = []
        next_action_list = []
        inflections = []

        infos = []  # [floor,scan_name,traj,n1,n2]
        for scan in splitScans:
            directory = self.args.base_dir + self.args.action_dir + scan
            trajFile = directory + "_gru_action_data.msg"
            featFile = directory + "_gru_action_feats.msg"
            trajs = msgpack_numpy.unpack(open(trajFile, "rb"), raw=False)
            feats = msgpack_numpy.unpack(open(featFile, "rb"), raw=False)
            for d in trajs:
                trajectory = d["traj"]
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

        inflections = pad_sequence(inflections).transpose(0, 1)
        next_action_list = pad_sequence(next_action_list).transpose(0, 1)
        prev_action_list = pad_sequence(prev_action_list).transpose(0, 1)
        batch_feats = pad_sequence(batch_feats).transpose(0, 1)
        return (
            inflections,
            goal_feats,
            batch_feats,
            next_action_list,
            prev_action_list,
            infos,
        )

    def build_dataset(self, split, dataset, sample_used):
        print("Loading {} dataset...".format(dataset))
        splitFile = self.args.data_splits + "scenes_" + split + ".txt"
        splitScans = [x.strip() for x in open(splitFile, "r").readlines()]
        print("[{}]: Using {} houses".format(split, splitScans))

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
        return inflection, goal_feat, batch_feat, next_action, prev_action, info

    def __len__(self):
        return len(self.infos)
