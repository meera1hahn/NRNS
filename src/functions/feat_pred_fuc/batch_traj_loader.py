import numpy as np
import msgpack_numpy
import torch
from torch_geometric.data import Data


class Loader:
    def __init__(self, args):
        self.args = args
        np.random.seed(args.seed)
        self.datasets = {}
        self.feat_size = 512
        self.edge_feat_size = 16

    def geo_dist(self, ind1, goalind, data_obj):
        geodesic = 0
        if ind1 <= goalind:
            ranges = range(ind1, goalind)
        else:
            ranges = range(goalind, ind1)
        for i in ranges:
            geodesic += np.linalg.norm(
                np.asarray(data_obj["nodes"]["pos"][i])
                - np.asarray(data_obj["nodes"]["pos"][i + 1])
            )
        return geodesic

    def geo_dist_valid(self, index, goalind, node_pos, data_obj):
        geodesic = 0
        if index == goalind:
            geodesic += np.linalg.norm(
                np.asarray(node_pos[-1]) - np.asarray(data_obj["nodes"]["pos"][goalind])
            )
        elif index < goalind:
            geodesic += np.linalg.norm(
                np.asarray(node_pos[-1])
                - np.asarray(data_obj["nodes"]["pos"][index + 1])
            )
            for i in range(index + 1, goalind):
                geodesic += np.linalg.norm(
                    np.asarray(data_obj["nodes"]["pos"][i])
                    - np.asarray(data_obj["nodes"]["pos"][i + 1])
                )
        else:
            geodesic += np.linalg.norm(
                np.asarray(node_pos[-1])
                - np.asarray(data_obj["nodes"]["pos"][index - 1])
            )
            for i in range(goalind, index - 1):
                geodesic += np.linalg.norm(
                    np.asarray(data_obj["nodes"]["pos"][i])
                    - np.asarray(data_obj["nodes"]["pos"][i + 1])
                )
        return geodesic

    def true_node_to_point(self, index, valid_points, goal_index, data_obj):
        """add graph info and valid node info"""
        node_ids = np.asarray(data_obj["nodes"]["ids"][0:index])
        node_pos = np.asarray(data_obj["nodes"]["pos"][0:index])
        node_feat = np.asarray(data_obj["nodes"]["feat"][0:index])

        adj_matrix = []
        edge_feats = []
        edge_pos = []
        for i in range(0, len(data_obj["edges"]), 2):
            edge1 = data_obj["edges"][i]
            if edge1[0] < index and edge1[1] < index:
                adj_matrix.append(edge1)
                edge_feats.append(data_obj["edge_attrs"][i])
                edge_pos.append([node_pos[int(edge1[0])], node_pos[int(edge1[1])]])

        # goal image and distances
        labels = []
        goal_feat = np.asarray(data_obj["nodes"]["feat"][goal_index])
        for ind in range(0, index):
            dist_score = (
                1 - min(self.geo_dist(ind, goal_index, data_obj) ** 2, 10.0) / 10.0
            )
            labels.append(dist_score)

        # valid_point
        pred_feat = []
        for v in range(0, len(valid_points)):
            valid_point = valid_points[v]
            new_index = index + v
            prev_index = int(valid_point[0])
            prev_pos = data_obj["nodes"]["pos"][prev_index]
            node_ids = np.append(node_ids, new_index)
            temp_pos = np.expand_dims(valid_point[1][0:3, 3], axis=0)
            node_pos = np.append(node_pos, temp_pos, axis=0)
            node_feat = np.append(node_feat, np.zeros((1, self.feat_size)), axis=0)
            edge_feats.append(valid_point[2])
            adj_matrix.append([prev_index, new_index])
            edge_pos.append([prev_pos, valid_point[1][0:3, 3]])
            valid_geo = self.geo_dist_valid(index, goal_index, node_pos, data_obj)
            valid_geo = 1 - min(valid_geo ** 2, 10.0) / 10.0
            labels.append(valid_geo)
            pred_feat.append(new_index)

        adj_matrix = self.create_adj_matrix(adj_matrix)
        edge_feats = torch.flatten(
            torch.tensor(edge_feats, dtype=torch.float), start_dim=1
        )
        edge_pos = torch.tensor(edge_pos)

        # create torch geometric object
        geo_data = Data(
            x=torch.tensor(node_feat, dtype=torch.float),
            edge_index=adj_matrix,
            edge_attr=edge_feats,
            edge_pos=edge_pos,
            pos=torch.tensor(node_pos, dtype=torch.float),
            goal_feat=torch.tensor(goal_feat, dtype=torch.float),
            pred_feat=torch.tensor(pred_feat),
            y=torch.tensor(labels),
            num_nodes=len(node_ids),
        )

        return geo_data

    def load_node_info(self, data):
        data_list = []

        for d in data:
            for data_obj in d:
                if len(data_obj["nodes"]["ids"]) < 10:
                    continue
                # graph starts with at least 2 nodes
                start_node_inx = 0
                end = len(data_obj["nodes"]["ids"]) - 1
                for i in range(start_node_inx + 1, end, 1):
                    length = len(data_obj["valid_points"][str(i - 1)])
                    if length > 1:
                        last_valid = data_obj["valid_points"][str(i - 1)]
                    else:
                        continue
                    # goal index can be anywhere between one behind to anywhere infront
                    goal_index = np.random.randint(
                        max(i - 2, start_node_inx + 1), min(i + 5, end + 1)
                    )
                    geo_data = self.true_node_to_point(
                        i, last_valid, goal_index, data_obj
                    )
                    data_list.append(geo_data)
        return data_list

    def create_adj_matrix(self, edge_list):
        edge_index = torch.tensor(edge_list, dtype=torch.long)
        edge_index = edge_index.t().contiguous()
        return edge_index

    def build_dataset(self, split):
        splitFile = self.args.data_splits + "scenes_" + split + ".txt"
        splitScans = [x.strip() for x in open(splitFile, "r").readlines()]
        data = []
        for house in splitScans:
            houseFile = self.args.clustered_graph_dir + house + "_graphs.msg"
            data.append(msgpack_numpy.unpack(open(houseFile, "rb"), raw=False))

        data_size = len(data)
        print("[{}]: Using {} houses".format("data", data_size))
        data_list = self.load_node_info(data)

        self.datasets[split] = Mp3dDataset(data_list)


class Mp3dDataset(Data):
    def __init__(
        self,
        data_list,
    ):
        self.data_list = data_list

    def __getitem__(self, index):
        data = self.data_list[index]
        return data

    def __len__(self):
        return len(self.data_list)
