import numpy as np
import torch


class Node(object):
    def __init__(self, idname, pos, rot, feat):
        self.is_start = False
        self.nodeid = idname
        self.pos = pos
        if isinstance(rot, list):
            self.rot = rot
        else:
            self.rot = [rot]
        if isinstance(feat, list):
            self.feat = feat
        else:
            self.feat = [feat]

        self.children = []

    def add_rot(self, rot, feat):
        if rot not in self.rot:
            self.rot.append(rot)
            self.feat.append(feat)

    def set_to_start(self):
        self.is_start = True

    def __lt__(self, other):
        return int(self.nodeid) < int(other.nodeid)


class Edge(object):
    def __init__(self, node1, node2, weight):
        self.nodes = (node1, node2)
        self.ids = (node1.nodeid, node2.nodeid)
        self.weight = weight


class GraphMap(object):
    def __init__(self, params):
        self.params = params
        self.frame_size = params["frame_size"]
        self.frame_height = params["feat_size"]

        self.nodes = set()
        self.node_by_id = {}
        self.poses = set()
        self.pose_to_id = {}
        self.edges = set()
        self.clustered = False
        self.clusterType = None
        self.graph_axes = None

    def set_axes(self):
        if len(self.nodes) == 0:
            self.graph_axes = {"x": [0, 1], "y": [0, 1]}
            return
        self.graph_axes = {}
        self.graph_axes["x"] = [
            min([i[0] for i in self.poses]) - 1,
            max([i[0] for i in self.poses]) + 1,
        ]
        self.graph_axes["y"] = [
            min([i[2] for i in self.poses]) - 1,
            max([i[2] for i in self.poses]) + 1,
        ]

    def total_nodes(self):
        return len(self.nodes)

    def get_node_by_id(self, nodeid):
        return self.node_by_id[nodeid]

    def get_node_by_pos(self, pos):
        pos = tuple([round(x, 4) for x in pos])
        return self.node_by_id[self.pose_to_id[pos]]

    def add_single_node(self, pos, rot, feat):
        pos = tuple([round(x, 4) for x in pos])
        if pos in self.poses:
            nodeid = self.pose_to_id[pos]
            node = self.node_by_id[nodeid]
            if rot in node.rot:
                return node

        nodeid = str(self.total_nodes())
        node = Node(nodeid, pos, rot, feat)
        self.nodes.add(node)
        self.node_by_id[nodeid] = node
        self.poses.add(pos)
        self.pose_to_id[pos] = nodeid
        return node

    def add_complex_node(self, pos, rots, feats):
        pos = tuple([round(x, 4) for x in pos])
        if pos in self.poses:
            nodeid = self.pose_to_id[pos]
            node = self.node_by_id[nodeid]
            return node

        nodeid = str(self.total_nodes())
        node = Node(nodeid, pos, rots[0], feats[0])
        for enum, x in enumerate(zip(rots, feats)):
            rot, feat = x
            if enum == 0:
                continue
            node.add_rot(rot, feat)

        self.nodes.add(node)
        self.node_by_id[nodeid] = node
        self.poses.add(pos)
        self.pose_to_id[pos] = nodeid
        return node

    def add_edge(self, node1, node2):
        if node1 not in self.nodes or node2 not in self.nodes:
            print("error one or more of the nodes not in the graph")
            return
        distance = np.linalg.norm(np.asarray(node1.pos) - np.asarray(node2.pos))
        edge = Edge(node1, node2, distance)
        self.edges.add(edge)
        node1.children.append(node2)
        node2.children.append(node1)

    def build_graph(self, episode, feats):
        """Build a graph from the rgb trajectory"""
        currNode = None
        lastNode = None
        poses = episode["poses"]
        rotations = episode["rotations"]
        for p, r, feat in zip(poses, rotations, feats):
            currNode = self.add_single_node(p, r, feat)
            if lastNode != None:
                self.add_edge(lastNode, currNode)
            else:
                currNode.set_to_start()
            lastNode = currNode


def affinity_cluster(G):
    # print('afinity propagation on G')

    """get feats for each node and cluster using sklearn kmeans"""
    X = []
    for i in range(len(G.nodes)):
        node = G.node_by_id[str(i)]
        feat = node.feat[0].detach().numpy()
        pos = node.pos
        full_feat = np.concatenate((feat, pos), axis=0)
        X.append(full_feat)
    from sklearn.cluster import AffinityPropagation

    clustering = AffinityPropagation(random_state=5).fit(X)
    labels = clustering.labels_

    """create new graph"""
    clusteredGraph = GraphMap(params=G.params)
    if G.graph_axes is None:
        G.set_axes()
    clusteredGraph.graph_axes = G.graph_axes

    """create the centriod nodes and add them to the new graph"""
    node_clusters = {i: [] for i in range(max(labels) + 1)}
    for i in range(len(G.nodes)):
        node = G.node_by_id[str(i)]
        node_clusters[labels[i]].append(node)

    clusteredGraph = centriod_from_cluster(clusteredGraph, node_clusters)
    clusteredGraph = add_edges_to_new_graph(G, clusteredGraph, labels)

    clusteredGraph.clustered = True
    clusteredGraph.clusterType = "affinity"
    return clusteredGraph


def add_edges_to_new_graph(G, clusteredGraph, labels):
    edgeList = []
    for edge in G.edges:
        id1 = str(int(labels[int(edge.ids[0])]))
        id2 = str(int(labels[int(edge.ids[1])]))
        if id1 == id2:
            continue
        if {id1, id2} in edgeList:
            continue
        edgeList.append({id1, id2})
        node1 = clusteredGraph.node_by_id[id1]
        node2 = clusteredGraph.node_by_id[id2]
        clusteredGraph.add_edge(node1, node2)

    for i in range(len(clusteredGraph.nodes) - 2):
        id1 = str(i)
        id2 = str(i + 1)
        if {id1, id2} in edgeList:
            continue
        edgeList.append({id1, id2})
        node1 = clusteredGraph.node_by_id[id1]
        node2 = clusteredGraph.node_by_id[id2]
        clusteredGraph.add_edge(node1, node2)

    return clusteredGraph


def centriod_from_cluster(clusteredGraph, node_clusters):
    mean = False
    for label in range(len(node_clusters)):
        cluster = node_clusters[label]
        middle = round(len(cluster) * 1.0 / 2)
        # mean
        if mean:
            centriod_pos = np.mean([x.pos for x in cluster], axis=0)
            centriod_rot = cluster[middle].rot
            centriod_feat = [x for n in cluster for x in n.feat]
            centriod_feat = torch.mean(torch.stack(centriod_feat), dim=0)
        # center
        else:
            centriod_pos = cluster[middle].pos
            centriod_rot = cluster[middle].rot
            centriod_feat = cluster[middle].feat[0]

        added_node = clusteredGraph.add_single_node(
            centriod_pos, centriod_rot, centriod_feat
        )
        has_start = [x.is_start for x in cluster]
        if True in has_start:
            added_node.set_to_start()

    return clusteredGraph
