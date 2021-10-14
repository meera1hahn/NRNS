import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import DataParallel, GCNConv, CGConv
from torch.nn import Linear as Lin, ReLU
from src.functions.feat_pred_fuc.layers import AttentionConv


class TopoGCN(nn.Module):
    def __init__(self, input_size=512, hidden_size=256, pos_size=3, edge_size=16):
        super(TopoGCN, self).__init__()
        """v1: graph attention layers"""
        self.conv1 = AttentionConv(input_size, dropout=0.6)
        self.conv2 = AttentionConv(input_size, dropout=0.6)
        self.distance_layers = nn.Sequential(
            Lin(2 * input_size, input_size),
            ReLU(True),
            Lin(input_size, 1),
        )

        """v2: graph conv layers"""
        self.conv1 = CGConv(
            channels=input_size, dim=edge_size, batch_norm="add", bias=True
        )
        self.conv2 = CGConv(
            channels=input_size, dim=edge_size, batch_norm="add", bias=True
        )
        self.conv3 = GCNConv(input_size, hidden_size)
        self.conv4 = GCNConv(hidden_size, hidden_size)
        self.distance_layers = nn.Sequential(
            Lin(hidden_size + input_size, input_size),  
            ReLU(True),
            Lin(input_size, 1),  
        )

    """v1: graph attention layers"""
    # def forward(self, data):
    #     num_nodes = data.batch.size()[0]
    #     x = F.relu(self.conv1(data.x, data.edge_index, data.edge_attr))
    #     x = F.dropout(x, training=self.training)
    #     x = F.relu(self.conv2(x, data.edge_index, data.edge_attr))
    #     x = F.dropout(x, training=self.training)
    #     pred_dist = self.distance_layers(
    #         torch.cat((x, data.goal_feat.repeat(num_nodes, 1)), dim=1)
    #     )
    #     pred_dist = F.sigmoid(
    #         pred_dist
    #     )  # distance score is between 0 - 1 but represents 0m - 10m+
    #     return pred_dist


    """v2: graph conv layers"""
    def forward(self, data):
        num_nodes = data.batch.size()[0]
        x = F.relu(self.conv1(data.x, data.edge_index, data.edge_attr))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.conv2(x, data.edge_index, data.edge_attr))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.conv3(x, data.edge_index))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.conv4(x, data.edge_index))
        pred_dist = self.distance_layers(
            torch.cat((x, data.goal_feat.repeat(num_nodes, 1)), dim=1)
        )

        return pred_dist


class XRN(object):
    def __init__(self, opt):
        self.dist_criterion = torch.nn.MSELoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TopoGCN()
        self.model = DataParallel(self.model)
        self.model = self.model.to(self.device)
        self.learning_rate = 0.0001
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), self.learning_rate, betas=(0.9, 0.999)
        )
        self.my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer, gamma=0.96
        )

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)

    def set_model(self, model):
        self.model = model

    def get_model(self):
        return self.model

    def train_start(self):
        self.model.train()

    def eval_start(self):
        self.model.eval()

    def calculate_error(self, output, dist_score):
        error = []

        for pred, true in zip(output, dist_score):
            error.append(abs(true - pred).item())
        acc = np.mean(np.where(np.asarray(error) <= 0.1, 1, 0))
        error = np.mean(error)
        return error, acc

    def top_k_acc(self, output, dist_score):
        if output.argmax() == dist_score.argmax():
            return 1
        return 0

    def eval_emb(self, data_list, *args):
        with torch.no_grad():
            pred_dist = self.model(data_list)
            batch_labels = (
                torch.cat([data.y for data in data_list])
                .to(self.device)
                .to(torch.float32)
            )
            loss = self.dist_criterion(
                pred_dist.squeeze(1)[data_list[0].pred_feat],
                batch_labels[data_list[0].pred_feat],
            )

            predictions = pred_dist.cpu().detach().squeeze(1)[data_list[0].pred_feat]
            labels = batch_labels.cpu().detach()[data_list[0].pred_feat]
            error, acc = self.calculate_error(predictions, labels)
            topk_acc = self.top_k_acc(predictions, labels)

        return loss.item(), error, acc, topk_acc

    def train_emb(self, data_list, *args):
        self.optimizer.zero_grad()
        pred_dist = self.model(data_list)
        batch_labels = (
            torch.cat([data.y for data in data_list]).to(self.device).to(torch.float32)
        )
        loss = self.dist_criterion(
            pred_dist.squeeze(1)[data_list[0].pred_feat],
            batch_labels[data_list[0].pred_feat],
        )
        loss.backward()
        self.optimizer.step()
        predictions = pred_dist.cpu().detach().squeeze(1)[data_list[0].pred_feat]
        labels = batch_labels.cpu().detach()[data_list[0].pred_feat]
        error, acc = self.calculate_error(predictions, labels)
        topk_acc = self.top_k_acc(predictions, labels)
        return loss.item(), error, acc, topk_acc