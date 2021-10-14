import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import math


class ActionNetwork(torch.nn.Module):
    def __init__(self):
        super(ActionNetwork, self).__init__()
        self.ntokens = 5
        self.dropout_amnt = 0.25
        self.nhid = 256
        self.dropout = nn.Dropout(p=self.dropout_amnt)
        self.encoder = nn.Embedding(self.ntokens, 32)
        self.map_resnet_model = models.resnet18(pretrained=True)
        self.map_resnet_model.avgpool = nn.AvgPool2d(3, stride=1)
        self.map_resnet_model.fc = nn.Linear(512, 128)
        self.rgb_linear = nn.Sequential(nn.Linear(512, 128), nn.ReLU(True))
        self.regression = nn.Sequential(
            nn.Linear(416, 128), nn.ReLU(True), nn.Linear(128, 4)
        )

    def forward(self, src, maps, rgb_curr, rgb_goal):
        src = self.encoder(src) * math.sqrt(self.nhid)
        batch_size, seq_len, H, W, C = maps.size()
        maps = maps.view(batch_size * seq_len, H, W, C)
        maps = maps.permute(0, 3, 1, 2)
        map_feats = self.map_resnet_model(maps.float())
        map_feats = map_feats.view(batch_size, seq_len, -1)
        curr_feats = self.dropout(self.rgb_linear(rgb_curr))
        goal_feats = self.dropout(self.rgb_linear(rgb_goal))
        output = torch.cat((map_feats, curr_feats, goal_feats, src), dim=-1)
        output = F.softmax(self.dropout(self.regression(output)), dim=2)
        return output


class XRN(object):
    def __init__(self, opt):
        self.opt = opt
        self.criterion = nn.CrossEntropyLoss(reduction="none")
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = ActionNetwork()
        self.model.to(self.device)
        self.learning_rate = 0.0001
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4
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

    def train_emb(
        self,
        inflection_weight,
        goal_feat,
        batch_feat,
        next_action,
        prev_action,
        maps,
        *args
    ):
        self.optimizer.zero_grad()
        # import ipdb

        # ipdb.set_trace()
        # get model output
        output = self.model(
            prev_action.to(self.device),
            maps.to(self.device),
            batch_feat.to(self.device),
            goal_feat.to(self.device, dtype=torch.float),
        ).permute(0, 2, 1)

        # losses
        loss = self.criterion(output, next_action.to(self.device, dtype=torch.int64))
        loss = (loss * inflection_weight.to(self.device)).sum() / inflection_weight.to(
            self.device
        ).sum()
        loss.backward()
        self.optimizer.step()

        # acc
        batch_size, seq_len = next_action.size()
        predicted_actions = torch.max(output.detach().cpu(), 1)[1].view(
            batch_size * seq_len
        )
        next_action = next_action.detach().cpu().view(batch_size * seq_len)
        acc = (
            torch.eq(next_action, predicted_actions).sum()
            * 1.0
            / (batch_size * seq_len)
        )
        ratios = []
        for i in range(4):
            r = torch.eq(predicted_actions, i).sum() * 1.0 / (batch_size * seq_len)
            ratios.append(r)
        return loss.item(), acc, ratios

    def eval_emb(
        self,
        inflection_weight,
        goal_feat,
        batch_feat,
        next_action,
        prev_action,
        maps,
        *args
    ):
        # get model output
        output = self.model(
            prev_action.to(self.device),
            maps.to(device=self.device, dtype=torch.float),
            batch_feat.to(self.device),
            goal_feat.to(self.device),
        ).permute(0, 2, 1)
        # losses
        loss = self.criterion(output, next_action.to(self.device, dtype=torch.int64))
        loss = (loss * inflection_weight.to(self.device)).sum() / inflection_weight.to(
            self.device
        ).sum()

        # acc
        batch_size, seq_len = next_action.size()
        predicted_actions = torch.max(output.detach().cpu(), 1)[1].view(
            batch_size * seq_len
        )
        next_action = next_action.detach().cpu().view(batch_size * seq_len)
        acc = (
            torch.eq(next_action, predicted_actions).sum()
            * 1.0
            / (batch_size * seq_len)
        )
        return loss.item(), acc
