import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import math


class SwitchMLP(torch.nn.Module):
    def __init__(self):
        super(SwitchMLP, self).__init__()
        self.regression = nn.Sequential(nn.Linear(1024, 512), nn.ReLU(True))
        self.dist_predict = torch.nn.Linear(512, 1)
        self.rot_predict = torch.nn.Linear(512, 2) 
        self.switch_predict = torch.nn.Linear(512, 1) 
        self.dropout = nn.Dropout(0.5)
        self.initialize()

    def initialize(self):
        nn.init.xavier_uniform_(self.dist_predict.weight.data)
        self.dist_predict.bias.data.zero_()
        nn.init.xavier_uniform_(self.rot_predict.weight.data)
        self.rot_predict.bias.data.zero_()
        nn.init.xavier_uniform_(self.switch_predict.weight.data)
        self.switch_predict.bias.data.zero_()

    def forward(self, x1, x2):
        x = self.dropout(self.regression(torch.cat((x1, x2), dim=1).squeeze()))
        dist_output = torch.sigmoid(self.dist_predict(x))
        rot_output = self.rot_predict(x)
        switch_output = torch.sigmoid(self.switch_predict(x))
        return (switch_output, dist_output, rot_output)


class XRN(object):
    def __init__(self, opt):
        self.opt = opt
        self.switch_criterion = nn.BCELoss()
        self.dist_criterion = torch.nn.SmoothL1Loss()
        self.rot_criterion = torch.nn.SmoothL1Loss()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = SwitchMLP()
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            self.model = nn.DataParallel(self.model)
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

    def eval_aux(self, rot_output, delta_rot, dist_output, delta_dist):
        rot_err = []
        for output_ang, target_ang in zip(rot_output, delta_rot):
            output_ang = math.degrees(math.atan2(output_ang[1], output_ang[0]))
            target_ang = math.degrees(math.atan2(target_ang[1], target_ang[0]))
            if output_ang < 0:
                output_ang = 360 + output_ang
            if target_ang < 0:
                target_ang = 360 + target_ang
            d = abs(output_ang - target_ang) % 360
            if d > 180:
                d = 360 - d
            rot_err.append(abs(d))
        dist_err = []
        for pred, true in zip(dist_output, delta_dist):
            true = self.opt.dist_max * (1 - true)
            pred = self.opt.dist_max * (1 - pred)
            dist_err.append(abs(true - pred))
        return np.mean(rot_err), np.mean(dist_err)

    def eval_emb(self, switch, batch_node1, batch_node2, angle_diff, delta_dist, *args):
        # get model output
        switch_output, dist_output, rot_output = self.model(
            batch_node1.to(self.device), batch_node2.to(self.device)
        )

        # losses
        switch_loss = self.switch_criterion(
            switch_output.squeeze(1), switch.to(self.device)
        )

        switches = switch.cpu().detach().numpy().nonzero()[0]
        dist_output = dist_output.squeeze(1)[switches]
        dist_loss = self.dist_criterion(
            dist_output,
            delta_dist[switches].to(self.device),
        )
        rot_output = rot_output.squeeze(1)[switches]
        rot_loss = self.rot_criterion(
            rot_output,
            angle_diff[switches].to(self.device),
        )
        losses = [dist_loss.item(), switch_loss.item(), rot_loss.item()]
        loss = switch_loss + dist_loss + rot_loss

        # err, acc, switch ratio
        location_err = self.eval_aux(
            rot_output.detach().cpu(),
            angle_diff[switches],
            dist_output.detach().cpu(),
            delta_dist[switches],
        )
        switch_output = np.round(switch_output.squeeze(1).detach().cpu())
        switch_acc = (
            torch.eq(switch.detach().cpu(), switch_output).sum()
            * 1.0
            / switch.size()[0]
        )
        true_ratio = switch.detach().cpu().sum() * 1.0 / switch.size()[0]
        pred_ratio = switch_output.sum() * 1.0 / switch_output.size()[0]

        return loss.item(), switch_acc, true_ratio, pred_ratio, location_err

    def train_emb(
        self, switch, batch_node1, batch_node2, angle_diff, delta_dist, *args
    ):
        self.optimizer.zero_grad()

        # get model output
        switch_output, dist_output, rot_output = self.model(
            batch_node1.to(self.device), batch_node2.to(self.device)
        )
        # losses
        switch_loss = self.switch_criterion(
            switch_output.squeeze(1), switch.to(self.device)
        )

        switches = switch.cpu().detach().numpy().nonzero()[0]
        dist_output = dist_output.squeeze(1)[switches]
        dist_loss = self.dist_criterion(
            dist_output,
            delta_dist[switches].to(self.device),
        )
        rot_output = rot_output[switches]
        rot_loss = self.rot_criterion(
            rot_output,
            angle_diff[switches].to(self.device),
        )
        losses = [dist_loss.item(), switch_loss.item(), rot_loss.item()]
        loss = switch_loss + dist_loss + rot_loss
        loss.backward()
        self.optimizer.step()

        # err, acc, switch ratio
        location_err = self.eval_aux(
            rot_output.detach().cpu(),
            angle_diff[switches],
            dist_output.detach().cpu(),
            delta_dist[switches],
        )
        switch_output = np.round(switch_output.squeeze(1).detach().cpu())
        switch_acc = (
            torch.eq(switch.detach().cpu(), switch_output).sum()
            * 1.0
            / switch.size()[0]
        )
        true_ratio = switch.detach().cpu().sum() * 1.0 / switch.size()[0]
        pred_ratio = switch_output.sum() * 1.0 / switch_output.size()[0]

        return losses, switch_acc, true_ratio, pred_ratio, location_err
