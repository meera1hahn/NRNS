import torch
import torch.nn as nn
from habitat_sim.utils.common import quat_from_angle_axis
from habitat_sim.geo import UP
import numpy as np
import quaternion
from src.utils.sim_utils import se3_to_mat


class GoalMLP(torch.nn.Module):
    def __init__(self):
        super(GoalMLP, self).__init__()
        self.f1 = nn.Sequential(nn.Linear(512, 512), nn.ReLU(True))
        self.regression = nn.Sequential(
            nn.Linear(512, 512), nn.ReLU(True), nn.Linear(512, 256), nn.ReLU(True)
        )
        self.dist_predict = torch.nn.Linear(256, 1)
        self.rot_predict = torch.nn.Linear(256, 1)
        self.dropout = nn.Dropout(0.5)
        self.initialize()

    def initialize(self):
        nn.init.xavier_uniform_(self.dist_predict.weight.data)
        self.dist_predict.bias.data.zero_()
        nn.init.xavier_uniform_(self.rot_predict.weight.data)
        self.rot_predict.bias.data.zero_()

    def forward(self, x1, x2):
        x = self.dropout(self.regression(torch.sub(self.f1(x1), self.f1(x2)).squeeze()))
        dist_output = self.dist_predict(x)
        rot_output = self.rot_predict(x)
        return dist_output, rot_output


class XRN(object):
    def __init__(self, opt):
        self.opt = opt
        self.dist_criterion = torch.nn.SmoothL1Loss()
        self.rot_criterion = torch.nn.SmoothL1Loss()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = GoalMLP()
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

    def eval_aux(
        self,
        rot_output,
        dist_output,
        true_phi,
        dist_score,
        start_poses,
        start_rots,
        goal_poses,
    ):
        goal_err = []
        for phi, rho, start_pos, start_rot, goal_pos in zip(
            rot_output, dist_output, start_poses, start_rots, goal_poses
        ):
            stateA = se3_to_mat(
                quaternion.from_float_array(start_rot),
                np.asarray(start_pos),
            )
            stateB = (
                stateA
                @ se3_to_mat(
                    quat_from_angle_axis(phi, UP),
                    np.asarray([0, 0, 0]),
                )
                @ se3_to_mat(
                    quaternion.from_float_array([1, 0, 0, 0]),
                    np.asarray([0, 0, -1 * rho]),
                )
            )
            final_pos = stateB[0:3, 3]
            goal_err.append(np.linalg.norm(goal_pos - final_pos))
        return np.mean(goal_err)

    def eval_emb(
        self,
        batch_node1,
        batch_node2,
        true_phi,
        angle_encoding,
        dist_score,
        start_poses,
        start_rots,
        goal_poses,
        *args
    ):
        # get model output
        dist_output, rot_output = self.model(
            batch_node1.to(self.device), batch_node2.to(self.device)
        )

        # losses
        dist_output = dist_output.squeeze(1)
        dist_loss = self.dist_criterion(
            dist_output,
            dist_score.to(self.device),
        )

        rot_output = rot_output.squeeze(1)
        rot_loss = self.rot_criterion(
            rot_output,
            true_phi.to(self.device),
        )
        losses = [dist_loss.item(), rot_loss.item()]
        loss = dist_loss + rot_loss

        # err, acc, switch ratio
        location_err = self.eval_aux(
            rot_output.detach().cpu(),
            dist_output.detach().cpu(),
            true_phi,
            dist_score,
            start_poses,
            start_rots,
            goal_poses,
        )

        phis = np.rad2deg(rot_output.cpu().detach().numpy())
        return (
            loss.item(),
            losses,
            location_err,
            dist_output.cpu().detach().numpy(),
            phis,
        )

    def train_emb(
        self,
        batch_node1,
        batch_node2,
        true_phi,
        angle_encoding,
        dist_score,
        start_poses,
        start_rots,
        goal_poses,
        *args
    ):

        self.optimizer.zero_grad()
        # get model output
        dist_output, rot_output = self.model(
            batch_node1.to(self.device), batch_node2.to(self.device)
        )

        # losses
        dist_output = dist_output.squeeze(1)
        dist_loss = self.dist_criterion(
            dist_output.float(),
            dist_score.to(self.device).float(),
        )
        rot_output = rot_output.squeeze(1)
        rot_loss = self.rot_criterion(
            rot_output,
            true_phi.to(self.device),
        )
        losses = [dist_loss.item(), rot_loss.item()]
        loss = dist_loss + rot_loss
        loss.backward()
        self.optimizer.step()

        # err, acc, switch ratio
        location_err = self.eval_aux(
            rot_output.detach().cpu(),
            dist_output.detach().cpu(),
            true_phi,
            dist_score,
            start_poses,
            start_rots,
            goal_poses,
        )
        phis = np.rad2deg(rot_output.cpu().detach().numpy())
        return (
            loss.item(),
            losses,
            location_err,
            dist_output.cpu().detach().numpy(),
            phis,
        )
