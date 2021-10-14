import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class ActionNetwork(torch.nn.Module):
    def __init__(self):
        super(ActionNetwork, self).__init__()
        self.type = "gru"
        self.dropout_amnt = 0.2
        ntokens = 5
        self.nhid = 256
        nlayers = 2
        self.f1 = nn.Sequential(nn.Linear(512, 256), nn.ReLU(True))
        self.dropout = nn.Dropout(self.dropout_amnt)
        self.regression = nn.Sequential(nn.Linear(544, 4))
        self.encoder = nn.Embedding(ntokens, 32)
        self.gru = torch.nn.GRU(544, 544, num_layers=nlayers, dropout=self.dropout_amnt)

    def forward(self, src, x1, x2):
        src = self.encoder(src) * math.sqrt(self.nhid)
        output = torch.cat((self.f1(x1), self.f1(x2), src), 2)
        output, hidden = self.gru(output)
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
        self, inflection_weight, goal_feat, batch_feat, next_action, prev_action, *args
    ):
        self.optimizer.zero_grad()
        # get model output
        output = self.model(
            prev_action.to(self.device),
            batch_feat.to(self.device),
            goal_feat.to(self.device),
        ).permute(0, 2, 1)

        # losses
        loss = self.criterion(output, next_action.to(self.device, dtype=torch.int64))
        loss = (loss * inflection_weight.to(self.device)).sum() / inflection_weight.to(
            self.device
        ).sum()
        loss.backward()
        self.optimizer.step()

        # acc
        predicted_actions = torch.max(output.detach().cpu(), 1)[1]
        acc = (
            torch.eq(next_action.detach().cpu(), predicted_actions).sum()
            * 1.0
            / next_action.size()[0]
        )
        return loss.item(), 0.5

    def eval_emb(
        self, inflection_weight, goal_feat, batch_feat, next_action, prev_action, *args
    ):
        # get model output
        output = self.model(
            prev_action.to(self.device),
            batch_feat.to(self.device),
            goal_feat.to(self.device),
        ).permute(0, 2, 1)
        # losses
        loss = self.criterion(output, next_action.to(self.device, dtype=torch.int64))
        loss = (loss * inflection_weight.to(self.device)).sum() / inflection_weight.to(
            self.device
        ).sum()

        # acc
        predicted_actions = torch.max(output.detach().cpu(), 1)[1]
        acc = (
            torch.eq(next_action.detach().cpu(), predicted_actions).sum()
            * 1.0
            / next_action.size()[0]
        )
        return loss.item(), acc
